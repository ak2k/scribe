from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import subprocess
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import typer
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from scribe.cli import app
from scribe.errors import ExternalServiceError
from scribe.parakeet import DEFAULT_TIMEOUT_S, ParakeetMlx, parse_output
from scribe.schema import Engine, Source, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable

UVX = "/opt/nowhere/bin/uvx"
SOURCE = Source(kind="audio", ref="clip.wav")
runner = CliRunner()


def _token(text: str, start: float, end: float) -> dict[str, object]:
    return {"text": text, "start": start, "end": end, "duration": end - start, "confidence": 0.9}


def _sentence(*tokens: dict[str, object]) -> dict[str, object]:
    return {
        "text": "".join(str(token["text"]) for token in tokens),
        **{key: 0.0 for key in ("start", "end", "duration")},
        "confidence": 0.9,
        "tokens": list(tokens),
    }


def _output(*sentences: dict[str, object]) -> str:
    return json.dumps({"text": "".join(str(s["text"]) for s in sentences), "sentences": sentences})


SAMPLE = _output(
    _sentence(_token(" Hel", 0.0, 0.2), _token("lo", 0.2, 0.4), _token(" there.", 0.5, 0.9)),
    # Opens a word without a leading space; a lone space opens an empty one
    # that the next piece fills, and a lone space before a boundary is dropped.
    _sentence(_token(".", 0.7, 0.8), _token(" ", 1.3, 1.4), _token("ok", 1.4, 1.6)),
    _sentence(_token(" So", 2.0, 2.2), _token(" ", 2.3, 2.4), _token(" n", 2.5, 2.6)),
)
SAMPLE_WORDS = [
    Word(text="Hello", start=0.0, end=0.4),
    Word(text="there.", start=0.5, end=0.9),
    Word(text=".", start=0.7, end=0.8),
    Word(text="ok", start=1.3, end=1.6),
    Word(text="So", start=2.0, end=2.2),
    Word(text="n", start=2.5, end=2.6),
]


class FakeParakeet:
    """Stands in for the child: writes `body` where the argv tells the tool to."""

    def __init__(
        self,
        body: str | None = SAMPLE,
        *,
        returncode: int = 0,
        stdout: str = "",
        error: Exception | None = None,
    ) -> None:
        self.body, self.returncode, self.stdout, self.error = body, returncode, stdout, error
        self.calls: list[list[str]] = []
        self.env: dict[str, str] = {}
        self.timeout = 0.0
        self.workdir = Path()

    def run(
        self, argv: list[str], *, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        self.env, self.timeout = env, timeout
        self.workdir = Path(argv[argv.index("--output-dir") + 1])
        if self.error is not None:
            raise self.error
        if self.body is not None:
            name = argv[argv.index("--output-template") + 1]
            (self.workdir / f"{name}.json").write_text(self.body, encoding="utf-8")
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, "uvx: resolved")


def _found(_name: str) -> str | None:
    return UVX


def _backend(
    fake: FakeParakeet,
    *,
    host: tuple[str, str] = ("Darwin", "arm64"),
    which: Callable[[str], str | None] = _found,
) -> ParakeetMlx:
    return ParakeetMlx(run=fake.run, which=which, host=lambda: host)


def _audio(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"pretend this is audio")
    return clip


def test_words_are_rebuilt_from_pieces_within_each_sentence(tmp_path: Path) -> None:
    transcript = _backend(FakeParakeet()).transcribe(_audio(tmp_path), source=SOURCE)

    assert transcript.words == SAMPLE_WORDS
    assert transcript.text == "Hello there. . ok So n"
    assert transcript.source == SOURCE
    assert transcript.engine == Engine(
        name="parakeet-mlx",
        model="mlx-community/parakeet-tdt-0.6b-v3",
        params={"package_version": "0.5.2", "exclude_newer": "2026-09-24T00:00:00Z"},
    )


def test_the_child_runs_the_pinned_tool_with_its_own_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PARAKEET_MAX_WORDS", "3")
    monkeypatch.setenv("PARAKEET_CACHE_DIR", str(tmp_path))
    fake = FakeParakeet()
    clip = _audio(tmp_path)

    _backend(fake).transcribe(clip, source=SOURCE)

    argv = fake.calls[0]
    pinned = [UVX, "--exclude-newer", "2026-09-24T00:00:00Z", "--from", "parakeet-mlx==0.5.2"]
    pinned += ["parakeet-mlx", str(clip.absolute())]
    pinned += ["--model", "mlx-community/parakeet-tdt-0.6b-v3", "--output-format", "json"]
    assert argv == [*pinned, "--output-dir", str(fake.workdir), "--output-template", argv[-1]]
    assert fake.timeout == DEFAULT_TIMEOUT_S
    assert "PARAKEET_MAX_WORDS" not in fake.env
    assert fake.env["PARAKEET_CACHE_DIR"] == str(tmp_path)
    assert int(fake.env["COLUMNS"]) >= 1000
    assert not fake.workdir.exists()


_pieces = st.text(alphabet="ab. \t", max_size=4)
_spans = st.floats(min_value=0, max_value=5)


@st.composite
def _sentences(draw: st.DrawFn) -> dict[str, object]:
    tokens: list[dict[str, object]] = []
    start = 0.0
    for text, gap, length in draw(st.lists(st.tuples(_pieces, _spans, _spans), min_size=1)):
        start += gap
        tokens.append(_token(text, start, start + length))
    return _sentence(*tokens)


@given(st.lists(_sentences(), max_size=4))
def test_every_piece_lands_in_a_word_and_starts_never_decrease(
    sentences: list[dict[str, object]],
) -> None:
    per_sentence = [parse_output(_output(sentence).encode()) for sentence in sentences]

    for sentence, words in zip(sentences, per_sentence, strict=True):
        joined = "".join(word.text for word in words)
        assert "".join(joined.split()) == "".join(str(sentence["text"]).split())
        starts = [word.start for word in words]
        assert starts == sorted(starts)
    flat = [word for words in per_sentence for word in words]
    floors = list(accumulate((word.start for word in flat), max))
    assert parse_output(_output(*sentences).encode()) == [
        Word(text=word.text, start=floor, end=max(word.end, floor))
        for word, floor in zip(flat, floors, strict=True)
    ]


def test_a_word_timed_before_an_earlier_one_keeps_its_place_with_a_clamped_start() -> None:
    raw = _output(
        _sentence(_token(" limit", 5.0, 5.4), _token(" Right?", 6.0, 6.3)),
        _sentence(_token(" So", 1.0, 1.2), _token(" I'm", 1.5, 6.1), _token(" and", 7.0, 7.2)),
    ).encode()

    assert [(word.text, word.start, word.end) for word in parse_output(raw)] == [
        ("limit", 5.0, 5.4),
        ("Right?", 6.0, 6.3),
        ("So", 6.0, 6.0),
        ("I'm", 6.0, 6.1),
        ("and", 7.0, 7.2),
    ]


def _swap(monkeypatch: pytest.MonkeyPatch, backend: ParakeetMlx) -> None:
    monkeypatch.setattr("scribe.cli.ParakeetMlx", lambda: backend)


def test_the_command_writes_beside_the_audio_and_prints_only_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeParakeet()
    _swap(monkeypatch, _backend(fake))
    clip = _audio(tmp_path)

    result = runner.invoke(app, ["parakeet", str(clip)])

    assert result.exit_code == 0, result.output
    written = tmp_path / "clip.parakeet.json"
    assert result.stdout == f"{written}\n"
    assert "can take minutes" in result.stderr
    transcript = Transcript.load(written)
    digest = hashlib.sha256(clip.read_bytes()).hexdigest()
    assert transcript.source == Source(kind="audio", ref=str(clip), sha256=digest)
    assert transcript.words == SAMPLE_WORDS
    assert not (tmp_path / "clip.transcript.json").exists()


def test_out_names_the_file_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _swap(monkeypatch, _backend(FakeParakeet()))
    out = tmp_path / "elsewhere.json"

    result = runner.invoke(app, ["parakeet", str(_audio(tmp_path)), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{out}\n"
    assert Transcript.load(out).words == SAMPLE_WORDS


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_read_only_output_is_refused_before_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeParakeet()
    _swap(monkeypatch, _backend(fake))
    locked = tmp_path / "clip.parakeet.json"
    locked.write_text("kept\n", encoding="utf-8")
    locked.chmod(0o444)
    try:
        result = runner.invoke(app, ["parakeet", str(_audio(tmp_path))])
    finally:
        locked.chmod(0o644)

    assert result.exit_code == 2
    assert "cannot write" in result.stderr
    assert fake.calls == []
    assert locked.read_text(encoding="utf-8") == "kept\n"


_TIMEOUT = subprocess.TimeoutExpired(["uvx"], DEFAULT_TIMEOUT_S)


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (FakeParakeet(returncode=1, stdout="Error loading model: gone"), "exited 1: Error loading"),
        (FakeParakeet(None, stdout="Error transcribing file: no ffmpeg"), "a transcript: Error"),
        (FakeParakeet(""), "exited 0 without writing a transcript: uvx: resolved"),
        (FakeParakeet("[not json"), "unexpected transcript"),
        (FakeParakeet('{"text": "", "sentences": [], "language": "en"}'), "at language"),
        (FakeParakeet(error=_TIMEOUT), f"did not finish within {DEFAULT_TIMEOUT_S:g} s"),
        (FakeParakeet(error=OSError("exec format error")), "cannot run uvx"),
    ],
    ids=["exit-1", "missing", "empty", "not-json", "extra-field", "timeout", "unrunnable"],
)
def test_a_failed_run_exits_2_with_one_line_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeParakeet, expected: str
) -> None:
    _swap(monkeypatch, _backend(fake))
    clip = _audio(tmp_path)

    result = runner.invoke(app, ["parakeet", str(clip)])

    assert result.exit_code == 2
    errors = [line for line in result.stderr.splitlines() if line.startswith("scribe:")]
    assert len(errors) == 1, result.stderr
    assert expected in errors[0]
    assert result.stdout == ""
    assert sorted(tmp_path.iterdir()) == [clip]
    assert not fake.workdir.exists()


@pytest.mark.parametrize(
    ("host", "found", "expected"),
    [
        (("Linux", "x86_64"), UVX, "needs Apple silicon"),
        (("Linux", "aarch64"), UVX, "needs Apple silicon"),
        (("Darwin", "x86_64"), UVX, "needs Apple silicon"),
        (("Darwin", "arm64"), None, "uvx is not on PATH"),
    ],
)
def test_an_unusable_machine_exits_2_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: tuple[str, str],
    found: str | None,
    expected: str,
) -> None:
    fake = FakeParakeet()
    _swap(monkeypatch, _backend(fake, host=host, which=lambda _name: found))

    result = runner.invoke(app, ["parakeet", str(_audio(tmp_path))])

    assert result.exit_code == 2
    assert result.stderr.startswith("scribe: ")
    assert expected in result.stderr
    assert len(result.stderr.splitlines()) == 1
    assert fake.calls == []


def test_the_backend_refuses_output_it_cannot_read() -> None:
    with pytest.raises(ExternalServiceError, match=r"at sentences\.0\.tokens\.0\.start"):
        parse_output(_output(_sentence(_token(" a", float("nan"), 1.0))).encode())


def test_the_help_states_the_timeout() -> None:
    result = runner.invoke(app, ["parakeet", "--help"], terminal_width=200)

    assert result.exit_code == 0
    assert f"stopped after {DEFAULT_TIMEOUT_S / 60:g} minutes" in " ".join(result.stdout.split())


@pytest.mark.parametrize("times_out", [False, True])
def test_the_default_runner_starts_its_own_group_and_kills_that_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, times_out: bool
) -> None:
    started: list[dict[str, object]] = []
    killed: list[tuple[int, int]] = []

    class Child:
        pid, returncode = 4242, 3

        def __init__(self, _argv: list[str], **kwargs: object) -> None:
            started.append(kwargs)

        def __enter__(self) -> Child:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def communicate(self, timeout: float) -> tuple[str, str]:
            if times_out:
                raise subprocess.TimeoutExpired("uvx", timeout)
            return "Error loading model", ""

    def killpg(pgid: int, sig: int) -> None:
        killed.append((pgid, sig))
        # The group may be gone already, which is no failure of its own.
        raise ProcessLookupError

    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(os, "killpg", killpg)
    backend = ParakeetMlx(which=_found, host=lambda: ("Darwin", "arm64"))

    expected = "did not finish" if times_out else "exited 3: Error loading model"
    with pytest.raises(ExternalServiceError, match=expected):
        backend.transcribe(_audio(tmp_path), source=SOURCE)
    assert started[0]["start_new_session"] is True
    assert killed == ([(4242, signal.SIGKILL)] if times_out else [])


def test_the_default_host_probe_reads_this_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    with pytest.raises(ExternalServiceError, match="this is Linux x86_64"):
        ParakeetMlx(run=FakeParakeet().run, which=_found).resolve()


def test_a_relative_audio_name_is_recorded_as_given_and_passed_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeParakeet()
    _swap(monkeypatch, _backend(fake))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "-x.wav").write_bytes(b"pretend this is audio")

    result = runner.invoke(app, ["parakeet", "--", "-x.wav"])

    assert result.exit_code == 0, result.output
    assert Transcript.load(tmp_path / "-x.parakeet.json").source.ref == "-x.wav"
    passed = Path(fake.calls[0][fake.calls[0].index("parakeet-mlx") + 1])
    assert passed.is_absolute()
    assert passed.samefile(tmp_path / "-x.wav")


def test_audio_that_is_not_a_regular_file_is_refused_before_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeParakeet()
    _swap(monkeypatch, _backend(fake))
    folder = tmp_path / "clip.wav"
    folder.mkdir()

    result = runner.invoke(app, ["parakeet", str(folder)])

    assert result.exit_code == 2
    assert "not an audio file" in result.stderr
    assert fake.calls == []


def test_the_first_run_notice_comes_before_the_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    fake = FakeParakeet()

    def run(
        argv: list[str], *, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        events.append("spawn")
        return fake.run(argv, env=env, timeout=timeout)

    echo = typer.echo

    def recording_echo(message: str, *, err: bool = False) -> None:
        events.append(message)
        echo(message, err=err)

    monkeypatch.setattr(typer, "echo", recording_echo)
    _swap(monkeypatch, ParakeetMlx(run=run, which=_found, host=lambda: ("Darwin", "arm64")))

    result = runner.invoke(app, ["parakeet", str(_audio(tmp_path))])

    assert result.exit_code == 0, result.output
    notice = next(i for i, event in enumerate(events) if "can take minutes" in event)
    assert notice < events.index("spawn")


_REPORT = (
    "Error transcribing file /a/clip.wav: Failed to load audio: ffmpeg version 8\n"
    + "  configuration: --enable-something\n" * 20
    + "Error opening input files: Invalid data found when processing input\n\n\n"
    + "parakeet-tdt-0.6b-v3 transcription complete. Outputs saved in '/tmp/x'.\n"
)


def test_a_long_tool_report_keeps_the_file_and_the_cause_but_not_the_closing_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _swap(monkeypatch, _backend(FakeParakeet(None, stdout=_REPORT)))

    result = runner.invoke(app, ["parakeet", str(_audio(tmp_path))])

    assert result.exit_code == 2
    line = result.stderr.splitlines()[-1]
    assert "exited 0 without writing a transcript: Error transcribing file /a/clip.wav" in line
    assert line.endswith("Error opening input files: Invalid data found when processing input")
    assert "transcription complete" not in line
