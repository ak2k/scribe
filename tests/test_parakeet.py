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
from scribe.parakeet import (
    DEFAULT_TIMEOUT_S,
    ParakeetBackend,
    ParakeetCpp,
    ParakeetMlx,
    parakeet_backend,
    parse_output,
    parse_segments,
)
from scribe.schema import Engine, Source, Transcript, Word
from tests.child_env_cases import ALL_BUT_THE_WORKER, DROPPED, KEPT, TOKENS

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
    # Bound first: an assert over the environment itself would print its values.
    names, cache, columns = set(fake.env), fake.env.get("PARAKEET_CACHE_DIR"), fake.env["COLUMNS"]
    assert "PARAKEET_MAX_WORDS" not in names
    assert cache == str(tmp_path)
    assert int(columns) >= 1000
    assert not fake.workdir.exists()


@pytest.mark.parametrize("name", [*DROPPED, *TOKENS])
def test_the_child_is_not_given(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "secret")
    fake = FakeParakeet()

    _backend(fake).transcribe(_audio(tmp_path), source=SOURCE)

    names = set(fake.env)
    assert name not in names


@pytest.mark.parametrize("name", [*KEPT, *ALL_BUT_THE_WORKER])
def test_the_child_is_given(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, str(tmp_path))
    fake = FakeParakeet()

    _backend(fake).transcribe(_audio(tmp_path), source=SOURCE)

    names = set(fake.env)
    assert name in names


def test_the_index_url_reaches_the_child_with_its_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # uv resolves through it, credentials and all.
    index = "https://user:pass@index.invalid/simple"
    monkeypatch.setenv("UV_INDEX_URL", index)
    fake = FakeParakeet()

    _backend(fake).transcribe(_audio(tmp_path), source=SOURCE)

    given = fake.env.get("UV_INDEX_URL")
    assert given == index


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


def _swap(monkeypatch: pytest.MonkeyPatch, backend: ParakeetBackend) -> None:
    monkeypatch.setattr("scribe.cli.parakeet_backend", lambda: backend)


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


def test_a_key_holding_a_line_break_is_named_on_one_line() -> None:
    raw = json.dumps({"text": "", "sentences": [], "a\nb": 1}).encode()

    with pytest.raises(ExternalServiceError) as caught:
        parse_output(raw)

    assert str(caught.value).endswith("(at a b)")


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


CLI = "/opt/nowhere/bin/parakeet-cli"
FFMPEG = "/opt/nowhere/bin/ffmpeg"
# The opening of a real `parakeet-cli --print-segments` run on a recorded call:
# its log cut to a few lines and its tokens to the first 60, the counts edited to match.
CAPTURED = Path(__file__).parent / "fixtures" / "parakeet_cli_segments.txt"
CAPTURED_OPENING = [
    ("Is", 10.56, 10.8),
    ("he", 10.8, 10.88),
    ("the", 11.2, 11.28),
    ("same", 11.28, 11.44),
    ("team", 11.6, 11.76),
    ("as", 11.92, 12.0),
    ("uh?", 12.24, 12.48),
    ("Yep,", 16.08, 16.56),
    ("yeah.", 16.72, 17.04),
    ("So", 17.12, 17.28),
]

type _Piece = tuple[str, int, int, bool]


def _token_line(index: int, piece: _Piece) -> str:
    text, t0, t1, start = piece
    flag = "true" if start else "false"
    return (
        f"  [{index:2d}] id= 1976 frame={t0 // 8:3d} dur_idx= 4 dur_val= 4 p=0.9996 "
        f'plog=-15.6206 t0={t0:4d} t1={t1:4d} word_start={flag} "{text}"'
    )


def _segments(*segments: list[_Piece]) -> str:
    lines = ["load_backend: loaded CPU backend", "Successfully loaded Parakeet model", ""]
    lines.append(f"Segments ({len(segments)}):")
    for number, pieces in enumerate(segments):
        lines.append(f'Segment {number}: [0 -> 1101] "{"".join(p[0] for p in pieces)}"')
        lines.append(f"Tokens [{len(pieces)}]:")
        lines += [_token_line(index, piece) for index, piece in enumerate(pieces)]
    return "\n".join(lines) + "\n"


SPOKEN: list[_Piece] = [
    ("▁And", 24, 56, True),
    ("▁so", 56, 88, True),
    (",", 88, 88, False),
    ("▁f", 128, 136, True),
    ("ell", 136, 152, False),
    ("ow", 152, 176, False),
    ("▁Amer", 176, 208, True),
    ("ic", 208, 240, False),
    ("ans", 240, 272, False),
    (".", 272, 272, False),
]
SPOKEN_WORDS = [
    Word(text="And", start=0.24, end=0.56),
    Word(text="so,", start=0.56, end=0.88),
    Word(text="fellow", start=1.28, end=1.76),
    Word(text="Americans.", start=1.76, end=2.72),
]


class FakeCpp:
    """Stands in for ffmpeg, which writes its output, and parakeet-cli, which prints `report`."""

    def __init__(
        self,
        report: str | None = None,
        *,
        returncode: int = 0,
        ffmpeg_exit: int = 0,
        error: Exception | None = None,
    ) -> None:
        self.report = _segments(SPOKEN) if report is None else report
        self.returncode, self.ffmpeg_exit, self.error = returncode, ffmpeg_exit, error
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.timeouts: list[float] = []

    def run(
        self, argv: list[str], *, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        self.envs.append(env)
        self.timeouts.append(timeout)
        if argv[0] == FFMPEG:
            if self.ffmpeg_exit == 0:
                Path(argv[-1]).write_bytes(b"RIFF")
            return subprocess.CompletedProcess(argv, self.ffmpeg_exit, "", "Invalid data found")
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(argv, self.returncode, "And so, fellow\n", self.report)


def _model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    model = tmp_path / "ggml-parakeet-tdt-0.6b-v3-f16.bin"
    model.write_bytes(b"pretend this is a model")
    monkeypatch.setenv("SCRIBE_PARAKEET_MODEL", str(model))
    monkeypatch.delenv("SCRIBE_PARAKEET_CLI", raising=False)
    monkeypatch.delenv("SCRIBE_PARAKEET_THREADS", raising=False)
    return model


def _tools(name: str) -> str | None:
    return {"parakeet-cli": CLI, "ffmpeg": FFMPEG}.get(name)


def _cpp(fake: FakeCpp, which: Callable[[str], str | None] = _tools) -> ParakeetCpp:
    return ParakeetCpp(run=fake.run, which=which)


def test_words_are_rebuilt_from_a_captured_parakeet_cli_run() -> None:
    words = parse_segments(CAPTURED.read_text(encoding="utf-8"))

    assert [(word.text, word.start, word.end) for word in words[:10]] == CAPTURED_OPENING
    assert len(words) == 37
    assert "don't" in [word.text for word in words]
    assert not [word for word in words if "\N{LOWER ONE EIGHTH BLOCK}" in word.text]
    starts = [word.start for word in words]
    assert starts == sorted(starts)


def test_pieces_join_into_words_and_punctuation_joins_the_word_before() -> None:
    assert parse_segments(_segments(SPOKEN)) == SPOKEN_WORDS


def test_times_are_hundredths_of_a_second() -> None:
    # The tool counts 10 ms mel frames, each encoder step being 8 of them.
    words = parse_segments(_segments([("▁late", 187_300, 187_340, True)]))

    assert [(word.start, word.end) for word in words] == [(1873.0, 1873.4)]


def test_a_token_that_is_a_double_quote_is_read_whole() -> None:
    pieces: list[_Piece] = [("▁said", 0, 40, True), ('"', 40, 40, False), ('▁"hi', 48, 80, True)]

    assert [word.text for word in parse_segments(_segments(pieces))] == ['said"', '"hi']


def test_a_word_timed_before_the_one_before_it_keeps_its_place_with_a_clamped_start() -> None:
    first: list[_Piece] = [("▁limit", 500, 540, True), ("▁right", 600, 630, True)]
    second: list[_Piece] = [("▁so", 100, 120, True), ("▁and", 700, 720, True)]

    words = parse_segments(_segments(first, second))

    assert [(word.text, word.start, word.end) for word in words] == [
        ("limit", 5.0, 5.4),
        ("right", 6.0, 6.3),
        ("so", 6.0, 6.0),
        ("and", 7.0, 7.2),
    ]


def test_a_piece_with_no_word_before_it_opens_one() -> None:
    pieces: list[_Piece] = [(".", 0, 0, False), ("▁", 8, 16, True), ("ok", 16, 32, False)]

    assert [word.text for word in parse_segments(_segments(pieces))] == [".", "ok"]


def test_a_run_that_heard_nothing_has_no_words() -> None:
    assert parse_segments(_segments()) == []


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ("Successfully loaded Parakeet model\n", "without printing its segments"),
        (_segments(SPOKEN).replace(" word_start=true", " word_start=maybe", 1), "word_start=maybe"),
        (_segments(SPOKEN).replace("Tokens [10]:", "Tokens [11]:"), "10 of 11 tokens"),
        (_segments(SPOKEN).replace("Segments (1):", "Segments (2):"), "1 of 2 segments"),
        (_segments(SPOKEN).replace("Segment 0: [0", "Segment 0 [0"), "Segment 0 [0 -> 1101]"),
    ],
    ids=["no-segments", "bad-token", "short-tokens", "short-segments", "bad-segment"],
)
def test_output_of_another_shape_is_refused_naming_the_line(report: str, expected: str) -> None:
    with pytest.raises(ExternalServiceError, match="parakeet-cli") as caught:
        parse_segments(report)

    assert expected in str(caught.value)
    assert "\n" not in str(caught.value)


@pytest.mark.parametrize(
    ("host", "kind"),
    [
        (("Darwin", "arm64"), ParakeetMlx),
        (("Darwin", "x86_64"), ParakeetCpp),
        (("Linux", "x86_64"), ParakeetCpp),
        (("Linux", "aarch64"), ParakeetCpp),
    ],
)
def test_apple_silicon_runs_parakeet_mlx_and_every_other_host_parakeet_cli(
    host: tuple[str, str], kind: type[object]
) -> None:
    assert type(parakeet_backend(host=lambda: host)) is kind


def test_the_cpp_backend_converts_the_audio_and_runs_the_tool_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path, monkeypatch)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setenv("XAI_API_KEY", "secret")
    fake = FakeCpp()
    clip = _audio(tmp_path)

    transcript = _cpp(fake).transcribe(clip, source=SOURCE)

    convert, transcribe = fake.calls
    wav = Path(convert[-1])
    assert convert[0] == FFMPEG
    assert convert[convert.index("-i") + 1] == str(clip.absolute())
    assert convert[convert.index("-ar") + 1] == "16000"
    assert convert[convert.index("-ac") + 1] == "1"
    assert wav.suffix == ".wav"
    assert transcribe == [CLI, "-m", str(model), "-t", "8", "--print-segments", "-f", str(wav)]
    assert not wav.parent.exists()
    assert all(0 < timeout <= DEFAULT_TIMEOUT_S for timeout in fake.timeouts)
    assert not [env for env in fake.envs if "XAI_API_KEY" in env]
    assert transcript.words == SPOKEN_WORDS
    assert transcript.text == "And so, fellow Americans."
    assert transcript.engine == Engine(name="parakeet-cpp", model=model.name)


@pytest.mark.parametrize(("cpus", "setting", "expected"), [(4, None, "4"), (32, "3", "3")])
def test_the_thread_count_is_capped_unless_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpus: int,
    setting: str | None,
    expected: str,
) -> None:
    _model(tmp_path, monkeypatch)
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    if setting is not None:
        monkeypatch.setenv("SCRIBE_PARAKEET_THREADS", setting)
    fake = FakeCpp()

    _cpp(fake).transcribe(_audio(tmp_path), source=SOURCE)

    argv = fake.calls[1]
    assert argv[argv.index("-t") + 1] == expected


def test_the_cli_named_by_the_environment_is_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _model(tmp_path, monkeypatch)
    monkeypatch.setenv("SCRIBE_PARAKEET_CLI", "/opt/elsewhere/parakeet-cli")
    fake = FakeCpp()

    def which(name: str) -> str | None:
        return name if name.startswith("/opt/elsewhere/") else _tools(name)

    _cpp(fake, which).transcribe(_audio(tmp_path), source=SOURCE)

    assert fake.calls[1][0] == "/opt/elsewhere/parakeet-cli"


def _no_model(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRIBE_PARAKEET_MODEL")


def _relative_model(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRIBE_PARAKEET_MODEL", "models/parakeet.bin")


def _absent_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRIBE_PARAKEET_MODEL", str(tmp_path / "gone.bin"))


def _named_cli_missing(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRIBE_PARAKEET_CLI", "/opt/gone/parakeet-cli")


def _bad_threads(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRIBE_PARAKEET_THREADS", "0")


def _as_is(_tmp_path: Path, _monkeypatch: pytest.MonkeyPatch) -> None:
    return None


@pytest.mark.parametrize(
    ("arrange", "missing", "expected"),
    [
        (_as_is, "parakeet-cli", "parakeet-cli is not on PATH"),
        (_named_cli_missing, None, "SCRIBE_PARAKEET_CLI names /opt/gone/parakeet-cli"),
        (_no_model, None, "SCRIBE_PARAKEET_MODEL is not set"),
        (_relative_model, None, "SCRIBE_PARAKEET_MODEL must be an absolute path"),
        (_absent_model, None, "which is not a file"),
        (_as_is, "ffmpeg", "ffmpeg is not on PATH"),
        (_bad_threads, None, "SCRIBE_PARAKEET_THREADS must be a whole number above 0"),
    ],
)
def test_a_host_without_what_parakeet_cli_needs_exits_2_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[Path, pytest.MonkeyPatch], None],
    missing: str | None,
    expected: str,
) -> None:
    _model(tmp_path, monkeypatch)
    arrange(tmp_path, monkeypatch)
    fake = FakeCpp()

    def which(name: str) -> str | None:
        return None if name == missing or name.startswith("/opt/gone/") else _tools(name)

    _swap(monkeypatch, _cpp(fake, which))

    result = runner.invoke(app, ["parakeet", str(_audio(tmp_path))])

    assert result.exit_code == 2
    assert result.stderr.startswith("scribe: ")
    assert expected in result.stderr
    assert len(result.stderr.splitlines()) == 1
    assert fake.calls == []


_CPP_TIMEOUT = subprocess.TimeoutExpired([CLI], DEFAULT_TIMEOUT_S)


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (FakeCpp(ffmpeg_exit=1), "ffmpeg exited 1: Invalid data found"),
        (FakeCpp("error: failed to load Parakeet model\n", returncode=1), "exited 1: error: fail"),
        (
            FakeCpp("load_backend: x\nerror: failed to read audio file 'a.wav'\n"),
            "without printing its segments: error: failed to read audio file 'a.wav'",
        ),
        (FakeCpp(error=_CPP_TIMEOUT), f"did not finish within {DEFAULT_TIMEOUT_S:g} s"),
        (FakeCpp(error=OSError("exec format error")), "cannot run parakeet-cli"),
    ],
    ids=["ffmpeg-exit-1", "exit-1", "unread", "timeout", "unrunnable"],
)
def test_a_failed_parakeet_cli_run_exits_2_with_one_line_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeCpp, expected: str
) -> None:
    _model(tmp_path, monkeypatch)
    _swap(monkeypatch, _cpp(fake))
    clip = _audio(tmp_path)

    result = runner.invoke(app, ["parakeet", str(clip)])

    assert result.exit_code == 2
    errors = [line for line in result.stderr.splitlines() if line.startswith("scribe:")]
    assert len(errors) == 1, result.stderr
    assert expected in errors[0]
    assert result.stdout == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "clip.wav",
        "ggml-parakeet-tdt-0.6b-v3-f16.bin",
    ]


def test_the_command_on_linux_runs_parakeet_cli_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _model(tmp_path, monkeypatch)
    _swap(monkeypatch, _cpp(FakeCpp()))

    result = runner.invoke(app, ["parakeet", str(_audio(tmp_path))])

    assert result.exit_code == 0, result.output
    assert "running parakeet-cli locally" in result.stderr
    assert "download" not in result.stderr
    assert Transcript.load(tmp_path / "clip.parakeet.json").words == SPOKEN_WORDS


def test_the_help_says_what_each_platform_runs_and_where_the_model_comes_from() -> None:
    result = runner.invoke(app, ["parakeet", "--help"], terminal_width=200)

    shown = " ".join(result.stdout.split())
    for needed in ["Apple silicon", "parakeet-mlx", "parakeet-cli", "SCRIBE_PARAKEET_MODEL"]:
        assert needed in shown, needed
