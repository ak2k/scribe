"""`scribe transcribe --vote`: three engines, each one's transcript kept, the words voted."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import httpx
import pytest
from typer.testing import CliRunner

from scribe import ensemble, gemini_stt
from scribe.cli import app
from scribe.errors import ExternalServiceError
from scribe.parakeet import ParakeetMlx
from scribe.schema import Source, Transcript

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typer.testing import Result

XAI_KEY = "xai-key-never-written"
GEMINI_KEY = "gemini-key-never-written"
# xAI hears "cat" where the other two hear "hat", and stops before the words
# both of them hear after it.
SAID = [("The", 0.0, 0.4), ("cat", 0.5, 0.9), ("sat.", 1.0, 1.4)]
HEARD = [
    (" The", 0.0, 0.4),
    (" hat", 0.5, 0.9),
    (" sat", 1.0, 1.4),
    (" on", 2.0, 2.3),
    (" mats.", 2.5, 2.9),
]
GEMINI_SAYS = "The hat sat on mats."
# One Gemini reply's usage at $2 / $12 per 1M tokens.
GEMINI_USD = 1000 * 2e-6 + 500 * 12e-6
runner = CliRunner()


def _gemini_reply(text: str = GEMINI_SAYS) -> httpx.Response:
    segments = {"segments": [{"start_seconds": 0.0, "speaker": "S1", "text": text}]}
    candidate = {"content": {"parts": [{"text": json.dumps(segments)}]}, "finishReason": "STOP"}
    usage = {"promptTokenCount": 1000, "candidatesTokenCount": 500, "thoughtsTokenCount": 0}
    return httpx.Response(200, json={"candidates": [candidate], "usageMetadata": usage})


def _parakeet_output(heard: Sequence[tuple[str, float, float]]) -> str:
    tokens = [
        {"text": text, "start": start, "end": end, "duration": end - start, "confidence": 0.9}
        for text, start, end in heard
    ]
    text = "".join(text for text, _, _ in heard)
    timing = {"start": 0.0, "end": 0.0, "duration": 0.0, "confidence": 0.9}
    sentences = [{"text": text, **timing, "tokens": tokens}] if tokens else []
    return json.dumps({"text": text, "sentences": sentences})


@dataclass
class Engines:
    """What each swapped-in backend was asked to do."""

    xai: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    parakeet: list[list[str]] = field(default_factory=list[list[str]])
    tools: list[list[str]] = field(default_factory=list[list[str]])
    gemini: list[httpx.Request] = field(default_factory=list[httpx.Request])


def _engines(
    monkeypatch: pytest.MonkeyPatch,
    *,
    xai_error: Exception | None = None,
    parakeet_exit: int = 0,
    heard: Sequence[tuple[str, float, float]] = tuple(HEARD),
    gemini: httpx.Response | None = None,
    host: tuple[str, str] = ("Darwin", "arm64"),
    missing: str | None = None,
) -> Engines:
    """Swap in all three backends and set both keys; return what they record."""
    calls = Engines()
    payload: dict[str, object] = {
        "text": " ".join(text for text, _, _ in SAID),
        "language": "English",
        "duration": 3.0,
        "words": [
            {"text": text, "start": start, "end": end, "speaker": 0} for text, start, end in SAID
        ],
    }

    class Xai:
        def __init__(self, api_key: str) -> None:
            assert api_key == XAI_KEY

        def transcribe(self, path: Path, **options: object) -> dict[str, object]:
            calls.xai.append({"path": path, **options})
            if xai_error is not None:
                raise xai_error
            return payload

    def run_parakeet(
        argv: list[str], *, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        calls.parakeet.append(argv)
        workdir = Path(argv[argv.index("--output-dir") + 1])
        name = argv[argv.index("--output-template") + 1]
        (workdir / f"{name}.json").write_text(_parakeet_output(heard), encoding="utf-8")
        return subprocess.CompletedProcess(argv, parakeet_exit, "Error loading model", "")

    def run_tool(argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
        calls.tools.append(argv)
        if argv[0] == "ffmpeg":
            Path(argv[-1]).write_bytes(b"mp3 bytes")
        return subprocess.CompletedProcess(argv, 0, stdout="3.0\n", stderr="")

    def answer(request: httpx.Request) -> httpx.Response:
        calls.gemini.append(request)
        return _gemini_reply() if gemini is None else gemini

    def which(name: str) -> str | None:
        return None if name == missing else f"/opt/bin/{name}"

    backend = ParakeetMlx(run=run_parakeet, which=which, host=lambda: host)
    monkeypatch.setattr("scribe.cli.XaiStt", Xai)
    monkeypatch.setattr("scribe.cli.ParakeetMlx", lambda: backend)
    voted = partial(
        ensemble.transcribe_voted,
        run=run_tool,
        transport=httpx.MockTransport(answer),
        which=which,
    )
    monkeypatch.setattr(ensemble, "transcribe_voted", voted)
    monkeypatch.setenv("XAI_API_KEY", XAI_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    return calls


def _clip(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"pretend this is audio")
    return clip


def _vote(clip: Path, *args: str) -> Result:
    return runner.invoke(app, ["transcribe", str(clip), "--vote", *args])


def _written(tmp_path: Path) -> list[str]:
    return sorted(path.name for path in tmp_path.iterdir())


def _failure(result: Result) -> str:
    """Return the one `scribe:` line a failed run prints, after its progress lines."""
    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert "Traceback" not in result.output
    lines = result.stderr.splitlines()
    assert [line for line in lines if line.startswith("scribe: ")] == lines[-1:], lines
    return lines[-1]


def test_vote_writes_each_engines_transcript_then_the_voted_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engines(monkeypatch)
    clip = _clip(tmp_path)

    result = _vote(clip, "--keyterm", "Acme", "--vad-threshold", "0.2")

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    assert _written(tmp_path) == [
        "clip.gemini.json",
        "clip.parakeet.json",
        "clip.transcript.json",
        "clip.wav",
        "clip.xai.json",
    ]
    voted = Transcript.load(out)
    assert [word.text for word in voted.words] == ["The", "hat", "sat.", "on", "mats."]
    assert voted.engine.name == "rover"
    assert voted.engine.params["backbone"] == "xai-stt"
    assert voted.engine.params["primary"] == "parakeet-mlx"
    assert voted.engine.params["secondary"] == "gemini"
    source = Source(
        kind="audio", ref=str(clip), sha256=hashlib.sha256(clip.read_bytes()).hexdigest()
    )
    engines = {
        name: Transcript.load(tmp_path / f"clip.{name}.json")
        for name in ("xai", "parakeet", "gemini")
    }
    assert {name: t.source for name, t in engines.items()} == dict.fromkeys(engines, source)
    assert voted.source == source
    assert engines["xai"].engine.params["vad_threshold"] == 0.2
    assert engines["xai"].engine.params["keyterms"] == '["Acme"]'
    assert calls.xai[0]["keyterms"] == ["Acme"]
    assert calls.xai[0]["vad_threshold"] == 0.2
    assert [word.text for word in engines["parakeet"].words] == ["The", "hat", "sat", "on", "mats."]
    assert [(word.text, word.start) for word in engines["gemini"].words] == [
        ("the", 0.0),
        ("hat", 0.5),
        ("sat", 1.0),
        ("on", 2.0),
        ("mats", 2.5),
    ]
    assert engines["gemini"].engine.params["anchor_engine"] == "parakeet-mlx"
    lines = result.stderr.splitlines()
    expected = round(gemini_stt.expected_usd(gemini_stt.plan_chunks(3.0)), 4)
    assert len(lines) == 4, lines
    assert re.fullmatch(r"Parakeet: 5 words in \d+\.\d s", lines[0])
    assert re.fullmatch(r"xAI: 3 words in \d+\.\d s", lines[1])
    assert re.fullmatch(
        rf"Gemini: 5 words in \d+\.\d s for \${round(GEMINI_USD, 4)} \(expected \${expected}\)",
        lines[2],
    )
    assert lines[3] == (
        "vote: 2 words inserted (0 with no speaker), 1 substituted; "
        "0 spans filled with 0 words from Parakeet, 0 unresolved"
    )
    assert len(calls.gemini) == 1
    everything = result.output + "".join(
        path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json")
    )
    assert XAI_KEY not in everything
    assert GEMINI_KEY not in everything


@pytest.mark.parametrize("flag", ["--cross-check", "--no-cross-check"])
def test_the_voted_words_are_filled_from_parakeet_unless_the_cross_check_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    # Words only Parakeet hears: the vote leaves them out, the fill does not.
    _engines(monkeypatch, heard=[*HEARD, (" we", 5.0, 5.3), (" lost", 6.0, 6.3), (" it", 7.0, 7.3)])

    result = _vote(_clip(tmp_path), flag)

    assert result.exit_code == 0, result.output
    voted = Transcript.load(tmp_path / "clip.transcript.json")
    line = result.stderr.splitlines()[-1]
    if flag == "--cross-check":
        assert [word.text for word in voted.words][5:] == ["we", "lost", "it"]
        assert line.endswith("; 1 span filled with 3 words from Parakeet, 0 unresolved")
    else:
        assert len(voted.words) == len(SAID) + 2
        assert line == "vote: 2 words inserted (0 with no speaker), 1 substituted"


def test_vote_reports_each_run_the_fill_moved_after_its_progress_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Parakeet hears xAI's three words 3 s later, after a passage xAI left out;
    # Gemini agrees with neither, so the vote inserts nothing.
    lost = [(" we", 0.5, 0.8), (" lost", 1.0, 1.3), (" this", 1.5, 1.8)]
    later = [(" The", 3.0, 3.4), (" cat", 3.5, 3.9), (" sat.", 4.0, 4.4)]
    _engines(monkeypatch, heard=[*lost, *later], gemini=_gemini_reply("Something else entirely."))

    result = _vote(_clip(tmp_path))

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines()[-2:] == [
        "vote: 0 words inserted (0 with no speaker), 0 substituted; "
        "1 span filled with 3 words from Parakeet, 0 unresolved",
        "scribe: re-timed 00:00:00.0-00:00:01.4 to 00:00:03.0-00:00:04.4 "
        "(3 words to where Parakeet heard them)",
    ]
    voted = Transcript.load(tmp_path / "clip.transcript.json")
    assert [(word.text, word.start) for word in voted.words] == [
        *(("we", 0.5), ("lost", 1.0), ("this", 1.5)),
        *(("The", 3.0), ("cat", 3.5), ("sat.", 4.0)),
    ]


def test_vote_points_at_each_unresolved_span_after_its_progress_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Words only Parakeet hears, among the voted ones, where no hole can take them.
    unheard = [(f" w{index}", 0.02 + 0.14 * index, 0.07 + 0.14 * index) for index in range(20)]
    _engines(monkeypatch, heard=sorted([*HEARD, *unheard], key=lambda word: word[1]))

    result = _vote(_clip(tmp_path))

    assert result.exit_code == 0, result.output
    vote, *after = result.stderr.splitlines()[-2:]
    assert vote.startswith("vote: ")
    assert vote.endswith("; 0 spans filled with 0 words from Parakeet, 1 unresolved")
    assert after == ["scribe: possible dropped speech 00:00:00.0-00:00:03.0; listen to that span"]
    voted = Transcript.load(tmp_path / "clip.transcript.json")
    assert voted.engine.params["fill_unresolved_ranges"] == "[[0.0, 3.0]]"


def test_every_file_goes_beside_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _engines(monkeypatch)
    clip = _clip(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = _vote(clip, "--out", str(elsewhere / "board.json"))

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{elsewhere / 'board.json'}\n"
    assert _written(elsewhere) == [
        "board.gemini.json",
        "board.json",
        "board.parakeet.json",
        "board.xai.json",
    ]
    assert _written(tmp_path) == ["clip.wav", "elsewhere"]


def _parakeet_fails(monkeypatch: pytest.MonkeyPatch) -> tuple[Engines, str, list[str]]:
    return _engines(monkeypatch, parakeet_exit=1), "parakeet-mlx exited 1", []


def _parakeet_hears_nothing(monkeypatch: pytest.MonkeyPatch) -> tuple[Engines, str, list[str]]:
    return _engines(monkeypatch, heard=()), "Parakeet heard no words", ["parakeet"]


def _xai_fails(monkeypatch: pytest.MonkeyPatch) -> tuple[Engines, str, list[str]]:
    down = ExternalServiceError("xAI transcription failed with HTTP 503: down")
    return _engines(monkeypatch, xai_error=down), "HTTP 503: down", ["parakeet"]


def _gemini_fails(monkeypatch: pytest.MonkeyPatch) -> tuple[Engines, str, list[str]]:
    refused = httpx.Response(400, text="bad request")
    return _engines(monkeypatch, gemini=refused), "Gemini failed with HTTP 400", ["parakeet", "xai"]


_PROGRESS = {"parakeet": "Parakeet", "xai": "xAI"}


@pytest.mark.parametrize(
    "failure", [_parakeet_fails, _parakeet_hears_nothing, _xai_fails, _gemini_fails]
)
def test_a_failed_engine_exits_two_keeping_the_transcripts_before_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Callable[[pytest.MonkeyPatch], tuple[Engines, str, list[str]]],
) -> None:
    calls, message, finished = failure(monkeypatch)
    clip = _clip(tmp_path)

    result = _vote(clip)

    line = _failure(result)
    assert message in line
    kept = [tmp_path / f"clip.{name}.json" for name in finished]
    if kept:
        assert line.endswith(f"; kept {', '.join(map(str, kept))}")
    else:
        assert "kept" not in line
    assert _written(tmp_path) == sorted(["clip.wav", *(path.name for path in kept)])
    progress = result.stderr.splitlines()[:-1]
    assert [line.partition(":")[0] for line in progress] == [_PROGRESS[n] for n in finished]
    assert bool(calls.xai) == ("xai" in finished or failure is _xai_fails)
    assert bool(calls.gemini) == (failure is _gemini_fails)


def test_a_transcript_that_cannot_be_written_exits_two_keeping_the_earlier_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engines(monkeypatch)
    dump = Transcript.dump

    def refuse(self: Transcript, path: Path) -> None:
        if path.name.endswith(".gemini.json"):
            raise OSError("disk full")
        dump(self, path)

    monkeypatch.setattr(Transcript, "dump", refuse)

    line = _failure(_vote(_clip(tmp_path)))

    kept = [tmp_path / "clip.parakeet.json", tmp_path / "clip.xai.json"]
    assert line.startswith(f"scribe: cannot write transcript to {tmp_path / 'clip.gemini.json'}")
    assert line.endswith(f"disk full; kept {kept[0]}, {kept[1]}")
    assert len(calls.gemini) == 1
    assert not (tmp_path / "clip.transcript.json").exists()


@pytest.mark.parametrize(
    ("line", "stop", "stays"),
    [
        ("vote: ", KeyboardInterrupt(), False),
        ("vote: ", BrokenPipeError(), False),
        ("Gemini: ", KeyboardInterrupt(), True),
    ],
    ids=["interrupted-after-out", "pipe-closed-after-out", "interrupted-before-out"],
)
def test_a_vote_stopped_once_it_wrote_out_leaves_no_earlier_disputes_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str, stop: BaseException, stays: bool
) -> None:
    _engines(monkeypatch)
    out = tmp_path / "clip.transcript.json"
    out.write_text("an earlier run's transcript\n", encoding="utf-8")
    listed = tmp_path / "clip.disputes.md"
    listed.write_text("an earlier run's list\n", encoding="utf-8")

    def report(progress: str) -> None:
        if progress.startswith(line):
            raise stop

    monkeypatch.setattr("scribe.cli._progress", report)

    result = _vote(_clip(tmp_path))

    assert result.exit_code != 0, result.output
    if stays:
        assert out.read_text(encoding="utf-8") == "an earlier run's transcript\n"
        assert listed.read_text(encoding="utf-8") == "an earlier run's list\n"
    else:
        assert [word.text for word in Transcript.load(out).words] == [
            *("The", "hat", "sat.", "on", "mats.")
        ]
        assert not listed.exists()


@pytest.mark.parametrize(
    ("host", "missing", "unset", "message"),
    [
        (("Linux", "x86_64"), None, None, "needs Apple silicon"),
        (("Darwin", "arm64"), "uvx", None, "uvx is not on PATH"),
        (("Darwin", "arm64"), "ffmpeg", None, "ffmpeg is not on PATH"),
        (("Darwin", "arm64"), "ffprobe", None, "ffprobe is not on PATH"),
        (("Darwin", "arm64"), None, "XAI_API_KEY", "XAI_API_KEY is not set"),
        (("Darwin", "arm64"), None, "GEMINI_API_KEY", "GEMINI_API_KEY is not set"),
    ],
)
def test_a_machine_that_cannot_run_every_engine_is_refused_before_any_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: tuple[str, str],
    missing: str | None,
    unset: str | None,
    message: str,
) -> None:
    calls = _engines(monkeypatch, host=host, missing=missing)
    if unset is not None:
        monkeypatch.delenv(unset)
    clip = _clip(tmp_path)

    result = _vote(clip)

    assert message in _failure(result)
    assert calls == Engines()
    assert _written(tmp_path) == ["clip.wav"]


@pytest.mark.parametrize(
    ("cap", "message"),
    [
        ("0.01", "the expected cost of $0.00 plus room for the last call's worst case"),
        ("nan", "--max-usd must be a finite amount above 0"),
    ],
)
def test_a_gemini_run_the_cap_refuses_makes_no_call_to_any_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: str, message: str
) -> None:
    calls = _engines(monkeypatch)
    clip = _clip(tmp_path)

    result = _vote(clip, "--max-usd", cap)

    assert message in _failure(result)
    assert (calls.xai, calls.parakeet, calls.gemini) == ([], [], [])
    assert all(argv[0] == "ffprobe" for argv in calls.tools)
    assert _written(tmp_path) == ["clip.wav"]


def test_an_engines_file_that_would_replace_an_input_is_refused_before_any_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engines(monkeypatch)
    clip = _clip(tmp_path)
    terms = tmp_path / "board.xai.json"
    terms.write_text("Acme\n", encoding="utf-8")

    result = _vote(clip, "--keyterm-file", str(terms), "--out", str(tmp_path / "board.json"))

    assert _failure(result) == f"scribe: the xAI transcript {terms} is the --keyterm-file"
    assert calls == Engines()
    assert terms.read_text(encoding="utf-8") == "Acme\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_an_unwritable_engines_file_is_refused_before_any_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engines(monkeypatch)
    clip = _clip(tmp_path)
    locked = tmp_path / "clip.gemini.json"
    locked.write_text("kept\n", encoding="utf-8")
    locked.chmod(0o444)
    try:
        result = _vote(clip)
    finally:
        locked.chmod(0o644)

    assert f"cannot write {locked}" in _failure(result)
    assert calls == Engines()
    assert locked.read_text(encoding="utf-8") == "kept\n"


def test_max_usd_without_vote_exits_two_before_the_key_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engines(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY")
    clip = _clip(tmp_path)

    result = runner.invoke(app, ["transcribe", str(clip), "--max-usd", "1"])

    assert _failure(result) == "scribe: --max-usd caps Gemini's spending, which only --vote runs"
    assert calls == Engines()


def test_without_vote_none_of_its_code_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engines(monkeypatch)

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("reached --vote code without --vote")

    for owner, name in [
        (ensemble, "transcribe_voted"),
        (gemini_stt, "preflight"),
        (gemini_stt, "transcribe"),
        (gemini_stt, "resolve_api_key"),
    ]:
        monkeypatch.setattr(owner, name, refuse)
    monkeypatch.setattr("scribe.cli.ParakeetMlx", refuse)
    monkeypatch.setattr("scribe.cli.configure", refuse)
    clip = _clip(tmp_path)

    result = runner.invoke(app, ["transcribe", str(clip), "--no-cross-check"])

    assert result.exit_code == 0, result.output
    assert _written(tmp_path) == ["clip.transcript.json", "clip.wav"]
    assert Transcript.load(tmp_path / "clip.transcript.json").engine.name == "xai-stt"
    assert len(calls.xai) == 1


def test_the_help_states_the_cost_the_needs_and_where_the_audio_goes() -> None:
    result = runner.invoke(app, ["transcribe", "--help"], terminal_width=200)

    assert result.exit_code == 0
    shown = " ".join(result.stdout.split())
    for needed in [
        "Gemini about $0.55 per audio hour",
        "xAI about $0.10",
        "Apple silicon",
        "uvx",
        "ffmpeg",
        "ffprobe",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "the audio goes to xAI and to Google",
    ]:
        assert needed in shown, needed
