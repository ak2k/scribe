"""`scribe transcribe` cross-checks xAI's words against Parakeet; no failure there fails the run."""

from __future__ import annotations

import errno
import json
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest
from typer.testing import CliRunner

from scribe import gaps
from scribe.cli import app
from scribe.errors import InputValidationError
from scribe.parakeet import ParakeetMlx
from scribe.schema import Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable

    from typer.testing import Result

SAID = [("Hello", 0.0, 0.4), ("there.", 0.5, 0.9), ("Bye.", 8.0, 8.4)]
XAI_WORDS = [Word(text=text, start=start, end=end, speaker=0) for text, start, end in SAID]
# Parakeet also hears the three words in the hole xAI left.
HEARD = [(" we", 3.0, 3.3), (" lost", 4.0, 4.3), (" this", 5.0, 5.3)]
ANNOUNCED = (
    "scribe: cross-checking against Parakeet, run locally "
    "(about 45 s per audio hour; a first run downloads a ~1.2 GB model)"
)
runner = CliRunner()


@dataclass
class Calls:
    """What the swapped-in Parakeet and ffmpeg were asked to run."""

    parakeet: list[list[str]] = field(default_factory=list[list[str]])
    ffmpeg: list[list[str]] = field(default_factory=list[list[str]])


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    host: tuple[str, str] = ("Darwin", "arm64"),
    parakeet_exit: int = 0,
    parakeet_hears: bool = True,
    ffmpeg: bool = True,
    ffmpeg_exit: int = 0,
) -> Calls:
    calls = Calls()
    words = [{"text": text, "start": start, "end": end, "speaker": 0} for text, start, end in SAID]
    payload: dict[str, object] = {"text": "Hello there. Bye.", "duration": 9.0, "words": words}

    class Xai:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def transcribe(self, _path: Path, **_options: object) -> dict[str, object]:
            return payload

    def run_parakeet(
        argv: list[str], *, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        calls.parakeet.append(argv)
        heard = [(f" {text}", start, end) for text, start, end in SAID[:2]] + HEARD
        heard.append((" Bye.", 8.0, 8.4))
        tokens = [
            {"text": text, "start": start, "end": end, "duration": end - start, "confidence": 0.9}
            for text, start, end in heard
        ]
        timing = {"start": 0.0, "end": 0.0, "duration": 0.0, "confidence": 0.9}
        sentences = [{"text": "", **timing, "tokens": tokens}] if parakeet_hears else []
        output = {"text": "", "sentences": sentences}
        workdir = Path(argv[argv.index("--output-dir") + 1])
        (workdir / "transcript.json").write_text(json.dumps(output), encoding="utf-8")
        return subprocess.CompletedProcess(argv, parakeet_exit, "Error loading model", "")

    # Loud throughout: xAI's 7.1 s hole sounds like speech.
    levels = "".join(
        f"frame:{i} pts:{i}\nlavfi.astats.Overall.RMS_level=-20.0\n" for i in range(180)
    )

    def run_ffmpeg(argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
        calls.ffmpeg.append(argv)
        return subprocess.CompletedProcess(argv, ffmpeg_exit, stdout=levels, stderr="bad audio")

    backend = ParakeetMlx(
        run=run_parakeet, which=lambda name: f"/opt/bin/{name}", host=lambda: host
    )
    monkeypatch.setattr("scribe.cli.XaiStt", Xai)
    monkeypatch.setattr("scribe.cli.ParakeetMlx", lambda: backend)
    found = "/usr/bin/ffmpeg" if ffmpeg else None
    checked = partial(gaps.check_gaps, run=run_ffmpeg, which=lambda _name: found)
    monkeypatch.setattr("scribe.cli.check_gaps", checked)
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    return calls


def _transcribe(tmp_path: Path, *args: str) -> Result:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"pretend this is audio")
    return runner.invoke(app, ["transcribe", str(clip), *args])


def test_the_cross_check_fills_the_hole_xai_left_with_parakeets_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _setup(monkeypatch)

    result = _transcribe(tmp_path)

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    filled = Transcript.load(out)
    assert [(word.text, word.speaker) for word in filled.words] == [
        *(("Hello", 0), ("there.", 0), ("we", None), ("lost", None), ("this", None), ("Bye.", 0))
    ]
    assert filled.engine.params["fill_ranges"] == "[[3.0, 5.3]]"
    reference = Transcript.load(tmp_path / "clip.parakeet.json")
    assert [word.text for word in reference.words] == [
        "Hello",
        "there.",
        "we",
        "lost",
        "this",
        "Bye.",
    ]
    assert result.stderr.splitlines()[1:] == [
        ANNOUNCED,
        "scribe: filled 00:00:03.0-00:00:05.3 (3 words from Parakeet, no speaker)",
        "scribe: cross-check: 1 span filled with 3 words from Parakeet, 0 unresolved",
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        *("clip.parakeet.json", "clip.transcript.json", "clip.wav")
    ]
    assert (len(calls.parakeet), calls.ffmpeg) == (1, [])


def test_no_cross_check_writes_xais_transcript_and_runs_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _setup(monkeypatch)

    result = _transcribe(tmp_path, "--no-cross-check")

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["clip.transcript.json", "clip.wav"]
    written = Transcript.load(out)
    assert (written.words, written.text) == (XAI_WORDS, "Hello there. Bye.")
    assert not [key for key in written.engine.params if key.startswith("fill_")]
    assert len(result.stderr.splitlines()) == 1
    assert calls == Calls()


def _sibling_is_a_directory(tmp_path: Path, _monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "clip.parakeet.json").mkdir()


def _refuse_writing(ending: str) -> Callable[[Path, pytest.MonkeyPatch], None]:
    def arrange(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dump = Transcript.dump

        def refuse(self: Transcript, path: Path) -> None:
            # A name the rewrite's temporary file starts with, or the sibling's ending.
            if path.name.startswith(ending) or path.name.endswith(ending):
                raise OSError("disk full")
            dump(self, path)

        monkeypatch.setattr(Transcript, "dump", refuse)

    return arrange


def _nothing(_tmp_path: Path, _monkeypatch: pytest.MonkeyPatch) -> None:
    return


def _no_temporary_directory(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_options: object) -> NoReturn:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", refuse)


def _not_of_this_audio(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_transcript: Transcript, _audio: Path) -> NoReturn:
        raise InputValidationError(
            "no word falls within the audio's 0.1 s; the transcript is not of this audio"
        )

    monkeypatch.setattr("scribe.cli.check_gaps", refuse)


LINUX = ("Linux", "x86_64")
LOUDNESS = [
    "scribe: possible dropped speech 00:00:00.9-00:00:08.0; listen to that span",
    "scribe: cross-check by loudness: 1 hole flagged",
]


@pytest.mark.parametrize(
    ("setup", "arrange", "message", "inserted", "after"),
    [
        (_setup, _sibling_is_a_directory, "is a directory", 3, []),
        (_setup, _refuse_writing(".parakeet.json"), "clip.parakeet.json: disk full", 3, []),
        (partial(_setup, host=LINUX), _nothing, "needs Apple silicon", 0, LOUDNESS),
        (partial(_setup, parakeet_exit=1), _nothing, "parakeet-mlx exited 1", 0, LOUDNESS),
        (partial(_setup, parakeet_hears=False), _nothing, "it heard no words", 0, LOUDNESS),
        (_setup, _no_temporary_directory, "No space left on device", 0, LOUDNESS),
        (partial(_setup, host=LINUX, ffmpeg=False), _nothing, "ffmpeg is not on PATH", 0, []),
        (partial(_setup, host=LINUX, ffmpeg_exit=1), _nothing, "ffmpeg exited 1", 0, []),
        (partial(_setup, host=LINUX), _not_of_this_audio, "not of this audio", 0, []),
        (_setup, _refuse_writing(".clip.transcript.json"), "keeps xAI's words", 0, []),
    ],
)
def test_a_cross_check_that_fails_is_one_line_and_keeps_xais_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[pytest.MonkeyPatch], Calls],
    arrange: Callable[[Path, pytest.MonkeyPatch], None],
    message: str,
    inserted: int,
    after: list[str],
) -> None:
    setup(monkeypatch)
    arrange(tmp_path, monkeypatch)

    result = _transcribe(tmp_path)

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    lines = result.stderr.splitlines()
    assert len([line for line in lines if message in line]) == 1, lines
    assert all(line.startswith("scribe: ") for line in lines[1:])
    assert lines[len(lines) - len(after) :] == after
    written = Transcript.load(out)
    assert [word for word in written.words if word.speaker is not None] == XAI_WORDS
    assert len(written.words) == len(XAI_WORDS) + inserted
    assert ("fill_spans" in written.engine.params) == bool(inserted)
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_the_rewrite_keeps_outs_mode_and_writes_through_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    (tmp_path / "kept").mkdir()
    target = tmp_path / "kept" / "clip.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o640)
    link = tmp_path / "clip.transcript.json"
    link.symlink_to(target)

    result = _transcribe(tmp_path)

    assert result.exit_code == 0, result.output
    assert link.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert len(Transcript.load(target).words) == len(XAI_WORDS) + len(HEARD)


def test_an_input_where_parakeets_transcript_would_go_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    terms = tmp_path / "clip.parakeet.json"
    terms.write_text("Acme\n", encoding="utf-8")

    result = _transcribe(tmp_path, "--keyterm-file", str(terms))

    assert result.exit_code == 0, result.output
    assert terms.read_text(encoding="utf-8") == "Acme\n"
    refused = [line for line in result.stderr.splitlines() if "is the --keyterm-file" in line]
    assert len(refused) == 1, result.stderr
    filled = Transcript.load(tmp_path / "clip.transcript.json")
    assert len(filled.words) == len(XAI_WORDS) + len(HEARD)


def test_the_help_says_what_the_cross_check_costs_and_how_to_turn_it_off() -> None:
    result = runner.invoke(app, ["transcribe", "--help"], terminal_width=200)

    assert result.exit_code == 0
    shown = " ".join(result.stdout.split())
    for needed in ["about 45 s per audio hour", "~1.2 GB model", "locally", "--no-cross-check"]:
        assert needed in shown, needed
