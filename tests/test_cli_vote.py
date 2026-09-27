from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from scribe.cli import app
from scribe.schema import Engine, Source, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

runner = CliRunner()


def _write(path: Path, engine: str, *said: tuple[str, float]) -> Path:
    """Write a transcript whose words each last 0.4 s from the given start."""
    words = [Word(text=text, start=start, end=start + 0.4, speaker=0) for text, start in said]
    Transcript(
        source=Source(kind="audio", ref="meeting.mp3"),
        engine=Engine(name=engine),
        language="en",
        duration=9.0,
        text=" ".join(text for text, _ in said),
        words=words,
    ).dump(path)
    return path


def _inputs(tmp_path: Path, backbone: str = "meeting.transcript.json") -> list[Path]:
    return [
        _write(tmp_path / backbone, "xai-stt", ("the", 0.0), ("cat", 0.5), ("sat", 1.0)),
        _write(tmp_path / "parakeet.json", "parakeet", ("the", 0.0), ("hat", 0.5), ("sat", 1.0)),
        _write(tmp_path / "gemini.json", "gemini", ("the", 0.0), ("hat", 0.5), ("sat", 1.0)),
    ]


def test_vote_writes_the_voted_transcript_beside_the_backbone(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    result = runner.invoke(app, ["vote", *map(str, inputs)])

    assert result.exit_code == 0, result.output
    written = tmp_path / "meeting.voted.json"
    assert result.stdout == f"{written}\n"
    voted = Transcript.load(written)
    assert voted.text == "the hat sat"
    assert voted.words[1] == Word(text="hat", start=0.5, end=0.9, speaker=0)
    assert voted.engine.name == "rover"
    assert voted.engine.params["primary"] == "parakeet"
    assert (voted.source, voted.language, voted.duration) == (
        Transcript.load(inputs[0]).source,
        "en",
        9.0,
    )


@pytest.mark.parametrize(
    ("backbone", "voted"),
    [
        ("xai.transcript.json", "xai.voted.json"),
        ("xai.json", "xai.voted.json"),
        ("xai", "xai.voted.json"),
    ],
)
def test_the_default_output_replaces_the_backbones_suffix(
    tmp_path: Path, backbone: str, voted: str
) -> None:
    result = runner.invoke(app, ["vote", *map(str, _inputs(tmp_path, backbone))])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{tmp_path / voted}\n"


def test_out_names_the_output(tmp_path: Path) -> None:
    out = tmp_path / "elsewhere.json"

    result = runner.invoke(app, ["vote", *map(str, _inputs(tmp_path)), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{out}\n"
    assert Transcript.load(out).text == "the hat sat"
    assert not (tmp_path / "meeting.voted.json").exists()


def _unsorted(inputs: list[Path]) -> str:
    _write(inputs[1], "parakeet", ("the", 0.0), ("hat", 0.9), ("sat", 0.5))
    return f"PRIMARY {inputs[1]} word 2 starts at 0.5 s"


def _wordless(inputs: list[Path]) -> str:
    _write(inputs[2], "gemini")
    return f"SECONDARY {inputs[2]} has no words"


def _not_a_transcript(inputs: list[Path]) -> str:
    inputs[0].write_text("{}", encoding="utf-8")
    return "is not a transcript"


def _missing(inputs: list[Path]) -> str:
    inputs[0].unlink()
    return "cannot read transcript"


@pytest.mark.parametrize("breakage", [_unsorted, _wordless, _not_a_transcript, _missing])
def test_a_bad_input_is_one_stderr_line_and_exit_two(
    tmp_path: Path, breakage: Callable[[list[Path]], str]
) -> None:
    inputs = _inputs(tmp_path)
    expected = breakage(inputs)

    result = runner.invoke(app, ["vote", *map(str, inputs)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith("scribe: ")
    assert len(result.stderr.splitlines()) == 1
    assert expected in result.stderr
    assert not (tmp_path / "meeting.voted.json").exists()


def test_an_out_that_is_an_input_is_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    result = runner.invoke(app, ["vote", *map(str, inputs), "--out", str(inputs[1])])

    assert result.exit_code == 2
    assert result.stderr == f"scribe: --out {inputs[1]} is PRIMARY\n"
    assert Transcript.load(inputs[1]).engine.name == "parakeet"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_an_unwritable_out_is_one_stderr_line_and_exit_two(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        result = runner.invoke(app, ["vote", *map(str, inputs), "--out", str(locked / "v.json")])
    finally:
        locked.chmod(0o755)

    assert result.exit_code == 2
    assert result.stderr.startswith(f"scribe: cannot write transcript to {locked / 'v.json'}: ")
    assert len(result.stderr.splitlines()) == 1
    assert list(locked.iterdir()) == []
