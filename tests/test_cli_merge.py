from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scribe.cli import app
from scribe.schema import Engine, Source, Transcript, Word

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def _write(path: Path, *said: tuple[str, float], speaker: int = 0, duration: float = 30.0) -> Path:
    Transcript(
        source=Source(kind="audio", ref=f"{path.stem}.wav"),
        engine=Engine(name="xai-stt"),
        duration=duration,
        text=" ".join(text for text, _ in said),
        words=[
            Word(text=text, start=start, end=start + 0.2, speaker=speaker) for text, start in said
        ],
    ).dump(path)
    return path


def _pair(tmp_path: Path, *, mic_shift: float = 0.05) -> tuple[Path, Path]:
    app_said = [("we", 1.0), ("can", 1.3), ("ship", 1.6), ("it.", 1.9), ("Yeah.", 4.0)]
    copies = [(text, start + mic_shift) for text, start in app_said]
    mic_said = [("Hello", 0.2), *copies[:4], ("right", 3.0), copies[4], ("thanks", 6.0)]
    mic = _write(tmp_path / "call.mic.transcript.json", *mic_said, speaker=1)
    return mic, _write(tmp_path / "call.app.json", *app_said, speaker=4)


def test_merge_writes_beside_mic_and_prints_only_the_path(tmp_path: Path) -> None:
    mic, app_path = _pair(tmp_path)

    result = runner.invoke(app, ["merge", str(mic), str(app_path)])

    assert result.exit_code == 0, result.output
    written = tmp_path / "call.mic.merged.json"
    assert result.stdout == f"{written}\n"
    assert result.stderr == (
        "scribe: merged 5 app words and 3 of 8 mic words; 5 mic words dropped as bleed-1 "
        "copies, by run length 1/2/3+: 1/0/4; offset +0.050 s\n"
    )
    merged = Transcript.load(written)
    assert merged.tracks is not None
    assert merged.tracks[0].label == "Me"


def test_out_and_me_name_the_output_and_the_operator(tmp_path: Path) -> None:
    mic, app_path = _pair(tmp_path)
    out = tmp_path / "elsewhere.json"

    result = runner.invoke(
        app, ["merge", str(mic), str(app_path), "--out", str(out), "--me", "Alice"]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{out}\n"
    tracks = Transcript.load(out).tracks
    assert tracks is not None
    assert tracks[0].label == "Alice"


def test_an_offset_past_half_a_second_is_warned(tmp_path: Path) -> None:
    mic, app_path = _pair(tmp_path, mic_shift=0.7)

    result = runner.invoke(app, ["merge", str(mic), str(app_path)])

    assert result.exit_code == 0, result.output
    assert "offset +0.700 s" in result.stderr
    assert "more than 0.5 s" in result.stderr


def _merged(tmp_path: Path) -> Path:
    mic, app_path = _pair(tmp_path)
    out = tmp_path / "merged.json"
    runner.invoke(app, ["merge", str(mic), str(app_path), "--out", str(out)])
    return out


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("missing", "cannot read transcript", id="missing"),
        pytest.param("not-a-transcript", "is not a transcript", id="not-a-transcript"),
        pytest.param("no-words", "has no words", id="no-words"),
        pytest.param("merged", "merges two tracks already", id="merged"),
        pytest.param("same-file", "are one transcript", id="same-file"),
        pytest.param("skew", "30.0 s and APP", id="skew"),
        pytest.param("out-is-mic", "--out", id="out-is-mic"),
    ],
)
def test_a_refused_merge_exits_two_and_writes_nothing(
    tmp_path: Path, case: str, message: str
) -> None:
    mic, app_path = _pair(tmp_path)
    extra: list[str] = []
    if case == "missing":
        app_path = tmp_path / "absent.json"
    elif case == "not-a-transcript":
        app_path.write_text('{"text": "hi"}', encoding="utf-8")
    elif case == "no-words":
        _write(app_path)
    elif case == "merged":
        shutil.copy(_merged(tmp_path), app_path)
    elif case == "same-file":
        app_path = mic
    elif case == "skew":
        _write(app_path, ("we", 1.0), duration=33.0)
    else:
        extra = ["--out", str(mic)]
    before = sorted(tmp_path.iterdir())

    result = runner.invoke(app, ["merge", str(mic), str(app_path), *extra])

    assert result.exit_code == 2
    assert message in result.stderr
    assert result.stderr.count("\n") == 1
    assert result.stdout == ""
    assert sorted(tmp_path.iterdir()) == before


def test_turns_on_a_merged_transcript_skip_both_speaker_passes(tmp_path: Path) -> None:
    merged = _merged(tmp_path)

    result = runner.invoke(app, ["turns", str(merged), "--attendees", "Bruno", "--stdout"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("**Me**")
    assert "**Speaker 1**" in result.stdout
    lines = result.stderr.splitlines()
    assert len(lines) == 2
    assert "speaker pass skipped" in lines[0]
    assert "--attendees names nobody" in lines[0]
    assert "audio naming skipped" in lines[1]


def test_turns_off_flags_on_a_merged_transcript_print_nothing_to_stderr(tmp_path: Path) -> None:
    merged = _merged(tmp_path)

    result = runner.invoke(
        app, ["turns", str(merged), "--no-llm-speakers", "--no-audio-speakers", "--stdout"]
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
