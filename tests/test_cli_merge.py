from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scribe.cli import app
from scribe.schema import Engine, Source, Transcript, Word
from tests.cleanup_fakes import patch_backend

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def _write(
    path: Path,
    *said: tuple[str, float],
    speaker: int = 0,
    duration: float = 30.0,
    params: dict[str, float | int | bool | str] | None = None,
) -> Path:
    Transcript(
        source=Source(kind="audio", ref=f"{path.stem}.wav"),
        engine=Engine(name="xai-stt", params=params or {}),
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
        "copies, by run length 1/2/3+: 1/0/4; offset +0.000 s, from --offset\n"
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


def test_the_offset_is_the_one_given_and_never_warned(tmp_path: Path) -> None:
    mic, app_path = _pair(tmp_path, mic_shift=0.7)
    out = tmp_path / "merged.json"

    result = runner.invoke(
        app, ["merge", str(mic), str(app_path), "--offset", "0.7", "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == (
        "scribe: merged 5 app words and 3 of 8 mic words; 5 mic words dropped as bleed-1 "
        "copies, by run length 1/2/3+: 1/0/4; offset +0.700 s, from --offset\n"
    )
    assert Transcript.load(out).engine.params["merge_offset_s"] == 0.7


def test_no_offset_given_is_zero_whatever_the_stems_hold(tmp_path: Path) -> None:
    mic, app_path = _pair(tmp_path, mic_shift=0.7)
    out = tmp_path / "merged.json"

    result = runner.invoke(app, ["merge", str(mic), str(app_path), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "8 of 8 mic words; 0 mic words dropped" in result.stderr
    assert Transcript.load(out).engine.params["merge_offset_s"] == 0.0


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir()}


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
        pytest.param("nan", "--offset", id="offset-nan"),
        pytest.param("inf", "--offset", id="offset-inf"),
        pytest.param("-inf", "--offset", id="offset-minus-inf"),
        pytest.param("Speaker 1", "--me", id="me-numbered"),
        pytest.param(" speaker ? ", "--me", id="me-unattributed"),
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
    elif case == "out-is-mic":
        extra = ["--out", str(mic)]
    elif "peaker" in case:
        extra = ["--me", case]
    else:
        extra = ["--offset", case]
    before = _snapshot(tmp_path)

    result = runner.invoke(app, ["merge", str(mic), str(app_path), *extra])

    assert result.exit_code == 2
    assert message in result.stderr
    assert result.stderr.count("\n") == 1
    assert result.stdout == ""
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["fill", "MERGED", "PLAIN"], id="fill-transcript"),
        pytest.param(["fill", "PLAIN", "MERGED"], id="fill-reference"),
        pytest.param(["vote", "MERGED", "PLAIN", "OTHER"], id="vote-backbone"),
        pytest.param(["vote", "PLAIN", "MERGED", "OTHER"], id="vote-primary"),
        pytest.param(["vote", "PLAIN", "OTHER", "MERGED"], id="vote-secondary"),
        pytest.param(["pick", "MERGED", "PLAIN", "--sides-from", "ABSENT"], id="pick-transcript"),
        pytest.param(["pick", "PLAIN", "MERGED", "--sides-from", "ABSENT"], id="pick-reference"),
        pytest.param(["gaps", "MERGED", "ABSENT"], id="gaps"),
        pytest.param(["gemini", "ABSENT", "--anchor", "MERGED"], id="gemini-anchor"),
        pytest.param(["disputes", "MERGED"], id="disputes"),
    ],
)
def test_a_merged_input_is_refused_where_its_tracks_would_be_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    # Nothing past the refusal may reach a model or the network.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    said = [("we", 1.0), ("can", 1.3), ("ship", 1.6)]
    paths = {
        "MERGED": _merged(tmp_path),
        "PLAIN": _write(tmp_path / "plain.json", *said),
        "OTHER": _write(tmp_path / "other.json", *said),
        "ABSENT": tmp_path / "absent.wav",
    }
    before = _snapshot(tmp_path)

    result = runner.invoke(app, [str(paths.get(arg, arg)) for arg in argv])

    assert result.exit_code == 2, result.output
    assert f"{paths['MERGED']} merges two tracks" in result.stderr
    assert result.stderr.count("\n") == 1
    assert result.stdout == ""
    assert _snapshot(tmp_path) == before


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


def test_the_recorded_sha256_is_of_the_bytes_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic, app_path = _pair(tmp_path)
    original = mic.read_bytes()
    replaced = _write(tmp_path / "replaced.json", ("Replaced", 0.2), speaker=1).read_bytes()
    read_bytes = Path.read_bytes
    reads = 0

    def replaced_after_one_read(self: Path) -> bytes:
        nonlocal reads
        if self != mic:
            return read_bytes(self)
        reads += 1
        return original if reads == 1 else replaced

    monkeypatch.setattr(Path, "read_bytes", replaced_after_one_read)
    out = tmp_path / "merged.json"

    result = runner.invoke(app, ["merge", str(mic), str(app_path), "--out", str(out)])

    assert result.exit_code == 0, result.output
    merged = Transcript.load(out)
    assert "Replaced" not in merged.text
    assert merged.engine.params["merge_mic_sha256"] == hashlib.sha256(original).hexdigest()


def _reading_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *app_said: tuple[str, float]
) -> str:
    """The reading copy of a call whose mic recovered "remote." at 2.05 s in a second pass."""
    patch_backend(monkeypatch)
    mic = _write(
        tmp_path / "call.mic.json",
        ("Hi", 0.2),
        ("remote.", 2.05),
        ("thanks", 6.0),
        speaker=1,
        params={"fill_ranges": "[[2.0, 2.3]]"},
    )
    app_path = _write(tmp_path / "call.app.json", *app_said, speaker=4)
    merged, copy = tmp_path / "merged.json", tmp_path / "merged.curated.md"
    for argv in (
        ["merge", str(mic), str(app_path), "--out", str(merged)],
        ["turns", str(merged), "--no-llm-speakers", "--no-audio-speakers"],
        ["cleanup", str(tmp_path / "merged.turns.json"), "--curated", str(copy)],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.output
    return copy.read_text(encoding="utf-8")


def test_a_fill_dropped_as_bleed_notes_no_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = _reading_copy(tmp_path, monkeypatch, ("We", 1.0), ("remote.", 2.0), ("ok", 3.0))

    assert "remote." in copy
    assert "recovered" not in copy


def test_a_kept_fill_notes_only_its_own_tracks_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = _reading_copy(tmp_path, monkeypatch, ("We", 1.0), ("talk", 2.1), ("ok", 3.0))

    assert copy.count("recovered") == 1
    assert "**Me | 00:00:02**\n[Includes speech recovered" in copy
    assert "**Speaker 1 | 00:00:02**\ntalk" in copy
