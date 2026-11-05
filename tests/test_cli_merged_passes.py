"""`scribe turns` on a merged call: both speaker passes for the call app's side, never the mic's."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner

from scribe.cli import app
from scribe.diarizer import DIMENSION, PyannoteDiarizer
from scribe.schema import Engine, Source, Track, Transcript, Word
from scribe.turns import turns_from_speakers
from tests.diarizer_fakes import FakeWorker, answer, found
from tests.speakers_fakes import FakeSpeakerBackend

if TYPE_CHECKING:
    from collections.abc import Callable

runner = CliRunner()
AXIS = {7: 0, 4: 1}
# The id `merge` gives the mic side: one past the app side's highest.
ME_ID = 8
MIC, APP = "mic", "app"
Runs = list[tuple[str, int | None, int]]
FAR: Runs = [(APP, 7, 3), (APP, 4, 3)] * 5
# Two unattributed app words with a mic turn between them, after a mic turn of its own.
TALK: Runs = [*FAR, (MIC, None, 3), (APP, None, 1), (MIC, None, 2), (APP, None, 1), *FAR]


def _layout(runs: Runs) -> list[tuple[str, int | None]]:
    return [(role, speaker) for role, speaker, count in runs for _ in range(count)]


def _side(tmp_path: Path, role: str, layout: list[tuple[str, int | None]]) -> None:
    audio = tmp_path / f"{role}.mp3"
    audio.write_bytes(f"pretend this is the {role} audio".encode())
    words = [
        Word(text=f"w{index}", start=index, end=index + 0.9, speaker=speaker)
        for index, (side, speaker) in enumerate(layout)
        if side == role
    ]
    Transcript(
        source=Source(
            kind="audio", ref=audio.name, sha256=hashlib.sha256(audio.read_bytes()).hexdigest()
        ),
        engine=Engine(name="xai-stt", params={"diarize": role == APP}),
        duration=float(len(layout)),
        text=" ".join(word.text for word in words),
        words=words,
    ).dump(tmp_path / f"{role}.json")


def _stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runs: Runs = TALK) -> None:
    """Merge a mic and an app side, a word a second, into call.json, from within tmp_path."""
    monkeypatch.chdir(tmp_path)
    layout = _layout(runs)
    _side(tmp_path, MIC, layout)
    _side(tmp_path, APP, layout)
    merged = runner.invoke(app, ["merge", "mic.json", "app.json", "--out", "call.json"])
    assert merged.exit_code == 0, merged.output


def _voice(speaker: int) -> list[float]:
    return [float(axis == AXIS[speaker]) for axis in range(DIMENSION)]


def _reply(runs: Runs = TALK) -> Callable[[list[list[float]]], dict[str, object]]:
    """The worker's answer: each unattributed app word in 7's cluster and sounding like 7."""
    layout = _layout(runs)

    def reply(intervals: list[list[float]]) -> dict[str, object]:
        exclusive = [
            {"start": float(index), "end": index + 0.9, "speaker": f"c{speaker or 7}"}
            for index, (role, speaker) in enumerate(layout)
            if role == APP
        ]
        voices: list[list[float]] = []
        for start, end in intervals:
            _, said = layout[min(int((start + end) / 2), len(layout) - 1)]
            voices.append(_voice(said or 7))
        return answer(intervals, exclusive=exclusive, embeddings=voices)

    return reply


def _swap(monkeypatch: pytest.MonkeyPatch, fake: FakeWorker) -> None:
    def factory() -> PyannoteDiarizer:
        return PyannoteDiarizer(run=fake.run, which=found, host=lambda: ("Darwin", "arm64"))

    monkeypatch.setattr("scribe.cli.PyannoteDiarizer", factory)


def _backends(
    monkeypatch: pytest.MonkeyPatch, **kwargs: Callable[[str], str]
) -> list[FakeSpeakerBackend]:
    made: list[FakeSpeakerBackend] = []

    def factory(*, model: str, disable_tools: bool) -> FakeSpeakerBackend:
        assert disable_tools
        made.append(FakeSpeakerBackend(model=model, **kwargs))
        return made[-1]

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)
    return made


def _swap_ranks(target: str) -> str:
    return (
        target.replace("<spk:0>", "<spk:x>")
        .replace("<spk:1>", "<spk:0>")
        .replace("<spk:x>", "<spk:1>")
    )


def _turns(*extra: str) -> list[str]:
    return ["turns", "call.json", "--out-dir", "out", *extra]


def _written() -> Transcript:
    return Transcript.load(Path("out") / "call.turns.json")


def _expected(runs: Runs, app_ids: Callable[[int | None], int | None]) -> list[int | None]:
    return [ME_ID if role == MIC else app_ids(speaker) for role, speaker in _layout(runs)]


def _mic_turns_are_mine(written: Transcript, runs: Runs) -> None:
    mic_text = [f"w{index}" for index, (role, _) in enumerate(_layout(runs)) if role == MIC]
    mine = " ".join(turn.text for turn in written.turns if turn.speaker == "Me").split()
    assert mine == mic_text


def test_merged_audio_naming_names_app_words_from_the_app_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, monkeypatch)
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)

    result = runner.invoke(app, _turns("--no-llm-speakers"))

    assert result.exit_code == 0, result.output
    decoded = fake.calls[0]
    assert any(arg.endswith("app.mp3") for arg in decoded)
    assert not any(arg.endswith("mic.mp3") for arg in decoded)
    # Only the two app words are counted: no mic word is ever named from the app recording.
    assert result.stderr.splitlines()[-1] == "scribe: the diarizer named 2 of 2 unattributed words"
    written = _written()
    expected = _expected(TALK, lambda speaker: speaker or 7)
    assert written.turns == turns_from_speakers(
        written.words, expected, tracks=written.tracks
    )
    _mic_turns_are_mine(written, TALK)
    assert written.engine.params["diarizer_named"] == 2


SPLIT: Runs = [*FAR[:6], (MIC, None, 3), *FAR[:6]]


def test_merged_speaker_pass_sees_mic_words_unlabeled_and_changes_only_app_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, monkeypatch, SPLIT)
    made = _backends(monkeypatch, reply=_swap_ranks)

    result = runner.invoke(app, _turns("--no-audio-speakers"))

    assert result.exit_code == 0, result.output
    assert len(made) == 1
    # The mic words are context for the far side's turns, with no speaker of their own.
    assert "<spk:?> w18 w19 w20" in made[0].calls[0][1]
    sidecar = cast(
        "dict[str, object]",
        json.loads((Path("out") / "call.speakers.json").read_text(encoding="utf-8")),
    )
    assert (sidecar["words"], sidecar["words_relabeled"]) == (36, 36)
    written = _written()
    swapped = {7: 4, 4: 7}
    expected = _expected(SPLIT, lambda speaker: None if speaker is None else swapped[speaker])
    assert written.turns == turns_from_speakers(written.words, expected, tracks=written.tracks)
    _mic_turns_are_mine(written, SPLIT)


# A and B on the far side; the operator answers twice between them.
ROUND: Runs = [
    (APP, 7, 3),
    (APP, 4, 3),
    (APP, 7, 3),
    (APP, 4, 3),
    (APP, 7, 3),
    (MIC, None, 3),
    (APP, 4, 3),
    (APP, 7, 3),
    (MIC, None, 3),
    (APP, 4, 3),
    (APP, 7, 3),
    (APP, 4, 3),
]


def _vocatives(_target: str) -> str:
    # w2 and w8 end A's turns before B's; w14 and w23 end A's turns before the operator's.
    return (
        "<names>\n"
        "Bob | w2 | next | w0 w1 w2\n"
        "Bob | w8 | next | w6 w7 w8\n"
        "Adam | w14 | next | w12 w13 w14\n"
        "Adam | w23 | next | w21 w22 w23\n"
        "</names>"
    )


def test_merged_attendees_name_a_far_speaker_and_never_hand_the_operators_name_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, monkeypatch, ROUND)
    _backends(monkeypatch, trailer=_vocatives)

    result = runner.invoke(app, _turns("--no-audio-speakers", "--attendees", "Adam,Bob"))

    assert result.exit_code == 0, result.output
    written = _written()
    # A pointer at the operator's reply lands on no far speaker, so B, who follows it,
    # is named only by the pointers that reach B.
    assert {turn.speaker for turn in written.turns} == {"Speaker 1", "Bob", "Me"}
    _mic_turns_are_mine(written, ROUND)
    assert json.loads(str(written.engine.params["speaker_names"])) == {"Speaker 2": "Bob"}


def test_merged_without_the_app_recording_warns_and_keeps_its_words_unattributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, monkeypatch)
    (tmp_path / "app.mp3").unlink()
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)

    result = runner.invoke(app, _turns("--no-llm-speakers"))

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == [
        "scribe: the diarizer cannot name 2 unattributed words: "
        "the source audio app.mp3 does not exist; pass --audio"
    ]
    assert fake.calls == []
    written = _written()
    assert written.turns == turns_from_speakers(
        written.words, _expected(TALK, lambda speaker: speaker), tracks=written.tracks
    )


def test_merged_audio_flag_must_be_the_app_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, monkeypatch)
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)

    result = runner.invoke(app, _turns("--no-llm-speakers", "--audio", "mic.mp3"))

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == [
        "scribe: the diarizer cannot name 2 unattributed words: "
        "mic.mp3 is not the audio the transcript was made from"
    ]
    assert fake.calls == []


def test_merged_speaker_pass_still_runs_without_the_app_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, monkeypatch)
    (tmp_path / "app.mp3").unlink()
    made = _backends(monkeypatch)

    result = runner.invoke(app, _turns())

    assert result.exit_code == 0, result.output
    assert len(made) == 1
    assert (Path("out") / "call.speakers.json").exists()
    assert "the source audio app.mp3 does not exist" in result.stderr


def test_a_transcript_with_two_app_tracks_skips_both_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Source(kind="audio", ref="x.mp3")
    words = [
        Word(text=f"w{index}", start=index, end=index + 0.9, speaker=speaker, track=track)
        for index, (track, speaker) in enumerate([(0, 9), (1, 7), (2, 4), (1, None)])
    ]
    Transcript(
        source=Source(kind="other", ref="three sides"),
        engine=Engine(name="xai-stt"),
        text=" ".join(word.text for word in words),
        words=words,
        tracks=[
            Track(role="mic", label="Me", source=source, transcript_sha256="0" * 64),
            Track(role="app", source=source, transcript_sha256="1" * 64),
            Track(role="app", source=source, transcript_sha256="2" * 64),
        ],
    ).dump(tmp_path / "call.json")
    made = _backends(monkeypatch)

    result = runner.invoke(app, ["turns", "call.json", "--stdout"])

    assert result.exit_code == 0, result.output
    assert made == []
    assert result.stderr.splitlines() == [
        "scribe: speaker pass skipped: call.json holds 2 call-app tracks, "
        "and the pass reads exactly one",
        "scribe: audio naming skipped: call.json holds 2 call-app tracks, "
        "and naming reads exactly one",
    ]
