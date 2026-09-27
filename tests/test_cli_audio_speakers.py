"""`scribe turns` naming 'Speaker ?' words from the audio, with ffmpeg and the worker faked."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

import pytest
from typer.testing import CliRunner

from scribe.cli import app
from scribe.diarizer import DIMENSION, FAILED, PyannoteDiarizer
from scribe.schema import Engine, Source, Transcript, Word
from scribe.turns import turns_from_speakers
from tests.diarizer_fakes import TOKEN, FakeWorker, answer, found
from tests.speakers_fakes import FakeSpeakerBackend

if TYPE_CHECKING:
    from collections.abc import Callable

runner = CliRunner()
AXIS = {7: 0, 4: 1}
# Three-word turns, each a centroid segment, around two unattributed words at 30-31.9 s.
TALK: list[tuple[int | None, int]] = [*[(7, 3), (4, 3)] * 5, (None, 2), *[(7, 3), (4, 3)] * 5]


def _said(runs: list[tuple[int | None, int]]) -> list[int | None]:
    return [speaker for speaker, count in runs for _ in range(count)]


def _voice(speaker: int) -> list[float]:
    return [float(axis == AXIS[speaker]) for axis in range(DIMENSION)]


def _stage(
    tmp_path: Path,
    runs: list[tuple[int | None, int]] = TALK,
    *,
    kind: str = "audio",
    ref: str | None = None,
    hashed: bool = True,
) -> Path:
    """Write t.json, a word a second, with meeting.mp3 beside it as its source audio."""
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"pretend this is audio")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest() if hashed else None
    words = [
        Word(text=f"w{index}", start=index, end=index + 0.9, speaker=speaker)
        for index, speaker in enumerate(_said(runs))
    ]
    source = Source.model_validate(
        {"kind": kind, "ref": str(audio) if ref is None else ref, "sha256": digest}
    )
    staged = tmp_path / "t.json"
    Transcript(
        source=source,
        engine=Engine(name="xai-stt", params={"diarize": True}),
        text=" ".join(word.text for word in words),
        words=words,
    ).dump(staged)
    return staged


def _reply(
    runs: list[tuple[int | None, int]] = TALK, *, cluster: int | None = 7, heard_as: int = 7
) -> Callable[[list[list[float]]], dict[str, object]]:
    """The worker's answer when each unattributed word is in `cluster`'s cluster and sounds
    like `heard_as`, and every other word in its own speaker's cluster, in that voice.
    `cluster=None` puts every word in one cluster.
    """
    speakers = _said(runs)

    def reply(intervals: list[list[float]]) -> dict[str, object]:
        exclusive = [
            {
                "start": float(index),
                "end": index + 0.9,
                "speaker": "one" if cluster is None else f"c{cluster if said is None else said}",
            }
            for index, said in enumerate(speakers)
        ]
        voices: list[list[float]] = []
        for start, end in intervals:
            said = speakers[min(int((start + end) / 2), len(speakers) - 1)]
            voices.append(_voice(heard_as if said is None else said))
        return answer(intervals, exclusive=exclusive, embeddings=voices)

    return reply


def _swap(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeWorker,
    *,
    host: tuple[str, str] = ("Darwin", "arm64"),
    which: Callable[[str], str | None] = found,
) -> list[PyannoteDiarizer]:
    made: list[PyannoteDiarizer] = []

    def factory() -> PyannoteDiarizer:
        made.append(PyannoteDiarizer(run=fake.run, which=which, host=lambda: host))
        return made[-1]

    monkeypatch.setattr("scribe.cli.PyannoteDiarizer", factory)
    return made


def _turns(tmp_path: Path, *extra: str, out: str = "out") -> list[str]:
    return ["turns", str(tmp_path / "t.json"), "--out-dir", str(tmp_path / out), *extra]


def _written(tmp_path: Path, out: str = "out") -> Transcript:
    return Transcript.load(tmp_path / out / "t.turns.json")


def _artifacts(tmp_path: Path, out: str) -> tuple[bytes, bytes]:
    return (tmp_path / out / "t.turns.json").read_bytes(), (tmp_path / out / "t.md").read_bytes()


def test_unattributed_words_take_the_speaker_both_readings_agree_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers"))

    assert result.exit_code == 0, result.output
    written = _written(tmp_path)
    original = Transcript.load(tmp_path / "t.json")
    assert (written.words, written.text) == (original.words, original.text)
    named = [7 if speaker is None else speaker for speaker in _said(TALK)]
    assert written.turns == turns_from_speakers(original.words, named)
    assert result.stderr.splitlines() == [
        "scribe: naming 2 unattributed words with pyannote Community-1, run locally "
        "(about 85 s per audio hour; a first run downloads ~960 MB)",
        "scribe: the diarizer named 2 of 2 unattributed words",
    ]
    assert written.engine.params == {
        "diarize": True,
        "diarizer": "pyannote.audio 4.0.7 pyannote/speaker-diarization-community-1"
        "@3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
        "diarizer_device": "mps",
        "diarizer_unattributed": 2,
        "diarizer_named": 2,
        "diarizer_runs": '[[30.0, 31.9, 2, {"7": 2}]]',
    }


def test_a_word_with_a_speaker_keeps_it_when_the_audio_would_name_every_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    # One cluster for every word, which maps to 7, and every stretch heard as 7.
    fake = FakeWorker(_reply(cluster=None, heard_as=7))
    _swap(monkeypatch, fake)

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers"))

    assert result.exit_code == 0, result.output
    written = _written(tmp_path)
    kept = [7 if speaker is None else speaker for speaker in _said(TALK)]
    assert written.turns == turns_from_speakers(written.words, kept)
    assert written.engine.params["diarizer_named"] == 2


def _swap_ranks(target: str) -> str:
    """Swap the pass's two speakers; it tags them by rank of first appearance, not by id."""
    return (
        target.replace("<spk:0>", "<spk:x>")
        .replace("<spk:1>", "<spk:0>")
        .replace("<spk:x>", "<spk:1>")
    )


def test_the_speaker_pass_sees_the_words_unattributed_and_cannot_move_a_name_from_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)
    made: list[FakeSpeakerBackend] = []

    def factory(*, model: str, disable_tools: bool) -> FakeSpeakerBackend:
        assert disable_tools
        made.append(FakeSpeakerBackend(model=model, reply=_swap_ranks))
        return made[-1]

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)

    result = runner.invoke(app, _turns(tmp_path))

    assert result.exit_code == 0, result.output
    assert "<spk:?> w30 w31" in made[0].calls[0][1]
    sidecar = (tmp_path / "out" / "t.speakers.json").read_text(encoding="utf-8")
    assert cast("dict[str, object]", json.loads(sidecar))["words_relabeled"] == 60
    # Every id swapped; the named words took what the audio says after the swap: the ids
    # of w0's voice, now 4, and they keep it.
    swapped = {7: 4, 4: 7, None: 4}
    expected = [swapped[speaker] for speaker in _said(TALK)]
    written = _written(tmp_path)
    assert written.turns == turns_from_speakers(written.words, expected)
    # Turns rank speakers by first appearance, so only the named id shows the swap was kept.
    assert written.engine.params["diarizer_runs"] == '[[30.0, 31.9, 2, {"4": 2}]]'


def _drop_audio(request: bytes) -> bytes:
    decoded = cast("dict[str, object]", json.loads(request))
    del decoded["audio"]
    return json.dumps(decoded).encode()


def test_the_same_input_sends_the_same_request_and_names_the_same_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)

    first = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers", out="a"))
    second = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers", out="b"))

    assert first.exit_code == second.exit_code == 0
    assert _artifacts(tmp_path, "a") == _artifacts(tmp_path, "b")
    assert json.loads(fake.requests[0])["audio"] != json.loads(fake.requests[1])["audio"]
    assert _drop_audio(fake.requests[0]) == _drop_audio(fake.requests[1])


def test_the_token_reaches_the_worker_only_through_its_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    _stage(tmp_path)
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers"))

    assert result.exit_code == 0, result.output
    # The decode and the worker, so the checks below look at real spawns.
    assert len(fake.calls) == 2
    decode_env, worker_env = fake.envs
    assert worker_env["HF_TOKEN"] == TOKEN
    assert "HF_TOKEN" not in decode_env
    assert not any(TOKEN in part for argv in fake.calls for part in argv)
    assert TOKEN not in result.output
    assert not any(TOKEN.encode() in request for request in fake.requests)
    assert not any(TOKEN.encode() in path.read_bytes() for path in (tmp_path / "out").iterdir())


def _nothing(_name: str) -> str | None:
    return None


def _uvx_only(name: str) -> str | None:
    return found(name) if name == "uvx" else None


def _setup(
    case: str, tmp_path: Path
) -> tuple[FakeWorker, tuple[str, str], Callable[[str], str | None]]:
    """Stage the transcript for a failure `case`; return the worker, host and lookup to use."""
    if case == "no-source":
        _stage(tmp_path, kind="other")
    elif case == "missing":
        _stage(tmp_path, ref="gone.mp3")
    else:
        _stage(tmp_path)
    if case == "hash":
        (tmp_path / "meeting.mp3").write_bytes(b"other audio")
    workers = {
        "exit": FakeWorker(returncode=1, stderr=f"noise\n{FAILED}set HF_TOKEN to a token\n"),
        "bad-answer": FakeWorker(lambda _: "[1, 2]"),
        "unrunnable": FakeWorker(error=OSError("exec format error")),
    }
    host = ("Darwin", "x86_64") if case == "intel" else ("Darwin", "arm64")
    which = {"no-uvx": _nothing, "no-ffmpeg": _uvx_only}.get(case, found)
    return workers.get(case, FakeWorker(_reply())), host, which


_FAILURES = {
    "no-source": "the transcript names no source audio; pass --audio",
    "missing": "the source audio gone.mp3 does not exist; pass --audio",
    "hash": "is not the audio the transcript was made from",
    "intel": "needs Apple silicon (macOS on arm64); this is Darwin x86_64",
    "no-uvx": "uvx is not on PATH",
    "no-ffmpeg": "ffmpeg is not on PATH",
    "exit": "the diarizer exited 1: set HF_TOKEN to a token",
    "bad-answer": "the diarizer wrote an unexpected answer",
    "unrunnable": "cannot run /opt/nowhere/bin/uvx: exec format error",
}
_LOOKUPS = {"no-source", "missing", "hash", "intel", "no-uvx", "no-ffmpeg"}


@pytest.mark.parametrize("case", list(_FAILURES))
def test_a_failure_is_one_line_and_leaves_what_no_audio_step_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    fake, host, which = _setup(case, tmp_path)
    _swap(monkeypatch, fake, host=host, which=which)

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers"))
    plain = runner.invoke(
        app, _turns(tmp_path, "--no-llm-speakers", "--no-audio-speakers", out="p")
    )

    assert result.exit_code == plain.exit_code == 0
    failed = [line for line in result.stderr.splitlines() if "cannot name" in line]
    assert len(failed) == 1
    assert failed[0].startswith("scribe: the diarizer cannot name 2 unattributed words: ")
    assert _FAILURES[case] in failed[0]
    if case in _LOOKUPS:
        assert result.stderr == f"{failed[0]}\n"
        assert fake.calls == []
    assert _artifacts(tmp_path, "out") == _artifacts(tmp_path, "p")


@pytest.mark.parametrize(
    ("runs", "flags"),
    [
        (TALK, ["--no-audio-speakers"]),
        ([(7, 3), (4, 3)], []),
        ([(7, 3), (None, 2), (7, 3)], []),
    ],
    ids=["off", "no-unattributed-word", "one-speaker"],
)
def test_nothing_is_looked_up_or_run_when_there_is_nothing_to_name(
    tmp_path: Path, runs: list[tuple[int | None, int]], flags: list[str]
) -> None:
    # The source audio is missing: looking for it would print a line. The autouse guard
    # fails any construction of the real backend.
    _stage(tmp_path, runs, ref="gone.mp3")

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers", *flags))

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert "diarizer" not in _written(tmp_path).engine.params


def test_an_explicit_audio_is_used_over_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, ref="gone.mp3")
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)
    audio = tmp_path / "meeting.mp3"

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers", "--audio", str(audio)))

    assert result.exit_code == 0, result.output
    assert fake.calls[0][fake.calls[0].index("-i") + 1] == str(audio.absolute())
    assert _written(tmp_path).engine.params["diarizer_named"] == 2


def test_an_explicit_audio_must_be_the_recording_the_transcript_was_made_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    other = tmp_path / "other.mp3"
    other.write_bytes(b"other audio")
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers", "--audio", str(other)))
    plain = runner.invoke(
        app, _turns(tmp_path, "--no-llm-speakers", "--no-audio-speakers", out="p")
    )

    assert result.exit_code == plain.exit_code == 0
    assert result.stderr == (
        f"scribe: the diarizer cannot name 2 unattributed words: {other} is not the audio "
        "the transcript was made from\n"
    )
    assert fake.calls == []
    assert _artifacts(tmp_path, "out") == _artifacts(tmp_path, "p")


def test_a_relative_source_is_read_from_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, ref="meeting.mp3")
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["turns", "t.json", "--no-llm-speakers"])

    assert result.exit_code == 0, result.output
    decoded = Path(fake.calls[0][fake.calls[0].index("-i") + 1])
    assert decoded.is_absolute()
    assert decoded.samefile(tmp_path / "meeting.mp3")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--no-audio-speakers", "--audio", "{audio}"], "--audio is read only by --audio-speakers"),
        (["--audio", "{missing}"], "does not exist"),
    ],
)
def test_an_unusable_audio_flag_exits_two_before_anything_is_written(
    tmp_path: Path, flags: list[str], expected: str
) -> None:
    _stage(tmp_path)
    paths = {"audio": str(tmp_path / "meeting.mp3"), "missing": str(tmp_path / "no.mp3")}

    result = runner.invoke(
        app, _turns(tmp_path, "--no-llm-speakers", *(flag.format(**paths) for flag in flags))
    )

    assert result.exit_code == 2
    assert expected in result.stderr
    assert len(result.stderr.splitlines()) == 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("rerun", ["off", "failing"])
def test_a_rerun_on_named_turns_keeps_no_counts_from_the_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rerun: str
) -> None:
    _stage(tmp_path)
    _swap(monkeypatch, FakeWorker(_reply()))
    first = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers"))
    assert first.exit_code == 0, first.output
    assert _written(tmp_path).engine.params["diarizer_named"] == 2
    _swap(monkeypatch, FakeWorker(returncode=1, stderr=f"{FAILED}the model cannot load\n"))
    flags = ["--no-audio-speakers"] if rerun == "off" else []

    # The words keep their own speakers, so the earlier run's names are not in them.
    named = str(tmp_path / "out" / "t.turns.json")
    again = runner.invoke(
        app, ["turns", named, "--no-llm-speakers", "--out-dir", str(tmp_path / "again"), *flags]
    )

    assert again.exit_code == 0, again.output
    written = Transcript.load(tmp_path / "again" / "t.turns.turns.json")
    assert written.turns == turns_from_speakers(written.words, _said(TALK))
    assert written.engine.params == {"diarize": True}


def test_a_run_directory_that_cannot_be_removed_keeps_the_answer_already_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)
    remove = shutil.rmtree
    refused: list[str] = []

    def refuse(
        path: str, *, onexc: Callable[[Callable[..., object], str, BaseException], object]
    ) -> None:
        # The directory still holds the run's files, so this fails as a removal can;
        # rmtree reports such a failure to `onexc`, inside the handler.
        refused.append(path)
        try:
            Path(path).rmdir()
        except OSError as exc:
            onexc(Path.rmdir, path, exc)

    monkeypatch.setattr(shutil, "rmtree", refuse)

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers"))

    for path in refused:
        remove(path)
    assert result.exit_code == 0, result.output
    assert [Path(path).name.startswith("scribe-diarize-") for path in refused] == [True]
    assert result.stderr.splitlines()[-1] == "scribe: the diarizer named 2 of 2 unattributed words"
    assert _written(tmp_path).engine.params["diarizer_named"] == 2


def _never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a run that opens the audio: opening a FIFO blocks until a writer comes."""

    def refuse(path: Path) -> NoReturn:
        raise AssertionError(f"read {path} before checking it is a regular file")

    monkeypatch.setattr("scribe.cli._sha256", refuse)


@pytest.mark.parametrize("device", ["fifo", "/dev/null"])
def test_an_explicit_audio_that_is_not_a_regular_file_exits_two_before_the_speaker_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    _stage(tmp_path)
    audio = tmp_path / "meeting.fifo"
    if device == "fifo":
        os.mkfifo(audio)
    else:
        audio = Path(device)
    _never_read(monkeypatch)

    # The speaker pass is on: the autouse guard fails the run if it reaches it.
    result = runner.invoke(app, _turns(tmp_path, "--audio", str(audio)))

    assert result.exit_code == 2, result.output
    assert result.stderr == f"scribe: not an audio file: {audio}\n"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("hashed", [True, False], ids=["hashed", "unhashed"])
def test_a_source_audio_that_is_not_a_regular_file_is_one_line_and_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hashed: bool
) -> None:
    fifo = tmp_path / "meeting.fifo"
    os.mkfifo(fifo)
    _stage(tmp_path, ref=str(fifo), hashed=hashed)
    fake = FakeWorker(_reply())
    _swap(monkeypatch, fake)
    _never_read(monkeypatch)

    result = runner.invoke(app, _turns(tmp_path, "--no-llm-speakers"))
    plain = runner.invoke(
        app, _turns(tmp_path, "--no-llm-speakers", "--no-audio-speakers", out="p")
    )

    assert result.exit_code == plain.exit_code == 0, result.output
    assert result.stderr == (
        f"scribe: the diarizer cannot name 2 unattributed words: not an audio file: {fifo}\n"
    )
    assert fake.calls == []
    assert _artifacts(tmp_path, "out") == _artifacts(tmp_path, "p")


def test_the_help_names_the_flags_the_token_and_the_rank_warning() -> None:
    result = runner.invoke(app, ["turns", "--help"], terminal_width=200)

    assert result.exit_code == 0
    shown = " ".join(result.stdout.split())
    for expected in ("--audio-speakers", "--no-audio-speakers", "--audio", "HF_TOKEN"):
        assert expected in shown
    assert 'changes the "Speaker N" ranks' in shown
