"""`scribe pick`: which reading is kept where xAI and Parakeet disagree, from the command line."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner

from scribe.claude_cli import ClaudeCliBackend
from scribe.cli import app
from scribe.ear import FAILED, RECOGNIZERS, LocalEars
from scribe.pick import DEFAULT_PICK_MODEL, PICK_PROMPT_VERSION, system_prompt
from scribe.schema import Transcript, Word
from tests.diarizer_fakes import FakeWorker, found
from tests.pick_fakes import answering, choosing, numbered, transcript
from tests.speakers_fakes import FakeSpeakerBackend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

runner = CliRunner()


def _said(*texts: str) -> list[Word]:
    return [
        Word(text=text, start=float(index), end=index + 0.4, speaker=0)
        for index, text in enumerate(texts)
    ]


def _inputs(tmp_path: Path) -> list[Path]:
    said = tmp_path / "meeting.transcript.json"
    heard = tmp_path / "meeting.parakeet.json"
    transcript(_said("the", "cat", "sat", "on", "the", "mat")).dump(said)
    transcript(_said("the", "hat", "sat", "on", "the", "bat"), engine="parakeet-mlx").dump(heard)
    return [said, heard]


def _long_inputs(tmp_path: Path) -> list[Path]:
    """Two spots, each alone in its own call."""
    said, heard = tmp_path / "long.json", tmp_path / "long.parakeet.json"
    transcript(numbered(4500)).dump(said)
    transcript(numbered(4500, {100: "x100", 4000: "x4000"}), engine="parakeet-mlx").dump(heard)
    return [said, heard]


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: Callable[[str], str] | None = None,
    fail_when: Callable[[str], bool] | None = None,
) -> list[FakeSpeakerBackend]:
    made: list[FakeSpeakerBackend] = []

    def factory(*, model: str, disable_tools: bool) -> FakeSpeakerBackend:
        assert disable_tools
        backend = FakeSpeakerBackend(
            model=model,
            reply=answering(choosing({"hat", "bat", "x100", "x4000"})) if reply is None else reply,
            fail_when=fail_when,
        )
        made.append(backend)
        return backend

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)
    return made


def test_pick_writes_the_picked_transcript_beside_the_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)

    result = runner.invoke(app, ["pick", *map(str, _inputs(tmp_path))])

    assert result.exit_code == 0, result.output
    written = tmp_path / "meeting.picked.json"
    assert result.stdout == f"{written}\n"
    assert result.stderr == (
        "scribe: picked the reference's reading at 2 of 2 disputed spots (0 unsure) "
        f"with {DEFAULT_PICK_MODEL}, prompt {PICK_PROMPT_VERSION}\n"
    )
    picked = Transcript.load(written)
    assert picked.text == "the hat sat on the bat"
    assert picked.engine.params["pick_to_reference"] == 2
    assert [system for system, _ in made[0].calls] == [system_prompt()]


def test_out_model_and_context_reach_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    out = tmp_path / "elsewhere.json"

    result = runner.invoke(
        app,
        [
            "pick",
            *map(str, _inputs(tmp_path)),
            "--out",
            str(out),
            "--model",
            "sonnet",
            "--context",
            " People at this meeting: Ann Lee ",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{out}\n"
    assert "with sonnet, prompt" in result.stderr
    assert [system for system, _ in made[0].calls] == [
        system_prompt("People at this meeting: Ann Lee")
    ]
    params = Transcript.load(out).engine.params
    assert (params["pick_model"], params["pick_context_chars"]) == ("sonnet", 31)
    assert not (tmp_path / "meeting.picked.json").exists()


def test_the_summary_names_the_picks_guarded_for_dropping_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, reply=answering(choosing({"purple"})))
    colors = ["red", "green", "blue", "pink", "gray", "brown", "black", "white", "gold", "teal"]
    said, heard = tmp_path / "said.json", tmp_path / "heard.json"
    transcript(_said("we", "saw", *colors, "at")).dump(said)
    purple = Word(text="purple", start=2.0, end=2.4)
    at = Word(text="at", start=12.0, end=12.4)
    transcript([*_said("we", "saw"), purple, at], engine="parakeet-mlx").dump(heard)

    result = runner.invoke(app, ["pick", str(said), str(heard)])

    assert result.exit_code == 0, result.output
    assert result.stderr == (
        "scribe: picked the reference's reading at 0 of 1 disputed spots (0 unsure) "
        f"with {DEFAULT_PICK_MODEL}, prompt {PICK_PROMPT_VERSION}; 1 guarded, keeping the "
        "transcript's words where the reading picked is 5 or more words shorter\n"
    )
    picked = Transcript.load(tmp_path / "said.picked.json")
    assert picked.words == Transcript.load(said).words
    assert picked.engine.params["pick_guarded"] == 1


def test_the_summary_names_the_spots_restored_for_words_the_transcript_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, reply=answering(choosing({"red"})))
    birds = ["robin", "wren", "finch", "crow", "owl", "hawk", "swan", "duck", "dove", "lark", "jay"]
    said, heard = tmp_path / "said.json", tmp_path / "heard.json"
    at = Word(text="at", start=13.0, end=13.4, speaker=0)
    transcript([*_said("we", "saw", "red"), at]).dump(said)
    transcript(_said("we", "saw", *birds, "at"), engine="parakeet-mlx").dump(heard)

    result = runner.invoke(app, ["pick", str(said), str(heard)])

    assert result.exit_code == 0, result.output
    assert result.stderr == (
        "scribe: picked the reference's reading at 0 of 1 disputed spots (0 unsure) "
        f"with {DEFAULT_PICK_MODEL}, prompt {PICK_PROMPT_VERSION}; 1 restored, putting in the "
        "reference's words where they are 10 or more words longer than the transcript's\n"
    )
    picked = Transcript.load(tmp_path / "said.picked.json")
    assert [word.text for word in picked.words] == ["we", "saw", *birds, "at"]
    assert picked.engine.params["pick_restored"] == 1


def test_no_spot_writes_the_transcript_and_asks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    said, _ = _inputs(tmp_path)
    same = tmp_path / "copy.json"
    same.write_bytes(said.read_bytes())

    result = runner.invoke(app, ["pick", str(said), str(same)])

    assert result.exit_code == 0, result.output
    assert Transcript.load(tmp_path / "meeting.picked.json").engine.params["pick_spots"] == 0
    assert result.stdout == f"{tmp_path / 'meeting.picked.json'}\n"
    assert made[0].calls == []


def test_a_failed_chunk_is_one_stderr_line_and_the_others_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, fail_when=lambda target: "[#1 " in target)

    result = runner.invoke(app, ["pick", *map(str, _long_inputs(tmp_path))])

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines()[-1] == (
        "scribe: the pick failed on 1 of 2 chunks (0); their 1 spots keep the transcript's words"
    )
    words = Transcript.load(tmp_path / "long.picked.json").words
    assert (words[100].text, words[4000].text) == ("w100", "x4000")


def test_every_chunk_failing_exits_four_with_the_input_words_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, fail_when=lambda _target: True)
    inputs = _long_inputs(tmp_path)

    result = runner.invoke(app, ["pick", *map(str, inputs)])

    assert result.exit_code == 4
    written = tmp_path / "long.picked.json"
    assert result.stdout == f"{written}\n"
    assert "failed on 2 of 2 chunks (0, 1)" in result.stderr
    picked = Transcript.load(written)
    assert picked.words == Transcript.load(inputs[0]).words
    assert picked.engine.params["pick_failed"] == 2


def _unsorted(inputs: list[Path]) -> str:
    transcript(_said("the", "cat")[::-1]).dump(inputs[0])
    return f"TRANSCRIPT {inputs[0]} word 1 starts at 0.0 s"


def _unsorted_reference(inputs: list[Path]) -> str:
    transcript(_said("the", "hat")[::-1]).dump(inputs[1])
    return f"REFERENCE {inputs[1]} word 1 starts at 0.0 s"


def _wordless(inputs: list[Path]) -> str:
    transcript([]).dump(inputs[1])
    return f"REFERENCE {inputs[1]} has no words"


def _not_a_transcript(inputs: list[Path]) -> str:
    inputs[0].write_text("{}", encoding="utf-8")
    return "is not a transcript"


def _picked_before(inputs: list[Path]) -> str:
    said = Transcript.load(inputs[0])
    said.model_copy(
        update={"engine": said.engine.model_copy(update={"params": {"pick_record": "[]"}})}
    ).dump(inputs[0])
    return f"TRANSCRIPT {inputs[0]} was picked already"


def _digested(path: Path, digest: str | None) -> None:
    loaded = Transcript.load(path)
    source = loaded.source.model_copy(update={"sha256": digest})
    loaded.model_copy(update={"source": source}).dump(path)


def _other_audio(inputs: list[Path]) -> str:
    _digested(inputs[0], "a" * 64)
    _digested(inputs[1], "b" * 64)
    return f"REFERENCE {inputs[1]} was made from other audio than TRANSCRIPT {inputs[0]}"


@pytest.mark.parametrize(
    "breakage",
    [_unsorted, _unsorted_reference, _wordless, _not_a_transcript, _picked_before, _other_audio],
)
def test_a_bad_input_is_one_stderr_line_and_exit_two_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breakage: Callable[[list[Path]], str]
) -> None:
    made = _patch(monkeypatch)
    inputs = _inputs(tmp_path)
    expected = breakage(inputs)

    result = runner.invoke(app, ["pick", *map(str, inputs)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert len(result.stderr.splitlines()) == 1
    assert expected in result.stderr
    assert made == []
    assert not (tmp_path / "meeting.picked.json").exists()


@pytest.mark.parametrize(
    ("said", "heard"),
    [("a" * 64, None), (None, "b" * 64), ("", "b" * 64), ("a" * 64, "a" * 64)],
    ids=["reference-none", "transcript-none", "transcript-empty", "equal"],
)
def test_inputs_not_both_recording_different_audio_are_picked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: str | None, heard: str | None
) -> None:
    made = _patch(monkeypatch)
    inputs = _inputs(tmp_path)
    _digested(inputs[0], said)
    _digested(inputs[1], heard)

    result = runner.invoke(app, ["pick", *map(str, inputs)])

    assert result.exit_code == 0, result.output
    assert len(made[0].calls) == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_an_unwritable_out_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        result = runner.invoke(
            app, ["pick", *map(str, _inputs(tmp_path)), "--out", str(locked / "p.json")]
        )
    finally:
        locked.chmod(0o755)

    assert result.exit_code == 2
    # The directory is named resolved, which can differ from tmp_path by a link.
    assert result.stderr.startswith("scribe: cannot write artifacts to ")
    assert f"{locked.name}: " in result.stderr
    assert made == []


def _no_proof(_plan: object) -> None:
    return None


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_write_failing_after_the_calls_still_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The proof cannot rule out a disk filling up or a mode changed mid-run.
    monkeypatch.setattr("scribe.cli.prove_writable", _no_proof)
    _patch(monkeypatch)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        result = runner.invoke(
            app, ["pick", *map(str, _inputs(tmp_path)), "--out", str(locked / "p.json")]
        )
    finally:
        locked.chmod(0o755)

    assert result.exit_code == 2
    assert result.stderr.startswith(f"scribe: cannot write transcript to {locked / 'p.json'}: ")
    assert result.stdout == ""


def _nowhere(_name: str) -> str | None:
    return None


def test_no_usable_claude_exits_two_with_a_message_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(*, model: str, disable_tools: bool) -> ClaudeCliBackend:
        return ClaudeCliBackend(model, disable_tools=disable_tools, which=_nowhere)

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)

    result = runner.invoke(app, ["pick", *map(str, _inputs(tmp_path))])

    assert result.exit_code == 2
    assert result.stderr == ("scribe: claude is not on PATH; the backend is the Claude Code CLI\n")
    assert result.stdout == ""
    assert not (tmp_path / "meeting.picked.json").exists()


def test_the_help_says_what_leaves_the_machine_and_what_exit_four_means() -> None:
    result = runner.invoke(app, ["pick", "--help"], terminal_width=200)

    assert result.exit_code == 0
    assert "not the audio, go to Anthropic" in " ".join(result.stdout.split())
    assert "Exit 4" in result.stdout


def _hat_then_unsure(number: int, a: str, _b: str) -> str:
    return "unsure" if number == 2 else ("A" if a == "hat" else "B")


def _picked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], Path]:
    """The inputs, and their pick: "hat" taken at spot 1, spot 2 unsure."""
    _patch(monkeypatch, reply=answering(_hat_then_unsure))
    inputs = _inputs(tmp_path)
    picked = tmp_path / "meeting.picked.json"
    assert runner.invoke(app, ["pick", *map(str, inputs), "--out", str(picked)]).exit_code == 0
    return inputs, picked


def test_sides_from_replays_a_pick_and_asks_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, picked = _picked(tmp_path, monkeypatch)
    made = _patch(monkeypatch)
    out = tmp_path / "replayed.json"

    result = runner.invoke(
        app, ["pick", *map(str, inputs), "--sides-from", str(picked), "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert made == []
    assert result.stdout == f"{out}\n"
    assert result.stderr == (
        f"scribe: replayed the sides {picked} records, asking no model: the reference's "
        f"reading at 1 of 2 disputed spots (1 unsure), as {DEFAULT_PICK_MODEL} picked them "
        f"with prompt {PICK_PROMPT_VERSION}\n"
    )
    replayed, original = Transcript.load(out), Transcript.load(picked)
    assert replayed.text == original.text == "the hat sat on the mat"
    assert replayed.words == original.words
    params, before = replayed.engine.params, original.engine.params
    assert params["pick_record"] == before["pick_record"]
    assert (params["pick_chunks"], params["pick_chunks_failed"], before["pick_chunks"]) == (0, 0, 1)
    same = ("pick_model", "pick_prompt_version", "pick_context_chars", "pick_unsure")
    assert {key: params[key] for key in same} == {key: before[key] for key in same}


def _fewer(record: list[list[object]], _params: dict[str, object]) -> str:
    del record[1]
    return "the record holds 1 spots where these transcripts have 2"


def _other_reading(record: list[list[object]], _params: dict[str, object]) -> str:
    record[1][3] = "cat"
    return 'its spot 2 is [5.0, 5.4, "mat", "cat"], where these transcripts have [5.0, 5.4, "mat", '


def _moved(record: list[list[object]], _params: dict[str, object]) -> str:
    record[0][0] = 0.5
    return "its spot 1 is [0.5, 1.4"


def _unrecorded(_record: list[list[object]], params: dict[str, object]) -> str:
    del params["pick_record"]
    return "has no pick_record"


def _unnamed(_record: list[list[object]], params: dict[str, object]) -> str:
    del params["pick_model"]
    return "names no pick_model"


def _ears_fewer(_record: list[list[object]], params: dict[str, object]) -> str:
    params["ear_record"] = json.dumps([["reference", ["hat", "hat"], ["reference", "reference"]]])
    return "its ear_record holds 1 spots where its pick_record holds 2"


def _ears_malformed(_record: list[list[object]], params: dict[str, object]) -> str:
    unheard = [None, None]
    params["ear_record"] = json.dumps([["heard", unheard, unheard], ["unsure", unheard, unheard]])
    return "has a malformed ear_record row 0"


@pytest.mark.parametrize(
    "breakage",
    [_fewer, _other_reading, _moved, _unrecorded, _unnamed, _ears_fewer, _ears_malformed],
)
def test_sides_from_a_record_of_other_spots_exits_two_asking_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    breakage: Callable[[list[list[object]], dict[str, object]], str],
) -> None:
    inputs, picked = _picked(tmp_path, monkeypatch)
    loaded = Transcript.load(picked)
    params: dict[str, object] = dict(loaded.engine.params)
    record = cast("list[list[object]]", json.loads(str(params["pick_record"])))
    expected = breakage(record, params)
    if "pick_record" in params:
        params["pick_record"] = json.dumps(record)
    engine = loaded.engine.model_copy(update={"params": params})
    loaded.model_copy(update={"engine": engine}).dump(picked)
    made = _patch(monkeypatch)
    out = tmp_path / "replayed.json"

    result = runner.invoke(
        app, ["pick", *map(str, inputs), "--sides-from", str(picked), "--out", str(out)]
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert len(result.stderr.splitlines()) == 1
    assert result.stderr.startswith("scribe: ")
    assert expected in result.stderr
    assert made == []
    assert not out.exists()


def _ears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *texts: list[str],
    host: tuple[str, str] = ("Darwin", "arm64"),
    returncode: int = 0,
) -> tuple[FakeWorker, Path]:
    """Both recognizers cached and faked, each saying its `texts` in turn; and the audio."""
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    for spec in RECOGNIZERS:
        snapshot = hub / f"models--{spec.model.replace('/', '--')}" / "snapshots" / spec.revision
        snapshot.mkdir(parents=True)
        for name in ("config.json", "model.safetensors"):
            (snapshot / name).write_text("{}", encoding="utf-8")
    said, unheard = iter(texts), list[str]()
    versions = {"python": "3.12.13", "transformers": "5.18.0", "torch": "2.14.1"}
    body = {"versions": versions, "device": "mps", "dtype": "bfloat16", "runtime_s": 2.5}
    fake = FakeWorker(
        lambda _: {**body, "texts": next(said, unheard)},
        returncode=returncode,
        stderr=f"{FAILED}out of memory\n",
    )
    ears = LocalEars(run=fake.run, which=found, host=lambda: host)
    monkeypatch.setattr("scribe.cli.LocalEars", lambda: ears)
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"pretend this is audio")
    return fake, audio


@pytest.mark.parametrize("replay", [False, True])
def test_audio_flips_a_spot_only_where_both_recognizers_heard_the_reading_set_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay: bool
) -> None:
    inputs, picked = _picked(tmp_path, monkeypatch)
    made = _patch(monkeypatch, reply=answering(_hat_then_unsure))
    # Spot 1, "cat" or "hat", picked "hat"; spot 2, "mat" or "bat", unsure.
    fake, audio = _ears(
        tmp_path, monkeypatch, ["the cat sat on the bat"] * 2, ["the cat sat on the mat"] * 2
    )
    out = tmp_path / "heard.json"
    sides = ["--sides-from", str(picked)] if replay else []

    result = runner.invoke(
        app, ["pick", *map(str, inputs), *sides, "--audio", str(audio), "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert len(made) == (0 if replay else 1)
    assert result.stderr.splitlines()[1:] == [
        "scribe: third ear heard 2 spots with cohere-transcribe, qwen3-asr in 5 s; "
        "0 flipped to the reference, 1 to the transcript"
    ]
    said = Transcript.load(inputs[0]).words
    windows = [[0.0, said[1].end + 5.0], [0.0, said[5].end + 5.0]]
    assert [json.loads(request)["intervals"] for request in fake.requests] == [windows] * 2
    heard = Transcript.load(out)
    assert heard.text == "the cat sat on the mat"
    params = heard.engine.params
    record = cast("list[list[object]]", json.loads(str(params["pick_record"])))
    assert [row[4] for row in record] == ["transcript", "unsure"]
    assert json.loads(str(params["ear_record"])) == [
        ["reference", ["cat", "cat"], ["transcript", "transcript"]],
        ["unsure", ["bat", "mat"], ["reference", "transcript"]],
    ]
    assert (params["pick_to_reference"], params["pick_unsure"]) == (1, 1)
    models = [f"{spec.name} {spec.model}@{spec.revision}" for spec in RECOGNIZERS]
    assert {key: value for key, value in params.items() if key.startswith("ear_")} == {
        "ear_rule": "ear-1",
        "ear_models": json.dumps(models),
        "ear_runtime": "transformers 5.18.0 torch 2.14.1 mps bfloat16",
        "ear_pad_s": 5.0,
        "ear_seconds": 5.0,
        "ear_to_reference": 0,
        "ear_to_transcript": 1,
        "ear_record": params["ear_record"],
    }


def test_sides_from_a_heard_pick_replays_the_picks_own_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, picked = _picked(tmp_path, monkeypatch)
    _patch(monkeypatch, reply=answering(_hat_then_unsure))
    # Both flip both spots: "hat" to "cat", and the unsure "mat" to "bat".
    _, audio = _ears(
        tmp_path, monkeypatch, ["the cat sat on the bat"] * 2, ["the cat sat on the bat"] * 2
    )
    heard = tmp_path / "heard.json"
    command = ["pick", *map(str, inputs), "--audio", str(audio), "--out", str(heard)]
    assert runner.invoke(app, command).exit_code == 0
    assert Transcript.load(heard).text == "the cat sat on the bat"
    made = _patch(monkeypatch)
    out = tmp_path / "replayed.json"

    result = runner.invoke(
        app, ["pick", *map(str, inputs), "--sides-from", str(heard), "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert made == []
    assert result.stderr == (
        f"scribe: replayed the sides {heard} records, asking no model: the reference's "
        f"reading at 1 of 2 disputed spots (1 unsure), as {DEFAULT_PICK_MODEL} picked them "
        f"with prompt {PICK_PROMPT_VERSION}\n"
    )
    replayed, original = Transcript.load(out), Transcript.load(picked)
    assert replayed.words == original.words
    # The pick's own record and counts, and no ear_* param: the ears did not run.
    params = dict(replayed.engine.params)
    assert params.pop("pick_chunks") == 0
    assert params == {
        key: value for key, value in original.engine.params.items() if key != "pick_chunks"
    }


@pytest.mark.parametrize(
    ("host", "returncode", "cause"),
    [
        (("Linux", "x86_64"), 0, "the recognizers need Apple silicon (macOS on arm64)"),
        (("Darwin", "arm64"), 1, "cohere-transcribe exited 1: out of memory"),
    ],
)
def test_when_the_ears_fail_the_picks_readings_stand_with_one_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: tuple[str, str],
    returncode: int,
    cause: str,
) -> None:
    inputs, picked = _picked(tmp_path, monkeypatch)
    _, audio = _ears(tmp_path, monkeypatch, host=host, returncode=returncode)
    out = tmp_path / "heard.json"

    result = runner.invoke(
        app, ["pick", *map(str, inputs), "--audio", str(audio), "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert out.read_bytes() == picked.read_bytes()
    _, *lines = result.stderr.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"scribe: no third ear: {cause}")
    assert lines[0].endswith("; the pick's readings stand")


@pytest.mark.parametrize("other", [False, True])
def test_audio_not_the_transcripts_recording_exits_two_asking_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other: bool
) -> None:
    inputs = _inputs(tmp_path)
    _digested(inputs[0], "a" * 64)
    made = _patch(monkeypatch)
    audio = tmp_path / "other.mp3"
    if other:
        audio.write_bytes(b"other audio")

    result = runner.invoke(app, ["pick", *map(str, inputs), "--audio", str(audio)])

    assert result.exit_code == 2
    assert len(result.stderr.splitlines()) == 1
    assert str(audio) in result.stderr
    assert made == []


def test_with_no_spot_to_hear_no_recognizer_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch)
    said, _ = _inputs(tmp_path)
    same = tmp_path / "copy.json"
    same.write_bytes(said.read_bytes())
    fake, audio = _ears(tmp_path, monkeypatch)

    result = runner.invoke(app, ["pick", str(said), str(same), "--audio", str(audio)])

    assert result.exit_code == 0, result.output
    assert fake.calls == []
    assert result.stderr.splitlines()[1:] == [
        "scribe: no third ear: there is no spot to hear; the pick's readings stand"
    ]
    assert not any(
        key.startswith("ear_")
        for key in Transcript.load(tmp_path / "meeting.picked.json").engine.params
    )
