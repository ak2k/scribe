"""`scribe pick`: which reading is kept where xAI and Parakeet disagree, from the command line."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from scribe.claude_cli import ClaudeCliBackend
from scribe.cli import app
from scribe.pick import DEFAULT_PICK_MODEL, PICK_PROMPT_VERSION, system_prompt
from scribe.schema import Transcript, Word
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
    assert Transcript.load(written).words == Transcript.load(inputs[0]).words


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


@pytest.mark.parametrize(
    "breakage", [_unsorted, _unsorted_reference, _wordless, _not_a_transcript, _picked_before]
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
