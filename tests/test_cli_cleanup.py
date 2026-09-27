from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner

from scribe.claude_cli import ClaudeCliBackend
from scribe.cleanup import CLEANUP_PROMPT_VERSION, PRIOR_TAIL_HEADER, chunk_turns
from scribe.cli import app
from scribe.errors import ExternalServiceError
from scribe.schema import Engine, Source, Transcript, Turn, Word
from tests.cleanup_fakes import keyed_reply, patch_backend

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()

TALK = "transcript_two_speakers.turns.json"
NUMBERS = "transcript_numbers.turns.json"


def _staged(tmp_path: Path, name: str = TALK) -> Path:
    staged = tmp_path / name
    shutil.copy(FIXTURES / name, staged)
    return staged


def _sidecar(path: Path) -> dict[str, object]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))  # pyright: ignore[reportAny]  # json.loads is Any
    assert isinstance(decoded, dict)
    return cast("dict[str, object]", decoded)


def _drop_2026(text: str) -> str:
    return text.replace("2026", "")


def test_it_writes_the_markdown_and_the_sidecar_beside_the_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 0, result.output
    document = (tmp_path / "transcript_two_speakers.clean.md").read_text(encoding="utf-8")
    assert result.output.strip().endswith(str(tmp_path / "transcript_two_speakers.clean.md"))
    assert document.startswith('---\ntitle: "transcript_two_speakers"\n')
    assert "numbers_checked: 0\n" in document
    assert "numbers_missing: []\n" in document
    assert 'cleanup_backend: "fake-backend"\n' in document
    assert 'cleanup_model: "opus"\n' in document
    assert f'cleanup_prompt_version: "{CLEANUP_PROMPT_VERSION}"\n' in document
    assert "speakers: {}\n" in document
    assert "Speaker 1: Okay, ready?" in document
    sidecar = _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")
    assert list(sidecar) == [
        "backend",
        "model",
        "prompt_version",
        "chunks",
        "truncated_chunks",
        "malformed_chunks",
        "emptied_turns",
        "stripped_labels",
        "completions",
        "numbers_checked",
        "numbers_missing",
        "numbers_reduced",
        "numbers_added",
        "fidelity",
    ]
    assert sidecar["chunks"] == 1
    assert sidecar["prompt_version"] == CLEANUP_PROMPT_VERSION
    assert (sidecar["truncated_chunks"], sidecar["malformed_chunks"]) == ([], [])
    assert sidecar["numbers_missing"] == []
    assert sidecar["completions"] == [
        {"model": "opus", "output_tokens": 44, "stop_reason": "end_turn", "is_error": False}
    ]
    # Metadata only: neither the prompts nor the replies' text land in it.
    assert "Okay, ready?" not in (tmp_path / "transcript_two_speakers.cleanup.json").read_text(
        encoding="utf-8"
    )
    assert len(made[0].calls) == 1


def test_words_no_engine_diarized_are_cleaned_under_speaker_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch)
    said = [("Okay,", 7), ("ready?", 7), ("filled", None), ("words.", None), ("Yes.", 4)]
    words = [
        Word(text=text, start=index, end=index + 0.5, speaker=speaker)
        for index, (text, speaker) in enumerate(said)
    ]
    source = Source(kind="audio", ref="a.mp3")
    Transcript(source=source, engine=Engine(name="xai-stt"), text="", words=words).dump(
        tmp_path / "t.json"
    )

    built = runner.invoke(app, ["turns", str(tmp_path / "t.json"), "--no-llm-speakers"])
    result = runner.invoke(app, ["cleanup", str(tmp_path / "t.turns.json")])

    assert built.exit_code == 0, built.output
    assert result.exit_code == 0, result.output
    document = (tmp_path / "t.clean.md").read_text(encoding="utf-8")
    assert "Speaker 1: Okay, ready?\n\nSpeaker ?: filled words.\n\nSpeaker 2: Yes." in document
    assert result.stderr == ""


def test_a_transcript_without_turns_exits_two_before_any_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path, "transcript_two_speakers.json")

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert "has no turns" in result.output
    assert made == []
    assert list(tmp_path.iterdir()) == [staged]


def test_a_missing_input_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_backend(monkeypatch)

    result = runner.invoke(app, ["cleanup", str(tmp_path / "absent.turns.json")])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1


def test_a_backend_failure_exits_two_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, fail_with=ExternalServiceError("claude -p exited 1: nope"))
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert list(tmp_path.iterdir()) == [staged]


def test_a_dropped_number_exits_three_and_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, mutate=_drop_2026)
    staged = _staged(tmp_path, NUMBERS)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 3
    assert "1 number(s) did not survive cleanup: 2026" in result.output
    document = (tmp_path / "transcript_numbers.clean.md").read_text(encoding="utf-8")
    assert "numbers_checked: 4\n" in document
    assert 'numbers_missing: ["2026"]\n' in document
    assert "$1,234.56" in document
    sidecar = _sidecar(tmp_path / "transcript_numbers.cleanup.json")
    assert sidecar["numbers_checked"] == 4
    assert sidecar["numbers_missing"] == ["2026"]


def test_an_italic_label_cannot_stand_in_for_a_dropped_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unstripped "*Speaker 1*:" would put a "1" on the cleaned side to
    # cancel the "1" the model dropped.
    staged = _staged(tmp_path, NUMBERS)
    transcript = Transcript.load(staged)
    turn = transcript.turns[0].model_copy(update={"text": "we shipped 1 unit"})
    transcript.model_copy(update={"turns": [turn]}).dump(staged)
    patch_backend(
        monkeypatch,
        mutate=lambda reply: reply.replace("<t id=1>", "<t id=1>*Speaker 1*: ").replace(
            " 1 ", " a "
        ),
    )

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 3
    assert "did not survive cleanup: 1" in result.output
    assert _sidecar(tmp_path / "transcript_numbers.cleanup.json")["stripped_labels"] == 1


def test_a_spoken_number_dropped_in_cleanup_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Word-level ASR text spells numbers out, so this is the main input path.
    staged = _staged(tmp_path, NUMBERS)
    transcript = Transcript.load(staged)
    turn = transcript.turns[0].model_copy(update={"text": "we ordered forty two jars"})
    transcript.model_copy(update={"turns": [turn]}).dump(staged)
    patch_backend(monkeypatch, mutate=lambda text: text.replace("forty two ", ""))

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 3
    assert "did not survive cleanup: 42" in result.output
    assert _sidecar(tmp_path / "transcript_numbers.cleanup.json")["numbers_checked"] == 1


def _carried(user: str) -> str:
    """The tail a prompt carries ahead of its turns, or "" when it carries none."""
    if not user.startswith(PRIOR_TAIL_HEADER):
        return ""
    return user.removeprefix(f"{PRIOR_TAIL_HEADER}\n\n").split("\n\n<t id=", 1)[0]


def _echoes_the_tail(user: str) -> str:
    """Copy the carried tail back ahead of the turns, as a model sometimes does."""
    return f"{_carried(user)}\n\n{keyed_reply(user)}"


def _echoes_the_tail_and_drops(user: str) -> str:
    """Copy the carried tail back, against the prompt, and lose chunk 2's figure."""
    return _echoes_the_tail(user).replace("approved 10 thousand dollars", "approved the same")


def test_an_echoed_tail_sets_its_chunk_aside_numbers_and_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tail carries chunk 1's "10 thousand dollars"; kept, a copy of it in
    # chunk 2's reply would count once more and cancel the one chunk 2 lost.
    staged = _staged(tmp_path, NUMBERS)
    transcript = Transcript.load(staged)
    texts = [
        "Good morning everyone let us get started with the first item on the agenda today",
        "Sure the budget for the pilot is 10 thousand dollars as we discussed last week",
        "Great and for the second phase we approved 10 thousand dollars as well so it matches",
    ]
    turns = [
        turn.model_copy(update={"text": text})
        for turn, text in zip(transcript.turns, texts, strict=False)
    ]
    transcript.model_copy(update={"turns": turns}).dump(staged)
    made = patch_backend(monkeypatch, respond=_echoes_the_tail_and_drops)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "32"])

    assert len(made[0].calls) == 2
    assert result.exit_code == 3
    assert result.stderr == (
        "scribe: chunk 1 came back with text outside its turn tags (outside_text); "
        "its input text is kept uncleaned\n"
    )
    document = (tmp_path / "transcript_numbers.clean.md").read_text(encoding="utf-8")
    assert document.count("Good morning everyone") == 1
    assert "\nSpeaker 1: Great and for the second phase we approved 10 thousand" in document
    sidecar = _sidecar(tmp_path / "transcript_numbers.cleanup.json")
    assert sidecar["numbers_reduced"] == []
    assert sidecar["malformed_chunks"] == [{"chunk": 1, "cause": "outside_text"}]


def _restaged(tmp_path: Path, texts: list[str]) -> Path:
    staged = _staged(tmp_path, NUMBERS)
    transcript = Transcript.load(staged)
    turns = [
        turn.model_copy(update={"text": text})
        for turn, text in zip(transcript.turns, texts, strict=False)
    ]
    transcript.model_copy(update={"turns": turns[: len(texts)]}).dump(staged)
    return staged


def test_a_merged_restatement_warns_but_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _restaged(tmp_path, ["There are two things. There are two things here."])
    patch_backend(
        monkeypatch,
        mutate=lambda user: user.replace(
            "There are two things. There are two things here.", "There are 2 things here."
        ),
    )

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 0, result.output
    assert "1 number(s) appear fewer times after cleanup: 2" in result.output
    assert "did not survive" not in result.output
    sidecar = _sidecar(tmp_path / "transcript_numbers.cleanup.json")
    assert sidecar["numbers_missing"] == []
    assert sidecar["numbers_reduced"] == ["2"]
    document = (tmp_path / "transcript_numbers.clean.md").read_text(encoding="utf-8")
    assert 'numbers_reduced: ["2"]\n' in document


def test_a_value_gone_entirely_still_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _restaged(tmp_path, ["We need 10 and 20 of them."])
    patch_backend(monkeypatch, mutate=lambda user: user.replace("10 and 20", "10"))

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 3
    assert "1 number(s) did not survive cleanup: 20" in result.output
    assert "fewer times" not in result.output
    sidecar = _sidecar(tmp_path / "transcript_numbers.cleanup.json")
    assert sidecar["numbers_missing"] == ["20"]
    assert sidecar["numbers_reduced"] == []


def _swap_one_copy(user: str) -> str:
    return user.replace("5 engineers and 5 designers", "5 engineers and 7 designers")


def test_one_copy_of_a_repeated_value_changed_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _restaged(tmp_path, ["We hired 5 engineers and 5 designers."])
    patch_backend(monkeypatch, mutate=_swap_one_copy)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 3
    assert "appear fewer times after cleanup: 5; new after cleanup: 7" in result.output
    sidecar = _sidecar(tmp_path / "transcript_numbers.cleanup.json")
    assert (sidecar["numbers_missing"], sidecar["numbers_reduced"]) == ([], ["5"])
    assert sidecar["numbers_added"] == ["7"]

    result = runner.invoke(app, ["cleanup", str(staged), "--allow-number-drift"])

    assert result.exit_code == 0, result.output
    assert "new after cleanup: 7" in result.output


def test_allow_number_drift_downgrades_the_exit_but_still_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, mutate=_drop_2026)
    staged = _staged(tmp_path, NUMBERS)

    result = runner.invoke(app, ["cleanup", str(staged), "--allow-number-drift"])

    assert result.exit_code == 0, result.output
    assert "did not survive cleanup: 2026" in result.output


def test_a_truncated_chunk_exits_three_whatever_the_drift_flag_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, stop_reasons=["max_tokens"])
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--allow-number-drift"])

    assert result.exit_code == 3
    assert result.stderr == (
        "scribe: chunk 0 stopped at the output limit (max_tokens); "
        "its input text is kept uncleaned; lower --chunk-words\n"
    )
    # The header over an empty document now says the clean bill rests on
    # nothing compared, where a bare `true` read as a verification.
    document = (tmp_path / "transcript_two_speakers.clean.md").read_text(encoding="utf-8")
    assert "numbers_checked: 0\n" in document
    assert _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")["truncated_chunks"] == [0]


def _no_turns(_reply: str) -> str:
    return ""


def _empty_turns(reply: str) -> str:
    return re.sub(r">[^<]*</t>", "></t>", reply)


def _labels_alone(reply: str) -> str:
    # Labels are not speech, so a reply of labels alone holds no words.
    return re.sub(r"(<t id=\d>)[^<]*", r"\1Speaker 1:\n**Speaker 2**:", reply)


def _misfiled(reply: str) -> str:
    """Return turn 2 ahead of turn 1."""
    tags = re.findall(r"<t id=\d+>[^<]*</t>", reply)
    return "".join([tags[1], tags[0], *tags[2:]])


def _closes_early(reply: str) -> str:
    return reply.replace("as planned.</t>", "as</t> planned.")


def _preamble(reply: str) -> str:
    return f"Sure, here it is: {reply}"


def _stray_tag(reply: str) -> str:
    return f"{reply}<t id=9>Stray.</t>"


def _stub_then_copy(reply: str) -> str:
    return reply.replace("<t id=3>", "<t id=3>Noted.</t><t id=3>")


def _drops_id_3(reply: str) -> str:
    return re.sub(r"<t id=3>[^<]*</t>", "", reply)


def _first_words(reply: str) -> str:
    return re.sub(r"(<t id=\d+>)(\S+)[^<]*", r"\1\2", reply)


def _as_said(path: Path) -> str:
    return (
        "\n\n".join(f"{turn.speaker}: {turn.text}" for turn in Transcript.load(path).turns) + "\n"
    )


@pytest.mark.parametrize(
    ("mutate", "cause", "what"),
    [
        pytest.param(
            _closes_early,
            "outside_text",
            "came back with text outside its turn tags",
            id="closed-early",
        ),
        pytest.param(
            _preamble,
            "outside_text",
            "came back with text outside its turn tags",
            id="preamble",
        ),
        pytest.param(
            _stray_tag,
            "foreign_id",
            "came back with an id it was not sent",
            id="foreign",
        ),
        pytest.param(
            _stub_then_copy,
            "repeated_id",
            "came back with one of its turns more than once",
            id="stub-then-a-full-copy",
        ),
        pytest.param(
            _drops_id_3,
            "missing_id",
            "came back without one of its turns",
            id="missing",
        ),
        pytest.param(_no_turns, "missing_id", "came back without one of its turns", id="no-turns"),
        pytest.param(
            _misfiled, "out_of_order", "came back with its turns out of order", id="out-of-order"
        ),
        pytest.param(_empty_turns, "wordless", "came back with no words", id="empty-turns"),
        pytest.param(_labels_alone, "wordless", "came back with no words", id="labels-alone"),
        pytest.param(
            _first_words,
            "short",
            "came back with far fewer words than it was sent",
            id="short",
        ),
    ],
)
def test_a_malformed_reply_exits_three_naming_its_chunk_and_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[str], str],
    cause: str,
    what: str,
) -> None:
    # Fifty-two words, so the word ratio is on.
    patch_backend(monkeypatch, mutate=mutate)
    staged = _staged(tmp_path, NUMBERS)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 3
    # The input stands in for the reply, so no number or word of it is lost.
    assert result.stderr == f"scribe: chunk 0 {what} ({cause}); its input text is kept uncleaned\n"
    document = (tmp_path / "transcript_numbers.clean.md").read_text(encoding="utf-8")
    assert document.split("---\n", 2)[2] == _as_said(staged)
    sidecar = _sidecar(tmp_path / "transcript_numbers.cleanup.json")
    assert sidecar["malformed_chunks"] == [{"chunk": 0, "cause": cause}]
    assert sidecar["truncated_chunks"] == [0]


def test_malformed_chunks_past_the_cap_are_counted_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch, mutate=_no_turns)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "1"])

    chunks = len(made[0].calls)
    assert chunks == 40
    assert result.exit_code == 3
    assert result.stderr.splitlines() == [
        *(
            f"scribe: chunk {index} came back without one of its turns (missing_id); "
            "its input text is kept uncleaned"
            for index in range(10)
        ),
        "scribe: 30 more chunk(s) kept their input text uncleaned; the sidecar names each",
    ]
    sidecar = _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")
    assert sidecar["truncated_chunks"] == list(range(chunks))


def test_a_turn_missing_from_a_finished_reply_exits_three_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, mutate=_drops_id_3)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "20"])

    assert result.exit_code == 3
    assert result.stderr == (
        "scribe: chunk 2 came back without one of its turns (missing_id); "
        "its input text is kept uncleaned\n"
    )
    document = (tmp_path / "transcript_two_speakers.clean.md").read_text(encoding="utf-8")
    assert "\nSpeaker 1: That works for me and I will send notes.\n" in document
    sidecar = _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")
    assert sidecar["malformed_chunks"] == [{"chunk": 2, "cause": "missing_id"}]
    assert sidecar["truncated_chunks"] == [2]


def test_the_sidecar_records_emptied_turns_and_malformed_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # At 20 words a chunk, ids 1, 2 and 3-4 go in three chunks: emptying id 2
    # leaves chunk 1 wordless, and emptying id 4 empties one turn of chunk 2.
    patch_backend(monkeypatch, mutate=lambda reply: re.sub(r"(<t id=[24]>)[^<]*", r"\1", reply))
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "20"])

    assert result.exit_code == 3
    sidecar = _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")
    assert sidecar["emptied_turns"] == 1
    assert sidecar["malformed_chunks"] == [{"chunk": 1, "cause": "wordless"}]


FILLED = " ".join(["alpha bravo charlie delta echo foxtrot golf hotel india juliet"] * 6)


def test_a_chunk_of_filler_alone_returned_empty_keeps_its_input_and_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The second chunk is fillers alone, cleaned to nothing: a reply of no
    # words is no evidence of a cleanup, whatever the chunk was sent.
    staged = _restaged(tmp_path, [FILLED, "um, uh, you know"])
    patch_backend(monkeypatch, mutate=lambda reply: reply.replace(">um, uh, you know</t>", "></t>"))

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "60"])

    assert result.exit_code == 3, result.output
    assert result.stderr == (
        "scribe: chunk 1 came back with no words (wordless); its input text is kept uncleaned\n"
    )
    document = (tmp_path / "transcript_numbers.clean.md").read_text(encoding="utf-8")
    assert document.split("---\n", 2)[2] == f"Speaker 1: {FILLED}\n\nSpeaker 2: um, uh, you know\n"
    sidecar = _sidecar(tmp_path / "transcript_numbers.cleanup.json")
    assert (sidecar["chunks"], sidecar["emptied_turns"]) == (2, 0)
    assert sidecar["malformed_chunks"] == [{"chunk": 1, "cause": "wordless"}]


def test_out_elsewhere_leaves_nothing_beside_the_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    out = tmp_path / "elsewhere" / "two.clean.md"
    out.parent.mkdir()

    result = runner.invoke(app, ["cleanup", str(staged), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["elsewhere", TALK]
    assert sorted(p.name for p in out.parent.iterdir()) == ["two.clean.md", "two.cleanup.json"]


def test_an_out_that_is_a_directory_exits_two_before_any_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.mkdir()

    result = runner.invoke(app, ["cleanup", str(staged), "--out", str(blocker)])

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    # The guard exists to fail before a paid call, not after one.
    assert made == []


def test_a_missing_out_directory_is_an_error_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A typo in the directory part of --out should fail fast, not invent a tree
    # and leave it behind when the run later fails.
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(
        app, ["cleanup", str(staged), "--out", str(tmp_path / "typo" / "out.md")]
    )

    assert result.exit_code == 2
    assert made == []
    assert [p.name for p in tmp_path.iterdir()] == [TALK]


def test_an_out_that_is_the_input_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The input is the output of a paid transcription run; overwriting it with
    # the cleaned markdown loses it with nothing else holding a copy.
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--out", str(staged)])

    assert result.exit_code == 2
    assert made == []
    assert staged.read_bytes() == (FIXTURES / TALK).read_bytes()


@pytest.mark.parametrize("out", ["meeting.md", "meeting.clean.md"])
def test_a_sidecar_that_would_be_the_input_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, out: str
) -> None:
    # The sidecar is named after --out, so an input named like a sidecar can be
    # the path it lands on even though --out itself is elsewhere.
    made = patch_backend(monkeypatch)
    staged = tmp_path / "meeting.cleanup.json"
    shutil.copy(FIXTURES / TALK, staged)

    result = runner.invoke(app, ["cleanup", str(staged), "--out", str(tmp_path / out)])

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert made == []
    assert staged.read_bytes() == (FIXTURES / TALK).read_bytes()
    assert [path.name for path in tmp_path.iterdir()] == [staged.name]


def test_a_sidecar_that_would_be_the_markdown_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    (tmp_path / "out.cleanup.json").symlink_to(tmp_path / "out.md")

    result = runner.invoke(app, ["cleanup", str(staged), "--out", str(tmp_path / "out.md")])

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert made == []
    assert not (tmp_path / "out.md").exists()


@pytest.mark.parametrize(
    ("existing", "link"),
    [(TALK, "out.md"), (TALK, "out.cleanup.json"), ("out.md", "out.cleanup.json")],
)
def test_an_output_hard_linked_to_another_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str, link: str
) -> None:
    # A hard link, like a case-only difference on a case-insensitive volume,
    # names the same file under a path that resolves to a different string.
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    (tmp_path / "out.md").write_text("earlier output\n", encoding="utf-8")
    (tmp_path / link).unlink(missing_ok=True)
    (tmp_path / link).hardlink_to(tmp_path / existing)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    result = runner.invoke(app, ["cleanup", str(staged), "--out", str(tmp_path / "out.md")])

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert made == []
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_a_sidecar_path_that_is_a_directory_exits_two_before_any_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    (tmp_path / "out.cleanup.json").mkdir()

    result = runner.invoke(app, ["cleanup", str(staged), "--out", str(tmp_path / "out.md")])

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert made == []
    assert not (tmp_path / "out.md").exists()


def test_an_unusable_out_directory_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file where a directory must go", encoding="utf-8")

    result = runner.invoke(app, ["cleanup", str(staged), "--out", str(blocker / "out.md")])

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1


def test_a_reply_that_cannot_be_encoded_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A lone surrogate is a str Python accepts and UTF-8 cannot encode, so it
    # fails at the write, after the backend has already answered.
    patch_backend(monkeypatch, mutate=lambda reply: reply.replace("Okay", "a\ud800b"))
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert "cannot write" in result.output


def test_the_relabel_key_and_the_glossary_reach_the_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    speakers = tmp_path / "speakers.txt"
    speakers.write_text("# who is who\nSpeaker 1=Ann Lee\nSpeaker 2=Bo Chen\n", encoding="utf-8")
    glossary = tmp_path / "glossary.txt"
    glossary.write_text("ackme=Acme\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "cleanup",
            str(staged),
            "--speakers-file",
            str(speakers),
            "--speaker",
            "Speaker 2=Bo Chen Jr",
            "--glossary-file",
            str(glossary),
            "--glossary",
            "widgit=widget",
            "--context",
            "a quarterly review",
            "--title",
            "Quarterly review",
        ],
    )

    assert result.exit_code == 0, result.output
    system, user = made[0].calls[0]
    assert user.startswith('<t id=1 speaker="Ann Lee">Okay, ready?')
    # A repeated flag beats the file it came with.
    assert 'speaker="Bo Chen Jr"' in user
    assert "- Ann Lee\n- Bo Chen Jr" in system
    assert '- "ackme" is written as "Acme"' in system
    assert '- "widgit" is written as "widget"' in system
    assert system.rstrip().endswith("a quarterly review")
    document = (tmp_path / "transcript_two_speakers.clean.md").read_text(encoding="utf-8")
    assert 'title: "Quarterly review"\n' in document
    assert '  "Speaker 1": "Ann Lee"\n  "Speaker 2": "Bo Chen Jr"\n' in document


@pytest.mark.parametrize("option", ["--speakers-file", "--glossary-file"])
def test_an_out_that_is_a_pairs_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("Speaker 1=Ann Lee\n", encoding="utf-8")

    result = runner.invoke(app, ["cleanup", str(staged), option, str(pairs), "--out", str(pairs)])

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert made == []
    assert pairs.read_text(encoding="utf-8") == "Speaker 1=Ann Lee\n"


def test_a_malformed_speaker_pair_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--speaker", "Ann Lee"])

    assert result.exit_code == 2
    assert "--speaker expects KEY=VALUE" in result.output
    assert made == []


def test_a_chunk_size_below_one_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing caps a run: --max-budget-usd is per call, so a chunk size of zero
    # buys one paid call per turn.
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "0"])

    assert result.exit_code == 2
    assert made == []


def test_chunk_words_model_and_budget_reach_the_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(
        app,
        [
            "cleanup",
            str(staged),
            "--chunk-words",
            "10",
            "--model",
            "sonnet",
            "--max-budget-usd",
            "2.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert made[0].model == "sonnet"
    assert made[0].max_budget_usd == 2.5
    # Four turns, the 14-word second one cut at a sentence into two pieces.
    assert len(made[0].calls) == 5
    assert _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")["chunks"] == 5


def _reply(argv: list[str], user: object) -> subprocess.CompletedProcess[str]:
    assert isinstance(user, str)
    reply = json.dumps({"is_error": False, "subtype": "success", "result": keyed_reply(user)})
    return subprocess.CompletedProcess(argv, 0, stdout=reply, stderr="")


def _succeed(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return _reply(argv, kwargs["input"])


def _patch_real_backend(
    monkeypatch: pytest.MonkeyPatch, run: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    def which(name: str) -> str | None:
        return f"/opt/nowhere/bin/{name}"

    def factory(*, model: str, max_budget_usd: float) -> ClaudeCliBackend:
        return ClaudeCliBackend(model, max_budget_usd, run=run, which=which)

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)


def test_the_real_backend_keeps_its_logs_off_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_real_backend(monkeypatch, _succeed)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "10"])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{tmp_path / 'transcript_two_speakers.clean.md'}\n"
    assert "claude_cli.executable" in result.stderr
    assert "claude_route" not in _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")


def test_log_lines_carry_no_color_codes_off_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_real_backend(monkeypatch, _succeed)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 0, result.output
    assert "claude_cli.executable" in result.stderr
    assert "\x1b[" not in result.stderr


def _then(
    failure: BaseException | subprocess.CompletedProcess[str],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    calls: list[object] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if len(calls) == 1:
            return _reply(argv, kwargs["input"])
        if isinstance(failure, BaseException):
            raise failure
        return failure

    return run


def test_a_later_call_failing_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    budget = (FIXTURES / "claude_p_budget_error.json").read_text(encoding="utf-8")
    _patch_real_backend(monkeypatch, _then(subprocess.CompletedProcess([], 1, budget, "")))
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "10"])

    assert result.exit_code == 2


def test_a_later_call_crashing_is_not_turned_into_an_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_real_backend(monkeypatch, _then(RuntimeError("not a scribe error")))
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "10"])

    assert isinstance(result.exception, RuntimeError)


FILLER = " ".join(["the team walked through the plan and talked about vendors"] * 5)
REST = "before launch we review the vendor list and the hiring plan with everyone"


def _seam_turns() -> list[tuple[str, str]]:
    """Chunk 1 fills DEFAULT_CHUNK_WORDS exactly; chunk 2 says its last exchange again."""
    exchange = [("Speaker 2", "And the budget?"), ("Speaker 1", "Two million dollars.")]
    opening = [(f"Speaker {1 + index % 2}", FILLER) for index in range(119)]
    closing = [(f"Speaker {1 + index % 2}", FILLER) for index in range(4)]
    return [
        *opening,
        ("Speaker 2", " ".join(FILLER.split()[:44])),
        *exchange,
        exchange[0],
        ("Speaker 1", f"Two million dollars. {REST}"),
        *closing,
    ]


def _rewrites_the_amount(reply: str) -> str:
    """Write the amount in figures."""
    return reply.replace("Two million dollars.", "$2 million.")


def test_a_repeat_really_said_at_a_real_chunk_seam_survives_under_its_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    turns = _seam_turns()
    staged = _staged(tmp_path, NUMBERS)
    transcript = Transcript.load(staged)
    built = [
        Turn(speaker=speaker, start=float(index), end=index + 0.9, text=text)
        for index, (speaker, text) in enumerate(turns)
    ]
    transcript.model_copy(update={"turns": built}).dump(staged)
    made = patch_backend(monkeypatch, mutate=_rewrites_the_amount)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert [len(chunk) for chunk in chunk_turns(built)] == [122, 6]
    assert len(made[0].calls) == 2
    assert result.exit_code == 0, result.output
    assert "fewer times" not in result.output
    assert "did not survive" not in result.output
    sidecar = _sidecar(tmp_path / "transcript_numbers.cleanup.json")
    assert sidecar["numbers_missing"] == []
    assert sidecar["numbers_reduced"] == []
    assert sidecar["numbers_added"] == []
    document = (tmp_path / "transcript_numbers.clean.md").read_text(encoding="utf-8")
    body = document.split("---\n", 2)[2]
    assert body.split("\n\n") == [
        f"{speaker}: {_rewrites_the_amount(text)}" for speaker, text in turns[:-1]
    ] + [f"{turns[-1][0]}: {turns[-1][1]}\n"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
@pytest.mark.parametrize("locked_name", ["out.clean.md", "out.cleanup.json"])
def test_a_read_only_output_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locked_name: str
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    locked = tmp_path / locked_name
    locked.write_text("kept\n", encoding="utf-8")
    locked.chmod(0o444)
    try:
        result = runner.invoke(
            app, ["cleanup", str(staged), "--out", str(tmp_path / "out.clean.md")]
        )
    finally:
        locked.chmod(0o644)

    assert result.exit_code == 2
    assert "cannot write" in result.output
    assert made == []
    assert locked.read_text(encoding="utf-8") == "kept\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_read_only_out_directory_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o555)
    try:
        result = runner.invoke(app, ["cleanup", str(staged), "--out", str(out / "t.clean.md")])
    finally:
        out.chmod(0o755)

    assert result.exit_code == 2
    assert "cannot write" in result.output
    assert made == []
    assert list(out.iterdir()) == []


def _fidelity(tmp_path: Path, stem: str) -> dict[str, object]:
    found = _sidecar(tmp_path / f"{stem}.cleanup.json")["fidelity"]
    assert isinstance(found, dict)
    return cast("dict[str, object]", found)


def _moves_a_sentence(reply: str) -> str:
    """Return the second turn's first sentence under the first turn's id."""
    return reply.replace(
        "</t>\n\n<t id=2>The first item is the budget.",
        " The first item is the budget.</t>\n\n<t id=2>",
    )


def test_a_sentence_moved_to_the_other_speaker_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, mutate=_moves_a_sentence)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 3
    assert (
        'scribe: 6 word(s) moved to another speaker in cleanup: turn 2 Speaker 2 to Speaker 1 "the'
        in result.stderr
    )
    fidelity = _fidelity(tmp_path, "transcript_two_speakers")
    assert fidelity["moved_words"] == 6
    assert fidelity["moved_spans"] == [
        {
            "turn": 2,
            "from_speaker": "Speaker 2",
            "to_speaker": "Speaker 1",
            "words": "the first item is the budget",
        }
    ]
    assert fidelity["content_edit_words"] == 0


def _moves_words_across_a_split_turn(reply: str) -> str:
    """Return the end of id 3, turn 2's second piece, under id 4, the next speaker's."""
    return reply.replace(" after lunch.", "").replace("<t id=4>", "<t id=4>After lunch. ")


def test_a_moved_span_names_the_id_of_the_piece_the_words_were_said_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, mutate=_moves_words_across_a_split_turn)
    staged = _staged(tmp_path)

    # At 10 words a chunk, turn 2's 14 words go as two pieces, ids 2 and 3.
    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "10"])

    assert result.exit_code == 3
    assert (
        "scribe: 2 word(s) moved to another speaker in cleanup: "
        'turn 3 Speaker 2 to Speaker 1 "after lunch"\n'
    ) in result.stderr
    assert _fidelity(tmp_path, "transcript_two_speakers")["moved_spans"] == [
        {"turn": 3, "from_speaker": "Speaker 2", "to_speaker": "Speaker 1", "words": "after lunch"}
    ]


def test_allow_speaker_moves_downgrades_the_exit_but_still_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, mutate=_moves_a_sentence)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--allow-speaker-moves"])

    assert result.exit_code == 0, result.output
    assert "6 word(s) moved to another speaker" in result.stderr


def test_a_faithful_cleanup_warns_about_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert _fidelity(tmp_path, "transcript_two_speakers") == {
        "words_checked": 40,
        "moved_words": 0,
        "moved_spans": [],
        "content_edit_words": 0,
        "content_edits_per_1000": 0.0,
        "content_spans": [],
    }


def test_fillers_stutters_and_case_are_not_content_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def copy_edits(user: str) -> str:
        for spoken, written in [
            ("Good morning, um, let", "GOOD MORNING. Let"),
            ("uptake, you know, the uptake", "uptake"),
            ("so we, we carry", "so we carry"),
        ]:
            user = user.replace(spoken, written)
        return user

    patch_backend(monkeypatch, mutate=copy_edits)
    staged = _staged(tmp_path, NUMBERS)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 0, result.output
    assert "GOOD MORNING. Let us" in (tmp_path / "transcript_numbers.clean.md").read_text(
        encoding="utf-8"
    )
    assert _fidelity(tmp_path, "transcript_numbers")["content_edit_words"] == 0
    assert "content word(s)" not in result.stderr


def test_a_paraphrase_with_no_number_warns_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(
        monkeypatch,
        mutate=lambda user: user.replace("We can review the numbers after lunch.", "Lunch first."),
    )
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 0, result.output
    assert "scribe: 7 content word(s) changed in cleanup (175.0 per 1,000)" in result.stderr
    fidelity = _fidelity(tmp_path, "transcript_two_speakers")
    assert fidelity["content_edit_words"] == 7
    assert fidelity["content_spans"] == [
        {"turn": 2, "before": "we can review the numbers after", "after": ""},
        {"turn": 2, "before": "", "after": "first"},
    ]
    assert fidelity["moved_words"] == 0


def test_a_glossary_substitution_is_not_a_content_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _restaged(tmp_path, ["We moved the ackme build to Friday."])
    patch_backend(monkeypatch, mutate=lambda user: user.replace("ackme", "Acme"))

    result = runner.invoke(app, ["cleanup", str(staged), "--glossary", "ackme=Acme"])

    assert result.exit_code == 0, result.output
    assert _fidelity(tmp_path, "transcript_numbers")["content_edit_words"] == 0


def test_a_two_chunk_run_with_a_carried_tail_moves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--chunk-words", "20"])

    assert len(made[0].calls) == 3
    assert all(_carried(user) for _, user in made[0].calls[1:])
    assert result.exit_code == 0, result.output
    fidelity = _fidelity(tmp_path, "transcript_two_speakers")
    assert fidelity["moved_words"] == 0
    assert fidelity["content_edit_words"] == 0
    assert _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")["malformed_chunks"] == []


def test_a_label_the_reply_writes_inside_a_turn_is_not_read_as_a_change_of_speaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Kept, "Speaker 1:" would open a line of Speaker 2's turn and give the
    # rest of it to Speaker 1.
    patch_backend(
        monkeypatch,
        mutate=lambda reply: reply.replace("Right. We can", "Right.\n\nSpeaker 1: We can"),
    )
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 0, result.output
    document = (tmp_path / "transcript_two_speakers.clean.md").read_text(encoding="utf-8")
    assert "Right.\n\nWe can review the numbers after lunch.\n" in document
    assert _fidelity(tmp_path, "transcript_two_speakers")["moved_words"] == 0
    assert _sidecar(tmp_path / "transcript_two_speakers.cleanup.json")["stripped_labels"] == 1


def test_speakers_renamed_by_the_key_are_not_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reply names the speaker itself, re-cased; the label written is the key's.
    patch_backend(monkeypatch, mutate=lambda reply: reply.replace("<t id=2>", "<t id=2>BO: "))
    staged = _staged(tmp_path)

    result = runner.invoke(
        app,
        ["cleanup", str(staged), "--speaker", "Speaker 1=Ann Lee", "--speaker", "Speaker 2=Bo"],
    )

    assert result.exit_code == 0, result.output
    document = (tmp_path / "transcript_two_speakers.clean.md").read_text(encoding="utf-8")
    assert "\nBo: The first item" in document
    assert "BO:" not in document
    fidelity = _fidelity(tmp_path, "transcript_two_speakers")
    assert fidelity["moved_words"] == 0
    assert fidelity["content_edit_words"] == 0


def _with_fill_ranges(path: Path, recorded: str) -> None:
    transcript = Transcript.load(path)
    engine = transcript.engine.model_copy(update={"params": {"fill_ranges": recorded}})
    transcript.model_copy(update={"engine": engine}).dump(path)


def _without_generated_at(document: bytes) -> bytes:
    lines = document.splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith(b"generated_at: ")]
    assert len(lines) - len(kept) == 1
    return b"".join(kept)


def _snapshot(directory: Path) -> dict[str, bytes | None]:
    return {
        path.name: path.read_bytes() if path.is_file() else None for path in directory.iterdir()
    }


_TALK_CLEANED = (
    "---\n"
    'title: "transcript_two_speakers"\n'
    'source_kind: "audio"\n'
    'source_ref: "fixture.mp3"\n'
    'stt_engine: "xai-stt"\n'
    'stt_model: "grok-voice-transcribe-2.0"\n'
    'cleanup_backend: "fake-backend"\n'
    'cleanup_model: "opus"\n'
    f'cleanup_prompt_version: "{CLEANUP_PROMPT_VERSION}"\n'
    "speakers: {}\n"
    "numbers_checked: 0\n"
    "numbers_missing: []\n"
    "numbers_reduced: []\n"
    "numbers_added: []\n"
    "---\n"
    "Speaker 1: Okay, ready? Yes, let us start with the agenda items.\n\n"
    "Speaker 2: The first item is the budget. Right. We can review the numbers after lunch.\n\n"
    "Speaker 1: That works for me and I will send notes.\n\n"
    "Speaker 2: Great, so we are done for today.\n"
)
_TALK_SIDECAR = f"""\
{{
  "backend": "fake-backend",
  "model": "opus",
  "prompt_version": "{CLEANUP_PROMPT_VERSION}",
  "chunks": 1,
  "truncated_chunks": [],
  "malformed_chunks": [],
  "emptied_turns": 0,
  "stripped_labels": 0,
  "completions": [
    {{
      "model": "opus",
      "output_tokens": 44,
      "stop_reason": "end_turn",
      "is_error": false
    }}
  ],
  "numbers_checked": 0,
  "numbers_missing": [],
  "numbers_reduced": [],
  "numbers_added": [],
  "fidelity": {{
    "words_checked": 40,
    "moved_words": 0,
    "moved_spans": [],
    "content_edit_words": 0,
    "content_edits_per_1000": 0.0,
    "content_spans": []
  }}
}}
"""


@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param(None, id="no-fill"),
        pytest.param("[[7.2, 9.4]]", id="fill"),
        # Read only for the reading copy, so not an error without one.
        pytest.param("[[9.4, 7.2]]", id="malformed-fill"),
    ],
)
def test_without_a_reading_copy_the_outputs_are_as_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str | None
) -> None:
    patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    if recorded is not None:
        _with_fill_ranges(staged, recorded)

    result = runner.invoke(app, ["cleanup", str(staged)])

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "transcript_two_speakers.clean.md",
        "transcript_two_speakers.cleanup.json",
        TALK,
    ]
    document = (tmp_path / "transcript_two_speakers.clean.md").read_bytes()
    assert _without_generated_at(document) == _TALK_CLEANED.encode()
    sidecar = (tmp_path / "transcript_two_speakers.cleanup.json").read_bytes()
    assert sidecar == _TALK_SIDECAR.encode()


@pytest.mark.parametrize(
    ("name", "mutate", "code"),
    [
        pytest.param(TALK, None, 0, id="faithful"),
        pytest.param(NUMBERS, _drop_2026, 3, id="number-dropped"),
        pytest.param(TALK, _moves_a_sentence, 3, id="speaker-moved"),
    ],
)
def test_the_reading_copy_changes_no_other_output_and_no_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mutate: Callable[[str], str] | None,
    code: int,
) -> None:
    patch_backend(monkeypatch, mutate=mutate)
    staged = _staged(tmp_path, name)
    _with_fill_ranges(staged, "[[7.2, 9.4]]")
    front = tmp_path / "front.md"
    front.write_bytes(b"Board meeting\n")
    copy = tmp_path / "with" / "t.curated.md"
    runs: dict[str, tuple[int, str, bytes, bytes]] = {}
    for run, extra in [("without", []), ("with", ["--curated", str(copy), "--front", str(front)])]:
        out = tmp_path / run / "t.clean.md"
        out.parent.mkdir()
        result = runner.invoke(app, ["cleanup", str(staged), "--out", str(out), *extra])
        runs[run] = (
            result.exit_code,
            result.stderr,
            _without_generated_at(out.read_bytes()),
            (out.parent / "t.cleanup.json").read_bytes(),
        )

    assert runs["with"] == runs["without"]
    assert runs["without"][0] == code
    assert copy.read_bytes().startswith(b"Board meeting\n\n---\n\n**Speaker 1 | 00:00:0")


def test_the_reading_copy_is_the_front_then_each_turn_under_its_label_and_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    _with_fill_ranges(staged, "[[7.2, 9.4]]")
    front = tmp_path / "front.md"
    front.write_bytes(b"# Board meeting\n\nAttendees: Ann Lee, Bo Chen")
    copy = tmp_path / "t.curated.md"

    result = runner.invoke(
        app,
        [
            "cleanup",
            str(staged),
            "--speaker",
            "Speaker 1=Ann Lee",
            "--curated",
            str(copy),
            "--front",
            str(front),
        ],
    )

    assert result.exit_code == 0, result.output
    assert copy.read_text(encoding="utf-8") == (
        "# Board meeting\n\nAttendees: Ann Lee, Bo Chen\n"
        "\n---\n\n"
        "**Ann Lee | 00:00:00**\n"
        "Okay, ready? Yes, let us start with the agenda items.\n\n"
        "**Speaker 2 | 00:00:06**\n"
        "[Includes speech recovered by a second transcription pass, "
        "00:00:07\N{EN DASH}00:00:10.] "
        "The first item is the budget. Right. We can review the numbers after lunch.\n\n"
        "**Ann Lee | 00:00:18**\n"
        "That works for me and I will send notes.\n\n"
        "**Speaker 2 | 00:00:24**\n"
        "Great, so we are done for today.\n"
    )
    # The copy is written last, and the path printed is still the markdown's.
    assert result.stdout.strip() == str(tmp_path / "transcript_two_speakers.clean.md")


@pytest.mark.parametrize(
    "recorded",
    [pytest.param(None, id="no-fill"), pytest.param("[]", id="empty-fill")],
)
def test_no_turn_is_noted_without_a_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str | None
) -> None:
    patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    if recorded is not None:
        _with_fill_ranges(staged, recorded)
    copy = tmp_path / "t.curated.md"

    result = runner.invoke(app, ["cleanup", str(staged), "--curated", str(copy)])

    assert result.exit_code == 0, result.output
    text = copy.read_text(encoding="utf-8")
    assert text.startswith("**Speaker 1 | 00:00:00**\nOkay, ready?")
    assert text.count("\n**Speaker") == 3
    assert "[" not in text


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("Zoë and Łukasz\n".encode(), id="non-ascii"),
        pytest.param(b"Board meeting", id="no-final-newline"),
        pytest.param(b"Board meeting\r\nAttendees\r\n", id="crlf"),
        pytest.param(b"\xef\xbb\xbfBoard meeting\n", id="bom"),
    ],
)
def test_the_front_file_opens_the_reading_copy_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    front = tmp_path / "front.md"
    front.write_bytes(raw)
    copy = tmp_path / "t.curated.md"

    result = runner.invoke(
        app, ["cleanup", str(staged), "--curated", str(copy), "--front", str(front)]
    )

    assert result.exit_code == 0, result.output
    written = copy.read_bytes()
    assert written[: len(raw)] == raw
    rule = b"\n---\n\n" if raw.endswith(b"\n") else b"\n\n---\n\n"
    assert written[len(raw) :].startswith(rule + b"**Speaker 1 | 00:00:00**\n")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(["--front", "{front}"], "--front needs --curated", id="no-copy"),
        pytest.param(
            ["--curated", "{dir}/t.curated.md", "--front", "{dir}/absent.md"],
            "cannot read --front",
            id="front-missing",
        ),
        pytest.param(
            ["--curated", "{dir}/t.curated.md", "--front", "{latin}"],
            "cannot read --front",
            id="front-not-utf-8",
        ),
        pytest.param(
            ["--curated", "{dir}/t.curated.md", "--front", "{dir}"],
            "cannot read --front",
            id="front-a-directory",
        ),
        pytest.param(["--curated", "{input}"], "is the input transcript", id="copy-is-input"),
        pytest.param(
            ["--curated", "{front}", "--front", "{front}"],
            "is the --front file",
            id="copy-is-front",
        ),
        pytest.param(
            ["--curated", "{dir}/transcript_two_speakers.clean.md"],
            "is the same file as --out",
            id="copy-is-out",
        ),
        pytest.param(
            ["--curated", "{dir}/Transcript_Two_Speakers.clean.md"],
            "is the same file as --out",
            id="copy-is-out-but-for-case",
        ),
        pytest.param(
            ["--curated", "{dir}/transcript_two_speakers.cleanup.json"],
            "is the same file as sidecar",
            id="copy-is-sidecar",
        ),
        pytest.param(["--curated", "{dir}"], "is a directory", id="copy-a-directory"),
        pytest.param(
            ["--curated", "{dir}/typo/t.curated.md"], "not a directory", id="copy-dir-missing"
        ),
    ],
)
def test_a_bad_reading_copy_option_exits_two_before_any_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str], message: str
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    front = tmp_path / "front.md"
    front.write_bytes(b"Board meeting\n")
    latin = tmp_path / "latin.md"
    latin.write_bytes("Zoë\n".encode("latin-1"))
    before = _snapshot(tmp_path)
    paths = {"dir": str(tmp_path), "front": str(front), "latin": str(latin), "input": str(staged)}

    result = runner.invoke(app, ["cleanup", str(staged), *(arg.format(**paths) for arg in args)])

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert message in result.output
    assert made == []
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param("[[9.4, 7.2]]", id="reversed"),
        pytest.param('[[7.2, "9.4"]]', id="quoted"),
        pytest.param("[[7.2, 9.4]", id="not-json"),
    ],
)
def test_a_malformed_fill_record_exits_two_before_any_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    _with_fill_ranges(staged, recorded)
    before = _snapshot(tmp_path)

    result = runner.invoke(
        app, ["cleanup", str(staged), "--curated", str(tmp_path / "t.curated.md")]
    )

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert "has a malformed fill_ranges" in result.output
    assert made == []
    assert _snapshot(tmp_path) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_read_only_reading_copy_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = patch_backend(monkeypatch)
    staged = _staged(tmp_path)
    copy = tmp_path / "t.curated.md"
    copy.write_text("kept\n", encoding="utf-8")
    copy.chmod(0o444)
    try:
        result = runner.invoke(app, ["cleanup", str(staged), "--curated", str(copy)])
    finally:
        copy.chmod(0o644)

    assert result.exit_code == 2
    assert "cannot write" in result.output
    assert made == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["t.curated.md", TALK]
    assert copy.read_text(encoding="utf-8") == "kept\n"


def test_a_backend_failure_writes_no_reading_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_backend(monkeypatch, fail_with=ExternalServiceError("claude -p exited 1: nope"))
    staged = _staged(tmp_path)

    result = runner.invoke(
        app, ["cleanup", str(staged), "--curated", str(tmp_path / "t.curated.md")]
    )

    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert list(tmp_path.iterdir()) == [staged]


def test_a_reading_copy_that_cannot_be_written_exits_two_after_the_other_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "t.curated.md"

    def respond(user: str) -> str:
        # Taken after the outputs were proven writable, as another process could.
        copy.mkdir(exist_ok=True)
        return keyed_reply(user)

    patch_backend(monkeypatch, respond=respond)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["cleanup", str(staged), "--curated", str(copy)])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert f"cannot write {copy}" in result.output
    document = (tmp_path / "transcript_two_speakers.clean.md").read_bytes()
    assert _without_generated_at(document) == _TALK_CLEANED.encode()
    sidecar = (tmp_path / "transcript_two_speakers.cleanup.json").read_bytes()
    assert sidecar == _TALK_SIDECAR.encode()
