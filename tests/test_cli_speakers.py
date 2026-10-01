from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner

from scribe.claude_cli import ClaudeCliBackend
from scribe.cli import app
from scribe.schema import Engine, Source, Transcript, Word
from scribe.speakers import DEFAULT_SPEAKER_MODEL, SPEAKER_PROMPT_VERSION
from scribe.turns import build_turns
from scribe.writers import to_markdown, to_srt, to_vtt
from tests.speakers_fakes import FakeSpeakerBackend, echo, service_error

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURES = Path(__file__).parent / "fixtures"
TALK = "transcript_two_speakers.json"
ALL_FORMATS = ["--format", "json", "--format", "md", "--format", "srt", "--format", "vtt"]
runner = CliRunner()


def _staged(tmp_path: Path) -> Path:
    staged = tmp_path / TALK
    shutil.copy(FIXTURES / TALK, staged)
    return staged


def _written(tmp_path: Path, *runs: tuple[int | None, int]) -> Path:
    """(speaker, word count) runs, one word per second, every tenth word ending a sentence."""
    said = [speaker for speaker, count in runs for _ in range(count)]
    words = [
        Word(
            text=f"w{index}." if index % 10 == 9 else f"w{index}",
            start=float(index),
            end=index + 0.9,
            speaker=speaker,
        )
        for index, speaker in enumerate(said)
    ]
    staged = tmp_path / "inline.json"
    Transcript(
        source=Source(kind="audio", ref="inline.mp3"),
        engine=Engine(name="xai-stt"),
        text=" ".join(word.text for word in words),
        words=words,
    ).dump(staged)
    return staged


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: Callable[[str], str] | None = None,
    fail_when: Callable[[str], bool] | None = None,
    fail_with: Callable[[], BaseException] | None = None,
    wrap: bool = True,
    trailer: Callable[[str], str] | None = None,
) -> list[FakeSpeakerBackend]:
    made: list[FakeSpeakerBackend] = []

    def factory(*, model: str, disable_tools: bool) -> FakeSpeakerBackend:
        assert disable_tools
        backend = FakeSpeakerBackend(
            model=model,
            reply=echo if reply is None else reply,
            fail_when=fail_when,
            fail_with=service_error if fail_with is None else fail_with,
            wrap=wrap,
            trailer=trailer,
        )
        made.append(backend)
        return backend

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)
    return made


def _sidecar(path: Path) -> dict[str, object]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))  # pyright: ignore[reportAny]  # json.loads is Any
    assert isinstance(decoded, dict)
    return cast("dict[str, object]", decoded)


def _chunks(sidecar: dict[str, object]) -> list[dict[str, object]]:
    chunks = sidecar["chunks"]
    assert isinstance(chunks, list)
    return cast("list[dict[str, object]]", chunks)


def _swap_ranks(target: str) -> str:
    return (
        target.replace("<spk:0>", "<spk:x>")
        .replace("<spk:1>", "<spk:0>")
        .replace("<spk:x>", "<spk:1>")
    )


def _artifacts(directory: Path, stem: str) -> dict[str, bytes]:
    return {
        suffix: (directory / f"{stem}{suffix}").read_bytes()
        for suffix in (".turns.json", ".md", ".srt", ".vtt")
    }


def test_the_skip_flag_writes_what_the_diarizer_alone_builds(tmp_path: Path) -> None:
    staged = _staged(tmp_path)
    out = tmp_path / "out"

    result = runner.invoke(
        app, ["turns", str(staged), "--no-llm-speakers", "--out-dir", str(out), *ALL_FORMATS]
    )

    assert result.exit_code == 0, result.output
    transcript = Transcript.load(staged)
    expected = transcript.model_copy(update={"turns": build_turns(transcript.words)})
    expected.dump(tmp_path / "expected.turns.json")
    assert _artifacts(out, "transcript_two_speakers") == {
        ".turns.json": (tmp_path / "expected.turns.json").read_bytes(),
        ".md": to_markdown(expected).encode(),
        ".srt": to_srt(expected).encode(),
        ".vtt": to_vtt(expected).encode(),
    }
    assert not (out / "transcript_two_speakers.speakers.json").exists()


def test_the_pass_runs_by_default_and_records_its_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    staged = _staged(tmp_path)
    plain, relabeled = tmp_path / "plain", tmp_path / "relabeled"

    skipped = runner.invoke(
        app, ["turns", str(staged), "--no-llm-speakers", "--out-dir", str(plain), *ALL_FORMATS]
    )
    result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(relabeled), *ALL_FORMATS])

    assert skipped.exit_code == 0, skipped.output
    assert result.exit_code == 0, result.output
    assert len(made) == 1
    assert made[0].model == DEFAULT_SPEAKER_MODEL
    assert len(made[0].calls) == 1
    # An echoed reply moves nothing.
    assert _artifacts(relabeled, "transcript_two_speakers") == _artifacts(
        plain, "transcript_two_speakers"
    )
    sidecar = _sidecar(relabeled / "transcript_two_speakers.speakers.json")
    assert sidecar == {
        "backend": "fake-backend",
        "model": DEFAULT_SPEAKER_MODEL,
        "prompt_version": SPEAKER_PROMPT_VERSION,
        "words": 40,
        "words_relabeled": 0,
        "chunks_failed": [],
        "chunks": [
            {
                "index": 0,
                "start": 0,
                "end": 40,
                "status": "ok",
                "reason": None,
                "stop_reason": "end_turn",
                "reply_words": 40,
                "aligned_words": 40,
                "relabeled": 0,
            }
        ],
    }
    assert result.stderr == ""


def test_the_speaker_model_reaches_the_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["turns", str(staged), "--speaker-model", "sonnet"])

    assert result.exit_code == 0, result.output
    assert made[0].model == "sonnet"
    assert _sidecar(tmp_path / "transcript_two_speakers.speakers.json")["model"] == "sonnet"


def test_every_artifact_carries_the_relabeled_speakers_and_the_input_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, reply=_swap_ranks)
    staged = _written(tmp_path, (7, 20), (4, 10), (7, 10))

    result = runner.invoke(app, ["turns", str(staged), *ALL_FORMATS])

    assert result.exit_code == 0, result.output
    original = Transcript.load(staged)
    written = Transcript.load(tmp_path / "inline.turns.json")
    # Word.speaker keeps the diarization id; only the turns carry the relabel.
    assert written.words == original.words
    assert [(turn.speaker, turn.text.split()[0]) for turn in written.turns] == [
        ("Speaker 1", "w0"),
        ("Speaker 2", "w20"),
        ("Speaker 1", "w30"),
    ]
    # Labels rank by first appearance, so a whole-speaker swap keeps their names.
    assert _sidecar(tmp_path / "inline.speakers.json")["words_relabeled"] == 40
    for suffix, render in ((".md", to_markdown), (".srt", to_srt), (".vtt", to_vtt)):
        assert (tmp_path / f"inline{suffix}").read_text(encoding="utf-8") == render(written)


def _moves_w20_to_rank_zero(target: str) -> str:
    return target.replace("<spk:1> w20", "<spk:0> w20 <spk:1>")


def test_a_moved_word_changes_the_turns_and_every_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, reply=_moves_w20_to_rank_zero)
    staged = _written(tmp_path, (7, 20), (4, 10))

    result = runner.invoke(app, ["turns", str(staged), *ALL_FORMATS])

    assert result.exit_code == 0, result.output
    written = Transcript.load(tmp_path / "inline.turns.json")
    assert [turn.text.split()[0] for turn in written.turns] == ["w0", "w21"]
    assert written.words == Transcript.load(staged).words
    assert "w19. w20\n" in (tmp_path / "inline.md").read_text(encoding="utf-8")
    assert "w19. w20\n" in (tmp_path / "inline.srt").read_text(encoding="utf-8")
    assert "w19. w20\n" in (tmp_path / "inline.vtt").read_text(encoding="utf-8")
    assert _sidecar(tmp_path / "inline.speakers.json")["words_relabeled"] == 1


def _fails_on_w900(target: str) -> bool:
    return " w900 " in f" {target} "


def test_a_failed_chunk_is_one_stderr_line_and_a_sidecar_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, reply=_swap_ranks, fail_when=_fails_on_w900)
    staged = _written(tmp_path, (1, 1000), (2, 1000))

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 0, result.output
    summary = [line for line in result.stderr.splitlines() if line.startswith("scribe:")]
    assert summary == [
        "scribe: the speaker pass failed on 1 of 3 chunks (1); "
        "their words keep the diarizer's speakers"
    ]
    assert "speakers.chunk_failed" in result.stderr
    sidecar = _sidecar(tmp_path / "inline.speakers.json")
    assert sidecar["chunks_failed"] == [1]
    assert [chunk["status"] for chunk in _chunks(sidecar)] == ["ok", "failed", "ok"]
    assert _chunks(sidecar)[1]["reason"] == "ExternalServiceError"
    assert result.stdout == ""


def _always(_target: str) -> bool:
    return True


def test_every_chunk_failing_exits_four_with_the_diarizer_speakers_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, fail_when=_always)
    staged = _written(tmp_path, (1, 1000), (2, 1000))
    plain = tmp_path / "plain"

    skipped = runner.invoke(
        app, ["turns", str(staged), "--no-llm-speakers", "--out-dir", str(plain)]
    )
    result = runner.invoke(app, ["turns", str(staged)])

    assert skipped.exit_code == 0, skipped.output
    assert result.exit_code == 4
    assert (tmp_path / "inline.turns.json").read_bytes() == (
        plain / "inline.turns.json"
    ).read_bytes()
    assert (tmp_path / "inline.md").read_bytes() == (plain / "inline.md").read_bytes()
    assert _sidecar(tmp_path / "inline.speakers.json")["chunks_failed"] == [0, 1, 2]
    assert "failed on 3 of 3 chunks" in result.stderr


def test_exit_four_is_documented() -> None:
    result = runner.invoke(app, ["turns", "--help"], terminal_width=200)

    assert result.exit_code == 0
    assert "Exit 4" in result.stdout
    assert "--no-llm-speakers" in result.stdout


def _nowhere(_name: str) -> str | None:
    return None


def test_no_usable_claude_exits_two_naming_the_skip_flag_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(*, model: str, disable_tools: bool) -> ClaudeCliBackend:
        return ClaudeCliBackend(model, disable_tools=disable_tools, which=_nowhere)

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)
    staged = _staged(tmp_path)
    out = tmp_path / "new"

    result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(out)])

    assert result.exit_code == 2
    assert "claude is not on PATH" in result.stderr
    assert "--no-llm-speakers" in result.stderr
    assert result.stdout == ""
    assert not out.exists()


def test_stdout_mode_prints_markdown_and_puts_provenance_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    staged = _staged(tmp_path)

    plain = runner.invoke(app, ["turns", str(staged), "--stdout", "--no-llm-speakers"])
    result = runner.invoke(app, ["turns", str(staged), "--stdout"])

    assert result.exit_code == 0, result.output
    assert result.stdout == plain.stdout
    assert result.stderr == (
        f"scribe: speakers relabeled by {DEFAULT_SPEAKER_MODEL} (prompt "
        f"{SPEAKER_PROMPT_VERSION}): 0 of 40 words changed, "
        "1 of 1 chunks answered\n"
    )
    assert len(made[0].calls) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [TALK]


def test_a_failed_chunk_under_stdout_keeps_warnings_off_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, fail_when=_always)
    staged = _staged(tmp_path)

    plain = runner.invoke(app, ["turns", str(staged), "--stdout", "--no-llm-speakers"])
    result = runner.invoke(app, ["turns", str(staged), "--stdout"])

    assert result.exit_code == 4
    assert result.stdout == plain.stdout
    assert "speakers.chunk_failed" in result.stderr


def test_one_speaker_asks_no_model(tmp_path: Path) -> None:
    # The autouse guard fails any construction of the real backend.
    staged = _written(tmp_path, (3, 30))

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "inline.speakers.json").exists()


def test_an_unusable_sidecar_path_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    staged = _staged(tmp_path)
    (tmp_path / "transcript_two_speakers.speakers.json").mkdir()

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 2
    assert "speakers sidecar" in result.stderr
    assert made == []


def test_an_out_dir_below_a_file_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    staged = _staged(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")

    result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(blocker / "sub")])

    assert result.exit_code == 2
    assert "cannot write artifacts" in result.stderr
    assert made[0].calls == []


def _refusal(_target: str) -> str:
    return "I can't help with that."


def test_replies_that_carry_no_labels_exit_four_like_failed_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, reply=_refusal, wrap=False)
    staged = _written(tmp_path, (1, 1000), (2, 1000))

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 4
    sidecar = _sidecar(tmp_path / "inline.speakers.json")
    assert sidecar["chunks_failed"] == [0, 1, 2]
    assert {chunk["reason"] for chunk in _chunks(sidecar)} == {"no_out_block"}
    assert "failed on 3 of 3 chunks" in result.stderr


def _undecodable() -> BaseException:
    return UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


def test_an_unexpected_error_in_one_chunk_keeps_the_others_and_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, reply=_swap_ranks, fail_when=_fails_on_w900, fail_with=_undecodable)
    staged = _written(tmp_path, (1, 1000), (2, 1000))

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 0, result.output
    sidecar = _sidecar(tmp_path / "inline.speakers.json")
    assert [chunk["status"] for chunk in _chunks(sidecar)] == ["ok", "failed", "ok"]
    assert _chunks(sidecar)[1]["reason"] == "error: UnicodeDecodeError"
    assert sidecar["words_relabeled"] == 1300
    assert (tmp_path / "inline.turns.json").is_file()
    assert (tmp_path / "inline.md").is_file()


def test_a_rerun_without_the_pass_removes_the_sidecar_the_last_run_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, reply=_moves_w20_to_rank_zero)
    staged = _written(tmp_path, (7, 20), (4, 10))
    plain = tmp_path / "plain"

    relabeled = runner.invoke(app, ["turns", str(staged), *ALL_FORMATS])
    rerun = runner.invoke(app, ["turns", str(staged), "--no-llm-speakers", *ALL_FORMATS])
    skipped = runner.invoke(
        app, ["turns", str(staged), "--no-llm-speakers", "--out-dir", str(plain), *ALL_FORMATS]
    )

    assert relabeled.exit_code == 0, relabeled.output
    assert rerun.exit_code == 0, rerun.output
    assert skipped.exit_code == 0, skipped.output
    assert not (tmp_path / "inline.speakers.json").exists()
    assert _artifacts(tmp_path, "inline") == _artifacts(plain, "inline")


def test_a_one_speaker_run_removes_a_stale_sidecar(tmp_path: Path) -> None:
    staged = _written(tmp_path, (3, 30))
    (tmp_path / "inline.speakers.json").write_text("{}\n", encoding="utf-8")

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "inline.speakers.json").exists()


def test_a_stale_sidecar_link_goes_and_its_target_stays(tmp_path: Path) -> None:
    staged = _written(tmp_path, (3, 30))
    target = tmp_path / "kept.json"
    target.write_text("{}\n", encoding="utf-8")
    (tmp_path / "inline.speakers.json").symlink_to(target)

    result = runner.invoke(app, ["turns", str(staged), "--no-llm-speakers"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "inline.speakers.json").is_symlink()
    assert target.read_text(encoding="utf-8") == "{}\n"


def test_a_directory_at_the_sidecar_path_is_left_alone_without_the_pass(tmp_path: Path) -> None:
    staged = _written(tmp_path, (3, 30))
    (tmp_path / "inline.speakers.json").mkdir()

    result = runner.invoke(app, ["turns", str(staged), "--no-llm-speakers"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "inline.speakers.json").is_dir()


def test_stdout_mode_leaves_a_sidecar_where_it_is(tmp_path: Path) -> None:
    staged = _written(tmp_path, (3, 30))
    (tmp_path / "inline.speakers.json").write_text("{}\n", encoding="utf-8")

    result = runner.invoke(app, ["turns", str(staged), "--stdout", "--no-llm-speakers"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "inline.speakers.json").read_text(encoding="utf-8") == "{}\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_read_only_artifact_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    staged = _staged(tmp_path)
    locked = tmp_path / "transcript_two_speakers.md"
    locked.write_text("kept\n", encoding="utf-8")
    locked.chmod(0o444)
    try:
        result = runner.invoke(app, ["turns", str(staged)])
    finally:
        locked.chmod(0o644)

    assert result.exit_code == 2
    assert "cannot write" in result.stderr
    assert made[0].calls == []
    assert locked.read_text(encoding="utf-8") == "kept\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_read_only_out_dir_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch)
    staged = _staged(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o555)
    try:
        result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(out)])
    finally:
        out.chmod(0o755)

    assert result.exit_code == 2
    assert "cannot write" in result.stderr
    assert made[0].calls == []
    assert list(out.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_existing_writable_artifacts_in_a_read_only_dir_are_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Artifacts are rewritten in place, so only a new one needs the directory.
    made = _patch(monkeypatch, reply=_moves_w20_to_rank_zero)
    staged = _written(tmp_path, (7, 20), (4, 10))
    out = tmp_path / "out"
    out.mkdir()
    for name in ("inline.turns.json", "inline.md", "inline.speakers.json"):
        (out / name).write_text("old\n", encoding="utf-8")
    out.chmod(0o555)
    try:
        result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(out)])
    finally:
        out.chmod(0o755)

    assert result.exit_code == 0, result.output
    assert len(made[0].calls) == 1
    assert _sidecar(out / "inline.speakers.json")["words_relabeled"] == 1
    assert sorted(path.name for path in out.iterdir()) == [
        "inline.md",
        "inline.speakers.json",
        "inline.turns.json",
    ]


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_stale_sidecar_that_cannot_go_stops_the_run_before_any_write(tmp_path: Path) -> None:
    staged = _written(tmp_path, (3, 30))
    out = tmp_path / "out"
    out.mkdir()
    for name in ("inline.turns.json", "inline.md", "inline.speakers.json"):
        (out / name).write_text("old\n", encoding="utf-8")
    out.chmod(0o555)
    try:
        result = runner.invoke(
            app, ["turns", str(staged), "--no-llm-speakers", "--out-dir", str(out)]
        )
    finally:
        out.chmod(0o755)

    assert result.exit_code == 2
    assert "cannot remove the stale" in result.stderr
    assert (out / "inline.md").read_text(encoding="utf-8") == "old\n"


def _no_proof(_plan: object) -> None:
    return None


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_write_failing_after_the_proof_still_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The proof cannot rule out a disk filling up or a mode changed mid-run.
    monkeypatch.setattr("scribe.cli.prove_writable", _no_proof)
    staged = _staged(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o555)
    try:
        result = runner.invoke(
            app, ["turns", str(staged), "--no-llm-speakers", "--out-dir", str(out)]
        )
    finally:
        out.chmod(0o755)

    assert result.exit_code == 2
    assert "cannot write artifacts" in result.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_an_artifact_linked_into_a_read_only_dir_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The write follows the link, so the file is created beside its target.
    made = _patch(monkeypatch)
    staged = _staged(tmp_path)
    out, locked = tmp_path / "out", tmp_path / "locked"
    out.mkdir()
    locked.mkdir()
    (out / "transcript_two_speakers.md").symlink_to(locked / "missing.md")
    locked.chmod(0o555)
    try:
        result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(out)])
    finally:
        locked.chmod(0o755)

    assert result.exit_code == 2
    assert "cannot write" in result.stderr
    assert made[0].calls == []
    assert sorted(path.name for path in out.iterdir()) == ["transcript_two_speakers.md"]


MEETING = (
    (7, "Okay, let's start. Connor, can you share the deck?"),
    (3, "Sure, sharing it now."),
    (7, "Thanks. Jose, what do you think of it?"),
    (5, "Looks good to me."),
    (7, "Great. Connor, one more thing about pricing."),
    (3, "Yes, pricing is settled."),
    (7, "And Jose, the timeline?"),
    (5, "End of month."),
)


def _meeting(tmp_path: Path, *runs: tuple[int, str]) -> Path:
    """Words one second apart from (speaker, "words of one run") pairs, as talk.json."""
    said = [(speaker, text) for speaker, line in runs for text in line.split()]
    words = [
        Word(text=text, start=float(index), end=index + 0.9, speaker=speaker)
        for index, (speaker, text) in enumerate(said)
    ]
    staged = tmp_path / "talk.json"
    Transcript(
        source=Source(kind="audio", ref="talk.mp3"),
        engine=Engine(name="xai-stt"),
        text=" ".join(word.text for word in words),
        words=words,
    ).dump(staged)
    return staged


def _answered(_target: str) -> str:
    return (
        "<names>\n"
        "Connor | Connor | next | Connor, can you share the deck?\n"
        "Jose | Jose | next | Jose, what do you think of it?\n"
        "Connor | Connor | next | Connor, one more thing about pricing.\n"
        "Jose | Jose | next | And Jose, the timeline?\n"
        "Maria | Maria | next | And Jose, the timeline?\n"
        "</names>"
    )


def _counted(name: str, word: int, by: str, points_to: str) -> dict[str, object]:
    return {
        "chunk": 0,
        "name": name,
        "said": name,
        "kind": "next",
        "word": word,
        "time": float(word),
        "by": by,
        "points_to": points_to,
        "status": "counted",
        "reason": None,
    }


def test_attendees_name_the_labels_the_words_point_at_in_every_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _patch(monkeypatch, trailer=_answered)
    staged = _meeting(tmp_path, *MEETING)
    plain = tmp_path / "plain"

    unnamed = runner.invoke(app, ["turns", str(staged), "--out-dir", str(plain), *ALL_FORMATS])
    result = runner.invoke(
        app, ["turns", str(staged), "--attendees", "Connor, Jose, Adam", *ALL_FORMATS]
    )

    assert unnamed.exit_code == 0, unnamed.output
    assert result.exit_code == 0, result.output
    assert "People at this meeting" not in made[0].calls[0][0]
    assert "People at this meeting: Connor, Jose, Adam." in made[1].calls[0][0]
    written = Transcript.load(tmp_path / "talk.turns.json")
    before = Transcript.load(plain / "talk.turns.json")
    assert written.words == before.words
    assert [(turn.start, turn.end, turn.text) for turn in written.turns] == [
        (turn.start, turn.end, turn.text) for turn in before.turns
    ]
    assert [turn.speaker for turn in written.turns] == [
        "Speaker 1",
        "Connor",
        "Speaker 1",
        "Jose",
        "Speaker 1",
        "Connor",
        "Speaker 1",
        "Jose",
    ]
    assert json.loads(str(written.engine.params["speaker_names"])) == {
        "Speaker 2": "Connor",
        "Speaker 3": "Jose",
    }
    for suffix, render in ((".md", to_markdown), (".srt", to_srt), (".vtt", to_vtt)):
        assert (tmp_path / f"talk{suffix}").read_text(encoding="utf-8") == render(written)
    sidecar = _sidecar(tmp_path / "talk.speakers.json")
    assert sidecar["prompt_version"] == SPEAKER_PROMPT_VERSION
    assert sidecar["naming"] == {
        "prompt_version": "names-1",
        "attendees": ["Connor", "Jose", "Adam"],
        "names": {"Speaker 2": "Connor", "Speaker 3": "Jose"},
        "unassigned": ["Adam"],
        "pointed": {"Connor": {"Speaker 2": 2}, "Jose": {"Speaker 3": 2}, "Adam": {}},
        "says": {"Connor": {"Speaker 1": 2}, "Jose": {"Speaker 1": 2}, "Adam": {}},
        "names_blocks_missing": [],
        "evidence": [
            _counted("Connor", 3, "Speaker 1", "Speaker 2"),
            _counted("Jose", 14, "Speaker 1", "Speaker 3"),
            _counted("Connor", 26, "Speaker 1", "Speaker 2"),
            _counted("Jose", 37, "Speaker 1", "Speaker 3"),
            {
                "chunk": 0,
                "name": "Maria",
                "said": "Maria",
                "kind": "next",
                "word": None,
                "time": None,
                "by": None,
                "points_to": None,
                "status": "dropped",
                "reason": "not_attendee",
            },
        ],
    }
    assert result.stderr == (
        "scribe: names from the words (prompt names-1): Speaker 2=Connor, Speaker 3=Jose; "
        "unassigned: Adam\n"
    )


def test_without_attendees_the_sidecar_has_no_naming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, trailer=_answered)
    staged = _meeting(tmp_path, *MEETING)

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 0, result.output
    assert "naming" not in _sidecar(tmp_path / "talk.speakers.json")
    assert "speaker_names" not in Transcript.load(tmp_path / "talk.turns.json").engine.params
    assert result.stderr == ""


def test_a_rerun_without_attendees_drops_the_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, trailer=_answered)
    staged = _meeting(tmp_path, *MEETING)
    named = runner.invoke(app, ["turns", str(staged), "--attendees", "Connor, Jose"])
    assert named.exit_code == 0, named.output
    assert "speaker_names" in Transcript.load(tmp_path / "talk.turns.json").engine.params

    again = runner.invoke(
        app, ["turns", str(tmp_path / "talk.turns.json"), "--out-dir", str(tmp_path / "again")]
    )

    assert again.exit_code == 0, again.output
    written = Transcript.load(tmp_path / "again" / "talk.turns.turns.json")
    assert "speaker_names" not in written.engine.params
    assert {turn.speaker for turn in written.turns} == {"Speaker 1", "Speaker 2", "Speaker 3"}


@pytest.mark.parametrize(
    ("flags", "complaint"),
    [
        pytest.param(["--attendees", "Connor", "--no-llm-speakers"], "--attendees", id="no-pass"),
        pytest.param(["--attendees", "Connor,,Jose"], "empty name", id="empty-name"),
        pytest.param(["--attendees", "Speaker 2"], "speaker label", id="label"),
    ],
)
def test_unusable_attendees_exit_two_before_any_call_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], complaint: str
) -> None:
    made = _patch(monkeypatch)
    staged = _meeting(tmp_path, *MEETING)
    out = tmp_path / "out"

    result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(out), *flags])

    assert result.exit_code == 2
    assert complaint in result.stderr
    assert made == []
    assert not out.exists()


def test_attendees_with_no_pass_to_ride_name_nobody_and_say_so(tmp_path: Path) -> None:
    staged = _meeting(tmp_path, (3, "Connor here, just me today."))

    result = runner.invoke(app, ["turns", str(staged), "--attendees", "Connor"])

    assert result.exit_code == 0, result.output
    assert "no speaker pass ran" in result.stderr
    assert "nobody was named" in result.stderr
    written = Transcript.load(tmp_path / "talk.turns.json")
    assert [turn.speaker for turn in written.turns] == ["Speaker 1"]
    assert "speaker_names" not in written.engine.params


def test_attendees_and_the_new_exits_are_documented() -> None:
    result = runner.invoke(app, ["turns", "--help"], terminal_width=200)

    assert result.exit_code == 0
    assert "--attendees" in result.stdout
    assert "with --no-llm-speakers" in " ".join(result.stdout.split())
