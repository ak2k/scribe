from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scribe.cli import app
from scribe.schema import Engine, Source, Transcript, Word

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def _staged(tmp_path: Path, name: str = "transcript_two_speakers.json") -> Path:
    staged = tmp_path / name
    shutil.copy(FIXTURES / name, staged)
    return staged


def _written(tmp_path: Path, *runs: tuple[int, str]) -> Path:
    """Write a transcript whose words, one per second, come in (speaker, "words") runs."""
    said = [(speaker, text) for speaker, line in runs for text in line.split()]
    words = [
        Word(text=text, start=float(index), end=index + 0.9, speaker=speaker)
        for index, (speaker, text) in enumerate(said)
    ]
    staged = tmp_path / "inline.json"
    Transcript(
        source=Source(kind="audio", ref="inline.mp3"),
        engine=Engine(name="xai-stt"),
        text=" ".join(text for _, text in said),
        words=words,
    ).dump(staged)
    return staged


def test_turns_writes_json_and_markdown_by_default(tmp_path: Path) -> None:
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["turns", "--no-llm-speakers", str(staged)])

    assert result.exit_code == 0, result.output
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [
        "transcript_two_speakers.json",
        "transcript_two_speakers.md",
        "transcript_two_speakers.turns.json",
    ]
    reloaded = Transcript.load(tmp_path / "transcript_two_speakers.turns.json")
    # Both one-word replies follow a sentence end, so each keeps its own turn.
    assert len(reloaded.turns) == 7
    assert reloaded.words == Transcript.load(staged).words


def test_turns_honors_out_dir_and_repeated_format(tmp_path: Path) -> None:
    staged = _staged(tmp_path)
    out = tmp_path / "artifacts"

    result = runner.invoke(
        app,
        [
            "turns",
            "--no-llm-speakers",
            str(staged),
            "--out-dir",
            str(out),
            "--format",
            "srt",
            "--format",
            "vtt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == [
        "transcript_two_speakers.srt",
        "transcript_two_speakers.vtt",
    ]
    assert (out / "transcript_two_speakers.vtt").read_text(encoding="utf-8").startswith("WEBVTT")


def test_turns_thresholds_reach_the_stage(tmp_path: Path) -> None:
    # A one-word flicker mid-sentence: a micro-turn at the defaults, merged away.
    staged = _written(tmp_path, (7, "so the plan"), (4, "is"), (7, "set for now"))

    merged = runner.invoke(app, ["turns", "--no-llm-speakers", str(staged), "--stdout"])
    kept = runner.invoke(
        app,
        [
            "turns",
            "--no-llm-speakers",
            str(staged),
            "--stdout",
            "--min-turn-words",
            "1",
            "--min-turn-seconds",
            "0",
        ],
    )

    assert merged.exit_code == 0, merged.output
    assert merged.stdout.count("**Speaker") == 1
    assert kept.exit_code == 0, kept.output
    assert kept.stdout.count("**Speaker") == 3


def test_turns_snap_words_reaches_the_stage(tmp_path: Path) -> None:
    staged = _written(tmp_path, (7, "we are done. and so"), (4, "the next item is here"))

    snapped = runner.invoke(app, ["turns", "--no-llm-speakers", str(staged), "--stdout"])
    unsnapped = runner.invoke(
        app, ["turns", "--no-llm-speakers", str(staged), "--stdout", "--snap-words", "0"]
    )

    assert snapped.exit_code == 0, snapped.output
    assert "we are done.\n" in snapped.stdout
    assert unsnapped.exit_code == 0, unsnapped.output
    assert "we are done. and so\n" in unsnapped.stdout


def test_a_negative_snap_words_exits_two(tmp_path: Path) -> None:
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["turns", str(staged), "--stdout", "--snap-words", "-1"])

    assert result.exit_code == 2
    assert result.stdout == ""


def test_stdout_prints_markdown_and_writes_nothing(tmp_path: Path) -> None:
    staged = _staged(tmp_path)

    result = runner.invoke(app, ["turns", "--no-llm-speakers", str(staged), "--stdout"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("**Speaker 1** [00:00:00]\n")
    assert [p.name for p in tmp_path.iterdir()] == ["transcript_two_speakers.json"]


def test_a_transcript_with_turns_but_no_words_keeps_its_turns(tmp_path: Path) -> None:
    # The shape an imported meeting arrives in: speaker turns, no word timings.
    staged = _staged(tmp_path, "transcript_numbers.turns.json")
    original = Transcript.load(staged)
    out = tmp_path / "artifacts"

    result = runner.invoke(
        app, ["turns", str(staged), "--out-dir", str(out), "--format", "json", "--format", "srt"]
    )
    printed = runner.invoke(app, ["turns", str(staged), "--stdout"])

    assert result.exit_code == 0, result.output
    assert Transcript.load(out / "transcript_numbers.turns.turns.json").turns == original.turns
    assert "Speaker 2: Sure. The total came to $1,234.56" in (
        out / "transcript_numbers.turns.srt"
    ).read_text(encoding="utf-8")
    assert printed.exit_code == 0, printed.output
    assert printed.stdout.count("**Speaker") == len(original.turns)


def test_a_transcript_with_neither_words_nor_turns_exits_two(tmp_path: Path) -> None:
    staged = _staged(tmp_path, "transcript_numbers.turns.json")
    Transcript.load(staged).model_copy(update={"turns": []}).dump(staged)

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert len(result.stderr.strip().splitlines()) == 1
    assert list(tmp_path.iterdir()) == [staged]


def test_missing_input_exits_two_without_a_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["turns", str(tmp_path / "absent.json")])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1


def test_malformed_input_exits_two_without_a_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"text": "no source here"}', encoding="utf-8")

    result = runner.invoke(app, ["turns", str(bad)])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert list(tmp_path.iterdir()) == [bad]


def test_a_non_finite_time_exits_two_with_one_stderr_line(tmp_path: Path) -> None:
    staged = _staged(tmp_path)
    raw = staged.read_text(encoding="utf-8")
    staged.write_text(raw.replace('"start": 0.2,', '"start": Infinity,', 1), encoding="utf-8")

    result = runner.invoke(app, ["turns", str(staged)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert len(result.stderr.strip().splitlines()) == 1
    assert list(tmp_path.iterdir()) == [staged]


def test_a_non_utf8_input_exits_two_without_a_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "not-utf8.json"
    bad.write_bytes(b"\xff\xfe")

    result = runner.invoke(app, ["turns", str(bad)])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1


def test_a_raw_api_payload_is_not_a_transcript(tmp_path: Path) -> None:
    staged = _staged(tmp_path, "xai_diarized_two_speakers.json")

    result = runner.invoke(app, ["turns", str(staged), "--stdout"])

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_an_unwritable_out_dir_exits_two_without_a_traceback(tmp_path: Path) -> None:
    staged = _staged(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file where a directory must go", encoding="utf-8")

    result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(blocker)])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1


def test_an_out_dir_below_a_regular_file_exits_two_without_a_traceback(tmp_path: Path) -> None:
    # The plan counts a missing --out-dir as one to create, so the mkdir is
    # what refuses this one.
    staged = _staged(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file where a directory must go", encoding="utf-8")

    result = runner.invoke(
        app, ["turns", "--no-llm-speakers", str(staged), "--out-dir", str(blocker / "sub")]
    )

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["blocker", staged.name]


def test_a_read_only_out_dir_exits_two_without_a_traceback(tmp_path: Path) -> None:
    staged = _staged(tmp_path)
    out = tmp_path / "read-only"
    out.mkdir()
    out.chmod(0o555)

    try:
        result = runner.invoke(
            app, ["turns", "--no-llm-speakers", str(staged), "--out-dir", str(out)]
        )
    finally:
        out.chmod(0o755)

    assert result.exit_code == 2
    assert "cannot write artifacts" in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_an_input_that_is_its_own_output_is_refused_before_any_write(tmp_path: Path) -> None:
    # The json artifact is written first; a check made per artifact would
    # already have written it by the time the markdown turned out to be the input.
    staged = tmp_path / "x.md"
    shutil.copy(FIXTURES / "transcript_two_speakers.json", staged)

    result = runner.invoke(app, ["turns", str(staged), "--format", "json", "--format", "md"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert len(result.stderr.strip().splitlines()) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["x.md"]
    assert staged.read_bytes() == (FIXTURES / "transcript_two_speakers.json").read_bytes()


@pytest.mark.parametrize(
    ("link", "target"), [("t.md", "t.json"), ("t.srt", "t.md")], ids=["input", "other-output"]
)
def test_a_hard_linked_output_is_refused_before_any_write(
    tmp_path: Path, link: str, target: str
) -> None:
    _staged(tmp_path).rename(tmp_path / "t.json")
    (tmp_path / "t.md").write_text("earlier output\n", encoding="utf-8")
    (tmp_path / link).unlink(missing_ok=True)
    (tmp_path / link).hardlink_to(tmp_path / target)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    result = runner.invoke(
        app, ["turns", str(tmp_path / "t.json"), "--format", "md", "--format", "srt"]
    )

    assert result.exit_code == 2
    assert len(result.stderr.strip().splitlines()) == 1
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_a_refused_out_dir_is_not_created(tmp_path: Path) -> None:
    # `new/..` names the input's own directory, so the markdown would be the
    # input; creating `new` first would leave it behind after the refusal.
    staged = tmp_path / "x.md"
    shutil.copy(FIXTURES / "transcript_two_speakers.json", staged)

    result = runner.invoke(app, ["turns", str(staged), "--out-dir", str(tmp_path / "new" / "..")])

    assert result.exit_code == 2
    assert [p.name for p in tmp_path.iterdir()] == ["x.md"]


def test_a_case_only_out_dir_through_a_missing_directory_is_refused(tmp_path: Path) -> None:
    # The planned markdown does not exist until `new` is created, so only
    # comparing it after resolving `new/..` sees that it names the input.
    work = tmp_path / "work"
    work.mkdir()
    if not (tmp_path / "WORK").exists():
        pytest.skip("the volume is case-sensitive")
    staged = work / "x.md"
    shutil.copy(FIXTURES / "transcript_two_speakers.json", staged)

    result = runner.invoke(
        app,
        [
            "turns",
            str(staged),
            "--out-dir",
            str(tmp_path / "WORK" / "new" / ".."),
            "--format",
            "md",
        ],
    )

    assert result.exit_code == 2
    assert [p.name for p in work.iterdir()] == ["x.md"]
    assert staged.read_bytes() == (FIXTURES / "transcript_two_speakers.json").read_bytes()


def test_schema_prints_parseable_json() -> None:
    result = runner.invoke(app, ["schema"])

    assert result.exit_code == 0, result.output
    printed: object = json.loads(result.stdout)
    assert printed == Transcript.model_json_schema()
    assert '"title": "Transcript"' in result.stdout
