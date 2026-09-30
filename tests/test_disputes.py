"""`scribe disputes`: every spot the two recognizers disagreed on, ranked, in one sidecar file."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from scribe import disputes
from scribe.cli import app
from scribe.coverage import fill_holes
from scribe.disputes import find_disputes, render_disputes
from scribe.errors import InputValidationError
from scribe.pick import Side, find_spots
from scribe.schema import Engine, Source, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typer.testing import Result

    from scribe.disputes import Dispute

Row = tuple[float, float, str, str, str]

REFERENCE = "parakeet-mlx mlx-community/parakeet-tdt-0.6b-v3"
LISTED = (
    "Listed: every spot where the pick compared the two recognizers' readings, "
    "and every span the fill filled or flagged."
)
UNLISTED = (
    "Not listed: short stretches only one recognizer heard (unless the fill filled or flagged "
    "them), and words both got wrong the same way. Unlisted text is unverified."
)
QUOTES = "Quotes are the words before cleanup; the reading copy may word them differently."
FOLDED = (
    "Same words once folded; the difference is with the words around the spot "
    "(for example a number split differently)"
)
SIDES: tuple[str, ...] = get_args(Side)
runner = CliRunner()


def _transcript(
    rows: Sequence[Row] = (),
    *,
    words: Sequence[Word] = (),
    params: Mapping[str, float | str | None] | None = None,
    duration: float | None = 60.0,
    ref: str = "recording.mp3",
    sha256: str | None = None,
) -> Transcript:
    """A picked transcript whose pick_record holds `rows`; a param set to None is left out."""
    recorded: dict[str, float | str | None] = {
        "pick_reference": REFERENCE,
        "pick_spots": len(rows),
        "pick_record": json.dumps([list(row) for row in rows]),
    }
    recorded |= params or {}
    return Transcript(
        source=Source(kind="audio", ref=ref, sha256=sha256),
        engine=Engine(
            name="xai-stt",
            model="grok-voice-transcribe-2.0",
            params={key: value for key, value in recorded.items() if value is not None},
        ),
        duration=duration,
        text=" ".join(word.text for word in words),
        words=list(words),
    )


def _entries(*rows: Row, **params: float | str | None) -> tuple[Dispute, ...]:
    return find_disputes(_transcript(rows, params=params), Path("meeting.json")).entries


def _numbered(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if re.match(r"\d+\. ", line)]


def _invoke(path: Path, *args: str) -> Result:
    return runner.invoke(app, ["disputes", str(path), *args])


@pytest.mark.parametrize(
    ("side", "mine", "theirs", "band"),
    [
        pytest.param("unsure", "go", "go and fetch my red hat", "A", id="unsure-surplus-5-is-A"),
        pytest.param("transcript", "go and fetch my red hat", "go", "A", id="surplus-minus-5-is-A"),
        pytest.param("transcript", "go", "go and fetch my hat", "C", id="surplus-4-is-not-A"),
        pytest.param(
            "reference", "see. And so that's the,", "think", "A", id="folded-surplus-5-is-A"
        ),
        pytest.param("restored", "cat", "hat", "A", id="restored-is-A"),
        pytest.param("guarded", "cat", "hat", "A", id="guarded-is-A"),
        pytest.param("failed", "the cat sat down", "a dog ran up", "B", id="failed-is-B"),
        pytest.param("failed", "go", "go and fetch my red hat", "A", id="failed-surplus-5-is-A"),
        pytest.param("unsure", "cat", "hat", "B", id="unsure-is-B"),
        pytest.param(
            "reference", "the cat sat down", "a dog ran up", "C", id="reference-cdiff-4-C"
        ),
        pytest.param("transcript", "the big red", "a small blue", "D", id="cdiff-3-is-D"),
        pytest.param("reference", "cat", "hat", "D", id="cdiff-1-is-D"),
        pytest.param("transcript", "a cat and a dog", "a cat and dog", "D", id="repeat-counts"),
        pytest.param("transcript", "gonna go", "going to go", "E", id="cdiff-0-is-E"),
        pytest.param("reference", "Cat.", "cat", "E", id="punctuation-only-is-E"),
    ],
)
def test_each_spot_takes_the_first_band_its_rule_matches(
    side: str, mine: str, theirs: str, band: str
) -> None:
    (entry,) = _entries((1.0, 2.0, mine, theirs, side))

    assert (entry.kind, entry.band, entry.side) == ("spot", band, side)


def _timed(*texts: str) -> list[Word]:
    return [Word(text=text, start=float(at), end=at + 0.5) for at, text in enumerate(texts)]


def test_a_number_split_across_a_spots_edge_is_the_same_words_once_folded() -> None:
    said = _timed("we", "had", "twenty", "five", "people")
    heard = _timed("we", "had", "twenty", "5", "people")
    (spot,) = find_spots(said, heard)
    mine = said[spot.transcript.start : spot.transcript.stop]
    theirs = heard[spot.reference.start : spot.reference.stop]
    row: Row = (
        mine[0].start,
        max(word.end for word in mine),
        " ".join(word.text for word in mine),
        " ".join(word.text for word in theirs),
        "transcript",
    )
    # Only the edge of the number is in the spot: 25 against 20 5 is outside it.
    assert row[2:4] == ("five", "5")

    found = find_disputes(_transcript([row]), Path("m.json"))

    (entry,) = found.entries
    assert entry.band == "E"
    assert f"\n## E. {FOLDED} (1)\n" in render_disputes(found)


@pytest.mark.parametrize("side", SIDES)
def test_the_delivered_reading_is_the_one_the_transcript_holds(side: str) -> None:
    (entry,) = _entries((1.0, 2.0, "said", "say", side))

    put_in = side in {"reference", "restored"}
    delivered = ("say", "parakeet-mlx") if put_in else ("said", "xai-stt")
    other = ("said", "xai-stt") if put_in else ("say", "parakeet-mlx")
    assert (entry.delivered, entry.delivered_by) == delivered
    assert (entry.other, entry.other_by) == other


def test_a_spot_line_names_the_delivered_reading_its_engine_and_the_rejected_one() -> None:
    found = find_disputes(
        _transcript([(520.05, 520.19, "said", "say", "reference")]), Path("m.json")
    )

    (line,) = _numbered(render_disputes(found))
    assert line == (
        '1. 00:08:40\N{EN DASH}00:08:41 \N{MIDDLE DOT} parakeet-mlx (reference): "say" ] '
        'xai-stt: "said"'
    )


def test_readings_collapse_whitespace_and_an_empty_one_is_nothing() -> None:
    found = find_disputes(
        _transcript([(1.0, 2.0, "  the\n cat\t sat ", "", "transcript")]), Path("m.json")
    )

    (line,) = _numbered(render_disputes(found))
    assert line.endswith('xai-stt (transcript): "the cat sat" ] parakeet-mlx: (nothing)')


def test_every_row_is_listed_once_ranked_by_band_then_time_and_numbered_throughout() -> None:
    rows: list[Row] = [
        (40.0, 41.0, "cat", "hat", "transcript"),
        (30.0, 31.0, "cat", "hat", "unsure"),
        (20.0, 22.0, "cat", "hat", "reference"),
        (20.0, 21.0, "cat", "hat", "transcript"),
        (10.0, 11.0, "cat", "hat", "guarded"),
        (5.0, 6.0, "gonna go", "going to go", "transcript"),
        (1.0, 2.0, "cat", "hat", "failed"),
    ]

    entries = _entries(*rows)

    assert [(entry.number, entry.band, entry.start, entry.end) for entry in entries] == [
        (1, "A", 10.0, 11.0),
        (2, "B", 1.0, 2.0),
        (3, "B", 30.0, 31.0),
        (4, "D", 20.0, 21.0),
        (5, "D", 20.0, 22.0),
        (6, "D", 40.0, 41.0),
        (7, "E", 5.0, 6.0),
    ]


def test_the_header_names_the_recording_both_engines_the_counts_and_the_caveats() -> None:
    rows: list[Row] = [
        (1.0, 2.0, "cat", "hat", "guarded"),
        (3.0, 4.0, "cat", "hat", "unsure"),
        (5.0, 6.0, "cat", "hat", "failed"),
        (7.0, 8.0, "cat", "hat", "transcript"),
    ]
    transcript = _transcript(
        rows,
        params={"fill_ranges": "[[10.0, 11.0]]", "fill_unresolved_ranges": "[[9.0, 20.0]]"},
        ref="/audio/meetings/recording.mp3",
    )

    markdown = render_disputes(find_disputes(transcript, Path("m.json")))

    assert markdown.startswith("# Disputes: recording.mp3\n")
    assert "- Transcript: xai-stt grok-voice-transcribe-2.0\n" in markdown
    assert f"- Reference: {REFERENCE}\n" in markdown
    assert "- Fill spans: 1; unresolved spans: 1 (both listed in A)\n" in markdown
    assert f"\n{LISTED}\n" in markdown
    assert f"\n{UNLISTED}\n" in markdown
    assert f"\n{QUOTES}\n" in markdown
    assert (
        "- A. Words missing on one side: 3\n"
        "- B. The pick was unsure or gave no answer: 2\n"
        "- C. 4 or more words differ: 0\n"
        "- D. 1 to 3 words differ: 1\n"
        f"- E. {FOLDED}: 0\n"
    ) in markdown
    sections = re.findall(r"^## .*$", markdown, re.MULTILINE)
    assert sections == [
        "## A. Words missing on one side (3)",
        "## B. The pick was unsure or gave no answer (2)",
        "## D. 1 to 3 words differ (1)",
    ]
    assert len(_numbered(markdown)) == len(rows) + 2


def test_a_fill_entry_carries_the_words_delivered_in_its_span() -> None:
    words = [
        Word(text=text, start=start, end=start + 0.2)
        for text, start in [("before", 9.7), ("we", 10.0), ("lost", 11.0), ("this", 11.8)]
    ]
    words.append(Word(text="after", start=12.0, end=12.3))
    transcript = _transcript(words=words, params={"fill_ranges": "[[10.0, 12.0]]"})

    found = find_disputes(transcript, Path("m.json"))

    (entry,) = found.entries
    assert (entry.kind, entry.band, entry.delivered, entry.words) == (
        "fill",
        "A",
        "we lost this",
        3,
    )
    (line,) = _numbered(render_disputes(found))
    assert line == (
        "1. 00:00:10\N{EN DASH}00:00:12 \N{MIDDLE DOT} xai-stt heard nothing; "
        'filled from parakeet-mlx: "we lost this"'
    )


def test_a_fill_entry_holds_only_the_words_the_fill_put_in() -> None:
    own = [Word(text="before", start=0.0, end=0.4), Word(text="after", start=3.0, end=3.3)]
    heard = [
        Word(text=text, start=start, end=end)
        for text, start, end in [
            ("one", 1.0, 1.4),
            ("two", 1.8, 2.2),
            ("three", 2.5, 3.0),
            ("after", 3.0, 3.3),
        ]
    ]
    reference = Transcript(
        source=Source(kind="audio", ref="recording.mp3"),
        engine=Engine(name="parakeet-mlx", model="mlx-community/parakeet-tdt-0.6b-v3"),
        duration=None,
        text="",
        words=heard,
    )
    filled, _ = fill_holes(_transcript(words=own), reference)
    # The word closing the hole starts where the last inserted word ends.
    assert filled.engine.params["fill_ranges"] == "[[1.0, 3.0]]"

    (entry,) = find_disputes(filled, Path("m.json")).entries

    assert (entry.kind, entry.delivered, entry.words) == ("fill", "one two three", 3)
    assert entry.words == filled.engine.params["fill_words"]


def test_an_unresolved_entry_counts_the_words_delivered_in_its_span() -> None:
    starts = [29.9, 30.0, 45.0, 60.0, 60.1]
    words = [Word(text="w", start=start, end=start + 0.1) for start in starts]
    transcript = _transcript(words=words, params={"fill_unresolved_ranges": "[[30.0, 60.0]]"})

    found = find_disputes(transcript, Path("m.json"))

    (entry,) = found.entries
    assert (entry.kind, entry.band, entry.words) == ("unresolved", "A", 3)
    (line,) = _numbered(render_disputes(found))
    assert line == (
        "1. 00:00:30\N{EN DASH}00:01:00 \N{MIDDLE DOT} possible dropped speech the fill "
        "could not repair; 3 words delivered here"
    )


def test_fill_and_unresolved_spans_sort_with_band_a_spots_and_overlaps_stay() -> None:
    entries = _entries(
        (12.0, 13.0, "cat", "hat", "guarded"),
        (50.0, 51.0, "cat", "hat", "transcript"),
        fill_ranges="[[11.0, 11.5]]",
        fill_unresolved_ranges="[[10.0, 40.0]]",
    )

    assert [(entry.number, entry.kind, entry.band, entry.start) for entry in entries] == [
        (1, "unresolved", "A", 10.0),
        (2, "fill", "A", 11.0),
        (3, "spot", "A", 12.0),
        (4, "spot", "D", 50.0),
    ]


def test_fill_words_are_credited_to_the_engine_the_fill_took_them_from() -> None:
    words = [Word(text="heard", start=3.0, end=3.2)]
    transcript = _transcript(
        words=words,
        params={"fill_ranges": "[[3.0, 3.2]]", "fill_reference": "gemini-stt gemini-3-pro"},
    )

    found = find_disputes(transcript, Path("m.json"))

    (entry,) = found.entries
    assert entry.delivered_by == "gemini-stt"
    assert "- Fill reference: gemini-stt gemini-3-pro\n" in render_disputes(found)


def test_a_record_from_before_restore_existed_is_read(tmp_path: Path) -> None:
    path = tmp_path / "meeting.picked.json"
    old = {"pick_to_reference": 1, "pick_guarded": 0, "pick_unsure": 1, "pick_failed": 0}
    _transcript(
        [(1.0, 2.0, "cat", "hat", "reference"), (3.0, 4.0, "cat", "hat", "unsure")], params=old
    ).dump(path)
    assert "pick_restored" not in Transcript.load(path).engine.params

    result = _invoke(path)

    assert result.exit_code == 0, result.output
    written = tmp_path / "meeting.picked.disputes.md"
    assert result.stdout == f"{written}\n"
    assert len(_numbered(written.read_text(encoding="utf-8"))) == 2


def test_the_summary_line_and_the_header_count_the_same_entries(tmp_path: Path) -> None:
    path = tmp_path / "meeting.json"
    rows: list[Row] = [
        (1.0, 2.0, "cat", "hat", "restored"),
        (3.0, 4.0, "cat", "hat", "unsure"),
        (5.0, 6.0, "the cat sat down", "a dog ran up", "transcript"),
        (7.0, 8.0, "cat", "hat", "transcript"),
        (9.0, 10.0, "cat", "hat", "reference"),
        (11.0, 12.0, "Cat.", "cat", "transcript"),
    ]
    spans = {"fill_ranges": "[[20.0, 21.0], [22.0, 23.0]]", "fill_unresolved_ranges": "[]"}
    _transcript(rows, params=spans).dump(path)

    result = _invoke(path, "--out", str(tmp_path / "list.md"))

    assert result.exit_code == 0, result.output
    assert result.stderr == (
        "scribe: 8 entries: A 3, B 1, C 1, D 2, E 1; 2 fill spans, 0 unresolved\n"
    )
    markdown = (tmp_path / "list.md").read_text(encoding="utf-8")
    assert "- A. Words missing on one side: 3\n" in markdown
    assert "- C. 4 or more words differ: 1\n" in markdown
    assert "- D. 1 to 3 words differ: 2\n" in markdown
    assert "- Fill spans: 2; unresolved spans: 0 (both listed in A)\n" in markdown
    assert len(_numbered(markdown)) == 8


@pytest.mark.parametrize(
    ("recorded", "named"),
    [
        pytest.param('[[1.0, 2.0, "a", "b"]]', "row 0", id="four-fields"),
        pytest.param('[[1.0, 2.0, "a", "b", "maybe"]]', "row 0", id="unknown-side"),
        pytest.param(
            '[[1.0, 2.0, "a", "b", "transcript"], [3.0, 2.0, "a", "b", "transcript"]]',
            "row 1",
            id="end-before-start",
        ),
        pytest.param('[[NaN, 2.0, "a", "b", "transcript"]]', "row 0", id="nan"),
        pytest.param('[[1.0, 1e400, "a", "b", "transcript"]]', "row 0", id="infinite"),
        pytest.param('[["1.0", 2.0, "a", "b", "transcript"]]', "row 0", id="quoted-number"),
        pytest.param('[[true, 2.0, "a", "b", "transcript"]]', "row 0", id="boolean"),
        pytest.param('[[1.0, 2.0, 3, "b", "transcript"]]', "row 0", id="number-as-text"),
        pytest.param('{"start": 1.0}', "pick_record", id="object"),
        pytest.param("[[1.0, 2.0", "pick_record", id="bad-json"),
        pytest.param(3.0, "pick_record", id="not-a-string"),
    ],
)
def test_a_malformed_pick_record_exits_2_naming_the_problem(
    tmp_path: Path, recorded: str | float, named: str
) -> None:
    path = tmp_path / "meeting.json"
    _transcript(params={"pick_record": recorded}).dump(path)

    result = _invoke(path)

    assert result.exit_code == 2
    assert result.stderr.startswith(f"scribe: {path} has a malformed pick_record")
    assert named in result.stderr
    assert not (tmp_path / "meeting.disputes.md").exists()


def test_an_end_before_its_start_is_named_as_such(tmp_path: Path) -> None:
    path = tmp_path / "meeting.json"
    _transcript([(3.0, 2.0, "a", "b", "transcript")]).dump(path)

    result = _invoke(path)

    assert result.exit_code == 2
    assert "row 0" in result.stderr
    assert "cannot end before it starts" in result.stderr


def test_a_spot_starting_before_the_recording_is_named_as_such(tmp_path: Path) -> None:
    path = tmp_path / "meeting.json"
    _transcript([(1.0, 2.0, "a", "b", "transcript"), (-2.0, -1.0, "a", "b", "transcript")]).dump(
        path
    )

    result = _invoke(path)

    assert result.exit_code == 2
    assert "row 1" in result.stderr
    assert "cannot start before the recording" in result.stderr
    assert not (tmp_path / "meeting.disputes.md").exists()


@pytest.mark.parametrize("key", ["fill_ranges", "fill_unresolved_ranges"])
def test_a_range_starting_before_the_recording_exits_2_naming_it(tmp_path: Path, key: str) -> None:
    path = tmp_path / "meeting.json"
    _transcript(params={key: "[[1.0, 2.0], [-0.5, 3.0]]"}).dump(path)

    result = _invoke(path)

    assert result.exit_code == 2
    assert f"has a malformed {key}: range 1 starts before the recording" in result.stderr
    assert not (tmp_path / "meeting.disputes.md").exists()


def test_a_transcript_with_no_pick_record_exits_2_saying_what_to_run(tmp_path: Path) -> None:
    path = tmp_path / "meeting.json"
    _transcript(params={"pick_record": None}).dump(path)

    result = _invoke(path)

    assert result.exit_code == 2
    assert "no pick_record" in result.stderr
    assert "`scribe transcribe`" in result.stderr
    assert "`scribe pick`" in result.stderr


@pytest.mark.parametrize("recorded", [None, 3.0, ""])
def test_a_record_with_no_reference_engine_exits_2(
    tmp_path: Path, recorded: float | str | None
) -> None:
    path = tmp_path / "meeting.json"
    _transcript(params={"pick_reference": recorded}).dump(path)

    result = _invoke(path)

    assert result.exit_code == 2
    assert "pick_reference" in result.stderr


def test_a_malformed_unresolved_range_exits_2_naming_its_key(tmp_path: Path) -> None:
    path = tmp_path / "meeting.json"
    _transcript(params={"fill_unresolved_ranges": "[[5.0, 4.0]]"}).dump(path)

    result = _invoke(path)

    assert result.exit_code == 2
    assert "fill_unresolved_ranges" in result.stderr


class Ffmpeg:
    """A stand-in for ffmpeg that records each argv and writes each clip."""

    def __init__(self, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.code = code

    def run(self, argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if self.code == 0:
            Path(argv[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(argv, self.code, stdout="", stderr="Invalid data")


def _ffmpeg(monkeypatch: pytest.MonkeyPatch, *, code: int = 0, found: bool = True) -> Ffmpeg:
    fake = Ffmpeg(code)
    located = "/usr/bin/ffmpeg" if found else None
    cut = partial(disputes.cut_clips, run=fake.run, which=lambda _name: located)
    monkeypatch.setattr("scribe.cli.cut_clips", cut)
    return fake


def _recording(tmp_path: Path, rows: Sequence[Row], *, sha256: str | None = None) -> Path:
    """Write an audio file and a transcript of it; return the transcript's path."""
    audio = tmp_path / "recording.mp3"
    audio.write_bytes(b"pretend this is audio")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest() if sha256 is None else sha256
    path = tmp_path / "meeting.json"
    _transcript(rows, ref=str(audio), sha256=digest).dump(path)
    return path


CLIPPED: list[Row] = [
    (1.0, 2.0, "cat", "hat", "transcript"),
    (30.0, 31.0, "cat", "hat", "transcript"),
    (58.5, 59.5, "cat", "hat", "transcript"),
]


def test_clips_cut_one_per_entry_clamped_to_the_recording_and_linked_by_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED)

    result = _invoke(path, "--clips")

    assert result.exit_code == 0, result.output
    audio = str((tmp_path / "recording.mp3").absolute())
    clips = tmp_path / "meeting.disputes.clips"
    names = ["001-000001.m4a", "002-000030.m4a", "003-000058.m4a"]
    windows = [("0.000", "5.000"), ("27.000", "7.000"), ("55.500", "4.500")]
    assert fake.calls == [
        [
            *("ffmpeg", "-nostdin", "-v", "error", "-ss", start, "-t", length, "-i", audio),
            *("-vn", "-c:a", "aac", str((clips / name).absolute())),
        ]
        for name, (start, length) in zip(names, windows, strict=True)
    ]
    assert sorted(child.name for child in clips.iterdir()) == names
    lines = _numbered((tmp_path / "meeting.disputes.md").read_text(encoding="utf-8"))
    assert [line.rsplit(" \N{MIDDLE DOT} ", 1)[1] for line in lines] == [
        f"clip: meeting.disputes.clips/{name}" for name in names
    ]


def test_without_a_duration_a_clip_runs_its_full_three_seconds_past_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    audio = tmp_path / "recording.mp3"
    audio.write_bytes(b"pretend this is audio")
    path = tmp_path / "meeting.json"
    _transcript([(58.5, 59.5, "cat", "hat", "transcript")], ref=str(audio), duration=None).dump(
        path
    )

    result = _invoke(path, "--clips")

    assert result.exit_code == 0, result.output
    (call,) = fake.calls
    assert call[call.index("-ss") + 1 : call.index("-t") + 2] == ["55.500", "-t", "7.000"]


def test_clips_read_the_source_audio_relative_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    (tmp_path / "recording.mp3").write_bytes(b"pretend this is audio")
    # Elsewhere than the working directory, so a ref read beside it is not found.
    path = tmp_path / "run" / "meeting.json"
    path.parent.mkdir()
    _transcript(CLIPPED[:1], ref="recording.mp3").dump(path)
    monkeypatch.chdir(tmp_path)

    result = _invoke(path, "--clips")

    assert result.exit_code == 0, result.output
    (call,) = fake.calls
    assert call[call.index("-i") + 1] == str((tmp_path / "recording.mp3").absolute())


def test_an_out_not_ending_md_gets_its_name_plus_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED[:1])
    out = tmp_path / "list.txt"

    result = _invoke(path, "--clips", "--out", str(out))

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{out}\n"
    assert [child.name for child in (tmp_path / "list.txt.clips").iterdir()] == ["001-000001.m4a"]
    assert out.read_text(encoding="utf-8").count("clip: list.txt.clips/001-000001.m4a") == 1


def test_clips_refuse_an_out_that_is_the_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED)
    audio = tmp_path / "recording.mp3"

    result = _invoke(path, "--clips", "--out", str(audio))

    assert result.exit_code == 2
    assert "is the audio file" in result.stderr
    assert fake.calls == []
    assert audio.read_bytes() == b"pretend this is audio"


def test_an_empty_clips_directory_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED[:1])
    (tmp_path / "meeting.disputes.clips").mkdir()

    result = _invoke(path, "--clips")

    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1


def _refused_before_anything(result: Result, fake: Ffmpeg, tmp_path: Path) -> None:
    assert result.exit_code == 2
    assert len(result.stderr.splitlines()) == 1
    assert fake.calls == []
    assert not (tmp_path / "meeting.disputes.md").exists()


def test_audio_of_another_hash_is_refused_before_any_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED, sha256="0" * 64)

    result = _invoke(path, "--clips")

    _refused_before_anything(result, fake, tmp_path)
    assert "is not the audio the transcript was made from" in result.stderr
    assert not (tmp_path / "meeting.disputes.clips").exists()


def test_a_missing_audio_option_is_refused_before_any_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED)
    missing = tmp_path / "gone.mp3"

    result = _invoke(path, "--clips", "--audio", str(missing))

    _refused_before_anything(result, fake, tmp_path)
    assert str(missing) in result.stderr
    assert not (tmp_path / "meeting.disputes.clips").exists()


def test_a_missing_source_audio_is_refused_before_any_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED)
    (tmp_path / "recording.mp3").unlink()

    result = _invoke(path, "--clips")

    _refused_before_anything(result, fake, tmp_path)
    assert "pass --audio" in result.stderr
    assert not (tmp_path / "meeting.disputes.clips").exists()


@pytest.mark.parametrize(
    ("start", "shown"),
    [
        pytest.param(60.0, "00:01:00", id="at-the-end"),
        pytest.param(61.0, "00:01:01", id="within-the-lead-in"),
        pytest.param(70.0, "00:01:10", id="past-the-lead-in"),
    ],
)
def test_an_entry_starting_at_or_past_the_recordings_end_is_refused_before_any_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: float, shown: str
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, [*CLIPPED, (start, start + 1.0, "cat", "hat", "transcript")])

    result = _invoke(path, "--clips")

    _refused_before_anything(result, fake, tmp_path)
    assert f"starts at {shown}, at or past the recording's end" in result.stderr
    assert not (tmp_path / "meeting.disputes.clips").exists()


def test_a_clips_directory_an_earlier_run_left_is_refused_and_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED)
    clips = tmp_path / "meeting.disputes.clips"
    clips.mkdir()
    (clips / "001-000001.m4a").write_bytes(b"old clip")

    result = _invoke(path, "--clips")

    _refused_before_anything(result, fake, tmp_path)
    assert "earlier run" in result.stderr
    assert [child.name for child in clips.iterdir()] == ["001-000001.m4a"]
    assert (clips / "001-000001.m4a").read_bytes() == b"old clip"


def test_a_file_where_the_clips_directory_goes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED)
    (tmp_path / "meeting.disputes.clips").write_bytes(b"not a directory")

    result = _invoke(path, "--clips")

    _refused_before_anything(result, fake, tmp_path)
    assert "not a directory" in result.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through any mode bits")
def test_an_unreadable_clips_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED)
    clips = tmp_path / "meeting.disputes.clips"
    clips.mkdir()
    clips.chmod(0o000)
    try:
        result = _invoke(path, "--clips")
    finally:
        clips.chmod(0o755)

    _refused_before_anything(result, fake, tmp_path)
    assert "cannot read the clips directory" in result.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_an_unwritable_list_is_refused_before_any_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch)
    path = _recording(tmp_path, CLIPPED)
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o555)
    try:
        result = _invoke(path, "--clips", "--out", str(out / "list.md"))
    finally:
        out.chmod(0o755)

    assert result.exit_code == 2
    assert "cannot write" in result.stderr
    assert fake.calls == []
    assert list(out.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode bits")
def test_a_clips_directory_that_cannot_be_made_is_an_input_error(tmp_path: Path) -> None:
    fake = Ffmpeg()
    entries = find_disputes(_transcript(CLIPPED), Path("m.json")).entries
    parent = tmp_path / "locked"
    parent.mkdir()
    parent.chmod(0o555)
    try:
        with pytest.raises(InputValidationError, match="cannot make the clips directory"):
            disputes.cut_clips(
                entries,
                tmp_path / "recording.mp3",
                parent / "list.clips",
                duration=60.0,
                run=fake.run,
                which=lambda _name: "/usr/bin/ffmpeg",
            )
    finally:
        parent.chmod(0o755)

    assert fake.calls == []


def test_missing_ffmpeg_exits_2_naming_it_before_the_clips_directory_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch, found=False)
    path = _recording(tmp_path, CLIPPED)

    result = _invoke(path, "--clips")

    _refused_before_anything(result, fake, tmp_path)
    assert "ffmpeg" in result.stderr
    assert not (tmp_path / "meeting.disputes.clips").exists()


def test_a_failed_clip_exits_2_and_writes_no_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ffmpeg(monkeypatch, code=1)
    path = _recording(tmp_path, CLIPPED)

    result = _invoke(path, "--clips")

    assert result.exit_code == 2
    assert result.stderr.startswith("scribe: ffmpeg exited 1")
    assert len(fake.calls) == 1
    assert not (tmp_path / "meeting.disputes.md").exists()


def test_audio_without_clips_exits_2(tmp_path: Path) -> None:
    path = _recording(tmp_path, CLIPPED)

    result = _invoke(path, "--audio", str(tmp_path / "recording.mp3"))

    assert result.exit_code == 2
    assert "--clips" in result.stderr
    assert not (tmp_path / "meeting.disputes.md").exists()


def test_without_clips_the_audio_is_never_read(tmp_path: Path) -> None:
    path = tmp_path / "meeting.json"
    _transcript(CLIPPED, ref=str(tmp_path / "nowhere.mp3"), sha256="0" * 64).dump(path)

    result = _invoke(path)

    assert result.exit_code == 0, result.output
    assert "clip:" not in (tmp_path / "meeting.disputes.md").read_text(encoding="utf-8")


def test_the_help_says_what_it_writes_and_that_it_sends_nothing() -> None:
    result = runner.invoke(app, ["disputes", "--help"], terminal_width=200)
    text = " ".join(result.stdout.split())

    assert result.exit_code == 0
    assert ".disputes.md" in text
    assert "sends nothing over the network" in text
    assert "Unlisted text is unverified" in text
    assert "an entry that starts at or past the audio's end" in text
    assert "5 or more words shorter than the other as the pick counts them" in text
    assert "fillers and repeats aside" in text
    assert "each entry names its clip" in text
    assert f"E: {FOLDED[0].lower()}{FOLDED[1:]}." in text


_WORDS = ["the", "cat", "hat", "a", "sat", "uh", "gonna", "going", "to", "Cat.", "mat", "on"]
_READING = st.lists(st.sampled_from(_WORDS), max_size=8).map(" ".join)
_TIME = st.floats(min_value=0, max_value=1e5, allow_nan=False)
_LENGTH = st.floats(min_value=0, max_value=60, allow_nan=False)
_ROWS = st.lists(st.tuples(_TIME, _LENGTH, _READING, _READING, st.sampled_from(SIDES)))
_SPANS = st.lists(st.tuples(_TIME, _LENGTH).map(lambda span: [span[0], span[0] + span[1]]))


def _recorded(entry: Dispute) -> Row:
    """The pick_record row an entry was made from."""
    put_in = entry.side in {"reference", "restored"}
    mine, theirs = (entry.other, entry.delivered) if put_in else (entry.delivered, entry.other)
    return (entry.start, entry.end, mine, theirs, str(entry.side))


@given(rows=_ROWS, fills=_SPANS, unresolved=_SPANS)
def test_every_row_is_listed_exactly_once_by_band_then_time(
    rows: list[tuple[float, float, str, str, str]],
    fills: list[list[float]],
    unresolved: list[list[float]],
) -> None:
    record: list[Row] = [
        (start, start + length, mine, theirs, side) for start, length, mine, theirs, side in rows
    ]
    spans = {"fill_ranges": json.dumps(fills), "fill_unresolved_ranges": json.dumps(unresolved)}

    found = find_disputes(_transcript(record, params=spans), Path("m.json"))

    entries = found.entries
    spots = [entry for entry in entries if entry.kind == "spot"]
    assert Counter(map(_recorded, spots)) == Counter(record)
    assert Counter(entry.kind for entry in entries if entry.kind != "spot") == Counter(
        fill=len(fills), unresolved=len(unresolved)
    )
    assert [entry.number for entry in entries] == list(range(1, len(entries) + 1))
    keys = [(entry.band, entry.start, entry.end) for entry in entries]
    assert keys == sorted(keys)
    assert len(_numbered(render_disputes(found))) == len(entries)
