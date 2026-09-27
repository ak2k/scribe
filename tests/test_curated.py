from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scribe.cleanup import CleanupRequest, clean, turn_label
from scribe.curated import fill_ranges, read_front, render_curated
from scribe.errors import InputValidationError
from scribe.schema import Engine, Turn
from tests.cleanup_fakes import FakeBackend, sent_turns

if TYPE_CHECKING:
    from pathlib import Path

Span = tuple[float, float]

_NOTE = re.compile(
    r"\[Includes speech recovered by a second transcription pass, "
    r"\d{2,}:\d{2}:\d{2}\N{EN DASH}\d{2,}:\d{2}:\d{2}\.\] "
)


def _note(start: str, end: str) -> str:
    return f"[Includes speech recovered by a second transcription pass, {start}\N{EN DASH}{end}.] "


def _turn(speaker: str, start: float, end: float) -> Turn:
    return Turn(speaker=speaker, start=start, end=end, text="as said")


def _request(speaker_key: dict[str, str] | None = None) -> CleanupRequest:
    return CleanupRequest(turns=[], speaker_key=speaker_key or {})


def test_each_kept_turn_is_a_block_under_its_label_and_start() -> None:
    kept = [
        (_turn("Speaker 1", 5.99, 9.0), "We ship Monday.\n\nThen we rest."),
        (_turn("Speaker 2", 3725.5, 3730.0), "Agreed."),
    ]

    copy = render_curated(_request({"Speaker 1": "Ann Lee"}), kept, [])

    assert copy == (
        "**Ann Lee | 00:00:05**\nWe ship Monday.\n\nThen we rest.\n\n"
        "**Speaker 2 | 01:02:05**\nAgreed.\n"
    )


def test_a_start_past_ninety_nine_hours_keeps_every_digit() -> None:
    copy = render_curated(_request(), [(_turn("A", 360_061.9, 360_070.0), "Late.")], [])

    assert copy == "**A | 100:01:01**\nLate.\n"


def test_no_kept_turn_is_a_copy_of_one_newline() -> None:
    assert render_curated(_request(), [], []) == "\n"


@pytest.mark.parametrize(
    "front",
    [
        pytest.param("Board meeting\n", id="final-newline"),
        pytest.param("Board meeting", id="no-final-newline"),
        pytest.param("Board meeting\r\nAttendees: Zoë\r\n", id="crlf"),
        pytest.param("\N{ZERO WIDTH NO-BREAK SPACE}Board meeting\n\n", id="bom"),
    ],
)
def test_the_front_opens_the_copy_verbatim_above_a_rule(front: str) -> None:
    copy = render_curated(_request(), [(_turn("A", 0.0, 1.0), "Hello.")], [], front)

    ruled = front if front.endswith("\n") else f"{front}\n"
    assert copy == f"{ruled}\n---\n\n**A | 00:00:00**\nHello.\n"


def test_without_a_front_the_copy_opens_on_the_first_turn() -> None:
    copy = render_curated(_request(), [(_turn("A", 0.0, 1.0), "Hello.")], [])

    assert copy.startswith("**A | 00:00:00**\n")
    assert "---" not in copy


def _noted(start: float, end: float, ranges: list[Span]) -> str:
    """The note the one-turn copy of a turn over `start`..`end` carries, or ""."""
    copy = render_curated(_request(), [(_turn("A", start, end), "Words.")], ranges)
    body = copy.split("\n", 1)[1]
    return body.removesuffix("Words.\n")


@pytest.mark.parametrize(
    ("turn", "ranges", "expected"),
    [
        pytest.param((10.0, 20.0), [(12.4, 15.2)], ("00:00:12", "00:00:16"), id="inside"),
        # Clipped to the turn, then floored and ceiled.
        pytest.param((10.7, 19.2), [(5.0, 30.0)], ("00:00:10", "00:00:20"), id="across"),
        pytest.param((10.0, 20.0), [(5.0, 12.5)], ("00:00:10", "00:00:13"), id="over-start"),
        pytest.param(
            (10.0, 20.0), [(11.0, 12.0), (17.5, 18.2)], ("00:00:11", "00:00:19"), id="union"
        ),
        # Ranges the turn does not touch never widen its note.
        pytest.param(
            (10.0, 20.0),
            [(1.0, 5.0), (12.0, 15.0), (25.0, 30.0)],
            ("00:00:12", "00:00:15"),
            id="untouched-ranges",
        ),
        pytest.param((10.0, 20.0), [(14.5, 14.5)], ("00:00:14", "00:00:15"), id="point-inside"),
        pytest.param((10.0, 20.0), [(20.0, 20.0)], ("00:00:20", "00:00:20"), id="point-at-end"),
        pytest.param((10.0, 20.0), [(10.0, 10.0)], ("00:00:10", "00:00:10"), id="point-at-start"),
        pytest.param((10.0, 20.0), [(20.0, 25.0), (1.0, 10.0)], None, id="touching-ends"),
        pytest.param((10.0, 20.0), [(20.5, 20.5), (2.0, 9.5)], None, id="outside"),
        pytest.param((10.0, 20.0), [], None, id="no-ranges"),
    ],
)
def test_a_turn_touching_a_fill_range_carries_the_note_whole(
    turn: Span, ranges: list[Span], expected: tuple[str, str] | None
) -> None:
    assert _noted(*turn, ranges) == ("" if expected is None else _note(*expected))


def test_a_point_range_on_a_boundary_marks_both_neighbors() -> None:
    kept = [(_turn("A", 0.0, 10.0), "First."), (_turn("B", 10.0, 20.0), "Second.")]

    copy = render_curated(_request(), kept, [(10.0, 10.0)])

    note = _note("00:00:10", "00:00:10")
    assert copy == f"**A | 00:00:00**\n{note}First.\n\n**B | 00:00:10**\n{note}Second.\n"


def _engine(**params: float | int | bool | str) -> Engine:
    return Engine(name="xai-stt", params=params)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param({}, [], id="absent"),
        pytest.param({"fill_ranges": "[]"}, [], id="empty"),
        pytest.param(
            {"fill_ranges": "[[3.0, 5.5], [7, 7]]"}, [(3.0, 5.5), (7.0, 7.0)], id="recorded"
        ),
    ],
)
def test_fill_ranges_are_read_from_the_engine_params(
    tmp_path: Path, params: dict[str, str], expected: list[Span]
) -> None:
    assert fill_ranges(_engine(**params), tmp_path / "t.turns.json") == expected


@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param("[[5.5, 3.0]]", id="reversed"),
        pytest.param("[[1, 2, 3]]", id="three-numbers"),
        pytest.param("[[1]]", id="one-number"),
        pytest.param('[[1, "2"]]', id="quoted-number"),
        pytest.param("[[true, 2]]", id="boolean"),
        pytest.param("[[NaN, 2]]", id="nan"),
        pytest.param("[[1e400, 2]]", id="infinite"),
        pytest.param('{"start": 1}', id="object"),
        pytest.param("[[1, 2]", id="bad-json"),
        pytest.param(3.0, id="not-a-string"),
    ],
)
def test_a_malformed_fill_record_is_a_caller_error(tmp_path: Path, recorded: str | float) -> None:
    source = tmp_path / "t.turns.json"

    with pytest.raises(InputValidationError, match="fill_ranges") as caught:
        fill_ranges(_engine(fill_ranges=recorded), source)

    assert str(source) in str(caught.value)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"Board meeting\n", id="plain"),
        pytest.param("Zoë and Łukasz — notes".encode(), id="non-ascii-no-final-newline"),
        pytest.param(b"Board\r\nmeeting\r\n", id="crlf"),
        pytest.param(b"Board\rmeeting\r", id="cr"),
        pytest.param(b"\xef\xbb\xbfBoard meeting\n", id="bom"),
    ],
)
def test_the_front_file_is_read_byte_for_byte(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "front.md"
    path.write_bytes(raw)

    assert read_front(path).encode("utf-8") == raw


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param("missing", id="missing"),
        pytest.param("latin-1", id="not-utf-8"),
        pytest.param("directory", id="directory"),
    ],
)
def test_an_unreadable_front_file_is_a_caller_error(tmp_path: Path, setup: str) -> None:
    path = tmp_path / "front.md"
    if setup == "latin-1":
        path.write_bytes("Zoë\n".encode("latin-1"))
    elif setup == "directory":
        path.mkdir()

    with pytest.raises(InputValidationError, match="--front"):
        read_front(path)


# No `*`, `|`, `&` or `<`, and single blank lines only: the layout stays
# unambiguous and the reply reads back as written.
_WORD = st.text(alphabet="abcdefgé.,[]\N{EN DASH}", min_size=1, max_size=5)
_PARAGRAPH = st.lists(_WORD, min_size=1, max_size=4).map(" ".join)
_TEXT = st.lists(_PARAGRAPH, min_size=1, max_size=3).map("\n\n".join) | st.just("")
_SECONDS = st.floats(min_value=0, max_value=5000, allow_nan=False, allow_infinity=False)


def _touches(turn: Turn, span: Span) -> bool:
    start, end = span
    if start == end:
        return turn.start <= start <= turn.end
    return start < turn.end and turn.start < end


def _body_blocks(body: str) -> list[str]:
    blocks = body.removesuffix("\n").split("\n\n**")
    return [blocks[0].removeprefix("**"), *blocks[1:]]


@settings(deadline=None)
@given(
    specs=st.lists(
        st.tuples(st.sampled_from(["A", "B", "Speaker 1"]), _SECONDS, _SECONDS, _TEXT),
        min_size=1,
        max_size=6,
    ),
    spans=st.lists(st.tuples(_SECONDS, _SECONDS), max_size=3),
    max_words=st.sampled_from([2, 6000]),
    front=st.none() | st.text(),
    data=st.data(),
)
def test_the_copy_carries_exactly_the_kept_texts_and_notes_exactly_the_touched_turns(
    specs: list[tuple[str, float, float, str]],
    spans: list[Span],
    max_words: int,
    front: str | None,
    data: st.DataObject,
) -> None:
    turns: list[Turn] = []
    start = 0.0
    for speaker, gap, length, _ in specs:
        start += gap
        turns.append(Turn(speaker=speaker, start=start, end=start + length, text="as said here"))
    # Ranges ending exactly on a turn's edge are where the overlap rule decides.
    edges = st.sampled_from([edge for turn in turns for edge in (turn.start, turn.end)])
    at_edges = data.draw(st.lists(st.tuples(edges, edges | _SECONDS), max_size=3))
    ranges = [(min(pair), max(pair)) for pair in [*spans, *at_edges]]
    replies = [text for *_, text in specs]
    backend = FakeBackend(
        respond=lambda user: "".join(
            f"<t id={turn_id}>{replies[(turn_id - 1) % len(replies)]}</t>"
            for turn_id, _ in sent_turns(user)
        )
    )
    request = CleanupRequest(turns=turns, speaker_key={"A": "Ann Lee"})

    result = clean(request, backend, max_words=max_words)
    copy = render_curated(request, result.kept, ranges, front)

    assert (
        result.text
        == "\n\n".join(f"{turn_label(request, turn)}: {text}" for turn, text in result.kept) + "\n"
    )
    ruled = ""
    if front is not None:
        ruled = (front if front.endswith("\n") else f"{front}\n") + "\n---\n\n"
    assert copy.startswith(ruled)
    blocks = _body_blocks(copy.removeprefix(ruled))
    assert len(blocks) == len(result.kept)
    for block, (turn, text) in zip(blocks, result.kept, strict=True):
        header, rest = block.split("\n", 1)
        hours, rest_of_hour = divmod(math.floor(turn.start), 3600)
        minutes, seconds = divmod(rest_of_hour, 60)
        assert header == f"{turn_label(request, turn)} | {hours:02d}:{minutes:02d}:{seconds:02d}**"
        note = _NOTE.match(rest)
        assert (note is not None) == any(_touches(turn, span) for span in ranges)
        assert rest.removeprefix(note.group() if note else "") == text
