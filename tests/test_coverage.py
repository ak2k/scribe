"""Filling a transcript's holes from a reference: only inserts, only inside holes."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from typer.testing import CliRunner

from scribe.cli import app
from scribe.coverage import Retime, Span, fill_holes
from scribe.gaps import MIN_DROP_SECONDS, find_holes
from scribe.schema import Engine, Source, Transcript, Turn, Word
from scribe.vote import first_decrease

if TYPE_CHECKING:
    from pathlib import Path

PARAKEET = Engine(name="parakeet-mlx", model="mlx-community/parakeet-tdt-0.6b-v3")
runner = CliRunner()


def _transcript(
    words: list[Word], *, duration: float | None = None, engine: Engine = PARAKEET
) -> Transcript:
    return Transcript(
        source=Source(kind="audio", ref="a.mp3"),
        engine=engine,
        duration=duration,
        text="",
        words=words,
    )


def _said(*spoken: tuple[str, float], speaker: int | None = 0) -> list[Word]:
    """Words half a second long, starting where given."""
    return [
        Word(text=text, start=start, end=start + 0.5, speaker=speaker) for text, start in spoken
    ]


def _heard(*spoken: tuple[str, float]) -> list[Word]:
    return _said(*spoken, speaker=None)


def _fill(own: list[Word], heard: list[Word], *, duration: float = 20.0) -> list[str]:
    """Fill `own` from `heard`; return the texts, each inserted word's marked with a +.

    The duration is set: without it the audio ends where `heard` does, which
    can cut a hole short of the transcript word closing it.
    """
    filled, _ = fill_holes(_transcript(own, duration=duration), _transcript(heard))
    return [word.text if word.speaker is not None else f"+{word.text}" for word in filled.words]


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        (_heard(("one", 3.0), ("two", 4.0), ("three", 5.0)), ["+one", "+two", "+three"]),
        (_heard(("one", 3.0), ("two", 4.0)), []),
        (_heard(("um", 2.0), ("one", 3.0), ("uh", 4.0), ("two", 5.0), ("—", 6.0)), []),
    ],
)
def test_a_hole_with_three_unmatched_words_takes_them_before_the_word_closing_it(
    heard: list[Word], expected: list[str]
) -> None:
    own = _said(("alpha", 0.0), ("omega", 10.0))

    assert _fill(own, heard) == ["alpha", *expected, "omega"]


@pytest.mark.parametrize(
    ("first", "last", "inserted"), [(1.25, 10.75, 3), (1.125, 10.75, 0), (1.25, 10.875, 0)]
)
def test_a_reference_word_counts_from_a_quarter_second_inside_the_hole(
    first: float, last: float, inserted: int
) -> None:
    own = _said(("alpha", 0.5), ("omega", 11.0))
    heard = _heard(("one", first), ("two", 5.0), ("three", last))

    assert len(_fill(own, heard)) - len(own) == inserted


@pytest.mark.parametrize(("closing", "inserted"), [(3.0, 3), (2.9375, 0)])
def test_a_hole_counts_from_two_seconds(closing: float, inserted: int) -> None:
    own = _said(("alpha", 0.5), ("omega", closing))
    heard = _heard(("one", 1.25), ("two", 1.75), ("three", 2.25))

    assert len(_fill(own, heard)) - len(own) == inserted


@pytest.mark.parametrize(("at", "inserted"), [(18.0, 0), (18.25, 3)])
def test_a_word_the_transcript_has_up_to_eight_seconds_away_is_not_missing(
    at: float, inserted: int
) -> None:
    own = _said(("alpha", 0.0), ("omega", 10.0), ("One,", at))
    heard = _heard(("one", 3.0), ("two", 4.0), ("three", 5.0))

    assert len(_fill(own, heard)) - len(own) == inserted


def test_a_diarized_references_words_go_in_with_no_speaker() -> None:
    own = _said(("alpha", 0.0), ("omega", 10.0))
    heard = _said(("one", 3.0), ("two", 4.0), ("three", 5.0), speaker=3)

    assert _fill(own, heard) == ["alpha", "+one", "+two", "+three", "omega"]


def test_the_span_runs_from_the_first_unmatched_word_to_the_last() -> None:
    own = _said(("go", 0.0), ("omega", 10.0))
    heard = _heard(
        ("um", 1.0), ("go", 2.0), ("go", 3.0), ("uh", 4.0), ("go", 5.0), ("new", 6.0), ("Um,", 7.0)
    )

    # The transcript's one "go" uses up the first; fillers inside the span go in with it.
    assert _fill(own, heard) == ["go", "+go", "+uh", "+go", "+new", "omega"]


def test_the_first_holes_words_go_first_and_the_last_holes_last() -> None:
    # Starts that decrease: the words keep their list order.
    own = _said(("mid", 10.0), ("low", 5.0))
    heard = _heard(
        *(
            (f"{hole}{index}", start + index)
            for hole, start in [("a", 1.0), ("b", 6.0), ("c", 12.0)]
            for index in range(3)
        )
    )

    assert _fill(own, heard, duration=20.0) == [
        *("+a0", "+a1", "+a2", "+b0", "+b1", "+b2"),
        *("mid", "low", "+c0", "+c1", "+c2"),
    ]


@pytest.mark.parametrize(
    ("starts", "unresolved"),
    [
        (range(19), []),
        (range(20), [Span(0.0, 30.0, 20)]),
        ([*range(20), *range(30, 50)], [Span(0.0, 60.0, 40)]),
    ],
)
def test_twenty_unmatched_words_in_a_window_with_no_hole_are_unresolved(
    starts: list[int], unresolved: list[Span]
) -> None:
    own = _said(*(("x", float(second)) for second in range(60)))
    heard = _heard(*((f"w{second}", second + 0.25) for second in starts))

    filled, fill = fill_holes(_transcript(own, duration=60.0), _transcript(heard))

    assert list(fill.unresolved) == unresolved
    assert (filled.words, fill.filled) == (own, ())


def _twenties(*firsts: float) -> list[tuple[str, float]]:
    """Ten words two tenths of a second apart from each of `firsts`."""
    starts = [first + index / 5 for first in firsts for index in range(10)]
    return [(f"w{index}", start) for index, start in enumerate(starts)]


@pytest.mark.parametrize(
    ("heard", "unresolved"),
    [
        (_heard(*_twenties(28.0, 30.0)), [Span(15.0, 45.0, 20)]),
        (_heard(*_twenties(0.0, 2.0, 46.0, 48.0)), [Span(0.0, 60.0, 40)]),
    ],
    ids=["across a window's edge", "in windows that only touch"],
)
def test_windows_every_fifteen_seconds_find_a_passage_wherever_it_falls(
    heard: list[Word], unresolved: list[Span]
) -> None:
    own = _said(*(("x", float(second)) for second in range(60)))

    filled, fill = fill_holes(_transcript(own, duration=60.0), _transcript(heard))

    # Flagged windows that overlap or touch are one span, each of its words counted once.
    assert list(fill.unresolved) == unresolved
    assert (filled.words, fill.filled) == (own, ())


_NINETEEN = [(f"w{second}", second + 0.25) for second in range(19)]


@pytest.mark.parametrize(
    ("heard", "extra", "unresolved"),
    [
        (_heard(*_NINETEEN, ("w30", 30.0)), [], []),
        (_heard(*_NINETEEN, ("w19", 19.25)), _said(("w0", 38.0)), [Span(0.0, 30.0, 20)]),
    ],
    ids=["a reference word at 30 s", "a transcript word at 38 s"],
)
def test_a_word_exactly_at_a_windows_end_belongs_to_the_next(
    heard: list[Word], extra: list[Word], unresolved: list[Span]
) -> None:
    own = [*_said(*(("x", float(second)) for second in range(60))), *extra]

    _, fill = fill_holes(_transcript(own, duration=60.0), _transcript(heard))

    assert list(fill.unresolved) == unresolved


def test_the_windows_are_judged_after_the_fill() -> None:
    # Filled, the hole's twenty words are the transcript's own and match.
    own = _said(("alpha", 0.0), ("omega", 29.0))
    heard = _heard(*((f"w{index}", 1.0 + index) for index in range(20)))

    _, fill = fill_holes(_transcript(own, duration=30.0), _transcript(heard))

    assert (fill.words, fill.unresolved) == (20, ())


def test_an_unresolved_span_ends_at_the_audios_end() -> None:
    own = _said(*(("x", float(second)) for second in range(45)))
    heard = _heard(*((f"w{index}", 30.25 + index / 2) for index in range(20)))

    _, fill = fill_holes(_transcript(own, duration=45.0), _transcript(heard))

    # The window from 15 s holds the twenty words too; the one from 30 s runs past the audio.
    assert fill.unresolved == (Span(15.0, 45.0, 20),)


def test_the_result_records_the_fill_and_drops_the_stale_turns() -> None:
    own = _transcript(
        _said(("alpha", 0.0), ("omega", 10.0)),
        duration=12.0,
        engine=Engine(name="xai-stt", model="grok", params={"diarize": True}),
    ).model_copy(update={"turns": [Turn(speaker="Speaker 1", start=0.0, end=10.5, text="x")]})
    heard = _transcript(_heard(("one", 3.0), ("two", 4.0), ("three", 5.0)))

    filled, fill = fill_holes(own, heard)
    unfilled, _ = fill_holes(own, _transcript([], engine=Engine(name="gemini")))

    assert filled.text == "alpha one two three omega"
    assert (filled.turns, filled.duration, filled.source) == ([], 12.0, own.source)
    assert filled.engine.params == {
        "diarize": True,
        "fill_reference": f"parakeet-mlx {PARAKEET.model}",
        "fill_spans": 1,
        "fill_words": 3,
        "fill_unresolved": 0,
        "fill_ranges": "[[3.0, 5.5]]",
        "fill_unresolved_ranges": "[]",
        "fill_retimed_words": 0,
        "fill_retimed_runs": "[]",
    }
    assert (fill.filled, fill.words) == ((Span(3.0, 5.5, 3),), 3)
    assert unfilled.engine.params == {
        "diarize": True,
        "fill_reference": "gemini",
        "fill_spans": 0,
        "fill_words": 0,
        "fill_unresolved": 0,
        "fill_ranges": "[]",
        "fill_unresolved_ranges": "[]",
        "fill_retimed_words": 0,
        "fill_retimed_runs": "[]",
    }


# The transcript slid four words into the time of a passage it dropped; the
# reference heard the passage there and the four words ten seconds later.
_SLID = _said(("alpha", 0.0), ("one", 2.0), ("two", 3.0), ("three", 4.0), ("four", 5.0))
_DROPPED = _heard(("we", 2.0), ("lost", 3.0), ("this", 4.0), ("passage", 5.0))
_LATER = _heard(("alpha", 0.0), ("one", 12.0), ("two", 13.0), ("three", 14.0), ("four", 15.0))


def _times(words: list[Word]) -> list[tuple[str, float, float, int | None]]:
    return [(word.text, word.start, word.end, word.speaker) for word in words]


def test_words_slid_into_a_dropped_passage_move_to_where_the_reference_heard_them() -> None:
    own = [*_SLID, *_said(("omega", 20.0))]
    heard = sorted([*_LATER, *_DROPPED, *_heard(("omega", 20.0))], key=lambda word: word.start)

    filled, fill = fill_holes(_transcript(own, duration=21.0), _transcript(heard))

    # Moved, they open the hole the passage fills.
    assert _times(filled.words) == [
        ("alpha", 0.0, 0.5, 0),
        *(("we", 2.0, 2.5, None), ("lost", 3.0, 3.5, None)),
        *(("this", 4.0, 4.5, None), ("passage", 5.0, 5.5, None)),
        *(("one", 12.0, 12.5, 0), ("two", 13.0, 13.5, 0)),
        *(("three", 14.0, 14.5, 0), ("four", 15.0, 15.5, 0)),
        ("omega", 20.0, 20.5, 0),
    ]
    assert fill.retimed == (Retime(2.0, 5.5, 12.0, 15.5, 4, 0, 0.0),)
    assert filled.engine.params["fill_retimed_words"] == 4
    assert json.loads(str(filled.engine.params["fill_retimed_runs"])) == [
        {"start": 2.0, "end": 5.5, "new_start": 12.0, "new_end": 15.5}
        | {"words": 4, "clamped": 0, "clamp_s": 0.0}
    ]


@pytest.mark.parametrize(
    ("later", "moved"),
    [
        (_heard(("one", 12.0), ("two", 13.0), ("three", 14.0)), 3),
        (_heard(("one", 12.0), ("two", 13.0), ("three", 4.5)), 0),
        (_heard(("one", 4.0), ("two", 5.0), ("three", 6.0)), 0),
    ],
    ids=["three words", "two words", "exactly two seconds"],
)
def test_a_run_needs_three_words_each_more_than_two_seconds_away(
    later: list[Word], moved: int
) -> None:
    own = _said(("alpha", 0.0), ("one", 2.0), ("two", 3.0), ("three", 4.0), ("omega", 20.0))
    heard = sorted([*_heard(("alpha", 0.0), ("omega", 20.0)), *later], key=lambda w: w.start)
    retimed = _said(("alpha", 0.0), ("one", 12.0), ("two", 13.0), ("three", 14.0), ("omega", 20.0))
    # Drifted words keep their times when the run is too short, and so do the words around a run.
    words, fill = fill_holes(_transcript(own, duration=21.0), _transcript(heard))
    expected = retimed if moved else own
    assert (words.words, fill.retimed_words) == (expected, moved)


def test_a_phrase_said_twice_stays_where_the_transcript_has_it() -> None:
    # The reference heard "go stop wait" once, where the transcript's second copy is.
    twice = _said(("omega", 20.0), ("go", 30.0), ("stop", 31.0), ("wait", 32.0))
    twice += _said(("go", 35.0), ("stop", 36.0), ("wait", 37.0), ("end", 45.0))
    once = _heard(("omega", 20.0), ("go", 35.0), ("stop", 36.0), ("wait", 37.0), ("end", 45.0))
    own = [*_SLID, *twice]
    heard = sorted([*_LATER, *_DROPPED, *once], key=lambda word: word.start)

    filled, fill = fill_holes(_transcript(own, duration=46.0), _transcript(heard))

    assert [run.start for run in fill.retimed] == [2.0]
    kept = [word for word in filled.words if word.speaker is not None]
    assert kept[len(_SLID) :] == twice


def test_a_run_most_of_whose_words_the_reference_also_heard_where_they_are_stays() -> None:
    # "one" and "two" are heard where the transcript has them, as well as ten seconds later.
    own = _said(("alpha", 0.0), ("one", 2.0), ("two", 3.0), ("three", 4.0), ("omega", 20.0))
    heard = _heard(("alpha", 0.0), ("one", 2.5), ("two", 3.5), ("one", 12.0), ("two", 13.0))
    heard += _heard(("three", 14.0), ("omega", 20.0))

    filled, fill = fill_holes(_transcript(own, duration=21.0), _transcript(heard))

    assert (filled.words, fill.retimed) == (own, ())


def test_each_repeat_test_needs_more_than_half_the_run_on_its_own() -> None:
    # "one" and "two" are said again near where the reference heard them; "three" and
    # "four" are heard again near where the transcript has them: half the run each.
    own = _said(("alpha", 0.0), ("one", 2.0), ("two", 3.0), ("three", 4.0), ("four", 5.0))
    own += _said(("one", 13.5), ("two", 14.5), ("omega", 30.0))
    heard = _heard(("alpha", 0.0), ("three", 4.5), ("four", 5.5), ("one", 12.0), ("two", 13.0))
    heard += _heard(("three", 14.0), ("four", 15.0), ("omega", 30.0))

    _, fill = fill_holes(_transcript(own, duration=31.0), _transcript(heard))

    assert fill.retimed_words == 4


def test_a_run_the_reference_heard_again_nearer_than_its_pairing_stays() -> None:
    # The alignment pairs the run with the copy 20 s before; another is 2.5 s after it.
    own = _said(("alpha", 0.0), ("one", 30.0), ("two", 31.0), ("three", 32.0), ("omega", 50.0))
    heard = _heard(("alpha", 0.0), ("one", 10.0), ("two", 11.0), ("three", 12.0))
    heard += _heard(("one", 32.5), ("two", 33.5), ("three", 34.5), ("omega", 50.0))

    filled, fill = fill_holes(_transcript(own, duration=51.0), _transcript(heard))

    assert (filled.words, fill.retimed) == (own, ())


def _timed(spoken: str, speaker: int | None) -> list[Word]:
    """Words 0.3 s long from "text@start" pairs."""
    pairs = [token.rsplit("@", 1) for token in spoken.split()]
    return [
        Word(text=text, start=float(at), end=round(float(at) + 0.3, 2), speaker=speaker)
        for text, at in pairs
    ]


@pytest.mark.parametrize(
    ("said", "heard"),
    [
        # "I" and "so" pair with the phrases around "I think so", which stays where it was said.
        (
            "no@0.25 no@0.65 no@1.05 thank@1.7 you@2.1 so@2.5 much@2.9 I@10.75 think@11.15 "
            "so@11.55 thank@13.0 you@13.4 so@13.8 much@14.2 no@21.45 no@21.85 no@22.25",
            "no@0.25 no@0.65 no@1.05 thank@1.7 you@2.1 so@2.5 much@2.9 I@8.55 don't@8.95 "
            "know@9.35 I@10.8 think@11.2 so@11.6 no@12.2 no@12.6 no@13.0 no@13.65 no@14.05 "
            "no@14.45 no@15.1 no@15.5 no@15.9 so@17.55 so@17.95 thank@18.6 you@19.0 so@19.4 "
            "much@19.8 no@21.45 no@21.85 no@22.25 no@22.9 no@23.3 no@23.7",
        ),
        # Moved, "I think so" uncovers "that's right" beside the "so so" the transcript
        # has 1.65 s early.
        (
            "so@0.0 so@0.4 I@6.08 think@6.48 so@6.88",
            "let's@0.3 move@0.7 on@1.1 so@1.65 so@2.05 that's@6.2 right@6.6 I@9.0 think@9.4 "
            "so@9.8 let's@13.3 move@13.7 on@14.1 no@20.5 no@20.9 no@21.3 let's@22.1 move@22.5 "
            "on@22.9",
        ),
    ],
    ids=["mixed pairings", "a neighbor timed early"],
)
def test_a_hole_a_move_opens_takes_no_copy_of_a_word_the_transcript_has(
    said: str, heard: str
) -> None:
    reference = _timed(heard, None)

    filled, _ = fill_holes(_transcript(_timed(said, 0)), _transcript(reference))

    held, spoken = Counter(w.text for w in filled.words), Counter(w.text for w in reference)
    assert {text: held[text] - spoken[text] for text in held if held[text] > spoken[text]} == {}


@pytest.mark.parametrize(
    ("said", "heard", "inserted"),
    [
        (
            "alpha@0 one@2 two@3 three@4 four@5 omega@20",
            "alpha@0 one@2 two@3 lost@4 one@12 two@13 three@14 four@15 omega@20",
            "one@2 two@3 lost@4",
        ),
        (
            "alpha@0 one@2 two@3 three@4 four@5 omega@20",
            "alpha@0 one@2.2 we@3 lost@3.6 one@4.2 one@12 two@13 three@14 four@15 omega@20",
            "one@2.2 we@3 lost@3.6 one@4.2",
        ),
        (
            "start@0 the@10 next@11 item@12 end@30",
            "start@0 the@10.2 alpha@10.8 beta@11.4 the@12 the@18 next@19 item@20 end@30",
            "the@10.2 alpha@10.8 beta@11.4 the@12",
        ),
        (
            "start@0 the@10 next@11 item@12 end@30",
            "start@0 the@10.2 the@10.8 alpha@11.4 beta@12 gamma@12.6 the@18 next@19 item@20 end@30",
            "the@10.2 the@10.8 alpha@11.4 beta@12 gamma@12.6",
        ),
    ],
    ids=["the run's start said before", "a word said twice", "a word said again", "a leading pair"],
)
def test_a_moved_word_stands_for_one_reference_word(said: str, heard: str, inserted: str) -> None:
    # No other transcript word could be the one the reference heard where each moved.
    filled, _ = fill_holes(_transcript(_timed(said, 0)), _transcript(_timed(heard, None)))

    assert [word for word in filled.words if word.speaker is None] == _timed(inserted, None)


def test_a_moved_word_stands_for_no_copy_near_where_it_went() -> None:
    # "go" moves onto the "go" at 17 and the transcript keeps the one at 24, so the
    # passage between them misses one of its two.
    said = "alpha@0 go@2 stop@3 wait@4 go@24 omega@32"
    heard = "alpha@0 go@17 stop@18 wait@19 p@20.5 go@21 go@22 q@23 go@24 omega@32"

    filled, _ = fill_holes(_transcript(_timed(said, 0)), _transcript(_timed(heard, None)))

    inserted = [word for word in filled.words if word.speaker is None]
    assert inserted == _timed("p@20.5 go@21 go@22 q@23", None)


def test_a_hole_within_eight_seconds_of_a_move_takes_no_copy_of_a_word_the_transcript_has() -> None:
    # Moved, "one two three" leave the reference's earlier copy unmatched, which
    # stretches the hole's span over the "go go" the transcript has 6 s early.
    said = "alpha@0 go@2 go@2.4 x@20 one@21 two@22 three@23 omega@40"
    heard = (
        "alpha@0 one@4 two@4.4 three@4.8 go@8 go@8.4 p@12 q@13 x@20 one@26 two@27 three@28 omega@40"
    )

    filled, _ = fill_holes(_transcript(_timed(said, 0)), _transcript(_timed(heard, None)))

    inserted = [word for word in filled.words if word.speaker is None]
    assert inserted == _timed("one@4 two@4.4 three@4.8 p@12 q@13", None)


def test_words_out_of_start_order_are_never_moved() -> None:
    own = [*_SLID, *_said(("omega", 20.0), ("late", 19.0))]
    heard = sorted([*_LATER, *_DROPPED, *_heard(("omega", 20.0))], key=lambda word: word.start)
    ordered = sorted(own, key=lambda word: word.start)

    unordered, _ = fill_holes(_transcript(own, duration=21.0), _transcript(heard))
    moved, _ = fill_holes(_transcript(ordered, duration=21.0), _transcript(heard))

    assert [word for word in unordered.words if word.speaker is not None] == own
    assert [word.start for word in moved.words if word.text == "one"] == [12.0]


def test_a_moved_start_is_held_between_the_words_around_the_run() -> None:
    # The reference heard "four" and "five" after "omega", which stays; they wait for it.
    own = [*_SLID, *_said(("five", 6.0), ("omega", 14.5), ("end", 30.0))]
    heard = [*_LATER, *_heard(("omega", 14.6), ("five", 16.0), ("end", 30.0))]
    heard = sorted(heard, key=lambda word: word.start)

    filled, fill = fill_holes(_transcript(own, duration=31.0), _transcript(heard))

    assert [(word.text, word.start, word.end) for word in filled.words] == [
        *(("alpha", 0.0, 0.5), ("one", 12.0, 12.5), ("two", 13.0, 13.5)),
        *(("three", 14.0, 14.5), ("four", 14.5, 15.5), ("five", 14.5, 16.5)),
        *(("omega", 14.5, 15.0), ("end", 30.0, 30.5)),
    ]
    assert fill.retimed == (Retime(2.0, 6.5, 12.0, 16.5, 5, 2, 1.5),)


def test_a_start_held_after_its_end_takes_the_start_as_its_end() -> None:
    # The reference heard the run before "x", which stays: the run waits at x's start.
    own = _said(("x", 10.0), ("one", 15.0), ("two", 16.0), ("three", 17.0), ("omega", 30.0))
    heard = _heard(("one", 5.0), ("two", 6.0), ("three", 7.0), ("x", 10.0), ("omega", 30.0))

    filled, fill = fill_holes(_transcript(own, duration=31.0), _transcript(heard))

    assert [(word.text, word.start, word.end) for word in filled.words] == [
        *(("x", 10.0, 10.5), ("one", 10.0, 10.0), ("two", 10.0, 10.0)),
        *(("three", 10.0, 10.0), ("omega", 30.0, 30.5)),
    ]
    assert fill.retimed == (Retime(15.0, 17.5, 10.0, 10.0, 3, 3, 5.0),)


def test_words_lined_up_by_one_shared_token_each_stay() -> None:
    # Each of "one", "two", "three" matches alone, between words the reference does not have.
    own = _said(("alpha", 0.0), ("u", 1.0), ("one", 2.0), ("x", 3.0), ("two", 4.0))
    own += _said(("y", 5.0), ("three", 6.0), ("v", 7.0), ("omega", 30.0))
    heard = _heard(("alpha", 0.0), ("one", 12.0), ("p", 13.0), ("two", 14.0), ("q", 15.0))
    heard += _heard(("three", 16.0), ("omega", 30.0))

    filled, fill = fill_holes(_transcript(own, duration=31.0), _transcript(heard))

    assert (filled.words, fill.retimed) == (own, ())


def test_words_the_reference_did_not_hear_share_the_stretch_between_moved_ones() -> None:
    own = _said(("alpha", 0.0), ("one", 2.0), ("two", 2.1), ("kids", 2.2), ("here", 2.3))
    own += _said(("three", 2.4), ("four", 2.5), ("omega", 30.0))
    heard = _heard(("alpha", 0.0), ("one", 12.0), ("two", 13.0), ("tube", 13.5))
    heard += _heard(("three", 16.0), ("four", 17.0), ("omega", 30.0))

    filled, _ = fill_holes(_transcript(own, duration=31.0), _transcript(heard))

    assert [(word.text, word.start, word.end) for word in filled.words[2:6]] == [
        *(("two", 13.0, 13.5), ("kids", 14.0, 15.0)),
        *(("here", 15.0, 16.0), ("three", 16.0, 16.5)),
    ]


def test_a_moved_word_ends_where_its_last_matched_token_does() -> None:
    own = [*_said(("alpha", 0.0), ("fifty-five", 2.0), ("nobody's", 3.0), ("got", 4.0))]
    own += _said(("any", 5.0), ("omega", 30.0))
    heard = _heard(("alpha", 0.0), ("fifty", 12.0), ("five", 12.5), ("nobody's", 13.0))
    heard += _heard(("got", 14.0), ("any", 15.0), ("omega", 30.0))

    filled, _ = fill_holes(_transcript(own, duration=31.0), _transcript(heard))

    assert (filled.words[1].start, filled.words[1].end) == (12.0, 13.0)


_QUARTERS = st.integers(min_value=0, max_value=240).map(lambda quarters: quarters / 4)


def _spoken(speaker: int | None) -> st.SearchStrategy[list[Word]]:
    def word(text: str, start: float, length: float) -> Word:
        return Word(text=text, start=start, end=start + length, speaker=speaker)

    texts = st.sampled_from(["go", "Go.", "stop", "wait", "now", "then", "um", "uh", "—"])
    lengths = st.integers(min_value=0, max_value=12).map(lambda quarters: quarters / 4)
    return st.lists(st.builds(word, texts, _QUARTERS, lengths), max_size=30)


def _in_order(words: list[Word], ordered: bool) -> list[Word]:
    return sorted(words, key=lambda word: word.start) if ordered else words


@given(own=_spoken(0), heard=_spoken(None), duration=st.none() | _QUARTERS, ordered=st.booleans())
def test_a_fill_only_inserts_and_only_inside_holes_of_two_seconds(
    own: list[Word], heard: list[Word], duration: float | None, ordered: bool
) -> None:
    own = _in_order(own, ordered)

    filled, fill = fill_holes(_transcript(own, duration=duration), _transcript(heard))

    kept = [word for word in filled.words if word.speaker is not None]
    assert [(word.text, word.speaker) for word in kept] == [
        (word.text, word.speaker) for word in own
    ]
    # Only words in start order move, and only in time: they stay in start order.
    assert kept == own if first_decrease(own) is not None else first_decrease(kept) is None
    assert sum(word != was for word, was in zip(kept, own, strict=True)) == fill.retimed_words
    inserted = [word for word in filled.words if word.speaker is None]
    assert len(inserted) == fill.words
    audio_end = max([duration or 0.0, *(word.end for word in heard)])
    holes = find_holes(kept, audio_end, MIN_DROP_SECONDS)
    assert all(any(hole.start < word.start < hole.end for hole in holes) for word in inserted)


@given(own=_spoken(0), data=st.data())
def test_a_reference_whose_words_the_transcript_has_inserts_nothing(
    own: list[Word], data: st.DataObject
) -> None:
    kept = data.draw(st.lists(st.booleans(), min_size=len(own), max_size=len(own)))
    heard = [
        word.model_copy(update={"speaker": None})
        for word, keep in zip(own, kept, strict=True)
        if keep
    ]

    assert fill_holes(_transcript(own), _transcript(own))[0].words == own
    assert fill_holes(_transcript(own), _transcript(heard))[0].words == own


# A run the alignment pairs across 30 s moves words away from copies the reference still has.
_ACROSS = [
    Word(text=text, start=start, end=start + length, speaker=0)
    for text, start, length in [
        *[("go", 0.0, 0.0)] * 3,
        *(("uh", 0.0, 3.0), ("stop", 4.5, 2.25), ("stop", 7.75, 1.75), ("stop", 19.25, 0.0)),
        *(("go", 22.0, 0.0), ("go", 22.5, 0.0), ("Go.", 38.5, 1.5), ("stop", 38.5, 0.0)),
        *(("stop", 38.5, 0.0), ("Go.", 47.5, 2.75), ("stop", 52.0, 3.0)),
    ]
]
# A run moved away from "go" and "wait" leaves them to the hole it opens unless counted where
# they were.
_LEFT = [
    Word(text=text, start=start, end=start + length, speaker=0)
    for text, start, length in [
        *(("go", 0.0, 0.0), ("go", 0.0, 0.0), ("go", 2.25, 0.0), ("go", 29.25, 0.0)),
        *(("wait", 29.75, 0.0), ("go", 31.75, 0.0), ("now", 31.75, 0.0), ("—", 31.75, 0.0)),
        *(("stop", 32.0, 1.5), ("stop", 35.0, 2.0)),
    ]
]
_MOVES = st.integers(min_value=-32, max_value=32).map(lambda quarters: quarters / 4)


def _shifted(own: list[Word], shifts: list[float]) -> list[Word]:
    """A reference copy of each of `own`'s words, moved by its shift but not before 0."""
    heard: list[Word] = []
    for word, shift in zip(own, shifts, strict=False):
        start = max(0.0, word.start + shift)
        heard.append(Word(text=word.text, start=start, end=start + word.end - word.start))
    return heard


@example(
    own=_ACROSS,
    shifts=[0, 0, 2.5, 2.25, 0.25, 0.25, 0.25, -2.75, 0.25, 2.25, 0.25, 0.25, 2.25, 0],
    ordered=True,
)
@example(own=_LEFT, shifts=[0, 0, 2.25, 2.5, 2.5, 0, -2.25, 0, 0, -5.5], ordered=True)
@given(own=_spoken(0), shifts=st.lists(_MOVES, min_size=30, max_size=30), ordered=st.booleans())
def test_a_word_the_transcript_has_within_eight_seconds_is_never_inserted(
    own: list[Word], shifts: list[float], ordered: bool
) -> None:
    own = _in_order(own, ordered)
    # Moved, a copy can land inside a hole, where only the match walk keeps it out.
    filled, fill = fill_holes(_transcript(own), _transcript(_shifted(own, shifts)))

    kept = [(word.text, word.speaker) for word in filled.words if word.speaker is not None]
    assert (kept, fill.filled) == ([(word.text, word.speaker) for word in own], ())


_GO_STOP = [("go", 0.0), ("go", 2.25), ("stop", 2.25), *[("go", 10.75)] * 3]
_STOP_GO = [("stop", 0.0), ("stop", 0.25), ("go", 0.0)]


@example(
    own=[*_SLID, *_said(("omega", 20.0))],
    heard=[*_LATER, *_DROPPED, *_heard(("omega", 20.0))],
    shifts=[],
    shifted=False,
    ordered=True,
)
# Filled, the "go" at 10.75 make "stop" at 8.5 look like a run heard at 0.25.
@example(
    own=[
        *(Word(text=text, start=2.25, end=2.25, speaker=0) for text in ("go", "stop")),
        Word(text="stop", start=8.5, end=10.5, speaker=0),
    ],
    heard=[
        *(Word(text=text, start=start, end=start) for text, start in _GO_STOP[:1]),
        *(Word(text="go", start=10.75, end=10.75) for _ in range(2)),
        Word(text="go", start=10.75, end=12.5),
        *(Word(text=text, start=start, end=start) for text, start in _STOP_GO),
    ],
    shifts=[],
    shifted=False,
    ordered=True,
)
@given(
    own=_spoken(0),
    heard=_spoken(None),
    shifts=st.lists(_MOVES, min_size=30, max_size=30),
    shifted=st.booleans(),
    ordered=st.booleans(),
)
def test_filling_again_from_the_same_reference_never_repeats_a_word(
    own: list[Word], heard: list[Word], shifts: list[float], shifted: bool, ordered: bool
) -> None:
    own = _in_order(own, ordered)
    if shifted:
        heard = [*_shifted(own, shifts), *heard]

    once, _ = fill_holes(_transcript(own), _transcript(heard))
    twice, again = fill_holes(once, _transcript(heard))

    assert again.retimed == ()
    remaining = iter(twice.words)
    assert all(any(word == later for later in remaining) for word in once.words)
    there = {(word.text, word.start) for word in once.words}
    added = Counter(twice.words) - Counter(once.words)
    assert not [word for word in added.elements() if (word.text, word.start) in there]


@pytest.mark.parametrize(
    ("own", "heard"),
    [
        ([*_SLID, *_said(("omega", 20.0))], [*_LATER, *_DROPPED, *_heard(("omega", 20.0))]),
        # Filled, the "go" at 10.75 pull the alignment of the words before them one copy back.
        (
            [
                *(Word(text="go", start=start, end=start, speaker=0) for start in (0.0, 2.25, 4.5)),
                Word(text="stop", start=8.5, end=10.5, speaker=0),
            ],
            [
                *(Word(text=text, start=start, end=start) for text, start in _GO_STOP),
                Word(text="go", start=10.75, end=12.5),
            ],
        ),
    ],
    ids=["a drifted run", "words the fill realigns"],
)
def test_filling_the_filled_words_again_moves_and_inserts_nothing(
    own: list[Word], heard: list[Word]
) -> None:
    once, _ = fill_holes(_transcript(own), _transcript(heard))
    twice, again = fill_holes(once, _transcript(heard))

    assert (twice.words, again.retimed, again.filled) == (once.words, (), ())


def _files(tmp_path: Path) -> tuple[Path, Path]:
    own = _said(
        ("alpha", 0.0), ("omega", 10.0), *(("x", float(second)) for second in range(30, 60))
    )
    heard = _heard(("one", 3.0), ("two", 4.0), ("three", 5.0))
    heard += _heard(*((f"w{second}", second + 0.25) for second in range(30, 50)))
    paths = tmp_path / "board.transcript.json", tmp_path / "board.parakeet.json"
    _transcript(own, duration=60.0).dump(paths[0])
    _transcript(heard).dump(paths[1])
    return paths


def test_fill_writes_beside_the_transcript_and_reports_every_span(tmp_path: Path) -> None:
    own, reference = _files(tmp_path)

    result = runner.invoke(app, ["fill", str(own), str(reference)])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{tmp_path / 'board.filled.json'}\n"
    assert result.stderr.splitlines() == [
        "scribe: filled 00:00:03.0-00:00:05.5 (3 words from Parakeet, no speaker)",
        "scribe: possible dropped speech 00:00:30.0-00:01:00.0; listen to that span",
        "scribe: cross-check: 1 span filled with 3 words from Parakeet, 1 unresolved",
    ]
    filled = Transcript.load(tmp_path / "board.filled.json")
    assert filled.engine.params["fill_ranges"] == "[[3.0, 5.5]]"
    assert filled.engine.params["fill_unresolved_ranges"] == "[[30.0, 60.0]]"


def test_fill_reports_each_run_it_moved_before_the_spans_it_filled(tmp_path: Path) -> None:
    own, reference = tmp_path / "board.transcript.json", tmp_path / "board.parakeet.json"
    _transcript([*_SLID, *_said(("omega", 20.0))], duration=21.0).dump(own)
    heard = sorted([*_LATER, *_DROPPED, *_heard(("omega", 20.0))], key=lambda word: word.start)
    _transcript(heard).dump(reference)

    result = runner.invoke(app, ["fill", str(own), str(reference)])

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == [
        "scribe: re-timed 00:00:02.0-00:00:05.5 to 00:00:12.0-00:00:15.5 "
        "(4 words to where Parakeet heard them)",
        "scribe: filled 00:00:02.0-00:00:05.5 (4 words from Parakeet, no speaker)",
        "scribe: cross-check: 1 span filled with 4 words from Parakeet, 0 unresolved",
    ]
    filled = Transcript.load(tmp_path / "board.filled.json")
    assert filled.engine.params["fill_retimed_words"] == 4


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["{own}", "{tmp}/absent.json"], "cannot read transcript"),
        (["{own}", "{reference}", "--out", "{reference}"], "--out {reference} is REFERENCE"),
        (["{turns}", "{reference}"], "has turns but no words"),
    ],
)
def test_fill_refuses_a_bad_input_with_one_line_and_exit_two(
    tmp_path: Path, args: list[str], message: str
) -> None:
    own, reference = _files(tmp_path)
    turns = tmp_path / "turns.json"
    stale = Turn(speaker="Speaker 1", start=0.0, end=1.0, text="hi")
    _transcript([]).model_copy(update={"turns": [stale]}).dump(turns)
    names = {"own": own, "reference": reference, "turns": turns, "tmp": tmp_path}

    result = runner.invoke(app, ["fill", *(arg.format(**names) for arg in args)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith("scribe: ")
    assert message.format(**names) in result.stderr
    assert len(result.stderr.splitlines()) == 1


def test_fill_that_cannot_write_its_result_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own, reference = _files(tmp_path)

    def refuse(self: Transcript, path: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Transcript, "dump", refuse)

    result = runner.invoke(app, ["fill", str(own), str(reference)])

    assert (result.exit_code, result.stdout) == (2, "")
    out = tmp_path / "board.filled.json"
    assert result.stderr == f"scribe: cannot write transcript to {out}: disk full\n"
