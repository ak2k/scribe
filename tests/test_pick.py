"""The pick: where xAI's and Parakeet's words disagree, and which reading is kept there."""

from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scribe.pick import Spot, find_spots
from scribe.schema import Word
from scribe.vote import align_words


def _said(*texts: str, speaker: int | None = 0) -> list[Word]:
    """One word a second, each 0.4 s long."""
    return [
        Word(text=text, start=float(index), end=index + 0.4, speaker=speaker)
        for index, text in enumerate(texts)
    ]


def _timed(*said: tuple[str, float, float]) -> list[Word]:
    return [Word(text=text, start=start, end=end, speaker=0) for text, start, end in said]


def test_align_words_names_the_word_each_step_consumes_a_token_of() -> None:
    backbone = _timed(("the", 0.0, 0.4), ("ice-cream", 1.0, 1.8), ("truck", 4.0, 4.4))
    hypothesis = _timed(
        ("the", 0.0, 0.4), ("ice", 1.0, 1.3), ("scream", 1.4, 1.8), ("big", 2.5, 2.9)
    )

    assert align_words(backbone, hypothesis) == [
        ("match", 0, 0),
        ("match", 1, 1),
        ("sub", 1, 2),
        ("ins", None, 3),
        ("del", 2, None),
    ]


def test_equal_readings_hold_no_spot() -> None:
    words = _said("the", "cat", "sat")

    assert find_spots(words, words) == []


@pytest.mark.parametrize(
    ("transcript", "reference"),
    [
        # A contraction, spelled out or not.
        (
            _timed(("it's", 0.0, 0.4), ("fine", 1.0, 1.4)),
            _timed(("it", 0.0, 0.2), ("is", 0.2, 0.4), ("fine", 1.0, 1.4)),
        ),
        (_said("yeah", "sure"), _said("yes", "sure")),
        # A number, in words or digits, and a percent sign.
        (
            _timed(("twenty", 0.0, 0.4), ("five", 0.5, 0.9), ("people", 1.0, 1.4)),
            _timed(("25", 0.0, 0.9), ("people", 1.0, 1.4)),
        ),
        (_timed(("5%", 0.0, 0.8)), _timed(("five", 0.0, 0.4), ("percent", 0.4, 0.8))),
        # A filler beside the same word.
        (_timed(("oh", 0.0, 0.2), ("okay", 0.3, 0.6)), _timed(("ok", 0.0, 0.6))),
        # A repeat that the neighboring words make the same.
        (_said("that", "that", "is"), _said("that", "is", "is")),
        # The same words in another order, or spaced differently, or accented.
        (_said("yes", "no"), _said("no", "yes")),
        (_timed(("every", 0.0, 0.4), ("day", 0.4, 0.8)), _timed(("everyday", 0.0, 0.8))),
        (_said("café"), _said("cafe")),
        # An ordinal in digits or in words.
        (_said("the", "1st", "quarter"), _said("the", "first", "quarter")),
    ],
    ids=[
        "contraction",
        "yeah",
        "number",
        "percent",
        "filler",
        "repeat",
        "order",
        "spacing",
        "accent",
        "ordinal",
    ],
)
def test_a_difference_in_form_alone_is_no_spot(
    transcript: list[Word], reference: list[Word]
) -> None:
    assert find_spots(transcript, reference) == []
    assert find_spots(reference, transcript) == []


@pytest.mark.parametrize(
    ("transcript", "reference"),
    [
        (_said("so", "um", "we"), _said("so", "and", "we")),
        (_said("so", "you know", "we"), _said("so", "and", "we")),
    ],
    ids=["um", "you-know"],
)
def test_a_side_of_only_fillers_is_no_spot(transcript: list[Word], reference: list[Word]) -> None:
    assert find_spots(transcript, reference) == []
    assert find_spots(reference, transcript) == []


@pytest.mark.parametrize(
    ("said", "heard"),
    [
        (("so", "I", "like", "cats"), ("so", "I", "hate", "cats")),
        (("it", "was", "like", "big"), ("it", "was", "light", "big")),
    ],
    ids=["hate", "light"],
)
def test_like_alone_against_another_word_is_a_spot(
    said: tuple[str, ...], heard: tuple[str, ...]
) -> None:
    assert find_spots(_said(*said), _said(*heard)) == [Spot(range(2, 3), range(2, 3))]


def test_like_beside_the_same_words_is_still_no_spot() -> None:
    transcript = _timed(
        ("so", 0.0, 0.4),
        ("like,", 1.0, 1.3),
        ("yeah,", 1.4, 1.8),
        ("you", 2.0, 2.2),
        ("know", 2.2, 2.4),
    )
    reference = _timed(("so", 0.0, 0.4), ("yes,", 1.4, 1.8), ("you", 2.0, 2.2), ("know", 2.2, 2.4))

    assert find_spots(transcript, reference) == []
    assert find_spots(reference, transcript) == []


def test_the_same_letters_split_at_another_word_boundary_are_a_spot() -> None:
    transcript = _said("it", "is", "an", "ice", "day")
    reference = _said("it", "is", "a", "nice", "day")

    assert find_spots(transcript, reference) == [Spot(range(2, 4), range(2, 4))]


def test_one_word_written_apart_among_the_same_words_is_still_no_spot() -> None:
    apart = _timed(
        ("we", 0.0, 0.4),
        ("meet", 0.5, 0.9),
        ("every", 1.0, 1.3),
        ("day", 1.3, 1.6),
        ("here", 2.0, 2.4),
    )
    joined = _timed(
        ("we", 0.0, 0.4), ("meet", 0.5, 0.9), ("everyday", 1.0, 1.6), ("here", 2.0, 2.4)
    )

    assert find_spots(apart, joined) == []
    assert find_spots(joined, apart) == []


def test_a_spot_holds_exactly_the_words_that_differ() -> None:
    transcript = _said("the", "cat", "sat", "on", "the", "mat")
    reference = _said("the", "hat", "sat", "on", "the", "mat")

    assert find_spots(transcript, reference) == [Spot(range(1, 2), range(1, 2))]


def test_differences_one_matched_word_apart_are_one_spot() -> None:
    transcript = _said("we", "saw", "the", "cat")
    reference = _said("he", "saw", "a", "cat")

    assert find_spots(transcript, reference) == [Spot(range(3), range(3))]


def test_differences_two_matched_words_apart_are_two_spots() -> None:
    transcript = _said("a", "cat", "sat", "on", "mat")
    reference = _said("a", "hat", "sat", "on", "bat")

    assert find_spots(transcript, reference) == [
        Spot(range(1, 2), range(1, 2)),
        Spot(range(4, 5), range(4, 5)),
    ]


def test_a_word_a_spot_takes_part_of_is_taken_whole_on_either_side() -> None:
    joined = _timed(("the", 0.0, 0.4), ("ice-cream", 1.0, 1.8), ("truck", 2.0, 2.4))
    split = _timed(("the", 0.0, 0.4), ("ice", 1.0, 1.3), ("scream", 1.4, 1.8), ("truck", 2.0, 2.4))

    assert find_spots(joined, split) == [Spot(range(1, 2), range(1, 3))]
    assert find_spots(split, joined) == [Spot(range(1, 3), range(1, 2))]


def test_spots_a_whole_word_brings_one_step_apart_are_one_spot() -> None:
    transcript = _timed(
        ("we", 0.0, 0.4), ("big-cat", 1.0, 1.8), ("sat", 2.0, 2.4), ("down", 3.0, 3.4)
    )
    reference = _timed(
        ("we", 0.0, 0.4),
        ("pig", 1.0, 1.3),
        ("cat", 1.4, 1.8),
        ("sat", 2.0, 2.4),
        ("town", 3.0, 3.4),
    )

    assert find_spots(transcript, reference) == [Spot(range(1, 4), range(1, 5))]


@pytest.mark.parametrize(
    ("said", "heard"),
    [
        (("мы", "видели", "кота", "вчера"), ("мы", "видели", "кита", "вчера")),
        (("είδαμε", "τη", "γάτα", "χθες"), ("είδαμε", "τη", "μάτα", "χθες")),
    ],
    ids=["cyrillic", "greek"],
)
def test_readings_in_another_script_that_differ_are_a_spot(
    said: tuple[str, ...], heard: tuple[str, ...]
) -> None:
    assert find_spots(_said(*said), _said(*heard)) == [Spot(range(2, 3), range(2, 3))]


@pytest.mark.parametrize(
    ("said", "heard"),
    [
        (("please", "do", "not", "note", "that"), ("please", "do", "no", "note", "that")),
        (("we", "read", "his", "history", "today"), ("we", "read", "hi", "history", "today")),
        (("we", "go", "now", "nowhere", "fast"), ("we", "go", "no", "nowhere", "fast")),
        # Neither reading begins the next word.
        (("we", "do", "not", "go", "there"), ("we", "do", "no", "go", "there")),
    ],
    ids=["not-note", "his-history", "now-nowhere", "not-go"],
)
def test_readings_that_each_begin_the_matched_word_after_them_are_a_spot(
    said: tuple[str, ...], heard: tuple[str, ...]
) -> None:
    assert find_spots(_said(*said), _said(*heard)) == [Spot(range(2, 3), range(2, 3))]


@pytest.mark.parametrize(
    ("transcript", "reference"),
    [
        # "that" said twice, once in the spot and once in the matched word before it.
        (
            _timed(("so", 0.0, 0.4), ("that", 1.0, 1.4), ("that", 2.0, 2.4), ("is", 3.0, 3.4)),
            _timed(("so", 0.0, 0.4), ("that", 1.0, 1.4), ("is", 2.0, 2.4), ("is", 3.0, 3.4)),
        ),
        # A false start that is the spot's first word, of the word after it in the spot.
        (
            _timed(("so", 0.0, 0.4), ("th", 1.0, 1.1), ("that's", 1.2, 1.6), ("fine", 2.0, 2.4)),
            _timed(("so", 0.0, 0.4), ("that", 1.0, 1.3), ("is", 1.3, 1.6), ("fine", 2.0, 2.4)),
        ),
    ],
    ids=["repeat", "false-start"],
)
def test_a_stutter_at_a_spots_edge_is_still_no_spot(
    transcript: list[Word], reference: list[Word]
) -> None:
    assert find_spots(transcript, reference) == []
    assert find_spots(reference, transcript) == []


def test_runs_one_long_word_joins_are_one_spot() -> None:
    # Two runs, each widened to the whole of one four-token word, overlap.
    transcript = _timed(("go", 0.0, 0.3), ("alpha-beta-gamma-delta", 1.0, 1.8), ("now", 3.0, 3.3))
    reference = _timed(
        ("go", 0.0, 0.3),
        ("zeta", 1.0, 1.1),
        ("beta", 1.2, 1.3),
        ("gamma", 1.4, 1.5),
        ("eta", 1.6, 1.8),
        ("now", 3.0, 3.3),
    )

    assert find_spots(transcript, reference) == [Spot(range(1, 2), range(1, 5))]


def test_a_word_only_one_side_heard_is_the_fills_not_a_spot() -> None:
    assert (
        find_spots(_said("the", "big", "cat"), _timed(("the", 0.0, 0.4), ("cat", 2.0, 2.4))) == []
    )
    assert (
        find_spots(_timed(("the", 0.0, 0.4), ("cat", 2.0, 2.4)), _said("the", "big", "cat")) == []
    )


def test_a_word_only_one_side_heard_one_matched_word_from_a_spot_stays_out_of_it() -> None:
    lovely = _said("please", "lovely", "cat", "sat")
    plain = _timed(("please", 0.0, 0.4), ("cat", 2.0, 2.4), ("slept", 3.0, 3.4))

    assert find_spots(lovely, plain) == [Spot(range(3, 4), range(2, 3))]
    assert find_spots(plain, lovely) == [Spot(range(2, 3), range(3, 4))]


def test_a_word_only_one_side_heard_between_two_widened_spots_stays_out_of_them() -> None:
    # Each spot widens to a whole hyphenated word, which leaves "lovely" one step from both.
    said = _timed(("big-cat", 0.0, 0.8), ("lovely", 1.0, 1.4), ("hat-sat", 2.0, 2.8))
    heard = _timed(("pig", 0.0, 0.3), ("cat", 0.4, 0.8), ("hat", 2.0, 2.3), ("sad", 2.4, 2.8))

    assert find_spots(said, heard) == [Spot(range(1), range(2)), Spot(range(2, 3), range(2, 4))]


_VOCAB = [
    *["the", "cat", "hat", "um", "it's", "it is", "twenty", "20", "sat", "e-mail", "Cat,", ""],
    "cat-the-hat-sat",
]


@st.composite
def _pairs(draw: st.DrawFn) -> tuple[list[Word], list[Word]]:
    def said(texts: list[str], offset: float) -> list[Word]:
        return [
            Word(text=text, start=index * 0.5 + offset, end=index * 0.5 + offset + 0.4, speaker=0)
            for index, text in enumerate(texts)
        ]

    return (
        said(draw(st.lists(st.sampled_from(_VOCAB), max_size=12)), 0.0),
        said(
            draw(st.lists(st.sampled_from(_VOCAB), max_size=12)), draw(st.sampled_from([0.0, 0.2]))
        ),
    )


@given(_pairs())
def test_spots_are_disjoint_ordered_and_hold_words_on_both_sides(
    pair: tuple[list[Word], list[Word]],
) -> None:
    transcript, reference = pair

    spots = find_spots(transcript, reference)

    for spot in spots:
        assert spot.transcript and spot.reference
        assert spot.transcript.stop <= len(transcript)
        assert spot.reference.stop <= len(reference)
    for before, after in pairwise(spots):
        assert before.transcript.stop <= after.transcript.start
        assert before.reference.stop <= after.reference.start


@given(st.lists(st.sampled_from(_VOCAB), max_size=12))
def test_a_transcript_against_itself_holds_no_spot(texts: list[str]) -> None:
    words = _said(*texts)

    assert find_spots(words, words) == []
