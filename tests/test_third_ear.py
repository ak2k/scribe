"""Rule ear-1: each recognizer's slot at a pick's spot, its verdict, and the side delivered."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scribe.ear import RECOGNIZERS, Heard
from scribe.pick import Spot, find_spots
from scribe.schema import Word
from scribe.third_ear import Slot, clip, clips, deliver, slot, verdict, vote
from tests.pick_fakes import transcript

if TYPE_CHECKING:
    from scribe.pick import Side
    from scribe.third_ear import Verdict

_NAMES = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango"
)
_ELEVEN = "black white green blue cyan pink gray brown amber coral ivory"


def _timed(text: str, start: float = 0.0, step: float = 1.0) -> list[Word]:
    """`text`'s words, one every `step` s from `start`, each lasting half that."""
    return [
        Word(text=word, start=start + n * step, end=start + (n + 0.5) * step)
        for n, word in enumerate(text.split())
    ]


def _heard(*texts: tuple[str, ...]) -> tuple[Heard, ...]:
    """Each recognizer's texts, one per spot heard."""
    versions = {"transformers": "5.18.0", "torch": "2.14.1"}
    return tuple(
        Heard(spec, versions, "mps", "bfloat16", 2.5, said)
        for spec, said in zip(RECOGNIZERS, texts, strict=True)
    )


def test_a_clip_pads_both_readings_by_five_seconds_within_the_recording() -> None:
    said, heard = _timed(_NAMES), _timed(_NAMES, start=0.25)

    assert clip(said, heard, Spot(range(6, 8), range(6, 7)), None) == (1.0, 12.5)
    assert clip(said, heard, Spot(range(6, 8), range(6, 7)), 10.0) == (1.0, 10.0)
    assert clip(said, heard, Spot(range(1, 2), range(1, 2)), None) == (0.0, 6.75)


def _spanned(*spans: tuple[str, float, float]) -> list[Word]:
    return [Word(text=text, start=start, end=end) for text, start, end in spans]


@pytest.mark.parametrize(
    ("said", "heard", "window"),
    [
        # "cat" runs on past "runs", the spot's last word.
        (
            _spanned(("alpha", 0, 0.5), ("cat", 1, 20), ("runs", 2, 2.5), ("omega", 30, 30.5)),
            _spanned(("alpha", 0, 0.5), ("hat", 1, 1.5), ("walks", 2, 2.5), ("omega", 30, 30.5)),
            (0.0, 25.0),
        ),
        # "walks" starts before "hat", the spot's first word.
        (
            _spanned(("alpha", 0, 0.5), ("cat", 10, 10.5), ("runs", 11, 11.5), ("omega", 30, 30.5)),
            _spanned(("alpha", 0, 0.5), ("hat", 9, 9.5), ("walks", 8, 8.5), ("omega", 30, 30.5)),
            (3.0, 16.5),
        ),
    ],
    ids=["end-overlapped", "start-out-of-order"],
)
def test_a_clip_spans_every_word_of_the_spot_whatever_their_order(
    said: list[Word], heard: list[Word], window: tuple[float, float]
) -> None:
    assert clip(said, heard, Spot(range(1, 3), range(1, 3)), None) == window


def test_a_spot_far_into_its_clip_gets_its_slot() -> None:
    # "papa", the first word after the spot, starts 8 s into the clip.
    text = "hotel india juliet kilo lima bike remember oscars papa quebec romeo sierra tango"

    found = slot(_timed(_NAMES), Spot(range(12, 15), range(12, 15)), (7.0, 19.5), text)

    assert found == Slot(("bike", "remember", "oscars"), (), ())


def test_with_no_word_matched_beside_the_spot_its_slot_runs_to_the_clip_edge() -> None:
    found = slot(_timed("we saw the cat sat down"), Spot(range(3, 4), range(3, 4)), (0, 9), "zulu")

    assert found == Slot(("zulu",), ("we", "saw", "the"), ("sat", "down"))


def test_a_word_beside_the_spot_heard_in_its_clip_is_never_its_slot() -> None:
    # The first "hat" starts inside the clip, (5.0, 15.5), and ends well past it.
    said = _spanned(("we", 0, 0.5), ("hat", 8, 24), ("cat", 10, 10.5), ("omega", 12, 12.5))
    heard = _spanned(("we", 0, 0.5), ("hat", 8, 24), ("hat", 10, 10.5), ("omega", 12, 12.5))
    spots = find_spots(said, heard)
    text = ("hat omega",)

    sides, params = vote(
        transcript(said), transcript(heard), spots, ["transcript"], _heard(text, text)
    )

    assert spots == [Spot(range(2, 3), range(2, 3))]
    assert sides == ("transcript",)
    assert json.loads(str(params["ear_record"])) == [["transcript", ["", ""], ["third", "third"]]]
    assert slot(said, spots[0], (5.0, 15.5), "hat omega") == Slot((), (), ())


@pytest.mark.parametrize("text", ["", "  ", "..."])
def test_an_empty_text_gives_no_slot(text: str) -> None:
    assert slot(_timed("we saw the cat"), Spot(range(3, 4), range(3, 4)), (0, 9), text) is None


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("We saw the cat sat down.", "transcript"),
        ("we saw the hat sat down", "reference"),
        # "the" went unmatched beside the spot: a reading is tried with and without it.
        ("we saw hat sat down", "reference"),
        ("we saw the bat sat down", "third"),
    ],
)
def test_a_slot_names_the_reading_it_is(text: str, named: Verdict) -> None:
    found = slot(_timed("we saw the cat sat down"), Spot(range(3, 4), range(3, 4)), (0, 11), text)

    assert found is not None
    assert verdict(found, ["cat"], ["hat"], (["saw", "the"], ["sat", "down"])) == named


def test_a_slot_that_is_both_readings_names_neither() -> None:
    assert verdict(Slot(("the", "hat"), ("the",), ()), ["hat"], ["the", "hat"], ([], [])) == "third"


_SAW_THE, _SAT_DOWN = ["saw", "the"], ["sat", "down"]


@pytest.mark.parametrize("heard", [["saw"], ["the"], ["down"], ["the", "saw"]])
@pytest.mark.parametrize("words", [(), ("um",), ("you", "know"), ("like",)])
def test_a_slot_of_no_word_names_no_reading_made_of_the_words_around_it(
    words: tuple[str, ...], heard: list[str]
) -> None:
    # Set beside "saw the" and "sat down", `heard` collapses into them as a repeat.
    assert verdict(Slot(words, (), ()), ["cat"], heard, (_SAW_THE, _SAT_DOWN)) == "third"


def test_a_slot_doubling_the_word_after_the_spot_names_no_reading_made_of_the_words_around_it() -> (
    None
):
    said = _timed("we saw the cat sat down")
    found = slot(said, Spot(range(3, 4), range(3, 4)), (0, 11), "we saw the sat sat down")

    assert found is not None
    assert found.words == ("sat",)
    assert verdict(found, ["cat"], ["the"], (_SAW_THE, _SAT_DOWN)) == "third"


@pytest.mark.parametrize("words", [("the",), ("The,",)])
def test_a_slot_that_is_a_reading_made_of_the_words_around_it_names_it(
    words: tuple[str, ...],
) -> None:
    assert verdict(Slot(words, (), ()), ["cat"], ["the"], (_SAW_THE, _SAT_DOWN)) == "reference"


@pytest.mark.parametrize("words", [(), ("um",), ("uh", "um"), ("you", "know")])
def test_a_slot_of_no_word_or_of_fillers_names_no_reading_of_like_alone(
    words: tuple[str, ...],
) -> None:
    # Compared as the pick compares readings, "like" drops out with the fillers.
    assert verdict(Slot(words, (), ()), ["want"], ["like"], (["i"], ["cats"])) == "third"
    assert verdict(Slot(words, (), ()), ["like"], ["want"], (["i"], ["cats"])) == "third"


@pytest.mark.parametrize("words", [("like",), ("Like,",)])
def test_a_slot_that_is_like_names_a_reading_of_like_alone(words: tuple[str, ...]) -> None:
    assert verdict(Slot(words, (), ()), ["want"], ["like"], (["i"], ["cats"])) == "reference"


def test_recognizers_that_heard_no_word_at_a_spot_never_flip_it_to_like() -> None:
    said, heard = _timed("we like the cats here"), _timed("we want the cats here")
    text = ("we the cats here",)

    sides, params = vote(
        transcript(said),
        transcript(heard),
        find_spots(said, heard),
        ["reference"],
        _heard(text, text),
    )

    assert sides == ("reference",)
    assert json.loads(str(params["ear_record"])) == [["reference", ["", ""], ["third", "third"]]]


@pytest.mark.parametrize(
    ("side", "verdicts", "delivered"),
    [
        ("unsure", ["reference", "reference"], "reference"),
        ("failed", ["reference", "reference"], "reference"),
        ("transcript", ["reference", "reference"], "reference"),
        ("reference", ["transcript", "transcript"], "transcript"),
        ("reference", ["reference", "reference"], "reference"),
        ("unsure", ["transcript", "transcript"], "unsure"),
        ("unsure", ["reference", "third"], "unsure"),
        ("unsure", ["reference", None], "unsure"),
        ("unsure", ["reference"], "unsure"),
        ("guarded", ["reference", "reference"], "guarded"),
        ("restored", ["transcript", "transcript"], "restored"),
    ],
)
def test_a_spot_takes_the_reading_set_aside_only_when_every_recognizer_heard_it(
    side: Side, verdicts: list[Verdict | None], delivered: Side
) -> None:
    assert deliver(side, verdicts, guard=False, restore=False) == delivered


@pytest.mark.parametrize(
    ("heard", "said", "side", "aside"),
    [
        # The reference's reading is 5 spoken words shorter: the guard keeps the transcript's.
        (
            _timed("alpha bravo") + _timed("black", 2.0) + _timed("charlie delta", 8.0),
            "alpha bravo red green blue cyan pink gray charlie delta",
            "unsure",
            "reference",
        ),
        # It is 10 words longer: the restore keeps it.
        (
            _timed("alpha bravo") + _timed(_ELEVEN, 2.0, 1 / 11) + _timed("charlie delta", 3.0),
            "alpha bravo red charlie delta",
            "reference",
            "transcript",
        ),
    ],
)
def test_no_flip_goes_against_the_guard_or_the_restore(
    heard: list[Word], said: str, side: Side, aside: Verdict
) -> None:
    text = " ".join(word.text for word in heard) if aside == "reference" else said
    spots = find_spots(_timed(said), heard)

    sides, params = vote(
        transcript(_timed(said)), transcript(heard), spots, [side], _heard((text,), (text,))
    )

    assert sides == (side,)
    assert json.loads(str(params["ear_record"]))[0][2] == [aside, aside]
    assert (params["ear_to_reference"], params["ear_to_transcript"]) == (0, 0)


def test_guarded_and_restored_spots_are_never_heard() -> None:
    said, heard = _timed("we saw the cat sat on the mat"), _timed("we saw the hat sat on the bat")
    spots = find_spots(said, heard)
    picked: list[Side] = ["guarded", "unsure"]
    text = ("we saw the cat sat on the bat",)

    windows = clips(transcript(said), transcript(heard), spots, picked)
    sides, params = vote(transcript(said), transcript(heard), spots, picked, _heard(text, text))

    assert windows == [clip(said, heard, spots[1], None)]
    assert sides == ("guarded", "reference")
    assert json.loads(str(params["ear_record"])) == [
        ["guarded", [None, None], [None, None]],
        ["unsure", ["bat", "bat"], ["reference", "reference"]],
    ]
    assert (params["ear_to_reference"], params["ear_to_transcript"]) == (1, 0)


@pytest.mark.parametrize(
    ("said", "heard", "side", "background", "delivered"),
    [
        # The pick took the reference's "Keigo", a name in its background; both ears heard "Kago".
        ("Kago", "Keigo", "reference", "Present: keigo.", "reference"),
        ("Kago", "Keigo", "reference", None, "transcript"),
        # Toward a word of the background is not away from one.
        ("Kago", "Keigo", "reference", "Present: Kago", "transcript"),
        # Unsure keeps the transcript's reading, which the background holds as the pick reads it.
        ("two", "too", "unsure", "The 2 of us", "unsure"),
    ],
)
def test_no_flip_goes_away_from_a_reading_holding_a_word_of_the_background_the_other_lacks(
    said: str, heard: str, side: Side, background: str | None, delivered: Side
) -> None:
    words, other = _timed(f"we met {said} today"), _timed(f"we met {heard} today")
    aside = heard if side != "reference" else said
    text = (f"we met {aside} today",)

    sides, params = vote(
        transcript(words),
        transcript(other),
        find_spots(words, other),
        [side],
        _heard(text, text),
        background=background,
    )

    kept = delivered == side
    assert sides == (delivered,)
    assert json.loads(str(params["ear_record"]))[0][3:] == (["context"] if kept else [])
    flips = (params["ear_to_reference"], params["ear_to_transcript"])
    assert flips == ((0, 0) if kept else (0, 1))
