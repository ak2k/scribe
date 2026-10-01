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


def test_a_spot_far_into_its_clip_gets_its_slot() -> None:
    # "papa", the first word after the spot, starts 8 s into the clip.
    text = "hotel india juliet kilo lima bike remember oscars papa quebec romeo sierra tango"

    found = slot(_timed(_NAMES), Spot(range(12, 15), range(12, 15)), (7.0, 19.5), text)

    assert found == Slot(("bike", "remember", "oscars"), (), ())


def test_with_no_word_matched_beside_the_spot_its_slot_runs_to_the_clip_edge() -> None:
    found = slot(_timed("we saw the cat sat down"), Spot(range(3, 4), range(3, 4)), (0, 9), "zulu")

    assert found == Slot(("zulu",), ("we", "saw", "the"), ("sat", "down"))


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
