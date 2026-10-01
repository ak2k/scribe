"""Rule ear-1: deliver the reading a pick set aside where every local recognizer heard it.

Each spot is heard in a clip of its own, both readings and PAD_S on each side,
by every recognizer `scribe.ear` runs. A recognizer's text has no word times,
so it is aligned to the transcript's words in the clip by edit distance alone.
Its slot at the spot is the words it heard between the last token matched
before the spot and the first matched after, or the clip's edge where none is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from scribe.pick import alike, guarded, normalized, restored, spot_contexts
from scribe.schema import Word
from scribe.vote import align_words, norm_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scribe.ear import Heard
    from scribe.pick import Side, Spot
    from scribe.schema import Transcript

RULE = "ear-1"
# Clips are never merged either: the audio around a spot changes what is heard
# in it, and 3 s instead of 5 moved 20 of 164 judged readings.
PAD_S = 5.0
# One recognizer alone flipped 7 or 8 right readings of 164 to wrong ones.
_EARS = 2
# Flipping these would undo what the guard or the restore kept.
_UNHEARD = frozenset({"guarded", "restored"})

Verdict = Literal["transcript", "reference", "third"]


@dataclass(frozen=True)
class Slot:
    """What a recognizer heard at a spot, and the transcript's unmatched words beside it."""

    words: tuple[str, ...]
    left: tuple[str, ...]
    right: tuple[str, ...]


def clip(
    said: Sequence[Word], heard: Sequence[Word], spot: Spot, duration: float | None
) -> tuple[float, float]:
    """Return the seconds a spot is heard in: both its readings, PAD_S each side, in the audio."""
    low = min(said[spot.transcript.start].start, heard[spot.reference.start].start)
    high = max(said[spot.transcript.stop - 1].end, heard[spot.reference.stop - 1].end) + PAD_S
    # Unbounded, a clip still ends where the worker's audio does.
    return max(0.0, low - PAD_S), high if duration is None else min(duration, high)


def clips(
    transcript: Transcript, reference: Transcript, spots: Sequence[Spot], picked: Sequence[Side]
) -> list[tuple[float, float]]:
    """Return the clip of each spot heard, in order: every one neither guarded nor restored."""
    return [
        clip(transcript.words, reference.words, spot, transcript.duration)
        for spot, side in zip(spots, picked, strict=True)
        if side not in _UNHEARD
    ]


def slot(said: Sequence[Word], spot: Spot, window: tuple[float, float], text: str) -> Slot | None:
    """Return what `text`, a recognizer's of the clip `window`, heard at `spot`; None if empty."""
    if not norm_tokens(text):
        return None
    start, end = window
    mine = [index for index, word in enumerate(said) if start <= (word.start + word.end) / 2 <= end]
    words = text.split()
    # Timed alike, every pair is near: the band and the tolerance never bind.
    steps = align_words(
        [Word(text=said[index].text, start=start, end=end) for index in mine],
        [Word(text=word, start=start, end=end) for word in words],
    )
    matched = [
        (step, mine[word])
        for step, (kind, word, _) in enumerate(steps)
        if kind == "match" and word is not None
    ]
    own = spot.transcript
    first, low = next(((n, index) for n, index in reversed(matched) if index < own.start), (-1, -1))
    last, high = next(
        ((n, index) for n, index in matched if index >= own.stop), (len(steps), len(said))
    )
    held = dict.fromkeys(theirs for _, _, theirs in steps[first + 1 : last] if theirs is not None)
    return Slot(
        tuple(words[index] for index in held),
        tuple(said[index].text for index in mine if low < index < own.start),
        tuple(said[index].text for index in mine if own.stop <= index < high),
    )


def verdict(
    slot: Slot, said: Sequence[str], heard: Sequence[str], context: tuple[list[str], list[str]]
) -> Verdict:
    """Name the reading `slot` is, the transcript's (`said`) or the reference's; "third" if neither.

    A reading is tried with and without the transcript's unmatched words
    beside it, which the recognizer may have kept, changed or dropped. A
    reading that compares alike to no words at all in its context is named
    only by a slot that is it token for token. A slot that is both readings
    names neither.
    """
    frames = ((slot.left, slot.right), ((), slot.right), (slot.left, ()), ((), ()))
    tokens = normalized(" ".join(slot.words))

    def names(reading: Sequence[str]) -> bool:
        # Such a reading repeats its context, so a slot of no word, of
        # fillers, or of a word beside it doubled collapses into it too.
        if alike((), reading, *context) and tokens != normalized(" ".join(reading)):
            return False
        return any(alike(slot.words, [*left, *reading, *right], *context) for left, right in frames)

    is_said, is_heard = names(said), names(heard)
    if is_said == is_heard:
        return "third"
    return "transcript" if is_said else "reference"


def deliver(side: Side, verdicts: Sequence[Verdict | None], *, guard: bool, restore: bool) -> Side:
    """Return the side delivered at a spot the pick gave `side`, `verdicts` one per recognizer.

    It is the reading the pick set aside, the transcript's where it picked the
    reference's and the reference's otherwise, where at least _EARS
    recognizers heard the spot and each named that reading; save where putting
    the reference's in would trip the guard (`guard`), or keeping the
    transcript's the restore (`restore`).
    """
    aside: Verdict = "transcript" if side == "reference" else "reference"
    if side in _UNHEARD or len(verdicts) < _EARS or any(each != aside for each in verdicts):
        return side
    return side if (guard if aside == "reference" else restore) else aside


def vote(
    transcript: Transcript,
    reference: Transcript,
    spots: Sequence[Spot],
    picked: Sequence[Side],
    heard: Sequence[Heard],
) -> tuple[tuple[Side, ...], dict[str, float | int | bool | str]]:
    """Apply rule ear-1 at `spots`, `heard` holding each recognizer's text of every `clips` clip.

    Returns:
        The side delivered at each spot, and the ear_* engine params: the
        rule, the recognizers, what they ran on, the pad, their seconds, the
        flips each way, and ear_record, each spot as [the pick's side, each
        recognizer's slot or null, each one's verdict or null].

    """
    said, other = transcript.words, reference.words
    contexts = spot_contexts(said, other)
    # Each recognizer's texts are of the spots heard, in order.
    at = 0
    sides: list[Side] = []
    record: list[list[object]] = []
    for spot, side in zip(spots, picked, strict=True):
        slots: list[Slot | None] = [None] * len(heard)
        if side not in _UNHEARD:
            window = clip(said, other, spot, transcript.duration)
            slots = [slot(said, spot, window, ear.texts[at]) for ear in heard]
            at += 1
        mine = [word.text for word in said[spot.transcript.start : spot.transcript.stop]]
        theirs = [word.text for word in other[spot.reference.start : spot.reference.stop]]
        verdicts: list[Verdict | None] = [
            None if found is None else verdict(found, mine, theirs, contexts[spot])
            for found in slots
        ]
        guard, restore = guarded(said, other, spot), restored(said, other, spot)
        sides.append(deliver(side, verdicts, guard=guard, restore=restore))
        heard_there = [None if found is None else " ".join(found.words) for found in slots]
        record.append([side, heard_there, verdicts])
    flips = [now for was, now in zip(picked, sides, strict=True) if now != was]
    runtimes = (
        f"transformers {ear.versions.get('transformers')} torch {ear.versions.get('torch')} "
        f"{ear.device} {ear.dtype}"
        for ear in heard
    )
    models = [
        f"{ear.recognizer.name} {ear.recognizer.model}@{ear.recognizer.revision}" for ear in heard
    ]
    return tuple(sides), {
        "ear_rule": RULE,
        "ear_models": json.dumps(models),
        "ear_runtime": "; ".join(dict.fromkeys(runtimes)),
        "ear_pad_s": PAD_S,
        "ear_seconds": round(sum(ear.runtime_s for ear in heard), 1),
        "ear_to_reference": flips.count("reference"),
        "ear_to_transcript": flips.count("transcript"),
        "ear_record": json.dumps(record),
    }
