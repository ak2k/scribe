"""Rule ear-1: deliver the reading a pick set aside where every local recognizer heard it.

Each spot is heard in a clip of its own, both readings and PAD_S on each side,
by every recognizer `scribe.ear` runs. A recognizer's text has no word times,
so it is aligned to the transcript's words in the clip by edit distance alone.
Its slot at the spot is the words it heard between the last token matched
before the spot and the first matched after, or the clip's edge where none is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Json, TypeAdapter, ValidationError

from scribe.errors import InputValidationError
from scribe.pick import Side, alike, guarded, restored, spot_contexts
from scribe.schema import Word
from scribe.vote import align_words, norm_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from scribe.ear import Heard
    from scribe.pick import Spot
    from scribe.schema import Transcript

RULE = "ear-1"
# Clips are never merged either: the audio around a spot changes what is heard
# in it, and 3 s instead of 5 moved 20 of 164 judged readings.
PAD_S = 5.0
# One recognizer alone flipped 7 or 8 right readings of 164 to wrong ones.
_EARS = 2
# Flipping these would undo what the guard or the restore kept.
_UNHEARD = frozenset({"guarded", "restored"})
_EDGES = re.compile(r"^[\W_]+|[\W_]+$")
# Capitalized only as "I", opening a sentence or in a title, these name nothing.
# Where one could be a name or term instead, the guard stays: a word of two or
# more capitals (US, IT) is never one of them, and a word that is also a common
# given name or month (Will, May, Can) is left out.
_FUNCTION = frozenset(
    {"a", "an", "the", "this", "that", "these", "those", "my", "our", "your", "his", "her"}
    | {"its", "their", "some", "any", "each", "every", "all", "both", "no"}
    | {"i", "me", "we", "us", "you", "he", "him", "she", "it", "they", "them", "there"}
    | {"who", "whom", "whose", "what", "which"}
    | {"and", "or", "but", "nor", "so", "yet", "if", "as", "because", "while", "when", "where"}
    | {"how", "why", "though", "although", "since"}
    | {"at", "by", "for", "from", "in", "of", "on", "to", "with", "about", "after", "before"}
    | {"during", "into", "over", "under", "through", "between", "without"}
    | {"am", "is", "are", "was", "were", "be", "been", "do", "does", "did", "has", "have", "had"}
    | {"would", "should", "could", "must", "shall", "might"}
)
# "We're" and "Don't" are function words too.
_CLITIC = re.compile(r"(?:n['\u2019]t|['\u2019](?:s|m|re|ll|ve|d))$", re.IGNORECASE)

Verdict = Literal["transcript", "reference", "third"]

# Strict: a side or verdict misspelled, or a slot that is not text, is drift.
_RECORD = TypeAdapter(
    Json[
        list[
            tuple[Side, list[str | None], list[Verdict | None]]
            | tuple[Side, list[str | None], list[Verdict | None], Literal["context"]]
        ]
    ],
    config=ConfigDict(strict=True),
)


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
    # Word times can overlap, so neither edge word need bound the spot.
    words = [
        *said[spot.transcript.start : spot.transcript.stop],
        *heard[spot.reference.start : spot.reference.stop],
    ]
    low = min(word.start for word in words)
    high = max(word.end for word in words) + PAD_S
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
    # A word any part of which is in the clip may be heard; left out, it
    # could not anchor, and what was heard of it would fall in the slot.
    mine = [index for index, word in enumerate(said) if word.start <= end and word.end >= start]
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
    tokens = norm_tokens(" ".join(slot.words))

    def names(reading: Sequence[str]) -> bool:
        # Such a reading repeats its context, so a slot of no word, of
        # fillers, or of a word beside it doubled collapses into it too.
        # Fillers count here: "like" alone is such a reading, and with
        # fillers dropped an empty slot would equal it.
        if alike((), reading, *context) and tokens != norm_tokens(" ".join(reading)):
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


def guard_words(background: str | None) -> frozenset[str]:
    """Return the names and terms of `background`, each as written, punctuation aside.

    They are the words the recognizers cannot know, which both engines
    capitalize: its capitalized words, function words aside.
    """
    return frozenset(
        word
        for word in _written(background or "")
        if word[0].isupper()
        and (_CLITIC.sub("", word).lower() not in _FUNCTION or (len(word) > 1 and word.isupper()))
    )


def vote(
    transcript: Transcript,
    reference: Transcript,
    spots: Sequence[Spot],
    picked: Sequence[Side],
    heard: Sequence[Heard],
    *,
    background: str | None = None,
) -> tuple[tuple[Side, ...], dict[str, float | int | bool | str]]:
    """Apply rule ear-1 at `spots`, `heard` holding each recognizer's text of every `clips` clip.

    The recognizers never see `background`, the pick's: no spot is flipped
    away from a reading holding a name or term of it (`guard_words`) that the
    other reading does not hold, each word compared as written, punctuation
    aside.

    Returns:
        The side delivered at each spot, and the ear_* engine params: the
        rule, the recognizers, what they ran on, the pad, their seconds, the
        flips each way, and ear_record, each spot as [the pick's side, each
        recognizer's slot or null, each one's verdict or null], then
        "context" where the background kept the pick's side.

    """
    said, other = transcript.words, reference.words
    contexts = spot_contexts(said, other)
    named = guard_words(background)
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
        delivered = deliver(side, verdicts, guard=guard, restore=restore)
        heard_there = [None if found is None else " ".join(found.words) for found in slots]
        row: list[object] = [side, heard_there, verdicts]
        kept, aside = (theirs, mine) if side == "reference" else (mine, theirs)
        if delivered != side and _holds(kept, aside, named):
            delivered = side
            row.append("context")
        sides.append(delivered)
        record.append(row)
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


def own_sides(transcript: Transcript, path: Path) -> list[Side] | None:
    """Return the side the pick gave each spot, from `transcript`'s ear_record, read from `path`.

    Its pick_record holds the side delivered, which the ears may have
    flipped. None when the ears did not run.
    """
    recorded = transcript.engine.params.get("ear_record")
    if recorded is None:
        return None
    try:
        rows = _RECORD.validate_python(recorded)
    except ValidationError as exc:
        first = exc.errors()[0]
        row = f" row {first['loc'][0]}" if first["loc"] else ""
        raise InputValidationError(
            f"{path} has a malformed ear_record{row}: {first['msg']}"
        ) from exc
    return [row[0] for row in rows]


def _holds(reading: Sequence[str], other: Sequence[str], words: frozenset[str]) -> bool:
    """Whether `reading` holds one of `words` that `other` lacks, each as written."""
    return not words.isdisjoint(_written(" ".join(reading)) - _written(" ".join(other)))


def _written(text: str) -> set[str]:
    """Return the words of `text` as written, punctuation at their edges aside."""
    return {bare for word in text.split() if (bare := _EDGES.sub("", word))}
