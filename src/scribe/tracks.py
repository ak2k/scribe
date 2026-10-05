"""Merge a call's two recorded sides, the operator's mic and the call app, into one transcript.

Each side is transcribed on its own first. The merge keeps every app word,
gives every mic word one speaker of its own, and drops the mic words that are
the far side leaking into the mic (rule bleed-1). Nothing is retimed, and the
offset between the sides is the one given, never fitted from the words.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, groupby
from typing import TYPE_CHECKING

from scribe.curated import fill_ranges
from scribe.schema import Engine, Source, Track, Transcript, Word
from scribe.vote import word_key

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

BLEED_RULE = "bleed-1"
# Leaked speech reaches the mic within tens of ms of the app; a repeat-back
# starts only after the word it repeats has ended.
BLEED_WINDOW_S = 0.25
# Two stems of one call differ in length by their start and stop skew, not by seconds.
MAX_SKEW_S = 2.0
DEFAULT_ME = "Me"
# A merged transcript's track indexes, in the order `tracks` lists them.
MIC, APP = 0, 1
# Each track's fill ranges, apart, so a recovered-speech note marks only that track's turns.
MIC_FILL, APP_FILL = "merge_mic_fill_ranges", "merge_app_fill_ranges"
_UNRESOLVED = "fill_unresolved_ranges"


@dataclass(frozen=True)
class TrackFile:
    """One side's transcript, the file it was read from, and that file's sha256."""

    transcript: Transcript
    path: Path
    sha256: str


@dataclass(frozen=True)
class Dropped:
    """A mic word dropped as a bleed copy, and the length of the dropped run holding it."""

    word: Word
    run: int


@dataclass(frozen=True)
class Merged:
    """The merged transcript, the offset applied, and the mic words dropped."""

    transcript: Transcript
    offset: float
    dropped: tuple[Dropped, ...]


def _sorted(words: Sequence[Word]) -> tuple[list[Word], int]:
    """Words stably sorted by start, and how many changed place."""
    order = sorted(range(len(words)), key=lambda index: words[index].start)
    return [words[index] for index in order], sum(at != index for at, index in enumerate(order))


def _pairs(
    mic: Sequence[Word], app: Sequence[Word], *, shift: float, window: float
) -> dict[int, int]:
    """Pair each keyed mic word, in start order, with the earliest free app word of its key.

    An app word qualifies when it starts within `window` of the mic word's
    start less `shift`. Both lists must be in start order.
    """
    by_key: defaultdict[str, list[int]] = defaultdict(list)
    for index, word in enumerate(app):
        by_key[word_key(word.text)].append(index)
    taken: set[int] = set()
    pairs: dict[int, int] = {}
    for at, word in enumerate(mic):
        key = word_key(word.text)
        if not key:
            continue
        candidates = by_key[key]
        target = word.start - shift
        # From well below the window: the exact test below decides.
        first = bisect.bisect_left(candidates, target - 2 * window, key=lambda i: app[i].start)
        for index in candidates[first:]:
            if app[index].start - target > window:
                break
            if index not in taken and abs(app[index].start - target) <= window:
                taken.add(index)
                pairs[at] = index
                break
    return pairs


def _runs(dropped: Sequence[int]) -> list[int]:
    """The length of the run of consecutive indexes each dropped index sits in."""
    lengths: list[int] = []
    for _, run in groupby(enumerate(dropped), key=lambda pair: pair[1] - pair[0]):
        held = len(list(run))
        lengths += [held] * held
    return lengths


def count_runs(dropped: Sequence[Dropped]) -> tuple[int, int, int]:
    """How many dropped words sit in runs of 1, of 2, and of 3 or more."""
    lengths = [drop.run for drop in dropped]
    alone, paired = lengths.count(1), lengths.count(2)
    return alone, paired, len(lengths) - alone - paired


def _union(mic: TrackFile, app: TrackFile, key: str) -> str | None:
    """Both sides' recorded spans under `key`, time-sorted; None where neither recorded any."""
    sides = (mic, app)
    if all(key not in side.transcript.engine.params for side in sides):
        return None
    spans = sorted(
        span for side in sides for span in fill_ranges(side.transcript.engine, side.path, key=key)
    )
    return json.dumps([list(span) for span in spans])


def _recovered(side: TrackFile, kept: Sequence[Word]) -> str | None:
    """The side's fill ranges still holding one of its words once merged; None where it has none."""
    if "fill_ranges" not in side.transcript.engine.params:
        return None
    held = [
        span
        for span in fill_ranges(side.transcript.engine, side.path)
        if any(span[0] <= word.start and word.end <= span[1] for word in kept)
    ]
    return json.dumps([list(span) for span in held])


def merge_tracks(
    mic: TrackFile, app: TrackFile, *, me: str = DEFAULT_ME, offset: float = 0.0
) -> Merged:
    """Merge the two sides of a call, dropping the mic's copies of the app's words.

    Each side's words are first sorted by start. A mic word is a bleed copy
    when an app word of the same key, not already answering for another,
    starts within BLEED_WINDOW_S of it once `offset`, how much later the mic
    runs than the app, is taken off.

    Raises:
        InputValidationError: a side's fill record is malformed.

    """
    mic_words, mic_moved = _sorted(mic.transcript.words)
    app_words, app_moved = _sorted(app.transcript.words)
    copies = sorted(_pairs(mic_words, app_words, shift=offset, window=BLEED_WINDOW_S))
    dropped = tuple(
        Dropped(mic_words[at], run) for at, run in zip(copies, _runs(copies), strict=True)
    )
    # One id for the whole mic side, held by no app word.
    me_id = max((word.speaker for word in app_words if word.speaker is not None), default=-1) + 1
    copied = set(copies)
    kept = [
        word.model_copy(update={"speaker": me_id, "track": MIC})
        for at, word in enumerate(mic_words)
        if at not in copied
    ]
    placed = [word.model_copy(update={"track": APP}) for word in app_words]
    # Stable, app words first: on an equal start the app word comes first.
    words = sorted(chain(placed, kept), key=lambda word: word.start)
    alone, paired, longer = count_runs(dropped)
    params: dict[str, float | int | bool | str] = {
        "merge_rule": BLEED_RULE,
        "merge_window_s": BLEED_WINDOW_S,
        "merge_offset_s": offset,
        "merge_mic_words": len(mic_words),
        "merge_app_words": len(app_words),
        "merge_mic_moved": mic_moved,
        "merge_app_moved": app_moved,
        "merge_dropped_run_1": alone,
        "merge_dropped_run_2": paired,
        "merge_dropped_run_3plus": longer,
        # A string, as engine params hold no lists.
        "merge_dropped": json.dumps(
            [[d.word.start, d.word.end, d.word.text, d.run] for d in dropped]
        ),
        "merge_mic_sha256": mic.sha256,
        "merge_app_sha256": app.sha256,
    }
    for key, side, held in ((MIC_FILL, mic, kept), (APP_FILL, app, placed)):
        if (recovered := _recovered(side, held)) is not None:
            params[key] = recovered
    if (unresolved := _union(mic, app, _UNRESOLVED)) is not None:
        params[_UNRESOLVED] = unresolved
    durations = [
        side.transcript.duration for side in (mic, app) if side.transcript.duration is not None
    ]
    transcript = Transcript(
        source=Source(
            kind="other", ref=f"{mic.transcript.source.ref} + {app.transcript.source.ref}"
        ),
        engine=Engine(
            name=app.transcript.engine.name, model=app.transcript.engine.model, params=params
        ),
        language=app.transcript.language,
        duration=max(durations, default=None),
        text=" ".join(word.text for word in words),
        words=words,
        tracks=[
            Track(role="mic", label=me, source=mic.transcript.source, transcript_sha256=mic.sha256),
            Track(role="app", source=app.transcript.source, transcript_sha256=app.sha256),
        ],
    )
    return Merged(transcript, offset, dropped)
