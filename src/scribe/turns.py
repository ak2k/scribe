"""Group diarized words into speaker turns.

Canonical shape for a stage: a pure function over the `Transcript` pieces it
needs, with the tuning knobs as keyword arguments so the CLI can expose them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING

from scribe.schema import Turn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scribe.schema import Word


@dataclass(frozen=True)
class _RawTurn:
    """A run of words still carrying the API's diarization id, not a label."""

    speaker_id: int | None
    words: tuple[Word, ...]

    # Overlapping speech leaves words out of end order, and a snap can hand a
    # turn a word that ends before one it already holds: the span covers them all.
    @property
    def start(self) -> float:
        return min(word.start for word in self.words)

    @property
    def end(self) -> float:
        return max(word.end for word in self.words)


def _group_by_speaker(words: Sequence[Word]) -> list[_RawTurn]:
    return [
        _RawTurn(speaker_id, tuple(run))
        for speaker_id, run in groupby(words, key=lambda word: word.speaker)
    ]


def _is_micro(turn: _RawTurn, min_seconds: float, min_words: int) -> bool:
    return (turn.end - turn.start) < min_seconds and len(turn.words) < min_words


_SENTENCE_ENDS = (".", "?", "!")


def _ends_sentence(word: Word) -> bool:
    return word.text.rstrip().endswith(_SENTENCE_ENDS)


def _merge_adjacent(turns: list[_RawTurn]) -> list[_RawTurn]:
    return [
        _RawTurn(speaker_id, tuple(word for run_turn in run for word in run_turn.words))
        for speaker_id, run in groupby(turns, key=lambda turn: turn.speaker_id)
    ]


def _merge_flickers_once(
    turns: list[_RawTurn], min_seconds: float, min_words: int
) -> list[_RawTurn]:
    """One left-to-right pass folding each flicker into the speaker around it.

    A flicker is a micro-turn between two turns of one speaker, where the turn
    before it stops mid-sentence: a diarization slip inside one person's
    sentence. After a sentence end the same micro-turn is a backchannel and
    keeps its own speaker.
    """
    kept: list[_RawTurn] = []
    index = 0
    while index < len(turns):
        current = turns[index]
        if (
            kept
            and index + 1 < len(turns)
            and _is_micro(current, min_seconds, min_words)
            and kept[-1].speaker_id == turns[index + 1].speaker_id
            and not _ends_sentence(kept[-1].words[-1])
        ):
            # The merged turn replaces the previous one in place, so the next
            # micro-turn is measured against the merged result, not the original.
            previous = kept.pop()
            following = turns[index + 1]
            kept.append(
                _RawTurn(previous.speaker_id, previous.words + current.words + following.words)
            )
            index += 2
        else:
            kept.append(current)
            index += 1
    return kept


def _merge_flickers(turns: list[_RawTurn], min_seconds: float, min_words: int) -> list[_RawTurn]:
    while True:
        passed = _merge_adjacent(_merge_flickers_once(turns, min_seconds, min_words))
        if passed == turns:
            return turns
        turns = passed


def _snap_switches(
    words: Sequence[Word], speakers: list[int | None], max_words: int
) -> list[int | None]:
    """Move each mid-sentence speaker switch to the nearest sentence end.

    The search tries distance 1..`max_words`, the left end before the right one
    at each distance. A move relabels only the words it crosses and applies only
    when those words hold no other switch; a switch at the crossed span's own
    edge does not block it, so a short run can be absorbed whole. The first and
    last words never change speaker.
    """
    snapped = list(speakers)
    ends = {index for index, word in enumerate(words) if _ends_sentence(word)}
    switch = 1
    while switch < len(snapped):
        before, after = snapped[switch - 1], snapped[switch]
        if before != after and switch - 1 not in ends:
            for distance in range(1, max_words + 1):
                left = switch - 1 - distance
                if (
                    left >= 0
                    and left in ends
                    and all(speaker == before for speaker in snapped[left + 1 : switch])
                ):
                    snapped[left + 1 : switch] = [after] * (switch - 1 - left)
                    break
                right = switch - 1 + distance
                if (
                    right < len(snapped) - 1
                    and right in ends
                    and all(speaker == after for speaker in snapped[switch : right + 1])
                ):
                    snapped[switch : right + 1] = [before] * (right + 1 - switch)
                    # The switch now follows `right`, a sentence end: nothing
                    # to snap there.
                    switch = right + 1
                    break
        switch += 1
    return snapped


UNATTRIBUTED = "Speaker ?"


def _labels(turns: Sequence[_RawTurn]) -> dict[int | None, str]:
    """Rank surviving speakers by first appearance.

    The API's diarization integers are not documented as 0- or 1-based and are
    not ordered by appearance, so the rank is the only stable display label.
    Where any word has a speaker, the words without one are UNATTRIBUTED and
    unranked: no engine said who spoke them, and a rank would pass them off
    as one more person.
    """
    order = dict.fromkeys(turn.speaker_id for turn in turns)
    if list(order) == [None]:
        return {None: "Speaker 1"}
    ranked = (speaker_id for speaker_id in order if speaker_id is not None)
    labels: dict[int | None, str] = {None: UNATTRIBUTED}
    labels.update(
        {speaker_id: f"Speaker {rank}" for rank, speaker_id in enumerate(ranked, start=1)}
    )
    return labels


# The CLI exposes these as option defaults, so they live beside `build_turns`
# rather than as a second literal in cli.py free to drift from this one.
DEFAULT_MIN_TURN_SECONDS = 1.2
DEFAULT_MIN_TURN_WORDS = 3
# ASR tends to place a speaker switch a few words off the sentence end it belongs to.
DEFAULT_SNAP_WORDS = 5


def word_speakers(
    words: Sequence[Word],
    *,
    min_turn_seconds: float = DEFAULT_MIN_TURN_SECONDS,
    min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
    snap_words: int = DEFAULT_SNAP_WORDS,
) -> list[int | None]:
    """Settle one speaker id per word: flickers merged, then switches snapped.

    Returns the diarization ids `build_turns` groups into turns, one per word
    in `words`; the arguments are `build_turns`'s. Where any word has a
    speaker, only those words are settled, as if the others were absent, and
    the others stay None.
    """
    # A word no engine diarized would otherwise join a speaker as a flicker,
    # or split one speaker's run so that a real flicker escapes its merge.
    attributed = [index for index, word in enumerate(words) if word.speaker is not None]
    attributed = attributed or list(range(len(words)))
    kept = [words[index] for index in attributed]
    merged = _merge_flickers(_group_by_speaker(kept), min_turn_seconds, min_turn_words)
    settled = [turn.speaker_id for turn in merged for _ in turn.words]
    settled = _snap_switches(kept, settled, snap_words)
    speakers: list[int | None] = [None] * len(words)
    for index, speaker in zip(attributed, settled, strict=True):
        speakers[index] = speaker
    return speakers


def turns_from_speakers(words: Sequence[Word], speakers: Sequence[int | None]) -> list[Turn]:
    """Group words into labeled turns by the speaker id given for each word.

    Unlike `build_turns` this merges and snaps nothing: every switch in
    `speakers` becomes a turn boundary.

    Raises:
        ValueError: `speakers` does not hold exactly one id per word.

    """
    raw = [
        _RawTurn(speaker_id, tuple(word for word, _ in run))
        for speaker_id, run in groupby(zip(words, speakers, strict=True), key=lambda pair: pair[1])
    ]
    labels = _labels(raw)
    return [
        Turn(
            speaker=labels[turn.speaker_id],
            start=turn.start,
            end=turn.end,
            text=" ".join(word.text for word in turn.words),
        )
        for turn in raw
    ]


def build_turns(
    words: Sequence[Word],
    *,
    min_turn_seconds: float = DEFAULT_MIN_TURN_SECONDS,
    min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
    snap_words: int = DEFAULT_SNAP_WORDS,
) -> list[Turn]:
    """Group consecutive same-speaker words into labeled turns.

    Words carrying `speaker=None` all belong to one speaker when no word has
    one (an undiarized transcript); otherwise they are UNATTRIBUTED, and take
    no part in the merge or the snap below. A turn shorter than
    `min_turn_seconds` AND under `min_turn_words` is a micro-turn. A
    micro-turn merges only as a flicker: between two turns of one speaker, the
    first of them stopping mid-sentence, it joins both under that speaker,
    repeated until nothing changes. Any other micro-turn (a backchannel, an
    opening or closing word) keeps its own speaker, so a speaker who only
    interjects still earns a label.

    Then each speaker switch that falls mid-sentence moves to the nearest
    sentence end within `snap_words` words, when the words it crosses hold no
    other switch (see `_snap_switches`). The snap can absorb a short run whole,
    so an interjector can still lose its label there. Turns are the maximal
    same-speaker runs after the snap.

    Args:
        words: Words in transcript order.
        min_turn_seconds: Duration below which a turn may be merged.
        min_turn_words: Word count below which a turn may be merged.
        snap_words: How far a switch may move to reach a sentence end; 0 is off.

    Returns:
        Turns in transcript order, no two adjacent ones sharing a speaker,
        labeled "Speaker 1", "Speaker 2", ... by order of first appearance.

    """
    speakers = word_speakers(
        words,
        min_turn_seconds=min_turn_seconds,
        min_turn_words=min_turn_words,
        snap_words=snap_words,
    )
    return turns_from_speakers(words, speakers)
