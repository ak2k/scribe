"""Fill the holes in a transcript's words with the words a reference heard there.

An engine can drop a passage of speech and return nothing for it, while a
second engine run over the same audio hears it. Reference words go in only
inside a hole, and only where enough of them are missing from the transcript
nearby: a word or two, or words the transcript has at a slightly different
time, are the engines disagreeing, not a dropped passage. Every word the
transcript has is kept, in order, with its text and speaker. So are its
times, except where the engine slid words into a dropped passage's time:
there the words the reference heard well away move to where it heard them
first, which opens the hole they covered.
"""

from __future__ import annotations

import bisect
import difflib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from scribe.gaps import MIN_DROP_SECONDS, MIN_DROP_WORDS, clock, find_holes
from scribe.schema import Word
from scribe.vote import first_decrease, is_filler, norm_tokens, word_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scribe.gaps import Hole
    from scribe.schema import Transcript

# These four and gaps.MIN_DROP_SECONDS and MIN_DROP_WORDS were measured together
# on 5 transcripts (6.2 h).
# A reference word this near a hole's edge is a word bounding it, timed differently.
GUARD_S = 0.25
# The engines can time one word this far apart, so a word the transcript has
# within this reach of a hole is not missing from it.
MATCH_S = 8.0
# Short enough to point at a passage to listen to, long enough to hold one.
WINDOW_S = 30.0
# Fewer unmatched words in a window are within the engines' ordinary disagreement.
MIN_UNRESOLVED = 20

# Half a window, so a passage of up to half a window lies whole in one window wherever it falls.
STEP_S = WINDOW_S / 2

# The engines time a word they share 0.07 s apart at the median, so one this
# far from where the reference heard it is mistimed.
DRIFT_S = 2.0
# Fewer mistimed words in a row can be common words the alignment paired by chance.
MIN_DRIFT_WORDS = 3
# A lone shared token lines up with any common word; two in a row place a word.
MIN_SHARED_TOKENS = 2


@dataclass(frozen=True)
class Span:
    """A stretch of the recording in seconds, and how many reference words it holds."""

    start: float
    end: float
    words: int


@dataclass(frozen=True)
class Retime:
    """A run of transcript words moved to where the reference heard them.

    Its span before and after the move, in seconds; how many of its words
    moved; and how many of their starts the words around the run held back,
    by at most `clamp_s` seconds.
    """

    start: float
    end: float
    new_start: float
    new_end: float
    words: int
    clamped: int
    clamp_s: float


@dataclass(frozen=True)
class Fill:
    """The spans a fill inserted words in, those it could not fill, and the runs it moved first."""

    filled: tuple[Span, ...]
    unresolved: tuple[Span, ...]
    retimed: tuple[Retime, ...]

    @property
    def words(self) -> int:
        """How many words were inserted."""
        return sum(span.words for span in self.filled)

    @property
    def retimed_words(self) -> int:
        """How many transcript words were moved."""
        return sum(run.words for run in self.retimed)


def describe(fill: Fill) -> str:
    """Summarize `fill` in one phrase, for a progress line."""
    spans, words = len(fill.filled), fill.words
    return (
        f"{spans} {'span' if spans == 1 else 'spans'} filled with {words} "
        f"{'word' if words == 1 else 'words'} from Parakeet, {len(fill.unresolved)} unresolved"
    )


def possible_drop(start: float, end: float) -> str:
    """Point at a span to listen to, for a progress line."""
    return f"possible dropped speech {clock(start)}-{clock(end)}; listen to that span"


def moved(run: Retime) -> str:
    """Say where `run`'s words went, for a progress line."""
    words = f"{run.words} {'word' if run.words == 1 else 'words'}"
    return (
        f"re-timed {clock(run.start)}-{clock(run.end)} to "
        f"{clock(run.new_start)}-{clock(run.new_end)} ({words} to where Parakeet heard them)"
    )


class _Timeline:
    """Words in start order, sliced by start."""

    def __init__(self, words: Sequence[Word]) -> None:
        self.words = sorted(words, key=lambda word: word.start)
        self._starts = [word.start for word in self.words]

    def positions(self, low: float, high: float, *, closed: bool = True) -> range:
        """Return where the words starting from `low` to `high` are, `high` included if `closed`."""
        cut = bisect.bisect_right if closed else bisect.bisect_left
        return range(bisect.bisect_left(self._starts, low), cut(self._starts, high))

    def between(self, low: float, high: float, *, closed: bool = True) -> list[Word]:
        """Return the words starting from `low` to `high`, `high` included if `closed`."""
        where = self.positions(low, high, closed=closed)
        return self.words[where.start : where.stop]


def _unmatched(heard: Sequence[Word], near: Sequence[Word]) -> list[int]:
    """Walk `heard` in order, each word using up one copy of its key in `near`.

    Returns:
        The indices of the words with no copy left. A filler, or a word with
        no letters or digits, is never among them.

    """
    copies = Counter(word_key(word.text) for word in near)
    missing: list[int] = []
    for index, word in enumerate(heard):
        key = word_key(word.text)
        if is_filler(key):
            continue
        if copies[key]:
            copies[key] -= 1
        else:
            missing.append(index)
    return missing


def _unresolved(heard: _Timeline, words: Sequence[Word], audio_end: float) -> tuple[Span, ...]:
    own = _Timeline(words)
    # Each span's start and end, and where in `heard` its unmatched words are.
    flagged: list[tuple[float, float, set[int]]] = []
    window = 0
    while (start := window * STEP_S) < audio_end:
        where = heard.positions(start, start + WINDOW_S, closed=False)
        missing = _unmatched(
            heard.words[where.start : where.stop],
            own.between(start - MATCH_S, start + WINDOW_S + MATCH_S, closed=False),
        )
        if len(missing) >= MIN_UNRESOLVED:
            end = min(start + WINDOW_S, audio_end)
            found = {where[index] for index in missing}
            if flagged and start <= flagged[-1][1]:
                flagged[-1] = (flagged[-1][0], end, flagged[-1][2] | found)
            else:
                flagged.append((start, end, found))
        window += 1
    # A set, so a word unmatched in two windows of a span counts once.
    return tuple(Span(start, end, len(found)) for start, end, found in flagged)


# For each paired transcript word, the reference words its first and last matched tokens are in.
type _Pairs = dict[int, tuple[int, int]]


def _tokens(words: Sequence[Word]) -> tuple[list[str], list[int]]:
    """Return `words`' tokens, and the index of the word each came from."""
    tokens: list[str] = []
    owners: list[int] = []
    for index, word in enumerate(words):
        for token in norm_tokens(word.text):
            tokens.append(token)
            owners.append(index)
    return tokens, owners


def _pairs(words: Sequence[Word], heard: Sequence[Word]) -> _Pairs:
    """Align the two lists' tokens and pair each transcript word the alignment places.

    Returns:
        For each word whose first token lies in a matching stretch of at
        least MIN_SHARED_TOKENS tokens, the index in `heard` of the word that
        token matched, and of the word its last such token matched.

    """
    own, owners = _tokens(words)
    ref, ref_owners = _tokens(heard)
    matched = {
        block.a + offset: ref_owners[block.b + offset]
        for block in difflib.SequenceMatcher(None, own, ref, autojunk=False).get_matching_blocks()
        if block.size >= MIN_SHARED_TOKENS
        for offset in range(block.size)
    }
    first: dict[int, int] = {}
    for position, index in enumerate(owners):
        first.setdefault(index, position)
    pairs: _Pairs = {}
    # In token order, so each word ends up with its last matched token.
    for position, index in enumerate(owners):
        if position in matched and first[index] in matched:
            pairs[index] = (matched[first[index]], matched[position])
    return pairs


def _drifted(words: Sequence[Word], heard: Sequence[Word], pairs: _Pairs) -> list[list[int]]:
    """Return the runs of MIN_DRIFT_WORDS paired words or more, in list order, all mistimed."""

    def drifted(index: int) -> bool | None:
        if index not in pairs:
            return None
        return abs(heard[pairs[index][0]].start - words[index].start) > DRIFT_S

    runs: list[list[int]] = []
    run: list[int] = []
    for index in range(len(words)):
        if (mistimed := drifted(index)) is None:
            continue
        if mistimed:
            run.append(index)
            continue
        if len(run) >= MIN_DRIFT_WORDS:
            runs.append(run)
        run = []
    if len(run) >= MIN_DRIFT_WORDS:
        runs.append(run)
    return runs


def _repeated(run: Sequence[int], own: _Timeline, heard: _Timeline, pairs: _Pairs) -> bool:
    """Tell whether `run` pairs copies of words said more than once nearby.

    It does when more than half its paired words are also said by another
    transcript word within DRIFT_S of the reference word paired with them,
    or when more than half are also heard by the reference nearer to where
    the transcript has them than the reference word paired with them. Each
    test is counted on its own: a run half of whose words meet one and half
    the other still moves. Moved, such a run would sit on another copy, or
    leave the one the reference heard where it was.
    """

    def said(index: int) -> bool:
        at, key = heard.words[pairs[index][0]].start, word_key(own.words[index].text)
        # Not the word itself: a drifted word is more than DRIFT_S from its partner.
        return any(
            word_key(own.words[other].text) == key
            for other in own.positions(at - DRIFT_S, at + DRIFT_S)
        )

    def heard_there(index: int) -> bool:
        word = own.words[index]
        key, reach = word_key(word.text), abs(heard.words[pairs[index][0]].start - word.start)
        near = heard.between(word.start - reach, word.start + reach)
        # A copy nearer than the one the alignment chose explains the word as well.
        return any(
            word_key(other.text) == key and abs(other.start - word.start) < reach for other in near
        )

    return 2 * max(sum(map(said, run)), sum(map(heard_there, run))) > len(run)


def _destinations(
    run: Sequence[int], words: Sequence[Word], heard: Sequence[Word], pairs: _Pairs
) -> dict[int, tuple[float, float]]:
    """Return the start and end each word from `run`'s first to its last moves to.

    A paired word takes the start of the reference word its first token
    matched and the end of the one its last matched token did. The words
    between two paired ones share the stretch between those two starts
    equally, in order.
    """
    times = {index: (heard[pairs[index][0]].start, heard[pairs[index][1]].end) for index in run}
    # Their own times are the mistimed ones, so only their order places them.
    for before, after in pairwise(run):
        share = (times[after][0] - times[before][0]) / (after - before)
        for index in range(before + 1, after):
            start = times[before][0] + share * (index - before)
            times[index] = (start, start + share)
    return times


def _alone(index: int, at: float, own: _Timeline) -> bool:
    """Tell whether no other of `own`'s words says word `index`'s text within MATCH_S of `at`."""
    key = word_key(own.words[index].text)
    return not any(
        other != index and word_key(own.words[other].text) == key
        for other in own.positions(at - MATCH_S, at + MATCH_S)
    )


def _retime(
    words: Sequence[Word], heard: _Timeline
) -> tuple[list[Word], tuple[Retime, ...], dict[int, float]]:
    """Move each mistimed run of `words` to where `heard` heard it.

    Words out of start order are returned as they are: a start held between
    its neighbors' needs neighbors that are in order.

    Returns:
        The words; the runs moved; and for each moved word `_alone` at the
        start of the reference word it was paired with, that start.

    """
    result = list(words)
    sole: dict[int, float] = {}
    if first_decrease(words) is not None:
        return result, (), sole
    # In start order already, so the sort keeps each word at its list index.
    own = _Timeline(words)
    pairs = _pairs(words, heard.words)
    runs: list[Retime] = []
    for run in _drifted(words, heard.words, pairs):
        if _repeated(run, own, heard, pairs):
            continue
        # The words just outside a run stay put, so a start held between
        # theirs leaves every word where it was in start order.
        low = words[run[0] - 1].start if run[0] > 0 else -math.inf
        high = words[run[-1] + 1].start if run[-1] + 1 < len(words) else math.inf
        clamps: list[float] = []
        for index, (start, end) in _destinations(run, words, heard.words, pairs).items():
            held = min(max(start, low), high)
            if held != start:
                clamps.append(abs(held - start))
            result[index] = words[index].model_copy(update={"start": held, "end": max(end, held)})
        for index in run:
            at = heard.words[pairs[index][0]].start
            if result[index] != words[index] and _alone(index, at, own):
                sole[index] = at
        span = range(run[0], run[-1] + 1)
        changed = sum(
            (result[index].start, result[index].end) != (words[index].start, words[index].end)
            for index in span
        )
        if changed:
            runs.append(
                Retime(
                    words[run[0]].start,
                    max(words[index].end for index in span),
                    result[run[0]].start,
                    max(result[index].end for index in span),
                    changed,
                    len(clamps),
                    round(max(clamps, default=0.0), 3),
                )
            )
    return result, tuple(runs), sole


# A moved word before and after its move, and the start from `_retime`'s third result, if any.
type _Move = tuple[Word, Word, float | None]


def _counted(moves: Sequence[_Move], hole: Hole) -> list[Word]:
    """Return the moved words that count as copies in `hole`'s walk.

    A moved word counts where it was, within MATCH_S of the hole: the
    alignment may have paired it with another copy of a word the reference
    heard there. Where it went, it is the reference word it moved to, not a
    copy of any other. One that no other transcript word could stand for
    (`_alone`) is that reference word for certain, so it counts only in a
    hole that word lies in.
    """
    low, high = hole.start - MATCH_S, hole.end + MATCH_S
    counted: list[Word] = []
    for old, now, at in moves:
        if at is None:
            there = low <= old.start <= high
        else:
            there = hole.start + GUARD_S <= at <= hole.end - GUARD_S
        if there:
            counted.append(now)
    return counted


def _uncovered(
    chosen: range,
    missing: set[int],
    interior: Sequence[Word],
    own: _Timeline,
    heard: _Timeline,
    hole: Hole,
) -> list[int]:
    """Keep `chosen`'s unmatched words, and each matched one the transcript lacks nearby.

    A matched word stays while fewer copies of it than the reference's start
    within MATCH_S of `hole`, counting the words kept before it.
    """
    low, high = hole.start - MATCH_S, hole.end + MATCH_S
    have = Counter(word_key(word.text) for word in own.between(low, high))
    spoken = Counter(word_key(word.text) for word in heard.between(low, high))
    kept: list[int] = []
    for index in chosen:
        key = word_key(interior[index].text)
        if index in missing or have[key] < spoken[key]:
            kept.append(index)
            have[key] += 1
    return kept


def fill_holes(transcript: Transcript, reference: Transcript) -> tuple[Transcript, Fill]:
    """Insert `reference`'s words into the holes of `transcript`'s words.

    First, unless the transcript was filled before (its params hold
    `fill_retimed_runs`) and if its words are in start order, the tokens
    (`vote.norm_tokens`) of its words and of the reference's, in start order,
    are aligned, and a word is paired when its first token lies in a matching
    stretch of MIN_SHARED_TOKENS or more. MIN_DRIFT_WORDS paired words or more
    in a row, each more than DRIFT_S from the reference word its first token
    matched, are a run, unless `_repeated` says it pairs copies of words said
    more than once. A run moves as `_destinations` says, each start held
    between the starts of the words just before and after it.

    The holes are `gaps.find_holes`'s of at least MIN_DROP_SECONDS, up to the later
    of the transcript's duration and the reference's last word end. In each,
    the reference words starting at least GUARD_S inside it are walked in
    start order, and each that is not a filler uses up one copy of its text
    among the transcript words starting within MATCH_S of the hole, a moved
    word counted as `_counted` says, or is unmatched. A hole with
    MIN_DROP_WORDS unmatched words or more takes every word from the first
    unmatched to the last, with the reference's text and times and no
    speaker, just before the transcript word whose start closes the hole, in
    list order; the first hole's words go first, the last's last.
    In a hole within MATCH_S of a run's span before or after its move, whose
    walk the move changes, a matched word among them goes in only as
    `_uncovered` says: a move can uncover speech next to words the
    transcript has, timed a little off.

    Then WINDOW_S windows starting every STEP_S from 0 are each walked the
    same way against the filled words starting within MATCH_S of it. A window
    with MIN_UNRESOLVED unmatched words or more is unresolved, and those that
    overlap or touch are joined, a word unmatched in more than one counted once.

    Returns:
        The transcript with the words moved and inserted, its text rebuilt
        from them, no turns, and its engine params recording the fill; and
        the fill.

    """
    heard = _Timeline(reference.words)
    # Words a fill inserted can pair with the reference where nobody spoke,
    # so filled words are never moved again.
    if "fill_retimed_runs" in transcript.engine.params:
        words, retimed, sole = list(transcript.words), (), {}
    else:
        words, retimed, sole = _retime(transcript.words, heard)
    own = _Timeline(words)
    versions = list(zip(transcript.words, words, strict=True))
    still = _Timeline([now for old, now in versions if old == now])
    moves = [(old, now, sole.get(index)) for index, (old, now) in enumerate(versions) if old != now]
    audio_end = max([transcript.duration or 0.0, *(word.end for word in reference.words)])
    reached = [
        (min(run.start, run.new_start) - MATCH_S, max(run.end, run.new_end) + MATCH_S)
        for run in retimed
    ]
    first = min(range(len(words)), key=lambda index: words[index].start, default=None)
    placed: dict[int, list[Word]] = {}
    filled: list[Span] = []
    for hole in find_holes(words, audio_end, MIN_DROP_SECONDS):
        interior = heard.between(hole.start + GUARD_S, hole.end - GUARD_S)
        near = [*still.between(hole.start - MATCH_S, hole.end + MATCH_S), *_counted(moves, hole)]
        missing = _unmatched(interior, near)
        if len(missing) < MIN_DROP_WORDS:
            continue
        chosen = range(missing[0], missing[-1] + 1)
        if any(low <= hole.end and hole.start <= high for low, high in reached):
            chosen = _uncovered(chosen, set(missing), interior, own, heard, hole)
        span = [
            Word(text=interior[index].text, start=interior[index].start, end=interior[index].end)
            for index in chosen
        ]
        # The first hole's words lead even where the word closing it comes later in the list.
        slot = len(words) if hole.before is None else -1 if hole.before == first else hole.before
        placed[slot] = span
        filled.append(Span(span[0].start, max(word.end for word in span), len(span)))
    result = list(placed.get(-1, []))
    for index, word in enumerate(words):
        result += [*placed.get(index, []), word]
    result += placed.get(len(words), [])
    fill = Fill(tuple(filled), _unresolved(heard, result, audio_end), retimed)
    engine = reference.engine
    params = transcript.engine.params | {
        "fill_reference": engine.name if engine.model is None else f"{engine.name} {engine.model}",
        "fill_spans": len(fill.filled),
        "fill_words": fill.words,
        "fill_unresolved": len(fill.unresolved),
        "fill_ranges": json.dumps([[span.start, span.end] for span in fill.filled]),
        # Beside the ranges, not in them, so their readers parse them as before:
        # a range's times cannot tell its words from the word that closed the hole.
        "fill_counts": json.dumps([span.words for span in fill.filled]),
        "fill_unresolved_ranges": json.dumps([[span.start, span.end] for span in fill.unresolved]),
        "fill_retimed_words": fill.retimed_words,
        "fill_retimed_runs": json.dumps([asdict(run) for run in fill.retimed]),
    }
    return transcript.model_copy(
        update={
            "engine": transcript.engine.model_copy(update={"params": params}),
            "text": " ".join(word.text for word in result),
            "words": result,
            "turns": [],
        }
    ), fill
