"""Vote three transcripts of one recording into one, word by word (ROVER).

The backbone's tokens are the slots. The primary and secondary hypotheses are
each aligned to the backbone, and a slot changes only when both pair it with
the same token, different from the backbone's; a gap between slots gains only
tokens both place there. Nothing is deleted: every backbone word survives, in
order.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Literal

from scribe.gaps import MIN_DROP_SECONDS, MIN_DROP_WORDS, find_holes
from scribe.schema import Engine, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scribe.gaps import Hole

# Substitutions and insertions; the deletion rule is left out because it
# removed words the audio holds.
VARIANT = "B"
# Two tokens pair only when their spans lie this close: at 0.5 s nearly every
# identical token the engines share does.
TOLERANCE_S = 0.5
# A backbone token is aligned only against hypothesis tokens starting this
# near it, which keeps the alignment linear in the transcript's length.
BAND_S = 6.0
# An agreed insertion this near an unpaired backbone token of the same text is
# that token mistimed, not a word the backbone missed.
_DUPLICATE_S = 1.0
FILLERS = frozenset(
    {"uh", "um", "er", "ah", "eh", "hmm", "hm", "mm", "mhm", "uhm", "erm", "mmm", "huh"}
    | {"uhhuh", "mmhmm"}
)
# Unordered: no phrase begins another, so at most one can match at a position.
_BACKCHANNELS = frozenset(
    {("yeah",), ("okay",), ("ok",), ("right",), ("mm", "hm"), ("uh", "huh"), ("yep",)}
    | {("sure",), ("cool",), ("got", "it"), ("i", "see"), ("exactly",), ("totally",)}
)
_DASHES = re.compile(r"[-\u2013\u2014/]")
_NOT_TOKEN = re.compile(r"[^\w\s']")
_OUTER_LEFT = re.compile(r"^[^\w']+")
_OUTER_RIGHT = re.compile(r"[^\w']+$")

_Kind = Literal["match", "sub", "del", "ins"]


def norm_tokens(text: str) -> list[str]:
    """Split text into the lowercase, punctuation-free tokens the vote compares."""
    folded = text.lower().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    folded = _NOT_TOKEN.sub("", _DASHES.sub(" ", folded))
    return [token.strip("'") for token in folded.split() if token.strip("'")]


def word_key(text: str) -> str:
    """Return a word's tokens joined: the key words are matched and judged fillers by."""
    return "".join(norm_tokens(text))


def is_filler(key: str) -> bool:
    """Tell whether a word key is a filler; so is an empty one, from no letter or digit."""
    return not key or key in FILLERS


def first_decrease(words: Sequence[Word]) -> int | None:
    """Return the index of the first word starting before the word ahead of it, if any."""
    return next(
        (index for index in range(1, len(words)) if words[index].start < words[index - 1].start),
        None,
    )


@dataclass(frozen=True)
class _Token:
    """One normalized token, timed by the word it came from."""

    text: str
    start: float
    end: float
    word: int


@dataclass(frozen=True)
class _Step:
    """One alignment step; on a side it consumes nothing, the count that side has consumed."""

    kind: _Kind
    ref: int
    hyp: int


def _tokens(words: Sequence[Word]) -> list[_Token]:
    return [
        _Token(text, word.start, word.end, index)
        for index, word in enumerate(words)
        for text in norm_tokens(word.text)
    ]


def _gap(first: _Token, second: _Token) -> float:
    return max(0.0, first.start - second.end, second.start - first.end)


def _align(ref: Sequence[_Token], hyp: Sequence[_Token]) -> list[_Step]:
    """Align by edit distance, pairing two tokens only within TOLERANCE_S of each other.

    Row r covers the hypothesis tokens from BAND_S before reference token r to
    BAND_S after the token that follows it, widened so that neither edge of the
    band moves backward. `hyp` must be in start order.
    """
    rows, cols = len(ref), len(hyp)
    starts = [token.start for token in hyp]
    low = [0] * (rows + 1)
    high = [0] * (rows + 1)
    high[0] = bisect.bisect_right(starts, ref[0].end + BAND_S) if rows else cols
    for row in range(1, rows + 1):
        token = ref[row - 1]
        near = bisect.bisect_left(starts, token.start - BAND_S)
        # From the token after the gap this row's insertions fill, as for row 0:
        # words heard late in a hole longer than the band belong before that token.
        ahead = ref[row] if row < rows else token
        high[row] = max(high[row - 1], bisect.bisect_right(starts, ahead.end + BAND_S))
        low[row] = max(low[row - 1], min(near, high[row - 1]))
    high[rows] = cols
    costs: list[dict[int, int]] = [{col: col for col in range(low[0], high[0] + 1)}]
    moves: list[dict[int, _Kind]] = [{col: "ins" for col in range(low[0], high[0] + 1)}]
    for row in range(1, rows + 1):
        token, above = ref[row - 1], costs[row - 1]
        cost: dict[int, int] = {}
        move: dict[int, _Kind] = {}
        for col in range(low[row], high[row] + 1):
            # Every cell is reachable: a row starts inside the row above, and a
            # column past that row's end follows the cell to its left.
            best: tuple[int, _Kind] = (
                (above[col] + 1, "del") if col in above else (cost[col - 1] + 1, "ins")
            )
            if col - 1 in above and _gap(token, hyp[col - 1]) <= TOLERANCE_S:
                same = hyp[col - 1].text == token.text
                paired = above[col - 1] + (0 if same else 1)
                # A tie with a deletion or an insertion pairs the tokens instead.
                if paired <= best[0]:
                    best = (paired, "match" if same else "sub")
            if col - 1 in cost and cost[col - 1] + 1 < best[0]:
                best = (cost[col - 1] + 1, "ins")
            cost[col], move[col] = best
        costs.append(cost)
        moves.append(move)
    steps: list[_Step] = []
    row, col = rows, cols
    while row or col:
        kind = moves[row][col]
        if kind == "ins":
            steps.append(_Step(kind, row, col - 1))
            col -= 1
        elif kind == "del":
            steps.append(_Step(kind, row - 1, col))
            row -= 1
        else:
            steps.append(_Step(kind, row - 1, col - 1))
            row, col = row - 1, col - 1
    steps.reverse()
    return steps


def align_words(
    backbone: Sequence[Word], hypothesis: Sequence[Word]
) -> list[tuple[_Kind, int | None, int | None]]:
    """Align two word lists token by token, as the vote aligns a hypothesis to its backbone.

    Each side's words are split by `norm_tokens`, and every token is consumed
    by exactly one step, in order. `hypothesis` must be in start order.

    Returns:
        One (kind, backbone word, hypothesis word) per step, in order, each
        word given by its index, and None on a side the step consumes nothing of.

    """
    backbone_tokens, hypothesis_tokens = _tokens(backbone), _tokens(hypothesis)
    return [
        (
            step.kind,
            None if step.kind == "ins" else backbone_tokens[step.ref].word,
            None if step.kind == "del" else hypothesis_tokens[step.hyp].word,
        )
        for step in _align(backbone_tokens, hypothesis_tokens)
    ]


def _slots(count: int, steps: Sequence[_Step]) -> tuple[list[int | None], list[list[int]]]:
    """Return the hypothesis token paired with each slot, and the ones in each gap.

    Gap g lies before slot g; gap `count` follows the last slot.
    """
    paired: list[int | None] = [None] * count
    extra: list[list[int]] = [[] for _ in range(count + 1)]
    for step in steps:
        if step.kind == "ins":
            extra[step.ref].append(step.hyp)
        elif step.kind != "del":
            paired[step.ref] = step.hyp
    return paired, extra


def _backchannel_only(tokens: Sequence[str]) -> bool:
    index = 0
    while index < len(tokens):
        phrase = next(
            (each for each in _BACKCHANNELS if tuple(tokens[index : index + len(each)]) == each),
            None,
        )
        if phrase is not None:
            index += len(phrase)
        elif tokens[index] in FILLERS:
            index += 1
        else:
            return False
    return True


def _runs(indices: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in indices:
        if runs and runs[-1][-1] == index - 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _near(starts: Sequence[float], at: float) -> bool:
    index = bisect.bisect_left(starts, at - _DUPLICATE_S)
    return index < len(starts) and starts[index] <= at + _DUPLICATE_S


def _insertable(token: _Token, unpaired: dict[str, list[float]]) -> bool:
    return token.text not in FILLERS and not _near(unpaired.get(token.text, []), token.start)


def nearest(neighbors: Sequence[Word], at: float) -> Word | None:
    """Return the word nearest `at` by either end, the earlier of equals; None if none."""
    # min keeps the first of equals, so a tie goes to the earlier neighbor.
    return min(
        neighbors, key=lambda word: min(abs(word.start - at), abs(word.end - at)), default=None
    )


def _strip_outer(text: str) -> str:
    return _OUTER_RIGHT.sub("", _OUTER_LEFT.sub("", text))


def _surface(backbone: str, primary: str, token: str, primary_tokens: int) -> str:
    """Spell a one-token word's change: the primary's core, the backbone's edges and capital."""
    core = (_strip_outer(primary) if primary_tokens == 1 else token) or token
    before = found.group(0) if (found := _OUTER_LEFT.match(backbone)) else ""
    after = found.group(0) if (found := _OUTER_RIGHT.search(backbone)) else ""
    if _strip_outer(backbone)[:1].isupper():
        core = core[:1].upper() + core[1:]
    return before + core + after


def _starting(word: Word, start: float) -> Word:
    return word.model_copy(update={"start": start, "end": max(word.end, start)})


@dataclass(frozen=True)
class _Ballot:
    """The backbone's slots, with each hypothesis's tokens aligned to them."""

    backbone: Sequence[Word]
    primary: Sequence[Word]
    slots: list[_Token]
    primary_tokens: list[_Token]
    secondary_tokens: list[_Token]
    primary_paired: list[int | None]
    secondary_paired: list[int | None]
    primary_extra: list[list[int]]
    secondary_extra: list[list[int]]

    @classmethod
    def cast(
        cls, backbone: Sequence[Word], primary: Sequence[Word], secondary: Sequence[Word]
    ) -> _Ballot:
        """Tokenize all three and align both hypotheses to the backbone."""
        slots, primary_tokens, secondary_tokens = map(_tokens, (backbone, primary, secondary))
        primary_paired, primary_extra = _slots(len(slots), _align(slots, primary_tokens))
        secondary_paired, secondary_extra = _slots(len(slots), _align(slots, secondary_tokens))
        return cls(
            backbone,
            primary,
            slots,
            primary_tokens,
            secondary_tokens,
            primary_paired,
            secondary_paired,
            primary_extra,
            secondary_extra,
        )

    def substitutions(self) -> dict[int, int]:
        """Return the primary token replacing each changed slot."""
        chosen: dict[int, int] = {}
        for slot, token in enumerate(self.slots):
            primary, secondary = self.primary_paired[slot], self.secondary_paired[slot]
            if primary is None or secondary is None:
                continue
            text = self.primary_tokens[primary].text
            if text == self.secondary_tokens[secondary].text != token.text and text not in FILLERS:
                chosen[slot] = primary
        return chosen

    def insertions(self) -> dict[int, list[int]]:
        """Return the primary tokens inserted in each gap that gains any."""
        unpaired: dict[str, list[float]] = {}
        for slot, token in enumerate(self.slots):
            if self.primary_paired[slot] is None or self.secondary_paired[slot] is None:
                unpaired.setdefault(token.text, []).append(token.start)
        for starts in unpaired.values():
            starts.sort()
        chosen: dict[int, list[int]] = {}
        for gap, (primary, secondary) in enumerate(
            zip(self.primary_extra, self.secondary_extra, strict=True)
        ):
            if not primary or not secondary:
                continue
            steps = _align(
                [self.primary_tokens[index] for index in primary],
                [self.secondary_tokens[index] for index in secondary],
            )
            agreed = [primary[step.ref] for step in steps if step.kind == "match"]
            kept = [
                index
                for run in _runs(agreed)
                if not _backchannel_only([self.primary_tokens[index].text for index in run])
                for index in run
                if _insertable(self.primary_tokens[index], unpaired)
            ]
            if kept:
                chosen[gap] = kept
        return chosen

    def slot_word(self, word: Word, slots: range, substitutions: dict[int, int]) -> Word:
        """Return a backbone word as its slots voted: kept, or respelled."""
        if not any(slot in substitutions for slot in slots):
            return word
        if len(slots) > 1:
            text = " ".join(
                self.primary_tokens[substitutions[slot]].text
                if slot in substitutions
                else self.slots[slot].text
                for slot in slots
            )
        else:
            token = self.primary_tokens[substitutions[slots[0]]]
            source = self.primary[token.word].text
            text = _surface(word.text, source, token.text, len(norm_tokens(source)))
        return word.model_copy(update={"text": text})

    def gap_words(self, gap: int, tokens: Sequence[int]) -> list[Word]:
        """Return the words a gap's inserted tokens form, each timed as its primary word."""
        by_word: dict[int, list[int]] = {}
        for index in tokens:
            by_word.setdefault(self.primary_tokens[index].word, []).append(index)
        neighbors = [
            self.backbone[self.slots[slot].word]
            for slot in (gap - 1, gap)
            if 0 <= slot < len(self.slots)
        ]
        words: list[Word] = []
        for index, members in by_word.items():
            source = self.primary[index]
            whole = len(members) == len(norm_tokens(source.text))
            closest = nearest(neighbors, source.start)
            words.append(
                Word(
                    text=source.text
                    if whole
                    else " ".join(self.primary_tokens[member].text for member in members),
                    start=source.start,
                    end=source.end,
                    speaker=None if closest is None else closest.speaker,
                )
            )
        return words


def _ordered(sequence: Sequence[tuple[Word, bool]]) -> list[Word]:
    """Keep each inserted word's start between the starts of the words around it."""
    placed: list[Word] = []
    for word, inserted in sequence:
        floor = placed[-1].start if placed else 0.0
        placed.append(_starting(word, max(word.start, floor)) if inserted else word)
    following = math.inf
    for position in reversed(range(len(placed))):
        if sequence[position][1]:
            placed[position] = _starting(placed[position], min(placed[position].start, following))
        else:
            following = placed[position].start
        following = min(following, placed[position].start)
    return placed


def _hole_of(holes: Sequence[Hole], at: float) -> int | None:
    # find_holes's holes are disjoint and in time order. A hole holds its start:
    # a word there, at 0 or at the end of the word before, begins what was dropped.
    index = bisect.bisect_right(holes, at, key=lambda hole: hole.start) - 1
    return index if index >= 0 and at < holes[index].end else None


def _audio_end(words: Sequence[Word], duration: float | None = None) -> float:
    return max((word.end for word in words), default=0.0) if duration is None else duration


def _placed(
    sequence: Sequence[tuple[Word, bool]], backbone: Sequence[Word], audio_end: float
) -> list[Word]:
    """Order the voted words, clearing the speakers of each passage inserted in a backbone hole."""
    holes = find_holes(backbone, audio_end, MIN_DROP_SECONDS)
    placed = _ordered(sequence)
    where = [
        _hole_of(holes, word.start) if inserted else None
        for word, (_, inserted) in zip(placed, sequence, strict=True)
    ]
    first = 0
    for hole, members in groupby(where):
        stop = first + len(list(members))
        run = placed[first:stop]
        if hole is not None and sum(not is_filler(word_key(w.text)) for w in run) >= MIN_DROP_WORDS:
            placed[first:stop] = [word.model_copy(update={"speaker": None}) for word in run]
        first = stop
    return placed


def _cast_votes(
    backbone: Sequence[Word], primary: Sequence[Word], secondary: Sequence[Word]
) -> list[tuple[Word, bool]]:
    """Return every voted word in order, each marked True if it was inserted."""
    for role, words in (("primary", primary), ("secondary", secondary)):
        if (index := first_decrease(words)) is not None:
            raise ValueError(f"{role} word {index} starts before word {index - 1}")
    ballot = _Ballot.cast(backbone, primary, secondary)
    substitutions = ballot.substitutions()
    inserted = {gap: ballot.gap_words(gap, tokens) for gap, tokens in ballot.insertions().items()}
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for slot, token in enumerate(ballot.slots):
        first.setdefault(token.word, slot)
        last[token.word] = slot
    sequence: list[tuple[Word, bool]] = []
    for index, word in enumerate(backbone):
        if index not in first:
            sequence.append((word, False))
            continue
        head, tail = first[index], last[index]
        sequence += [(each, True) for each in inserted.get(head, [])]
        sequence.append((ballot.slot_word(word, range(head, tail + 1), substitutions), False))
        for gap in range(head + 1, tail + 1):
            sequence += [(each, True) for each in inserted.get(gap, [])]
    sequence += [(each, True) for each in inserted.get(len(ballot.slots), [])]
    return sequence


def vote_words(
    backbone: Sequence[Word], primary: Sequence[Word], secondary: Sequence[Word]
) -> list[Word]:
    """Vote the backbone's words against two hypotheses of the same speech.

    A slot takes the token both hypotheses pair with it when that token differs
    from the backbone's and is not a filler; the word keeps the backbone's times
    and speaker. A run of tokens both hypotheses place in the same gap is
    inserted, less its fillers, unless it holds only backchannels; each inserted
    token must also lie over 1 s from every backbone token of the same text
    that either hypothesis left unpaired. An inserted word takes the primary's
    times, moved as far as needed to keep it between its neighbors' starts, and
    the speaker of the backbone word nearest in time, the earlier one on a tie.
    Inserted words in a row whose starts, so moved, lie inside one hole of
    MIN_DROP_SECONDS or more in the backbone's words, its start included, as
    `gaps.find_holes` finds them up to the latest backbone word end, take no
    speaker instead when MIN_DROP_WORDS or more of them are not fillers: the
    backbone may have dropped a whole turn of another speaker's there. Fewer
    keep the nearest speaker, as too little speech for its speaker to matter.

    Args:
        backbone: The words that are the slots, in transcript order.
        primary: The hypothesis whose spelling a changed word takes and whose
            times an inserted word takes.
        secondary: The second voter. Its speakers, like the primary's, are unused.

    Returns:
        Every backbone word, kept or respelled, with inserted words among them.

    Raises:
        ValueError: A hypothesis word starts before the word ahead of it.

    """
    return _placed(_cast_votes(backbone, primary, secondary), backbone, _audio_end(backbone))


def vote_transcripts(
    backbone: Transcript, primary: Transcript, secondary: Transcript
) -> Transcript:
    """Vote three transcripts of one recording into one; see `vote_words`.

    The backbone's holes run to its duration when it has one.

    Returns:
        A transcript with the backbone's source, language and duration, the
        voted words, and no turns: voting precedes the turns stage. Its engine
        params count the words inserted, those of them with no speaker, and the
        backbone words respelled.

    Raises:
        ValueError: A hypothesis word starts before the word ahead of it.

    """
    sequence = _cast_votes(backbone.words, primary.words, secondary.words)
    # One uninserted word per backbone word, in the backbone's order.
    kept = [word for word, inserted in sequence if not inserted]
    words = _placed(sequence, backbone.words, _audio_end(backbone.words, backbone.duration))
    return Transcript(
        source=backbone.source,
        engine=Engine(
            name="rover",
            params={
                "variant": VARIANT,
                "tolerance_s": TOLERANCE_S,
                "band_s": BAND_S,
                "backbone": backbone.engine.name,
                "primary": primary.engine.name,
                "secondary": secondary.engine.name,
                "words_inserted": len(sequence) - len(kept),
                "words_inserted_unattributed": sum(
                    inserted and word.speaker is None
                    for word, (_, inserted) in zip(words, sequence, strict=True)
                ),
                "words_substituted": sum(
                    new.text != old.text for new, old in zip(kept, backbone.words, strict=True)
                ),
            },
        ),
        language=backbone.language,
        duration=backbone.duration,
        text=" ".join(word.text for word in words),
        words=words,
    )
