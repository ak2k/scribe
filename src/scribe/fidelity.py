"""Check a cleaned transcript against its input: speakers kept, content unchanged.

Both sides are folded the way the number check reads them and reduced by the
edits cleanup is asked to make, then aligned word by word. A word matched
across a speaker change is a move; any other difference is a content edit.
A move cleanup also reordered past kept speech is found only as a run of
`_MOVE_WORDS` or more within `_MOVE_REACH` turns: shorter runs of common words
recur by chance, and pairing them would report moves nobody made.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import groupby
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from scribe.cleanup import (
    DEFAULT_CHUNK_WORDS,
    chunk_turns,
    final_speakers,
    folded_words,
    label_prefix,
    spoken_values,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from difflib import Match

    from scribe.cleanup import CleanupRequest

# Folded words before "like" that make it the verb ("we like it", "I'd like").
# Not "to" or "do": in speech "to like keep" and "do like a refactor" are filler.
_LIKE_SUBJECTS = frozenset(
    [
        "i",
        "we",
        "they",
        "he",
        "she",
        "you",
        "id",
        "wed",
        "theyd",
        "youd",
        "hed",
        "shed",
        "would",
        "dont",
        "doesnt",
        "didnt",
    ]
)
# "kind of" and "sort of" are filler before one of these ("kind of a mess") or
# at the end of a line; anywhere else they qualify what follows ("kind of
# agree") or name a type ("that kind of budget"), and dropping them is an edit.
_DETERMINERS = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "some",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
    ]
)


def _always(_before: str | None, _after: str | None) -> bool:
    return True


def _like_is_filler(before: str | None, _after: str | None) -> bool:
    return before not in _LIKE_SUBJECTS


def _hedge_is_filler(_before: str | None, after: str | None) -> bool:
    return after is None or after in _DETERMINERS


# The fillers SYSTEM_PROMPT rule 2 names, as folded words, each with when
# removing it is allowed, judged on its neighbors once the fillers before it
# are gone. "you know" and "i mean" mark discourse rather than qualify a claim,
# so they are allowed wherever they stand. "like" comes before the hedges so
# that "sort of like a" is read as two fillers.
_FILLERS: tuple[tuple[tuple[str, ...], Callable[[str | None, str | None], bool]], ...] = (
    (("um",), _always),
    (("uh",), _always),
    (("er",), _always),
    (("you", "know"), _always),
    (("i", "mean"), _always),
    (("like",), _like_is_filler),
    (("sort", "of"), _hedge_is_filler),
    (("kind", "of"), _hedge_is_filler),
)
# Longer repeats than this are a speaker restating, not a stutter.
_REPEAT_WORDS = 4
# A stutter's cut-off start, written with a hyphen and begun again by the next
# word ("bu- budget"). A whole word the next one begins with ("no nobody") is
# speech, and dropping it would hide its deletion.
_FRAGMENT = re.compile(r"(?<!\S)([^\W_]+)-(?=\s+\1)", re.IGNORECASE)
# A reordered move is found as this many words in a row, no further than this
# many turns from where they were said. Reach counts diarized turns, not the
# pieces a long turn is sent as: a turn cut in five is still one turn away
# from the speaker after it.
_MOVE_WORDS = 3
_MOVE_REACH = 2
# Enough of a span to find it again in the input, and no more of the meeting.
_SPAN_WORDS = 8
_SPANS_KEPT = 20


class MovedSpan(BaseModel):
    """Consecutive words of one input turn that cleanup put under another speaker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The id cleanup sent the turn under: from 1, a split turn's pieces each
    # their own.
    turn: int
    from_speaker: str
    to_speaker: str
    words: str


class ContentSpan(BaseModel):
    """One content edit: input words replaced, deleted or inserted near `turn`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # An id, as in `MovedSpan`.
    turn: int
    before: str
    after: str


class Fidelity(BaseModel):
    """What the alignment of cleaned text to input turns found.

    Counts are of folded words left after the allowed edits are taken out of
    both sides. Cleanup takes every label from the input, so a real move in
    `moved_words` is text the reply returned under another turn's id; the
    word-alone match can still pair common words across speakers and report a
    move nobody made.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # A count, like `numbers_checked`: zero moves over zero words is no evidence.
    words_checked: int = 0
    moved_words: int = 0
    moved_spans: list[MovedSpan] = Field(default_factory=list)
    content_edit_words: int = 0
    content_edits_per_1000: float = 0.0
    content_spans: list[ContentSpan] = Field(default_factory=list)


def _fold(text: str, glossary: Sequence[tuple[list[str], list[str]]]) -> list[str]:
    words = [word for word, _ in folded_words(spoken_values(_FRAGMENT.sub("", text)))]
    for wrong, right in glossary:
        words = _replace(words, wrong, right)
    for filler, allowed in _FILLERS:
        words = _drop_filler(words, list(filler), allowed)
    return _collapse_repeats(words)


def _drop_filler(
    words: list[str], filler: list[str], allowed: Callable[[str | None, str | None], bool]
) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(words):
        end = index + len(filler)
        if words[index:end] == filler and allowed(
            out[-1] if out else None, words[end] if end < len(words) else None
        ):
            index = end
        else:
            out.append(words[index])
            index += 1
    return out


def _replace(words: list[str], old: list[str], new: list[str]) -> list[str]:
    if not old:
        return words
    out: list[str] = []
    index = 0
    while index < len(words):
        if words[index : index + len(old)] == old:
            out.extend(new)
            index += len(old)
        else:
            out.append(words[index])
            index += 1
    return out


def _collapse_repeats(words: list[str]) -> list[str]:
    """Keep one copy of a run of up to `_REPEAT_WORDS` words said twice in a row."""
    out: list[str] = []
    for word in words:
        out.append(word)
        for size in range(1, _REPEAT_WORDS + 1):
            if len(out) >= 2 * size and out[-size:] == out[-2 * size : -size]:
                del out[-size:]
                break
    return out


def _output_lines(text: str, labels: Sequence[str]) -> Iterator[tuple[str | None, str]]:
    """Each line of cleaned text with the label it is read under, prefix removed.

    A line with no label of its own continues the one above it, as a reader
    takes it. Cleanup's text opens with a label; a line before any has None,
    and its words are never read as moved.
    """
    by_name = {label.lower(): label for label in labels}
    current: str | None = None
    for line in text.split("\n"):
        prefix = label_prefix(line, labels)
        if prefix:
            current = by_name.get(prefix.strip(" \t*:").lower())
        yield current, line[len(prefix) :]


type _Word = tuple[str | None, str, int]
"""A folded word: the label it is read under, the word, and its turn's id or line index."""


def _joined(words: Sequence[_Word]) -> str:
    return " ".join(word for _, word, _ in words[:_SPAN_WORDS])


class _Findings:
    """Running counts, with the first `_SPANS_KEPT` spans of each kind."""

    def __init__(self) -> None:
        self.moved = 0
        self.moved_spans: list[MovedSpan] = []
        self.edited = 0
        self.content_spans: list[ContentSpan] = []

    def move(self, turn: int, origin: str, target: str, words: list[_Word]) -> None:
        self.moved += len(words)
        if len(self.moved_spans) < _SPANS_KEPT:
            self.moved_spans.append(
                MovedSpan(turn=turn, from_speaker=origin, to_speaker=target, words=_joined(words))
            )

    def edit(self, turn: int, removed: list[_Word], added: list[_Word]) -> None:
        self.edited += max(len(removed), len(added))
        if len(self.content_spans) < _SPANS_KEPT:
            self.content_spans.append(
                ContentSpan(turn=turn, before=_joined(removed), after=_joined(added))
            )


@dataclass
class _Leftover:
    """What one aligned region removed and added that nothing matched.

    `turn` is where it is reported when no removed word says where.
    """

    turn: int
    removed: list[_Word]
    added: list[_Word]


def _record_moves(pairs: Iterable[tuple[_Word, _Word]], findings: _Findings) -> None:
    """Record each matched (input, output) pair whose output label is another speaker's."""
    for (turn, origin, target), group in groupby(
        pairs, key=lambda pair: (pair[0][2], pair[0][0], pair[1][0])
    ):
        if origin is not None and target is not None and origin != target:
            findings.move(turn, origin, target, [word for word, _ in group])


def _compare_words(
    removed: list[_Word], added: list[_Word], turn: int, findings: _Findings
) -> list[_Leftover]:
    """Match what speaker-aware alignment left over on the words alone.

    `turn` is the turn of the input word before `removed`, where an insertion
    with no input word of its own before it is reported.

    Returns:
        What still matched nothing.

    """
    matcher = SequenceMatcher(
        None, [word for _, word, _ in removed], [word for _, word, _ in added], autojunk=False
    )
    leftovers: list[_Leftover] = []
    for tag, low, high, out_low, out_high in matcher.get_opcodes():
        old, new = removed[low:high], added[out_low:out_high]
        if tag == "equal":
            _record_moves(zip(old, new, strict=True), findings)
        # A compound joined or split ("follow up", "follow-up") is punctuation.
        elif "".join(word for _, word, _ in old) != "".join(word for _, word, _ in new):
            where = old[0][2] if old else removed[low - 1][2] if low else turn
            leftovers.append(_Leftover(where, old, new))
    return leftovers


type _Run = tuple[int, list[_Word]]
"""Consecutive unmatched words, with the index of the leftover they belong to."""


def _longest_move(
    leftovers: Sequence[_Leftover],
    removed: Sequence[_Run],
    added: Sequence[_Run],
    origin: Mapping[int, int],
) -> tuple[int, int, Match] | None:
    """The longest removed run found again, within reach, under other speakers only.

    `origin` maps each id to the diarized turn it was cut from.
    """
    best: tuple[int, int, Match] | None = None
    for low, (_, old) in enumerate(removed):
        if len(old) < _MOVE_WORDS:
            continue
        for out_low, (owner, new) in enumerate(added):
            where = origin[leftovers[owner].turn]
            if len(new) < _MOVE_WORDS or not (
                origin[old[0][2]] - _MOVE_REACH <= where <= origin[old[-1][2]] + _MOVE_REACH
            ):
                continue
            match = SequenceMatcher(
                None, [word for _, word, _ in old], [word for _, word, _ in new], autojunk=False
            ).find_longest_match()
            if match.size < max(_MOVE_WORDS, best[2].size + 1 if best else 0):
                continue
            said = old[match.a : match.a + match.size]
            shown = new[match.b : match.b + match.size]
            if abs(origin[said[0][2]] - where) <= _MOVE_REACH and all(
                target is not None and target != origin
                for (origin, _, _), (target, _, _) in zip(said, shown, strict=True)
            ):
                best = (low, out_low, match)
    return best


def _settle(leftovers: list[_Leftover], findings: _Findings, origin: Mapping[int, int]) -> None:
    """Find moves cleanup reordered past kept speech; count the rest as content edits."""
    removed: list[_Run] = [(index, item.removed) for index, item in enumerate(leftovers)]
    added: list[_Run] = [(index, item.added) for index, item in enumerate(leftovers)]
    while (found := _longest_move(leftovers, removed, added, origin)) is not None:
        low, out_low, match = found
        (owner, old), (out_owner, new) = removed[low], added[out_low]
        end, out_end = match.a + match.size, match.b + match.size
        _record_moves(zip(old[match.a : end], new[match.b : out_end], strict=True), findings)
        removed[low : low + 1] = [(owner, part) for part in (old[: match.a], old[end:])]
        added[out_low : out_low + 1] = [
            (out_owner, part) for part in (new[: match.b], new[out_end:])
        ]
    old_left: defaultdict[int, list[_Word]] = defaultdict(list)
    new_left: defaultdict[int, list[_Word]] = defaultdict(list)
    for index, words in removed:
        old_left[index].extend(words)
    for index, words in added:
        new_left[index].extend(words)
    for index, item in enumerate(leftovers):
        old, new = old_left[index], new_left[index]
        if old or new:
            findings.edit(old[0][2] if old else item.turn, old, new)


def check_fidelity(
    request: CleanupRequest, text: str, *, max_words: int = DEFAULT_CHUNK_WORDS
) -> Fidelity:
    """Align cleaned text to the request's turns; find moved and edited words.

    Allowed edits are taken out of both sides before aligning: the fillers the
    prompt lists, except a hedge or a "like" that carries meaning, stutter
    fragments, a run repeated in a row, case and
    punctuation, value rewrites the number check accepts, and glossary
    substitutions. Words are matched with their speaker first, so a word
    repeated across a turn boundary pairs with the copy under its own speaker;
    only what is left is matched on words alone, where a match under another
    label is a move, first inside each region and then, for longer runs,
    across nearby regions.

    Args:
        request: The turns cleanup was given, with its relabel key and glossary.
        text: The cleaned text, without the provenance header.
        max_words: The word ceiling cleanup chunked the turns under, so that
            each span names the id its turn, or the piece of a split turn,
            was sent under.

    Returns:
        Counts and capped spans of moved and content-edited words.

    """
    glossary = [
        (
            [word for word, _ in folded_words(spoken_values(wrong))],
            [word for word, _ in folded_words(spoken_values(right))],
        )
        for wrong, right in request.glossary.items()
    ]
    # Each piece cleanup sent, in order, with the diarized turn it was cut from.
    sent = [
        (index, piece)
        for index, whole in enumerate(request.turns)
        for chunk in chunk_turns([whole], max_words=max_words)
        for piece in chunk
    ]
    origin = {turn_id: index for turn_id, (index, _) in enumerate(sent, start=1)}
    before: list[_Word] = [
        (request.speaker_key.get(turn.speaker, turn.speaker), word, turn_id)
        for turn_id, (_, turn) in enumerate(sent, start=1)
        for word in _fold(turn.text, glossary)
    ]
    after: list[_Word] = [
        (label, word, index)
        for index, (label, line) in enumerate(_output_lines(text, final_speakers(request)))
        for word in _fold(line, glossary)
    ]
    findings = _Findings()
    leftovers: list[_Leftover] = []
    by_speaker = SequenceMatcher(
        None,
        [(label, word) for label, word, _ in before],
        [(label, word) for label, word, _ in after],
        autojunk=False,
    )
    for tag, low, high, out_low, out_high in by_speaker.get_opcodes():
        if tag != "equal":
            turn = before[max(low - 1, 0)][2] if before else 1
            leftovers += _compare_words(before[low:high], after[out_low:out_high], turn, findings)
    _settle(leftovers, findings, origin)
    return Fidelity(
        words_checked=len(before),
        moved_words=findings.moved,
        moved_spans=findings.moved_spans,
        content_edit_words=findings.edited,
        content_edits_per_1000=round(1000 * findings.edited / len(before), 1) if before else 0.0,
        content_spans=findings.content_spans,
    )
