"""Find where two transcripts of one recording disagree, to pick a reading at each.

A spot is where the transcript's words and a reference's differ in what was
said. The two are aligned token by token as the vote aligns them; a spot starts
as a maximal run of unmatched steps, runs MERGE_GAP matched steps apart or
fewer joined, and only a run holding tokens of both sides counts: words only
one side heard are the fill's to handle. It then widens until it holds whole
words on both sides. A spot whose readings match once numbers, contractions,
fillers, stutters, order, spacing and accents are set aside, or where one side
holds only fillers, is not a disagreement worth asking about.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scribe.spoken_numbers import digitize
from scribe.vote import FILLERS, align_words, norm_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scribe.schema import Word

# Differences one matched word apart ("we saw the cat" against "he saw a cat")
# are one reading, not two.
MERGE_GAP = 1
# Matched tokens on each side of a spot that its readings are compared with, so
# a repeat or a number across the spot's edge collapses as it does inside.
CONTEXT_TOKENS = 2

_CONTRACTIONS = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "hafta": "have to",
    "kinda": "kind of",
    "sorta": "sort of",
    "lemme": "let me",
    "gimme": "give me",
    "dunno": "do not know",
    "cause": "because",
    "cuz": "because",
    "coz": "because",
    "cos": "because",
    "ya": "you",
    "y'all": "you all",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "how's": "how is",
    "he's": "he is",
    "she's": "she is",
    "let's": "let us",
    "i'm": "i am",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are",
    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "they'll": "they will",
    "it'll": "it will",
    "that'll": "that will",
    "he'll": "he will",
    "she'll": "she will",
    "i'd": "i would",
    "we'd": "we would",
    "you'd": "you would",
    "they'd": "they would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "that'd": "that would",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "ain't": "is not",
    "there're": "there are",
    "what're": "what are",
    "gotcha": "got you",
    "alright": "all right",
    "ok": "okay",
    "k": "okay",
    "yeah": "yes",
    "yep": "yes",
    "yup": "yes",
    "yea": "yes",
    "yah": "yes",
    "nope": "no",
    "nah": "no",
    "til": "until",
    "till": "until",
    "em": "them",
}
_FILLERS = FILLERS | {"uhm", "umm", "uhh", "mhm", "mmhmm", "er", "erm", "ah", "oh", "like"}
_FILLER_PAIRS = frozenset({("you", "know"), ("i", "mean"), ("mm", "hmm"), ("uh", "huh")})
# A one-letter token ("a" before "about") is a word too often to be a false start.
_FALSE_START_LETTERS = 2


@dataclass(frozen=True)
class Spot:
    """A disagreement: the transcript's words and the reference's there, by index."""

    transcript: range
    reference: range


def _without_fillers(tokens: Sequence[str]) -> list[str]:
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        if tuple(tokens[index : index + 2]) in _FILLER_PAIRS:
            index += 2
            continue
        if tokens[index] not in _FILLERS:
            kept.append(tokens[index])
        index += 1
    return kept


def _without_stutters(tokens: Sequence[str]) -> list[str]:
    """Drop a word said twice running, a false start the next word begins, and a repeated pair."""
    kept = list(tokens)
    while True:
        repeat = next(
            (
                index
                for index in range(len(kept) - 1)
                if kept[index] == kept[index + 1]
                or (
                    len(kept[index]) >= _FALSE_START_LETTERS
                    and kept[index + 1].startswith(kept[index])
                )
            ),
            None,
        )
        if repeat is not None:
            del kept[repeat]
            continue
        pair = next(
            (
                index
                for index in range(len(kept) - 3)
                if kept[index : index + 2] == kept[index + 2 : index + 4]
            ),
            None,
        )
        if pair is None:
            return kept
        del kept[pair : pair + 2]


def _folded(token: str) -> str:
    return unicodedata.normalize("NFKD", token.replace("'", "")).encode("ascii", "ignore").decode()


def _normalized(text: str) -> list[str]:
    """Return the tokens two readings are compared by: what was said, not how it was written."""
    tokens = norm_tokens(digitize(text.replace("%", " percent")))
    expanded = [part for token in tokens for part in _CONTRACTIONS.get(token, token).split()]
    return [_folded(token) for token in _without_stutters(_without_fillers(expanded))]


def _same(first: Sequence[str], second: Sequence[str]) -> bool:
    return (
        list(first) == list(second)
        or set(first) == set(second)
        or "".join(first) == "".join(second)
    )


@dataclass(frozen=True)
class _Alignment:
    """The token alignment, one entry per step in each list."""

    # The transcript token a matched step consumes; None on any other step.
    matched: list[str | None]
    transcript: list[int | None]
    reference: list[int | None]
    # Each word's steps, [first, last + 1), on each side.
    transcript_extents: dict[int, tuple[int, int]]
    reference_extents: dict[int, tuple[int, int]]

    @classmethod
    def of(cls, transcript: Sequence[Word], reference: Sequence[Word]) -> _Alignment:
        steps = align_words(transcript, reference)
        # align_words consumes each transcript token once, in order.
        tokens = iter([token for word in transcript for token in norm_tokens(word.text)])
        matched: list[str | None] = []
        for kind, word, _ in steps:
            token = None if word is None else next(tokens)
            matched.append(token if kind == "match" else None)
        said, heard = [word for _, word, _ in steps], [word for _, _, word in steps]
        return cls(matched, said, heard, _extents(said), _extents(heard))

    def _runs(self) -> list[tuple[int, int]]:
        """Return each maximal run of unmatched steps, those MERGE_GAP apart or fewer joined."""
        runs: list[tuple[int, int]] = []
        for index, token in enumerate(self.matched):
            if token is not None:
                continue
            if runs and index - runs[-1][1] <= MERGE_GAP:
                runs[-1] = (runs[-1][0], index + 1)
            else:
                runs.append((index, index + 1))
        return runs

    def _whole(self, start: int, stop: int) -> tuple[int, int]:
        """Widen steps [start, stop) until every word they touch on either side is inside."""
        sides = (
            (self.transcript, self.transcript_extents),
            (self.reference, self.reference_extents),
        )
        while True:
            low, high = start, stop
            for side, extent in sides:
                for word in side[start:stop]:
                    if word is not None:
                        low, high = min(low, extent[word][0]), max(high, extent[word][1])
            if (low, high) == (start, stop):
                return start, stop
            start, stop = low, high

    def spans(self) -> list[tuple[int, int]]:
        """Return the step ranges of the spots, before any is judged a difference in form."""
        spans: list[tuple[int, int]] = []
        for first, last in self._runs():
            both = any(word is not None for word in self.transcript[first:last]) and any(
                word is not None for word in self.reference[first:last]
            )
            if not both:
                continue
            start, stop = self._whole(first, last)
            # Widened, a spot can reach the one before or come within MERGE_GAP of it.
            while spans and start - spans[-1][1] <= MERGE_GAP:
                before = spans.pop()
                start, stop = self._whole(min(start, before[0]), max(stop, before[1]))
            spans.append((start, stop))
        return spans

    def context(self, start: int, stop: int) -> tuple[list[str], list[str]]:
        """Return the matched tokens among the CONTEXT_TOKENS steps before and after a span."""
        before = self.matched[max(0, start - CONTEXT_TOKENS) : start]
        after = self.matched[stop : stop + CONTEXT_TOKENS]
        return [token for token in before if token is not None], [
            token for token in after if token is not None
        ]


def _extents(side: Sequence[int | None]) -> dict[int, tuple[int, int]]:
    """Map each word to the steps its tokens take, [first, last + 1)."""
    extents: dict[int, tuple[int, int]] = {}
    for index, word in enumerate(side):
        if word is not None:
            extents[word] = (extents.get(word, (index, index))[0], index + 1)
    return extents


def _words(side: Sequence[int | None]) -> range:
    held = [word for word in side if word is not None]
    return range(held[0], held[-1] + 1)


def find_spots(transcript: Sequence[Word], reference: Sequence[Word]) -> list[Spot]:
    """Find where two transcripts of one recording disagree in what was said.

    Args:
        transcript: The words a pick can change.
        reference: The other transcript's words, in start order.

    Returns:
        The spots, disjoint and in order on both sides, each holding words of
        both. Readings the same once normalized, with up to CONTEXT_TOKENS
        matched tokens on each side, are left out: equal, the same set of
        tokens, or the same letters spaced differently. So is a spot where
        either side, alone, normalizes to nothing: fillers only.

    """
    aligned = _Alignment.of(transcript, reference)
    spots: list[Spot] = []
    for start, stop in aligned.spans():
        spot = Spot(_words(aligned.transcript[start:stop]), _words(aligned.reference[start:stop]))
        said = [word.text for word in transcript[spot.transcript.start : spot.transcript.stop]]
        heard = [word.text for word in reference[spot.reference.start : spot.reference.stop]]
        if not _normalized(" ".join(said)) or not _normalized(" ".join(heard)):
            continue
        before, after = aligned.context(start, stop)
        if not _same(
            _normalized(" ".join([*before, *said, *after])),
            _normalized(" ".join([*before, *heard, *after])),
        ):
            spots.append(spot)
    return spots
