"""Pick, where two transcripts of one recording disagree, which reading was said.

A spot is where the transcript's words and a reference's differ in what was
said. The two are aligned token by token as the vote aligns them; a spot starts
as a maximal run of unmatched steps, runs MERGE_GAP matched steps apart or
fewer joined, and only a run holding tokens of both sides counts: words only
one side heard are the fill's to handle. It then widens until it holds whole
words on both sides. A spot whose readings match once numbers, contractions,
fillers, stutters, order, spacing and accents are set aside, or where one side
holds only fillers, is not a disagreement worth asking about.

A model is shown the transcript around each spot, both readings marked in
place, and picks one of the two or neither; it is never asked for words of its
own. A reply is used whole or not at all: one that breaks its format keeps the
transcript's words at every spot its chunk asked about.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal, Protocol

import anyio
import structlog
from pydantic import BaseModel, ConfigDict

from scribe.errors import AppError
from scribe.schema import Word
from scribe.speakers import DEFAULT_CONCURRENCY, SpeakerBackend, ask_all, cut_points, render
from scribe.spoken_numbers import digitize
from scribe.vote import FILLERS, align_words, first_decrease, norm_tokens

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from structlog.stdlib import BoundLogger

    from scribe.claude_cli import Completion
    from scribe.schema import Transcript

# Recorded beside every pick: the pass is nondeterministic, so a result is only
# comparable to one made with the same prompt.
PICK_PROMPT_VERSION = "pick-1"
DEFAULT_PICK_MODEL = "opus"
# About 50 spots a call on the meeting the pick was measured on.
TARGET_WORDS = 1500
MAX_WORDS = 2000
CONTEXT_WORDS = 200

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
# `digitize` leaves ordinals as written, so "1st" and "first" would differ.
_ORDINALS = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "ninth",
    "10th": "tenth",
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
    # Only the accents go: dropping every non-ASCII letter would make any two
    # Cyrillic or Greek readings equal.
    decomposed = unicodedata.normalize("NFKD", token.replace("'", ""))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _normalized(text: str) -> list[str]:
    """Return the tokens two readings are compared by: what was said, not how it was written."""
    tokens = norm_tokens(digitize(text.replace("%", " percent")))
    expanded = [
        part
        for token in tokens
        for part in _CONTRACTIONS.get(token, _ORDINALS.get(token, token)).split()
    ]
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


Side = Literal["transcript", "reference", "unsure", "failed"]
Label = Literal["A", "B", "unsure"]
# Why a reply is unusable, in the order they are checked: the first that holds.
Cause = Literal[
    "max_tokens",
    "no_out_block",
    "unclosed_out_block",
    "bad_json",
    "foreign_id",
    "repeated_id",
    "missing_id",
]

_SYSTEM = """\
You are checking a machine transcript of a recorded conversation. Two speech
recognizers transcribed the same audio; the text shows one recognizer's words,
which may themselves hold errors. Where the two heard something different, the
spot is marked in place as [#N A: <one reading> | B: <the other>], N being the
spot's id; which recognizer is A varies from spot to spot. Each speaker
turn starts with a tag like <spk:0>; <spk:?> marks words with no known speaker.

For each marked spot, decide which reading the speaker more likely said, from
the conversation: its topic, names and terms used elsewhere in it, grammar,
and what is said before and after. Either recognizer can be the one that is
right, and both often garble names and jargon. Both readings came from the
same audio, so each reflects real sounds.
- "A" or "B": the reading that fits the conversation better.
- "unsure" when neither fits better or you cannot tell.
Fillers, repeats and false starts alone never decide a spot. Never propose
other words. BEFORE and AFTER are neighboring text for reference; they hold
no spots.

Reply with one JSON object between <out> and </out>, and nothing else:
{"picks": [{"id": N, "pick": "A", "reason": "..."}]}
with one entry per marked spot, each id exactly once, "pick" one of "A", "B",
"unsure", and "reason" at most 15 words.
"""

# Kept symmetric, as the measured pickers were: it favors a name or term from
# the background, never one recognizer.
_BACKGROUND = (
    "Background on this recording: names of people at it and terms used in it. "
    "A reading that matches one of these names or terms, or sounds like one, is "
    "more likely right. Never add any of it to the transcript:\n"
)

_USER = """\
<before>
{before}
</before>

<target>
{target}
</target>

<after>
{after}
</after>
"""

_START = "(start of recording)"
_END = "(end of recording)"
_OUT = re.compile(r"<out>(.*)</out>", re.DOTALL)
_FENCE = re.compile(r"```[^\n]*\n(.*?)\n?```", re.DOTALL)
# A reply cut off at the output limit can still parse, with picks for only
# the spots it reached.
_TRUNCATED = "max_tokens"


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


class PickBackend(SpeakerBackend, Protocol):
    """Whatever answers one prompt, naming the model it asks for the provenance."""

    model: str


# Strict: an id of true or "1" would otherwise be read as spot 1.
class _Pick(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int
    pick: Label
    reason: str


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    picks: list[_Pick]


@dataclass(frozen=True)
class ChunkPick:
    """How one call went."""

    index: int
    # The ids of the spots it asked about.
    spots: tuple[int, ...]
    # Why its reply went unused: a Cause, the backend's AppError class, or
    # "error: <class>" for any other exception. None when it was used.
    reason: str | None = None


@dataclass(frozen=True)
class Picking:
    """The transcript with the picks applied, and what each spot and call came to."""

    transcript: Transcript
    spots: tuple[Spot, ...]
    # One per spot: whose reading it keeps, or "unsure" or "failed" for the
    # transcript's kept without a pick.
    picked: tuple[Side, ...]
    chunks: tuple[ChunkPick, ...]

    @property
    def failed(self) -> tuple[ChunkPick, ...]:
        """The calls whose reply went unused."""
        return tuple(chunk for chunk in self.chunks if chunk.reason is not None)


def system_prompt(context: str | None = None) -> str:
    """Build the system prompt, with a background section when `context` holds any text."""
    background = (context or "").strip()
    return f"{_SYSTEM}\n{_BACKGROUND}{background}\n" if background else _SYSTEM


def reference_first(spot_id: int, said: str, heard: str) -> bool:
    """Whether a spot shows the reference's reading as A: drawn per spot, the same on every run.

    Shown transcript first every time, a model unsure between the two could
    lean toward the transcript's reading.
    """
    # A str seed is hashed with SHA-512, so the draw is the same in every
    # process, which `hash()` is not.
    draw = random.Random(f"{spot_id}:{said}|{heard}").random()  # noqa: S311  # an order, not a secret
    return draw < 0.5  # noqa: PLR2004  # even odds


def chunk_spans(
    texts: Sequence[str],
    spots: Sequence[Spot],
    *,
    target: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
) -> list[tuple[int, int]]:
    """Split word offsets into calls' chunks, as `speakers.cut_points` does, cutting no spot.

    A cut inside a spot moves to the spot's first word, so the spot goes whole
    to the later chunk; a cut that would so leave a chunk with no words is
    dropped instead.

    Returns:
        (start, end) word offsets, end exclusive, covering every word once.

    """
    inside = {
        index: spot.transcript.start
        for spot in spots
        for index in range(spot.transcript.start + 1, spot.transcript.stop)
    }
    cuts = [0]
    for start, _ in cut_points(texts, target=target, max_words=max_words)[1:]:
        cut = inside.get(start, start)
        if cut > cuts[-1]:
            cuts.append(cut)
    return list(pairwise([*cuts, len(texts)]))


def _parsed(text: str) -> _Reply | Cause:
    if "<out>" not in text:
        return "no_out_block"
    found = _OUT.search(text)
    if found is None:
        return "unclosed_out_block"
    body = found.group(1).strip()
    fenced = _FENCE.fullmatch(body)
    try:
        return _Reply.model_validate_json(fenced.group(1) if fenced else body)
    # ValidationError is a ValueError; so is text UTF-8 cannot encode.
    except ValueError:
        return "bad_json"


def read_reply(answer: Completion, asked: Collection[int]) -> dict[int, Label] | Cause:
    """Read a reply's pick for each spot id asked, or the first cause that makes it unusable.

    The picks are the JSON object in the reply's `<out>` block, which may sit
    in one ``` fence, holding each id asked exactly once and no other.
    """
    reply: _Reply | Cause = (
        "max_tokens" if answer.stop_reason == _TRUNCATED else _parsed(answer.text)
    )
    if isinstance(reply, str):
        return reply
    ids = [each.id for each in reply.picks]
    # In order: the first that holds names the failure.
    checks: tuple[tuple[bool, Cause], ...] = (
        (not set(ids) <= set(asked), "foreign_id"),
        (len(set(ids)) < len(ids), "repeated_id"),
        (len(ids) < len(asked), "missing_id"),
    )
    found: Cause | None = next((cause for failed, cause in checks if failed), None)
    return {each.id: each.pick for each in reply.picks} if found is None else found


def _joined(words: Sequence[Word]) -> str:
    return " ".join(word.text for word in words)


def _target(words: Sequence[Word], start: int, end: int, marks: dict[int, tuple[int, str]]) -> str:
    """Render words [start, end), each spot as its mark, under its first word's speaker."""
    texts: list[str] = []
    ids: list[int | None] = []
    index = start
    while index < end:
        stop, mark = marks.get(index, (index + 1, words[index].text))
        texts.append(mark)
        ids.append(words[index].speaker)
        index = stop
    return render(texts, ids)


def _user(words: Sequence[Word], start: int, end: int, marks: dict[int, tuple[int, str]]) -> str:
    before, after = words[max(0, start - CONTEXT_WORDS) : start], words[end : end + CONTEXT_WORDS]
    return _USER.format(
        before=render([word.text for word in before], [word.speaker for word in before])
        if before
        else _START,
        target=_target(words, start, end, marks),
        after=render([word.text for word in after], [word.speaker for word in after])
        if after
        else _END,
    )


def _nearest(words: Sequence[Word], at: float) -> Word:
    # min keeps the first of equals, so a tie goes to the earlier word.
    return min(words, key=lambda word: min(abs(word.start - at), abs(word.end - at)))


def _apply(
    transcript: Sequence[Word],
    reference: Sequence[Word],
    spots: Sequence[Spot],
    picked: Sequence[Side],
) -> list[Word]:
    """Put the reference's words in place of the transcript's at every spot picked for them.

    A word put in keeps its text; its start is held between the starts of the
    nearest kept words on either side, its end no earlier than its start, and
    it takes the speaker of the replaced word nearest it in time.
    """
    chosen = [spot for spot, side in zip(spots, picked, strict=True) if side == "reference"]
    gone = {index for spot in chosen for index in spot.transcript}
    kept = [index for index in range(len(transcript)) if index not in gone]
    at = {spot.transcript.start: spot for spot in chosen}
    words: list[Word] = []
    index = 0
    while index < len(transcript):
        spot = at.get(index)
        if spot is None:
            words.append(transcript[index])
            index += 1
            continue
        after = bisect.bisect_left(kept, index)
        low = transcript[kept[after - 1]].start if after else -math.inf
        high = transcript[kept[after]].start if after < len(kept) else math.inf
        replaced = transcript[spot.transcript.start : spot.transcript.stop]
        for word in reference[spot.reference.start : spot.reference.stop]:
            start = min(max(word.start, low), high)
            words.append(
                Word(
                    text=word.text,
                    start=start,
                    end=max(word.end, start),
                    speaker=_nearest(replaced, word.start).speaker,
                )
            )
        index = spot.transcript.stop
    return words


def _raised(index: int, error: Exception) -> str:
    """Name what a call raised, logged without the call's text."""
    if isinstance(error, AppError):
        reason = type(error).__name__
        _logger().info("pick.chunk_failed", chunk=index, reason=reason, error=str(error))
        return reason
    # Its message may quote the call's input or output, so only the class is logged.
    reason = f"error: {type(error).__name__}"
    _logger().info("pick.chunk_failed", chunk=index, reason=reason)
    return reason


def _side(label: Label, *, reference_is_a: bool) -> Side:
    if label == "unsure":
        return "unsure"
    return "reference" if (label == "A") == reference_is_a else "transcript"


def pick_readings(
    transcript: Transcript,
    reference: Transcript,
    backend: PickBackend,
    *,
    context: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Picking:
    """Ask the backend which reading was said at each spot, and apply its picks.

    Args:
        transcript: The transcript whose words are kept except where a pick
            names the reference's reading.
        reference: Another transcript of the same audio.
        backend: Answers each chunk's prompt; called from worker threads.
        context: Background on the recording, such as who was at it; blank
            is none.
        concurrency: Calls in flight at most.

    Returns:
        The picking. Its transcript has the reference's words at each spot
        picked for them, `_apply`'s way, every other word as it was, its text
        rebuilt from its words and no turns; its engine params record the
        model, the prompt version, the reference, the counts, and each spot
        as [start, end, transcript words, reference words, picked], its start
        and end the transcript words' own. A chunk whose call raised or whose
        reply is unusable keeps the transcript's words at all its spots.

    Raises:
        ValueError: A word of either transcript starts before the word ahead of it.

    """
    for role, words in (("transcript", transcript.words), ("reference", reference.words)):
        if (index := first_decrease(words)) is not None:
            raise ValueError(f"{role} word {index} starts before word {index - 1}")
    said, heard = transcript.words, reference.words
    spots = find_spots(said, heard)
    readings = [
        (
            _joined(said[spot.transcript.start : spot.transcript.stop]),
            _joined(heard[spot.reference.start : spot.reference.stop]),
        )
        for spot in spots
    ]
    # A spot's id is its place in the transcript, from 1.
    orders = [reference_first(number, *pair) for number, pair in enumerate(readings, 1)]
    marks: dict[int, tuple[int, str]] = {}
    for number, (spot, (mine, theirs), flipped) in enumerate(
        zip(spots, readings, orders, strict=True), 1
    ):
        first, second = (theirs, mine) if flipped else (mine, theirs)
        marks[spot.transcript.start] = (
            spot.transcript.stop,
            f"[#{number} A: {first} | B: {second}]",
        )
    starts = [spot.transcript.start for spot in spots]
    spans = [
        (
            start,
            end,
            tuple(range(bisect.bisect_left(starts, start), bisect.bisect_left(starts, end))),
        )
        for start, end in chunk_spans([word.text for word in said], spots)
    ]
    # A chunk with no spot has nothing to ask.
    spans = [(start, end, tuple(index + 1 for index in held)) for start, end, held in spans if held]
    system = system_prompt(context)
    asked = [(system, _user(said, start, end, marks)) for start, end, _ in spans]
    answers = anyio.run(ask_all, backend, asked, concurrency)

    picked: list[Side] = ["failed"] * len(spots)
    chunks: list[ChunkPick] = []
    for index, ((_, _, ids), answer) in enumerate(zip(spans, answers, strict=True)):
        if isinstance(answer, Exception):
            chunks.append(ChunkPick(index, ids, _raised(index, answer)))
            continue
        labels = read_reply(answer, ids)
        if isinstance(labels, str):
            _logger().info(
                "pick.chunk_failed", chunk=index, reason=labels, stop_reason=answer.stop_reason
            )
            chunks.append(ChunkPick(index, ids, labels))
            continue
        for number in ids:
            picked[number - 1] = _side(labels[number], reference_is_a=orders[number - 1])
        chunks.append(ChunkPick(index, ids))

    words = _apply(said, heard, spots, picked)
    engine = reference.engine
    failed = [chunk for chunk in chunks if chunk.reason is not None]
    params = transcript.engine.params | {
        "pick_model": backend.model,
        "pick_prompt_version": PICK_PROMPT_VERSION,
        "pick_reference": engine.name if engine.model is None else f"{engine.name} {engine.model}",
        "pick_context_chars": len((context or "").strip()),
        "pick_spots": len(spots),
        "pick_to_reference": picked.count("reference"),
        "pick_unsure": picked.count("unsure"),
        "pick_failed": picked.count("failed"),
        "pick_chunks": len(chunks),
        "pick_chunks_failed": len(failed),
        "pick_record": json.dumps(
            [
                [
                    said[spot.transcript.start].start,
                    max(word.end for word in said[spot.transcript.start : spot.transcript.stop]),
                    mine,
                    theirs,
                    side,
                ]
                for spot, (mine, theirs), side in zip(spots, readings, picked, strict=True)
            ]
        ),
    }
    return Picking(
        transcript.model_copy(
            update={
                "engine": transcript.engine.model_copy(update={"params": params}),
                "text": _joined(words),
                "words": words,
                "turns": [],
            }
        ),
        tuple(spots),
        tuple(picked),
        tuple(chunks),
    )
