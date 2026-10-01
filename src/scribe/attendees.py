"""Name speaker labels from the people a meeting's words address.

The speaker pass's model lists each place a chunk names an attendee, and how;
this module checks every such place against the words and decides, by count,
which label is which attendee. The model only classifies a mention: whether it
is in the words, which turn it points at and what that adds up to is decided
here.

A name off the list, or a listed name never said, names nothing. Whether a
spoken form ("Dimitri") is a listed name ("Dmitri") is the model's judgment,
recorded per mention as what was said.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from scribe.errors import InputValidationError
from scribe.speakers import text_keys, word_keys

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from scribe.schema import Word
    from scribe.speakers import Claim

# One pointer can be a misheard name, or a question someone else took.
MIN_POINTERS = 2
# A label must draw twice every other label's pointers, so one name can lead
# at most one label.
RIVAL_FACTOR = 2

# The run a mention of each kind points at, from the run it is said in.
_POINTS = {"next": 1, "previous": -1, "self": 0}
_KINDS = (*_POINTS, "about")

DropReason = Literal[
    "bad_line",
    "not_attendee",
    "bad_kind",
    "unlocated",
    "ambiguous",
    "said_outside_quote",
    "unattributed",
    "no_turn",
    "repeat",
]
# The reasons, None for none, of a mention found in the words: whoever may have
# said it is not the person it names. A repeat's run is counted already.
_SAID: tuple[DropReason | None, ...] = (None, "ambiguous", "no_turn", "unattributed")

# A name shaped like a label could not be told from an unnamed one downstream.
_LABEL = re.compile(r"speaker\s+(\d+|\?)", re.IGNORECASE)
# The names block is one `|`-separated line per mention, and the list is shown
# to the model inside the tagged prompt.
_FORBIDDEN = ("|", "<", ">")


def parse_attendees(text: str) -> tuple[str, ...]:
    """Read a comma-separated attendee list, each name trimmed and NFC-normalized, in order.

    Raises:
        InputValidationError: the list names nobody, an item is empty, a name
            repeats in any case or encoding, holds a line break, `|`, `<` or
            `>`, or reads like a speaker label.

    """
    if not text.strip():
        raise InputValidationError("--attendees names nobody")
    names = tuple(unicodedata.normalize("NFC", item.strip()) for item in text.split(","))
    seen: set[str] = set()
    for name in names:
        if not name:
            raise InputValidationError(f"--attendees has an empty name in {text!r}")
        if any(mark in name for mark in _FORBIDDEN) or name.splitlines() != [name]:
            raise InputValidationError(
                f"--attendees name {name!r} may not hold a line break, '|', '<' or '>'"
            )
        if _LABEL.fullmatch(name):
            raise InputValidationError(f"--attendees name {name!r} looks like a speaker label")
        if _folded(name) in seen:
            raise InputValidationError(f"--attendees lists {name!r} twice")
        seen.add(_folded(name))
    return names


def _folded(name: str) -> str:
    """The form names are compared in: an accent typed either way, and any case, match."""
    return unicodedata.normalize("NFC", name).casefold()


@dataclass(frozen=True)
class Mention:
    """One names line checked against the words: where it is and what it points at."""

    chunk: int
    # The attendee as listed, or the line's NAME where that is no attendee.
    name: str
    said: str
    kind: str
    # The word SAID starts at, and its start time; None where not located.
    word: int | None = None
    time: float | None = None
    # The speaker id of the run the mention is said in, and of the run it points at.
    by: int | None = None
    points_to: int | None = None
    reason: DropReason | None = None
    # Where the line fits more than one place, the speaker id of each place.
    by_one_of: tuple[int | None, ...] = ()

    @property
    def status(self) -> Literal["counted", "dropped"]:
        """Whether the mention takes part in the decision."""
        return "counted" if self.reason is None else "dropped"


@dataclass(frozen=True)
class SpeakerNames:
    """Which speaker id each attendee names, and what that was decided from."""

    names: Mapping[int, str]
    unassigned: tuple[str, ...]
    # Per attendee and id: counted mentions pointing at the id, and mentions
    # spoken, or that may be spoken, in the id's runs other than the speaker
    # naming themself.
    pointed: Mapping[str, Mapping[int, int]]
    says: Mapping[str, Mapping[int, int]]
    evidence: tuple[Mention, ...]


def _runs(speakers: Sequence[int | None]) -> tuple[list[int], list[int | None]]:
    """Each word's run and each run's id, over the maximal one-id runs turns are built from."""
    run_of: list[int] = []
    ids: list[int | None] = []
    for index, speaker in enumerate(speakers):
        if not index or speaker != speakers[index - 1]:
            ids.append(speaker)
        run_of.append(len(ids) - 1)
    return run_of, ids


def _line_fault(claim: Claim, attendee: str | None) -> DropReason | None:
    """What makes a line unusable before it is looked for in the words, if anything."""
    # In order: the first that holds names the drop.
    checks: tuple[tuple[bool, DropReason], ...] = (
        (not (claim.name and text_keys(claim.said) and text_keys(claim.quote)), "bad_line"),
        (attendee is None, "not_attendee"),
        (claim.kind not in _KINDS, "bad_kind"),
    )
    return next((reason for failed, reason in checks if failed), None)


def _locate(words: Sequence[Word], claim: Claim) -> list[int] | DropReason:
    """Each word SAID may start at, found through QUOTE among its chunk's words, or why none."""
    tokens = word_keys([word.text for word in words[claim.start : claim.end]])
    keys = [key for _, key in tokens]
    quote, said = text_keys(claim.quote), text_keys(claim.said)
    starts = [at for at in range(len(keys) - len(quote) + 1) if keys[at : at + len(quote)] == quote]
    if not starts:
        return "unlocated"
    offsets = [at for at in range(len(quote) - len(said) + 1) if quote[at : at + len(said)] == said]
    if not offsets:
        return "said_outside_quote"
    return sorted({claim.start + tokens[at + offset][0] for at in starts for offset in offsets})


def _checked(
    claim: Claim,
    words: Sequence[Word],
    listed: Mapping[str, str],
    runs: tuple[list[int], list[int | None]],
) -> Mention:
    """Everything one claim can be checked for on its own, before repeats are counted."""
    attendee = listed.get(_folded(claim.name))
    mention = Mention(claim.chunk, attendee or claim.name, claim.said, claim.kind)
    found = _line_fault(claim, attendee) or _locate(words, claim)
    if isinstance(found, str):
        return replace(mention, reason=found)
    run_of, ids = runs
    if len(found) > 1:
        # Whichever place was meant, its speaker may be the one saying the name.
        speakers = tuple(dict.fromkeys(ids[run_of[at]] for at in found))
        return replace(mention, reason="ambiguous", by_one_of=speakers)
    word = found[0]
    mention = replace(mention, word=word, time=words[word].start, by=ids[run_of[word]])
    if claim.kind not in _POINTS:
        return mention
    target = run_of[word] + _POINTS[claim.kind]
    if not 0 <= target < len(ids):
        return replace(mention, reason="no_turn")
    # Words nobody was found for may hold the answer, or the question: from them,
    # a pointer could name the asker's label, past the check that it never says the name.
    if ids[run_of[word]] is None or ids[target] is None:
        return replace(mention, reason="unattributed")
    return replace(mention, points_to=ids[target])


def _winner(pointed: Mapping[int, int], says: Mapping[int, int]) -> int | None:
    """The id enough mentions point at, well ahead of any other, whose own turns never say it."""
    return next(
        (
            speaker
            for speaker, count in pointed.items()
            if count >= MIN_POINTERS
            and not says.get(speaker)
            and all(
                count >= RIVAL_FACTOR * other
                for rival, other in pointed.items()
                if rival != speaker
            )
        ),
        None,
    )


def name_speakers(
    words: Sequence[Word],
    speakers: Sequence[int | None],
    claims: Sequence[Claim],
    attendees: Sequence[str],
) -> SpeakerNames:
    """Decide which attendee, if any, each speaker id is, from where the words name them.

    A claim counts once it names an attendee, is one of the known kinds, and
    its QUOTE occurs exactly once among its chunk's words with SAID once
    inside it. A claim that fits more than one place points at nothing, but
    counts against the speaker of every place it fits, any of whom may be
    the one saying the name. A `next` mention points at the run after the
    one it is said in, `previous` at the run before, `self` at its own run,
    and `about` at none. In a run, an attendee's first pointer at each id
    counts and so does their first other mention; later ones repeat. A
    pointer spoken in, or pointing at, words no speaker was found for does
    not count, nor does one with no run to point at, but either still
    counts against the speaker who said it. An id is named for an attendee
    when at least MIN_POINTERS counted mentions point at it, at least
    RIVAL_FACTOR times as many as at any other id, and none of its own turns
    names the attendee; an id two attendees win stays unnamed.

    Args:
        words: The transcript's words.
        speakers: The speaker id of each word, as turns will be built from.
        claims: The speaker pass's names lines.
        attendees: The names the claims may name, as given.

    Returns:
        The names by id, every attendee no id was named for, the tallies, and
        one checked mention per claim, in claim order.

    """
    listed = {_folded(name): name for name in attendees}
    runs = _runs(speakers)
    checked = [_checked(claim, words, listed, runs) for claim in claims]
    located = sorted(
        (mention.word, at)
        for at, mention in enumerate(checked)
        if mention.reason is None and mention.word is not None
    )
    # A run counts once per id pointed at and once for mentions, so neither a
    # mention nor a pointer hides a later pointer that disagrees with it.
    once: set[tuple[str, int, int | None]] = set()
    for word, at in located:
        key = (checked[at].name, runs[0][word], checked[at].points_to)
        if key in once:
            checked[at] = replace(checked[at], reason="repeat")
        once.add(key)

    pointed = {name: Counter[int]() for name in attendees}
    says = {name: Counter[int]() for name in attendees}
    for mention in checked:
        if mention.reason is None and mention.points_to is not None:
            pointed[mention.name][mention.points_to] += 1
        if mention.kind == "self" or mention.reason not in _SAID:
            continue
        for speaker in (mention.by, *mention.by_one_of):
            if speaker is not None:
                says[mention.name][speaker] += 1
    won: dict[int, list[str]] = {}
    for name in attendees:
        speaker = _winner(pointed[name], says[name])
        if speaker is not None:
            won.setdefault(speaker, []).append(name)
    names = {speaker: found[0] for speaker, found in won.items() if len(found) == 1}
    return SpeakerNames(
        names=names,
        unassigned=tuple(name for name in attendees if name not in names.values()),
        pointed={name: dict(counts) for name, counts in pointed.items()},
        says={name: dict(counts) for name, counts in says.items()},
        evidence=tuple(checked),
    )
