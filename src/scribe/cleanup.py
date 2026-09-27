"""Light-touch LLM cleanup of a diarized transcript.

The model is asked for a clean verbatim edit, never a rewrite, and returns each
turn under the id it was sent with. A chunk's reply is used only when it is
well formed: finished, nothing but one tag for each of the chunk's turns in
order, and holding words enough, even for a chunk of fillers alone. Its text is
then rebuilt from those tags, every label taken from the input. Any other reply
is set aside whole: its chunk keeps its input text and is reported. The result
is checked by code: every numeric token in the input has to survive.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from scribe.claude_cli import DEFAULT_MODEL, Completion
from scribe.errors import InputValidationError
from scribe.schema import Turn
from scribe.spoken_numbers import digitize

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime
    from pathlib import Path

    from scribe.schema import Engine, Source

DEFAULT_CLEANUP_MODEL = DEFAULT_MODEL
# Words per chunk, under the 8,000 the prior art found workable, leaving the reply
# and the carried-over tail room inside the same context.
DEFAULT_CHUNK_WORDS = 6000

# Recorded beside every cleaned transcript: the pass is nondeterministic, so a
# result is only comparable to one made with the same prompt.
CLEANUP_PROMPT_VERSION = "verbatim-3"

SYSTEM_PROMPT = """\
You are a professional transcript editor. You receive a raw automatic speech
recognition (ASR) transcript of a recorded conversation, meeting, call or talk,
and you return a clean verbatim version of it. Faithful words and the right
speaker matter more than polish.

Rules:
1. Clean verbatim, not a copy-edit: do not correct the speakers' grammar or
   word choice, and never rewrite or summarize. Keep the speakers' own words
   and their order. When in doubt, keep the original wording.
2. Always remove "um", "uh", "er" and stutters. Remove "you know", "like",
   "I mean", "sort of" and "kind of" only where they carry no meaning. Keep a
   hedge that qualifies a claim.
3. Remove a false start only when it adds no information; keep one that does.
   A complete sentence is not a false start.
4. If a passage makes no sense, leave its words as they were recognized, apart
   from the glossary substitutions in rule 7. Do not guess what was meant.
5. Add punctuation, sentence boundaries and paragraph breaks at natural
   transitions. Add nothing else: no headings, no bullets, no markup other
   than the turn tags described below.
6. Each turn's speaker is kept from its id: never write a speaker label, and
   never move words from one turn to another.
7. Apply the glossary substitutions below wherever the ASR garbled a name or a
   term.
8. Preserve every number, date, amount, percentage, time and identifier exactly
   as it appears, even when it looks implausible or wrong. Never round, convert
   or correct one.
9. Do not add, remove or rephrase substantive content, and never add commentary
   of your own.

The transcript arrives as turns, each written as <t id=N speaker="Label">text</t>,
with ids numbered from 1 across the whole recording. Return every turn exactly
once, in the order given, as <t id=N>cleaned text</t> with the same id. Never
merge two turns, split a turn, drop a turn or add one; a turn with no words left
after cleaning comes back as <t id=N></t>. Paragraph breaks inside a turn are
allowed. Text shown before the turns as context has no id: never return it or
any part of it.

Output only the tagged turns, with nothing outside the tags:
no preamble, no headings, no analysis."""

PRIOR_TAIL_HEADER = "(context from the previous section, already cleaned, do not repeat)"

# Below this, measured against the WHOLE transcript, the ratio test is noise: in
# a handful of words, fillers alone can account for the shortfall a truncated
# reply would show.
_RATIO_FLOOR_WORDS = 50
_TRUNCATION_RATIO = 0.6
_PRIOR_TAIL_PARAGRAPHS = 2
# Two paragraphs are enough to pick up mid-thought, but a turn with no blank
# lines is one paragraph, and a split monologue would then carry a whole chunk.
_PRIOR_TAIL_WORDS = 200

# One returned turn. Loose where a model varies without meaning anything (a
# quoted id, a copied speaker attribute) and strict where it matters: the text
# cannot run across another turn's opening tag, so an unclosed tag takes no
# other turn's words with it.
_TAG = re.compile(r'<t\s+id\s*=\s*"?(\d+)"?[^<>]*>((?:(?!<t[\s>]|</t>).)*)</t>', re.S)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_NON_SPACE = re.compile(r"\S+")
_NOT_ALNUM = re.compile(r"[\W_]+")
_BLANKS = re.compile(r"[ \t]+")
# A minus is a sign only where nothing but a space or "(" precedes it: between
# two numbers ("5-10", "2026-09-09") it joins a range or a date.
_SIGN = r"(?:(?<![^\s(])[-\N{MINUS SIGN}])"
# "C$" and "A$" are other currencies than "$", so a letter prefix stays in the
# token; `_normalize` folds "US$" back to "$".
_CURRENCY = r"(?:(?<![A-Za-z])[A-Z]{1,2}(?=\$))?[$€£] ?"
# A clock time first: otherwise the number branch matches "10" out of "10:30" and
# leaves ":30" behind. The optional spaces let a copy-edit insert one after a
# currency symbol or before a percent sign without it reading as a lost number.
_NUMERIC_TOKEN = re.compile(
    rf"\d{{1,2}}:\d{{2}}|{_SIGN}?(?:{_CURRENCY})?\d[\d,]*(?:\.\d+)?(?: ?%)?"
)
# Turns are joined with a blank line, so a newline in the gap is a turn boundary
# and a repeat across it is two values, not a stutter. A hyphen or slash is
# how cleanup writes a spoken "fifty fifty", which collapses the same way.
_STUTTER_GAP = re.compile(r"[ \t,/-]*")
_FRACTION_ZEROS = re.compile(r"(\.\d*?)0+(?=%?$)")
# A spoken "minus" signs the number after it ("minus five" is "-5"), except as
# subtraction after an amount ("$10 minus $5", "5% minus 3%") or in a tolerance
# ("plus or minus 5%"). Any other word before it ("zero or negative five") leaves
# it a sign. It runs after amount folding, so "ten dollars" already reads "$10"
# and one rule covers both sides.
_SPOKEN_SIGN = re.compile(
    r"(?<![\d%][ \t])"
    r"\b(?:(?<!\b(?i:plus)[ \t])(?<!\b(?i:plus[ \t]or)[ \t])(?i:minus)|(?i:negative))"
    rf"(?:[ \t]+|-)(?=(?:{_CURRENCY})?\d)"
)
# Cleanup may rewrite "2 million dollars" as "$2M" or "ten percent" as "10%";
# both sides fold to one form of the value ("$2000000", "10%") so a faithful
# rewrite compares equal and a changed value still differs.
_AMOUNT = re.compile(
    rf"(?P<sign>{_SIGN})?(?P<currency>{_CURRENCY})?(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"(?:(?P<suffix>[kK]|M|B|bn)(?![A-Za-z])|[ \t]+(?P<scale>(?i:thousand|million|billion))\b)?"
    r"(?:[ \t]+(?P<unit>(?i:dollars?|bucks|percent|per cent))\b)?"
)
_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
}


class CleanupRequest(BaseModel):
    """Everything the prompts are rendered from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turns: list[Turn]
    speaker_key: dict[str, str] = Field(default_factory=dict)
    glossary: dict[str, str] = Field(default_factory=dict)
    context: str | None = None


# Why a reply was set aside, one name per rule, in the order the rules are
# checked.
MalformedCause = Literal[
    "max_tokens",
    "outside_text",
    "foreign_id",
    "repeated_id",
    "missing_id",
    "out_of_order",
    "wordless",
    "short",
]


class MalformedChunk(BaseModel):
    """A chunk whose reply was set aside, with the first rule the reply broke."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: int
    cause: MalformedCause


class CleanResult(BaseModel):
    """The cleaned text and what producing it cost and risked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    completions: list[Completion]
    # Chunks whose reply was not well formed, in order: each keeps its input
    # text, uncleaned.
    malformed_chunks: list[MalformedChunk] = Field(default_factory=list)
    # Turns a well-formed reply returned empty, which leave no line.
    emptied_turns: int = 0
    # Speaker labels a well-formed reply wrote inside a turn, dropped: the
    # input names the speaker.
    stripped_labels: int = 0

    @property
    def chunks(self) -> int:
        """How many chunks were sent: one completion each."""
        return len(self.completions)

    @property
    def truncated_chunks(self) -> list[int]:
        """Indexes of the chunks whose reply was set aside."""
        return [item.chunk for item in self.malformed_chunks]


class NumberDiff(BaseModel):
    """Numeric tokens that did not survive cleanup, and ones it invented.

    Entries are normalized tokens ("$10000", "6:30"). `missing` holds values
    `after` lacks entirely, each repeated as often as `before` carried it;
    `reduced` holds, once each, values `after` keeps but fewer times;
    `added` holds each surplus copy `after` carries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    missing: list[str] = Field(default_factory=list)
    # Cleanup merging a restated phrase keeps the value once; that is a
    # fidelity signal, not a lost number.
    reduced: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    # How many the input carried. An empty `missing` over a `checked` of zero is
    # no evidence; over a count it is a check that ran and found nothing.
    checked: int = 0

    @property
    def drifted(self) -> bool:
        """Whether a number did not survive: one is missing, or one changed.

        A count drop alone is a merged restatement. A count drop beside a new
        value is one copy rewritten as another ("5 and 5" to "5 and 7"), so
        it fails like a missing value.
        """
        return bool(self.missing) or bool(self.reduced and self.added)


class CleanupBackend(Protocol):
    """One completion per chunk. `clean` needs nothing else from a model."""

    def complete(self, system: str, user: str) -> Completion:
        """Answer `user` under the instructions in `system`.

        Raises:
            ExternalServiceError: the backend failed, or replied unusably.

        """
        ...


def final_speakers(request: CleanupRequest) -> list[str]:
    """Labels the user prompt will carry, in order of first appearance."""
    return list(
        dict.fromkeys(request.speaker_key.get(turn.speaker, turn.speaker) for turn in request.turns)
    )


def render_system_prompt(request: CleanupRequest) -> str:
    """Build the system prompt: the rules, then the request's labeled sections.

    Args:
        request: Turns, relabel key, glossary and optional context.

    Returns:
        The prompt text. The relabel key is never handed over as a substitution
        to perform — only the final labels appear, because `render_user_prompt`
        has already applied the key.

    """
    sections = [SYSTEM_PROMPT]
    speakers = final_speakers(request)
    if speakers:
        listed = "\n".join(f"- {name}" for name in speakers)
        sections.append(f"Speakers, as the turns name them:\n{listed}")
    if request.glossary:
        listed = "\n".join(
            f'- "{wrong}" is written as "{right}"' for wrong, right in request.glossary.items()
        )
        sections.append(f"Glossary of ASR garbles to correct:\n{listed}")
    if request.context:
        sections.append(
            "Background on this recording, to disambiguate what you hear. Never add "
            f"any of it to the transcript:\n{request.context.strip()}"
        )
    return "\n\n".join(sections) + "\n"


def render_user_prompt(
    request: CleanupRequest, *, prior_tail: str | None = None, first_id: int = 1
) -> str:
    """Render each turn as `<t id=N speaker="Label">text</t>`, relabel key applied.

    Args:
        request: Turns to clean, with the relabel key to apply to them.
        prior_tail: Already-cleaned text closing the previous chunk, so the
            model can pick up mid-thought without being asked to repeat it. It
            carries no id, so nothing the reply says under an id can be it.
        first_id: Id of the first turn: ids number every turn of the recording
            from 1, so a later chunk starts where the one before it stopped.

    Returns:
        The user prompt text. Text is escaped as in XML, so no turn's words can
        open or close a tag, and a `"` in a label cannot end its attribute.

    """
    body = "\n\n".join(
        f'<t id={turn_id} speaker="{_escape(_label(request, turn), quote=True)}">'
        f"{_escape(turn.text)}</t>"
        for turn_id, turn in enumerate(request.turns, start=first_id)
    )
    if prior_tail:
        return f"{PRIOR_TAIL_HEADER}\n\n{_escape(prior_tail.strip())}\n\n{body}\n"
    return f"{body}\n"


def _label(request: CleanupRequest, turn: Turn) -> str:
    return request.speaker_key.get(turn.speaker, turn.speaker)


def _escape(text: str, *, quote: bool = False) -> str:
    # Only `"` needs escaping inside the attribute; escaping `'` there too would
    # show "O'Brien" to the model as "O&#x27;Brien".
    escaped = html.escape(text, quote=False)
    return escaped.replace('"', "&quot;") if quote else escaped


def _split_turn(turn: Turn, max_words: int) -> list[Turn]:
    """Cut a turn over `max_words` into pieces under it, each keeping the label."""
    if len(turn.text.split()) <= max_words:
        return [turn]
    pieces: list[list[str]] = [[]]
    for sentence in _SENTENCE_BREAK.split(turn.text.strip()):
        words = sentence.split()
        if len(pieces[-1]) + len(words) <= max_words:
            pieces[-1].extend(words)
            continue
        # A sentence over the ceiling on its own, or text with no sentence
        # punctuation at all, can only be cut between words.
        while len(words) > max_words:
            pieces.extend([words[:max_words], []])
            words = words[max_words:]
        pieces.append(words)
    return [turn.model_copy(update={"text": " ".join(piece)}) for piece in pieces if piece]


def _word_count(turns: Iterable[Turn]) -> int:
    return sum(len(turn.text.split()) for turn in turns)


def chunk_turns(turns: Sequence[Turn], *, max_words: int = DEFAULT_CHUNK_WORDS) -> list[list[Turn]]:
    """Group turns into chunks of at most `max_words` words.

    Args:
        turns: Turns in transcript order.
        max_words: Word ceiling per chunk.

    Returns:
        Chunks in transcript order, split on turn boundaries. A turn longer than
        `max_words` is first cut between sentences, or between words where a
        sentence alone is too long, into pieces that each keep its label.

    """
    chunks: list[list[Turn]] = []
    current: list[Turn] = []
    words = 0
    for turn in (piece for whole in turns for piece in _split_turn(whole, max_words)):
        length = len(turn.text.split())
        if current and words + length > max_words:
            chunks.append(current)
            current = []
            words = 0
        current.append(turn)
        words += length
    if current:
        chunks.append(current)
    return chunks


def label_prefix(paragraph: str, labels: Sequence[str]) -> str:
    """The `Label:` prefix opening `paragraph`'s first line, as written, or ""."""
    first_line = paragraph.split("\n", 1)[0]
    return first_line[: len(first_line) - len(strip_speaker_labels(first_line, labels))]


def _prior_tail(text: str, labels: Sequence[str]) -> str | None:
    paragraphs = [block.strip() for block in _PARAGRAPH_BREAK.split(text.strip()) if block.strip()]
    if not paragraphs:
        return None
    reply = "\n\n".join(paragraphs)
    first = max(len(paragraphs) - _PRIOR_TAIL_PARAGRAPHS, 0)
    start = sum(len(paragraph) + 2 for paragraph in paragraphs[:first])
    words = list(_NON_SPACE.finditer(reply, start))
    if len(words) > _PRIOR_TAIL_WORDS:
        start = words[-_PRIOR_TAIL_WORDS].start()
    line_start = reply.rfind("\n", 0, start) + 1
    own = label_prefix(reply[line_start:], labels)
    # Starting on or inside its own line's label, the tail keeps that label whole.
    if start < line_start + len(own):
        return reply[line_start:]
    # Only the first line of a turn is labeled, so a tail that opens mid-turn
    # takes the label of the last labeled line before it.
    label = own or next(
        (
            prefix
            for line in reversed(reply[:line_start].split("\n"))
            if (prefix := label_prefix(line, labels))
        ),
        "",
    )
    return f"{label.rstrip()} {reply[start:]}" if label else reply[start:]


def folded_words(text: str) -> list[tuple[str, int]]:
    """Each word lowercased to its letters and digits, with the offset it ends at.

    A copy-edit re-cases and re-punctuates, so only this much of a word says
    whether two passages are the same words. Punctuation alone folds to nothing
    and is not a word.
    """
    return [
        (folded, match.end())
        for match in _NON_SPACE.finditer(text)
        if (folded := _NOT_ALNUM.sub("", match.group().lower()))
    ]


@dataclass(frozen=True)
class _Reply:
    """A well-formed reply's text for each turn of its chunk, in input order."""

    # Labels dropped, trimmed; "" for a turn returned empty.
    texts: list[str]
    stripped_labels: int


def _read_reply(
    completion: Completion,
    chunk: Sequence[Turn],
    ids: range,
    labels: Sequence[str],
    *,
    apply_ratio: bool,
) -> _Reply | MalformedCause:
    # Nothing of a reply that breaks a rule is read: which of its words are
    # which turn's would be a guess, and a wrong guess loses or doubles speech.
    if completion.stop_reason == "max_tokens":
        return "max_tokens"
    if _TAG.sub("", completion.text).strip():
        return "outside_text"
    matches = list(_TAG.finditer(completion.text))
    broken = _id_rule([int(match.group(1)) for match in matches], ids)
    if broken is not None:
        return broken
    texts: list[str] = []
    stripped = 0
    for match in matches:
        # The input names the speaker. A label written inside the turn, read in
        # place, would pass for a change of speaker at that line, and one can
        # stand behind another, so labels go until none opens a line.
        text, count = html.unescape(match.group(2)), 1
        while count:
            text, count = _strip_labels(text, labels)
            stripped += count
        texts.append(_paragraphs(text))
    # A chunk of fillers alone is held to these too: a filler can be a glossary
    # term, and a chunk with no words to keep would accept any reply at all.
    words = sum(len(text.split()) for text in texts)
    if not words:
        return "wordless"
    if apply_ratio and words < _TRUNCATION_RATIO * _word_count(chunk):
        return "short"
    return _Reply(texts=texts, stripped_labels=stripped)


def _id_rule(returned: list[int], ids: range) -> MalformedCause | None:
    """The first id rule `returned` breaks: each of `ids` once, in order, and no other."""
    if any(turn_id not in ids for turn_id in returned):
        return "foreign_id"
    if len(set(returned)) < len(returned):
        return "repeated_id"
    if len(returned) < len(ids):
        return "missing_id"
    if returned != list(ids):
        return "out_of_order"
    return None


def _paragraphs(text: str) -> str:
    return "\n\n".join(block.strip() for block in _PARAGRAPH_BREAK.split(text) if block.strip())


def clean(
    request: CleanupRequest,
    backend: CleanupBackend,
    *,
    max_words: int = DEFAULT_CHUNK_WORDS,
) -> CleanResult:
    """Clean a transcript one chunk at a time.

    A chunk's text is rebuilt from its reply's turns, in input order and under
    the input's labels, only when the reply is well formed. Any other reply is
    set aside whole and the chunk keeps its input text, so no reply can lose or
    repeat a word of the input, or mix cleaned and uncleaned turns in a chunk.

    Args:
        request: Turns to clean, with the relabel key, glossary and context.
        backend: Model to send each chunk to.
        max_words: Word ceiling per chunk.

    Returns:
        The joined cleaned text, one `Completion` per chunk, each chunk whose
        reply was set aside with the first rule that reply broke, and what the
        well-formed replies returned empty or labeled. A set-aside chunk is
        reported, not retried: re-splitting it is the caller's decision.

    Raises:
        ExternalServiceError: the backend failed on a chunk.

    """
    system = render_system_prompt(request)
    labels = final_speakers(request)
    # A reply can write the label a turn had before the key renamed it.
    strip = [*labels, *(turn.speaker for turn in request.turns)]
    # Decided once for the transcript, never per chunk, so a small chunk size
    # cannot switch the check off for every chunk at once.
    apply_ratio = _word_count(request.turns) >= _RATIO_FLOOR_WORDS
    completions: list[Completion] = []
    malformed: list[MalformedChunk] = []
    emptied = 0
    stripped = 0
    lines: list[str] = []
    prior_tail: str | None = None
    first_id = 1
    for index, chunk in enumerate(chunk_turns(request.turns, max_words=max_words)):
        ids = range(first_id, first_id + len(chunk))
        first_id = ids.stop
        section = request.model_copy(update={"turns": chunk})
        completion = backend.complete(
            system, render_user_prompt(section, prior_tail=prior_tail, first_id=ids.start)
        )
        completions.append(completion)
        reply = _read_reply(completion, chunk, ids, strip, apply_ratio=apply_ratio)
        if isinstance(reply, _Reply):
            texts = reply.texts
            emptied += texts.count("")
            stripped += reply.stripped_labels
        else:
            malformed.append(MalformedChunk(chunk=index, cause=reply))
            texts = [_paragraphs(turn.text) for turn in chunk]
        chunk_lines = [
            f"{_label(request, turn)}: {text}"
            for turn, text in zip(chunk, texts, strict=True)
            if text
        ]
        lines += chunk_lines
        prior_tail = _prior_tail("\n\n".join(chunk_lines), labels)
    return CleanResult(
        text="\n\n".join(lines) + "\n",
        completions=completions,
        malformed_chunks=malformed,
        emptied_turns=emptied,
        stripped_labels=stripped,
    )


def _normalize(token: str) -> str:
    # Thousands separators, an inserted space and a trailing period are
    # typography, not value: "$ 1,234.56." and "$1234.56" have to compare equal.
    plain = token.replace(",", "").replace(" ", "").rstrip(".,")
    # Only the time branch produces a colon, and an hour keeps its value when a
    # copy-edit drops its leading zero.
    if ":" in plain:
        hour, _, minute = plain.partition(":")
        return f"{hour.lstrip('0') or '0'}:{minute}"
    sign = "-" if plain[0] in "-\N{MINUS SIGN}" else ""
    plain = plain.removeprefix(plain[0]) if sign else plain
    plain = plain.removeprefix("US") if plain.startswith("US$") else plain
    # Money and a percentage have no dotted-time reading, so "$5.00" is "$5";
    # a bare "10.30" keeps its zero, since it may be a time written with a dot.
    if not plain[0].isdigit() or plain.endswith("%"):
        plain = _FRACTION_ZEROS.sub(lambda zeros: zeros.group(1).rstrip("."), plain)
    return f"{sign}{plain}"


def _fold_amount(match: re.Match[str]) -> str:
    sign, currency, number, suffix, scale, unit = match.group(
        "sign", "currency", "number", "suffix", "scale", "unit"
    )
    if suffix is None and scale is None and unit is None:
        return match.group()
    multiplier = suffix or scale
    if multiplier is not None:
        value = Decimal(number.replace(",", "")) * _MULTIPLIERS[multiplier.lower()]
        number = f"{value.normalize():f}"
    unit = (unit or "").lower()
    if unit.startswith("per"):
        return f"{sign or ''}{currency or ''}{number}%"
    if unit:
        currency = currency or "$"
    return f"{sign or ''}{currency or ''}{number}"


def spoken_values(text: str) -> str:
    """Rewrite every value in `text` to the one form both sides are compared in."""
    # One blank between words, so the sign rule's fixed-width look-behinds see
    # "plus  or minus" as the tolerance it is.
    spaced = _BLANKS.sub(" ", digitize(text))
    return _SPOKEN_SIGN.sub("-", _AMOUNT.sub(_fold_amount, spaced))


def _numeric_tokens(text: str) -> list[str]:
    return _value_tokens(spoken_values(text))


def _value_tokens(digits: str) -> list[str]:
    tokens: list[str] = []
    previous_end = 0
    for match in _NUMERIC_TOKEN.finditer(digits):
        token = _normalize(match.group())
        # Speech disfluency repeats a value ("six, six, six"); cleanup removing
        # the stutter is not a dropped number, so a repeat counts once.
        stutter = (
            bool(tokens)
            and tokens[-1] == token
            and _STUTTER_GAP.fullmatch(digits, previous_end, match.start()) is not None
        )
        if not stutter:
            tokens.append(token)
        previous_end = match.end()
    return tokens


def verify_numbers(before: str, after: str) -> NumberDiff:
    """Compare the numeric tokens of two texts, with multiplicity.

    Args:
        before: Text as spoken.
        after: Text as cleaned.

    Returns:
        Tokens `before` carries that `after` lacks entirely (missing), tokens
        `after` keeps fewer times (reduced), and surplus tokens `after`
        carries (added), each in order of first appearance, with how many
        `before` carried in all. Two "10" in and one out reports "10"
        reduced; two in and none out reports two missing.

    """
    spoken = Counter(_numeric_tokens(before))
    written = Counter(_numeric_tokens(after))
    return NumberDiff(
        missing=[token for token in spoken.elements() if token not in written],
        reduced=[token for token in spoken if 0 < written[token] < spoken[token]],
        added=list((written - spoken).elements()),
        checked=sum(spoken.values()),
    )


def strip_speaker_labels(text: str, names: Iterable[str]) -> str:
    """Drop a leading `Label:` prefix from every line carrying one.

    A label is not speech: "Speaker 1" would otherwise put a "1" on the cleaned
    side of the number check that the spoken side never had.

    Args:
        text: Cleaned transcript text.
        names: Speaker labels to strip.

    Returns:
        The text with those prefixes removed.

    """
    return _strip_labels(text, names)[0]


def _strip_labels(text: str, names: Iterable[str]) -> tuple[str, int]:
    """`strip_speaker_labels`, with how many prefixes it removed."""
    # Longest first, so "Ann Lee" is stripped whole where "Ann" also matches.
    ordered = sorted(set(names), key=len, reverse=True)
    if not ordered:
        return text, 0
    label = "(?:" + "|".join(re.escape(name) for name in ordered) + ")"
    # Asterisks after the label are taken only when they close the ones that
    # opened it, before or after the colon: emphasis opening the speech is
    # speech. A model may also upper-case the label.
    forms = rf"(\*{{0,2}}){label}\1[ \t]*:|(\*{{1,2}}){label}[ \t]*:\2"
    return re.subn(rf"^[ \t]*(?:{forms})[ \t]*", "", text, flags=re.M | re.I)


def _yaml_scalar(value: str) -> str:
    # JSON is a subset of YAML 1.2, so json.dumps quotes and escapes exactly as a
    # double-quoted YAML scalar needs — without a YAML dependency.
    return json.dumps(value)


def provenance_header(
    *,
    title: str,
    source: Source,
    engine: Engine,
    backend_name: str,
    backend_model: str,
    speaker_key: dict[str, str],
    number_diff: NumberDiff,
    generated_at: datetime,
) -> str:
    """Render the YAML front matter that heads a cleaned transcript.

    Args:
        title: Human title for the document.
        source: Where the audio came from.
        engine: Which engine transcribed it.
        backend_name: Which cleanup backend edited it.
        backend_model: Which model that backend was configured with.
        speaker_key: Relabel key that was applied, recorded as given.
        number_diff: Result of the number check.
        generated_at: Timestamp to record; injected so callers can pin it.

    Returns:
        The front matter, delimiters included, ending in a newline.

    """
    lines = [
        "---",
        f"title: {_yaml_scalar(title)}",
        f"source_kind: {_yaml_scalar(source.kind)}",
        f"source_ref: {_yaml_scalar(source.ref)}",
        f"stt_engine: {_yaml_scalar(engine.name)}",
        f"stt_model: {'null' if engine.model is None else _yaml_scalar(engine.model)}",
        f"cleanup_backend: {_yaml_scalar(backend_name)}",
        f"cleanup_model: {_yaml_scalar(backend_model)}",
        f"cleanup_prompt_version: {_yaml_scalar(CLEANUP_PROMPT_VERSION)}",
        f"generated_at: {_yaml_scalar(generated_at.isoformat())}",
    ]
    if speaker_key:
        lines.append("speakers:")
        lines.extend(
            f"  {_yaml_scalar(raw)}: {_yaml_scalar(name)}" for raw, name in speaker_key.items()
        )
    else:
        lines.append("speakers: {}")
    lines.extend(
        [
            # A count, not a flag: zero says the clean bill below rests on
            # nothing, which a bare `true` over an empty document would hide.
            f"numbers_checked: {number_diff.checked}",
            f"numbers_missing: {json.dumps(number_diff.missing)}",
            f"numbers_reduced: {json.dumps(number_diff.reduced)}",
            f"numbers_added: {json.dumps(number_diff.added)}",
            "---",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_pairs(items: Iterable[str], option: str) -> dict[str, str]:
    """Parse `KEY=VALUE` strings into a mapping.

    Args:
        items: Raw `KEY=VALUE` strings.
        option: Option name to name in an error message.

    Returns:
        The mapping, later items winning.

    Raises:
        InputValidationError: an item carries no `=`, an empty key, or an
            empty value.

    """
    pairs: dict[str, str] = {}
    for item in items:
        # First `=` only: a replacement value may contain one.
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise InputValidationError(f"{option} expects KEY=VALUE, got {item!r}")
        pairs[key.strip()] = value.strip()
    return pairs


def read_pairs_file(path: Path, option: str) -> dict[str, str]:
    """Read a `KEY=VALUE` per line file, skipping blanks and `#` comments.

    Args:
        path: File to read.
        option: Option name to name in an error message.

    Returns:
        The mapping.

    Raises:
        InputValidationError: the file is unreadable, not UTF-8, or malformed.

    """
    try:
        # utf-8-sig: a byte order mark some editors write is not part of a key.
        raw = path.read_text(encoding="utf-8-sig")
    # ValueError too: a non-UTF-8 file raises UnicodeDecodeError, not OSError.
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"cannot read {option} {path}: {exc}") from exc
    lines = [line.strip() for line in raw.splitlines()]
    return parse_pairs([line for line in lines if line and not line.startswith("#")], option)
