"""Correct per-word speaker ids with an LLM, never touching the words.

The model gets each chunk of the transcript tagged with speaker ids and returns
it re-tagged. Its reply is only ever read for labels, and only when its words
are the chunk's words: same tokens, same order, compared case-, quote- and
punctuation-blind. Then each word takes the reply's label on its first token
when that label is one of the recording's ids, and keeps the label it came in
with otherwise. A reply that drops, adds, reorders or rewords a word fails its
chunk, which keeps its input labels: with the words out of step there is no
telling which of two identical lines a label was meant for. Words no speaker was
found for are shown to the model as `<spk:?>` and stay unattributed whatever it
replies.

Given the meeting's attendees, the same calls also ask where each chunk names
one of them; those lines come back unchecked, for `scribe.attendees` to verify.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import anyio
import anyio.to_thread
import structlog

from scribe.errors import AppError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from structlog.stdlib import BoundLogger

    from scribe.claude_cli import Completion

# Recorded beside every relabeled transcript: the pass is nondeterministic, so
# a result is only comparable to one made with the same prompt.
SPEAKER_PROMPT_VERSION = "tpst-2"
NAMES_PROMPT_VERSION = "names-1"
DEFAULT_SPEAKER_MODEL = "opus"

# The chunking the pass was measured with: at 700-900 words every reply copied
# its chunk word for word. Larger chunks are untested.
TARGET_WORDS = 700
MAX_WORDS = 900
CONTEXT_WORDS = 150
DEFAULT_CONCURRENCY = 6

_SENTENCE_ENDS = (".", "?", "!")
_TAG = re.compile(r"<spk:(\d+|\?)>")
_OUT = re.compile(r"<out>(.*)</out>", re.DOTALL)
_NAMES_BLOCK = re.compile(r"<names>(.*?)</names>", re.DOTALL)
_SPLIT = re.compile(r"(<spk:(?:\d+|\?)>)|\s+")
_NOT_WORD = re.compile(r"[^\w']")
# A reply cut off at the output limit can still hold a word-exact prefix, and
# its labels would pass for the model's verdict on the whole chunk.
_TRUNCATED = "max_tokens"

_SYSTEM = """\
You are correcting speaker labels in a machine transcript of a meeting.
The transcript words are correct. Some speaker labels are wrong: an automatic
diarizer sometimes keeps an answer with the person who asked the question, switches
speaker a few words too early or too late, or attaches a short backchannel
("Yeah.", "Right.", "Okay.") to the wrong person.

Each speaker turn starts with a tag like <spk:N>. Speaker ids in this call: {ids}.{unattributed}

Task: return the TARGET text with corrected speaker tags.
Rules:
- Copy every word of TARGET exactly, in the same order, with the same punctuation.
  Do not add, remove, fix, or reorder any word. Only move, add, or remove <spk:N> tags.
- Use only the speaker ids listed above. Do not invent new ids.
- Change a tag only when the conversation makes it clear: a question and its answer,
  "I" vs "you" references, a person addressed by name, a backchannel reply,
  a sentence that continues another speaker's thought. When unsure, keep the tag.
- What matters most is substantive speech (sentences, answers, explanations) carrying
  the right speaker. Short backchannels matter less; do not spend effort on them.
- CONTEXT is earlier text for reference only. Do not return it.
- Reply with the corrected TARGET text between <out> and </out>, and nothing else.
"""

_USER = """\
<context>
{context}
</context>

<target>
{target}
</target>
"""

_NO_CONTEXT = "(start of call)"
# Only added when the recording has such words, so a fully diarized recording
# gets exactly the prompt the pass was measured with.
_UNATTRIBUTED = (
    "\n<spk:?> marks words no speaker was detected for; leave every <spk:?> tag where it is."
)
# Swapped in, and the names section appended, only when attendees are given.
_REPLY_RULE = "- Reply with the corrected TARGET text between <out> and </out>, and nothing else.\n"
_NAMES_RULE = (
    "- Reply with the corrected TARGET text between <out> and </out>, "
    "then the <names> block described below, and nothing else.\n"
)
_NAMES = """\
Naming them does not change the rules for <spk:N> tags above.
After </out>, list each place in TARGET where one of these people is named, one line per \
place, between <names> and </names>:
NAME | SAID | KIND | QUOTE
- NAME: the person, written exactly as in the list above.
- SAID: the word or words in TARGET that name the person, copied as written there; speech \
recognition may have misspelled the name.
- KIND, from what the words show:
  next: the speaker addresses the person, who is expected to speak next;
  previous: the speaker addresses the person who spoke just before;
  self: the speaker names themself;
  about: any other mention.
- QUOTE: 4 to 12 consecutive words copied exactly from TARGET, including SAID.
List only people on the list, and only names said in TARGET, not in CONTEXT. When no one on \
the list is named, write <names></names>.
"""


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


class _PromptMismatchError(AppError):
    """The speaker prompt no longer holds the rule the names request replaces."""


class SpeakerBackend(Protocol):
    """Whatever answers one prompt; `ClaudeCliBackend` in production."""

    def complete(self, system: str, user: str) -> Completion:
        """Return the model's reply to one prompt.

        Raises:
            AppError: the backend failed, or replied unusably.

        """
        ...


@dataclass(frozen=True)
class ChunkOutcome:
    """What one chunk's call did, for the provenance sidecar."""

    index: int
    # Word offsets into the transcript, end exclusive.
    start: int
    end: int
    status: Literal["ok", "failed"]
    # Why the chunk failed: the backend's AppError class, "error: <class>" for
    # any other exception, or what made the reply unusable. None on success.
    reason: str | None = None
    stop_reason: str | None = None
    reply_words: int = 0
    aligned_words: int = 0
    relabeled: int = 0


@dataclass(frozen=True)
class Claim:
    """One line of a reply's names block, as the model wrote it and not yet checked.

    The fields are the line's `NAME | SAID | KIND | QUOTE`, trimmed; a line
    with fewer leaves the missing ones empty.
    """

    chunk: int
    # The chunk's word offsets into the transcript, end exclusive.
    start: int
    end: int
    name: str
    said: str
    kind: str
    quote: str


@dataclass(frozen=True)
class Relabeling:
    """Per-word speaker ids after the pass, and how each chunk went."""

    speakers: tuple[int | None, ...]
    chunks: tuple[ChunkOutcome, ...]
    # Asked for only with attendees, and read only from usable replies.
    claims: tuple[Claim, ...] = ()
    # Chunks whose usable reply held no names block after its <out> block.
    names_blocks_missing: tuple[int, ...] = ()

    @property
    def relabeled(self) -> int:
        """Words whose speaker the pass changed."""
        return sum(chunk.relabeled for chunk in self.chunks)

    @property
    def failed(self) -> tuple[int, ...]:
        """Indexes of the chunks whose call failed."""
        return tuple(chunk.index for chunk in self.chunks if chunk.status == "failed")


def needs_relabeling(speakers: Sequence[int | None]) -> bool:
    """Whether the pass could change anything: one speaker has nobody to trade with.

    Unattributed words don't count, since the pass never gives them a speaker.
    """
    return len(set(speakers) - {None}) > 1


def render(texts: Sequence[str], ids: Sequence[int | None]) -> str:
    """Tag words as the prompt shows them: one line per run, opened by `<spk:N>`.

    A run of unattributed words opens with `<spk:?>`.
    """
    out: list[str] = []
    for index, (text, speaker) in enumerate(zip(texts, ids, strict=True)):
        if not index or speaker != ids[index - 1]:
            tag = "?" if speaker is None else speaker
            out.append(("\n" if out else "") + f"<spk:{tag}>")
        out.append(text)
    return " ".join(out).replace(" \n", "\n")


def cut_points(
    texts: Sequence[str], *, target: int = TARGET_WORDS, max_words: int = MAX_WORDS
) -> list[tuple[int, int]]:
    """Split word offsets into chunks: at least `target` words, ended at a sentence end.

    A chunk runs to the first sentence end at or past its `target`-th word, or
    to `max_words` words when none comes first. A remainder of `max_words` or
    fewer is the last chunk whatever its length.

    Returns:
        (start, end) word offsets, end exclusive, covering every word once.

    """
    spans: list[tuple[int, int]] = []
    low = 0
    while low < len(texts):
        if len(texts) - low <= max_words:
            spans.append((low, len(texts)))
            break
        high = next(
            (
                index + 1
                for index in range(low + target - 1, low + max_words)
                if texts[index].endswith(_SENTENCE_ENDS)
            ),
            low + max_words,
        )
        spans.append((low, high))
        low = high
    return spans


def _key(word: str) -> str:
    """The form reply words are matched on: case, curly quotes and punctuation don't count."""
    folded = _NOT_WORD.sub("", word.lower().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'"))
    return folded or word


def _body(text: str) -> str:
    """The part of a reply its words are read from: the `<out>` block, or all of it."""
    found = _OUT.search(text)
    return found.group(1) if found else text


def parse_reply(text: str) -> list[tuple[int | None, str]]:
    """Read a reply as (label, word) pairs, one per word, tagged or not.

    Words before any tag and words under `<spk:?>` are labeled None: they move
    no label, but the word check and the alignment still see them.
    """
    label: int | None = None
    pairs: list[tuple[int | None, str]] = []
    for token in _SPLIT.split(_body(text)):
        # `split` yields None for the tag group wherever whitespace matched.
        if not token:
            continue
        tag = _TAG.fullmatch(token)
        if tag:
            label = None if tag.group(1) == "?" else int(tag.group(1))
        else:
            pairs.append((label, token))
    return pairs


def word_keys(texts: Sequence[str]) -> list[tuple[int, str]]:
    """(word index, key) for each token the prompt shows.

    `render` joins raw word texts, so a word holding a space reaches the model
    as two tokens, and an empty one as none.
    """
    return [(index, _key(token)) for index, text in enumerate(texts) for token in text.split()]


def text_keys(text: str) -> list[str]:
    """The keys of free text the model copied from the prompt, read as `word_keys` reads words.

    A `<spk:N>` tag counts as a space, so words copied across a run boundary
    still read as the words they are, even with the tag glued between them.
    """
    return [_key(token) for token in _TAG.sub(" ", text).split()]


def _same_words(texts: Sequence[str], reply: Sequence[tuple[int | None, str]]) -> bool:
    """Whether the reply's words are `texts`' tokens, in order, as matching sees them."""
    return [key for _, key in word_keys(texts)] == [_key(word) for _, word in reply]


def align_labels(
    texts: Sequence[str],
    labels: Sequence[int | None],
    reply: Sequence[tuple[int | None, str]],
    allowed: set[int],
) -> tuple[list[int | None], int]:
    """Carry the reply's labels onto the words they match; the words stay as given.

    A word takes the label on its first token, if that token matched. An
    unattributed word (label None) keeps None, whatever the reply says.

    Returns:
        One label per word in `texts`, and how many reply words matched a token.

    """
    tokens = word_keys(texts)
    matcher = difflib.SequenceMatcher(
        None, [key for _, key in tokens], [_key(word) for _, word in reply], autojunk=False
    )
    out = list(labels)
    aligned = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            label = reply[block.b + offset][0]
            aligned += 1
            token = block.a + offset
            word = tokens[token][0]
            first = not token or tokens[token - 1][0] != word
            if first and label is not None and label in allowed and out[word] is not None:
                out[word] = label
    return out, aligned


def _asking_names(system: str, attendees: Sequence[str]) -> str:
    """`system` with its reply rule asking for the names block too, and the block described.

    Raises:
        AppError: `system` lacks the reply rule, which would leave the block unasked for.

    """
    if _REPLY_RULE not in system:
        raise _PromptMismatchError("the speaker prompt has no reply rule to ask for names in")
    asked = system.replace(_REPLY_RULE, _NAMES_RULE)
    # Joined, never formatted, so a brace in a name is just a brace.
    return asked + "\nPeople at this meeting: " + ", ".join(attendees) + ".\n" + _NAMES


def prompts(
    texts: Sequence[str],
    ids: Sequence[int | None],
    spans: Sequence[tuple[int, int]],
    attendees: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Build the (system, user) prompt for each chunk.

    Context shows the input's labels, not an earlier chunk's corrected ones,
    so every chunk's call can run at once. With `attendees` the system prompt
    also asks, after the corrected text, where TARGET names any of them.

    Raises:
        AppError: the names request could not be added to the system prompt.

    """
    known = {speaker for speaker in ids if speaker is not None}
    listed = ", ".join(str(speaker) for speaker in sorted(known))
    system = _SYSTEM.format(
        ids=listed, unattributed=_UNATTRIBUTED if len(known) < len(set(ids)) else ""
    )
    if attendees:
        system = _asking_names(system, attendees)
    built: list[tuple[str, str]] = []
    for start, end in spans:
        context_start = max(0, start - CONTEXT_WORDS)
        context = (
            render(texts[context_start:start], ids[context_start:start]) if start else _NO_CONTEXT
        )
        target = render(texts[start:end], ids[start:end])
        built.append((system, _USER.format(context=context, target=target)))
    return built


async def ask_all(
    backend: SpeakerBackend, asked: Sequence[tuple[str, str]], concurrency: int
) -> list[Completion | Exception]:
    """Ask every (system, user) prompt, `concurrency` at a time, from worker threads.

    Returns:
        One answer per prompt, in order: its completion, or the Exception its
        call raised.

    """
    limiter = anyio.CapacityLimiter(concurrency)
    answers: dict[int, Completion | Exception] = {}

    async def ask(index: int, system: str, user: str) -> None:
        try:
            answers[index] = await anyio.to_thread.run_sync(
                backend.complete, system, user, limiter=limiter
            )
        # One chunk's failure costs that chunk's corrections, not the others'
        # paid answers; an interrupt or a cancellation is no Exception and still
        # stops the pass.
        except Exception as exc:  # noqa: BLE001  # contained per chunk, recorded as failed
            answers[index] = exc

    async with anyio.create_task_group() as group:
        for index, (system, user) in enumerate(asked):
            group.start_soon(ask, index, system, user)
    return [answers[index] for index in range(len(asked))]


def _unusable(
    answer: Completion,
    texts: Sequence[str],
    reply: Sequence[tuple[int | None, str]],
    aligned: int,
) -> str | None:
    """Why a reply can carry no correction, or None when it can.

    A reply that aligns yet moves no label is usable: it says the labels are right.
    """
    # In order: the first that holds names the failure.
    checks = (
        (answer.stop_reason == _TRUNCATED, "truncated_reply"),
        (not answer.text.strip(), "empty_reply"),
        ("<out>" not in answer.text, "no_out_block"),
        (_OUT.search(answer.text) is None, "unclosed_out_block"),
        (_TAG.search(_body(answer.text)) is None, "no_speaker_tags"),
        (not aligned, "no_words_aligned"),
        (not _same_words(texts, reply), "words_changed"),
    )
    return next((reason for failed, reason in checks if failed), None)


def _split_reply(text: str, texts: Sequence[str]) -> tuple[str, str]:
    """A reply up to the `</out>` closing exactly the chunk's words, and what follows that.

    A names line may quote `</out>`, which the greedy match would take for the
    block's end. Where no `</out>` closes the chunk's words, the reply is whole
    and nothing follows it.
    """
    for found in re.finditer("</out>", text):
        if _same_words(texts, parse_reply(text[: found.end()])):
            return text[: found.end()], text[found.end() :]
    return text, ""


def _claims(after: str, chunk: int, start: int, end: int) -> tuple[Claim, ...] | None:
    """The lines of the names block after a usable reply's <out> block; None without one."""
    found = _NAMES_BLOCK.search(after)
    if found is None:
        return None
    claims: list[Claim] = []
    for line in found.group(1).splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|", 3)]
        name, said, kind, quote = fields + [""] * (4 - len(fields))
        claims.append(Claim(chunk, start, end, name, said, kind, quote))
    return tuple(claims)


def _warn_failed(index: int, reason: str, **logged: object) -> None:
    _logger().warning("speakers.chunk_failed", chunk=index, reason=reason, **logged)


def relabel(
    texts: Sequence[str],
    speakers: Sequence[int | None],
    backend: SpeakerBackend,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    attendees: Sequence[str] = (),
) -> Relabeling:
    """Ask the backend to correct each chunk's speaker ids, all chunks at once.

    The prompt names speakers by rank of first appearance, not by diarization
    id, and a returned rank maps back to the id it stands for. Every rank in
    the recording is allowed in every chunk.

    Args:
        texts: Word texts in transcript order.
        speakers: One diarization id per word.
        backend: Answers each chunk's prompt; called from worker threads.
        concurrency: Calls in flight at most.
        attendees: People at the meeting; when given, each reply is also asked
            where its chunk names them.

    Returns:
        One id per word, drawn from `speakers`' ids, and each chunk's outcome.
        A chunk whose call failed or whose reply was unusable keeps its input
        ids, and a word whose id is None always keeps None. With `attendees`,
        the usable replies' names lines, and which usable replies had none.

    Raises:
        AppError: the names request could not be added to the prompt.

    """
    if not needs_relabeling(speakers):
        return Relabeling(tuple(speakers), ())
    order = [speaker for speaker in dict.fromkeys(speakers) if speaker is not None]
    rank = {speaker: index for index, speaker in enumerate(order)}
    ids = [None if speaker is None else rank[speaker] for speaker in speakers]
    allowed = set(range(len(order)))
    spans = cut_points(texts)
    answers = anyio.run(ask_all, backend, prompts(texts, ids, spans, attendees), concurrency)

    out = list(ids)
    chunks: list[ChunkOutcome] = []
    claims: list[Claim] = []
    missing: list[int] = []
    for index, ((start, end), answer) in enumerate(zip(spans, answers, strict=True)):
        if isinstance(answer, Exception):
            if isinstance(answer, AppError):
                reason = type(answer).__name__
                _warn_failed(index, reason, error=str(answer))
            else:
                # Its message may quote the call's input or output, so only the class is logged.
                reason = f"error: {type(answer).__name__}"
                _warn_failed(index, reason)
            chunks.append(ChunkOutcome(index, start, end, "failed", reason=reason))
            continue
        kept, after = (
            _split_reply(answer.text, texts[start:end]) if attendees else (answer.text, "")
        )
        reply = parse_reply(kept)
        labels, aligned = align_labels(texts[start:end], ids[start:end], reply, allowed)
        reason = _unusable(
            answer.model_copy(update={"text": kept}), texts[start:end], reply, aligned
        )
        if reason is not None:
            _warn_failed(index, reason, stop_reason=answer.stop_reason)
            chunks.append(
                ChunkOutcome(
                    index,
                    start,
                    end,
                    "failed",
                    reason=reason,
                    stop_reason=answer.stop_reason,
                    reply_words=len(reply),
                    aligned_words=aligned,
                )
            )
            continue
        out[start:end] = labels
        named = _claims(after, index, start, end) if attendees else ()
        if named is None:
            missing.append(index)
        else:
            claims.extend(named)
        chunks.append(
            ChunkOutcome(
                index,
                start,
                end,
                "ok",
                stop_reason=answer.stop_reason,
                reply_words=len(reply),
                aligned_words=aligned,
                relabeled=sum(
                    before != after for before, after in zip(ids[start:end], labels, strict=True)
                ),
            )
        )
    return Relabeling(
        tuple(None if label is None else order[label] for label in out),
        tuple(chunks),
        tuple(claims),
        tuple(missing),
    )
