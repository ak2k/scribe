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


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


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
class Relabeling:
    """Per-word speaker ids after the pass, and how each chunk went."""

    speakers: tuple[int | None, ...]
    chunks: tuple[ChunkOutcome, ...]

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


def _tokens(texts: Sequence[str]) -> list[tuple[int, str]]:
    """(word index, key) for each token the prompt shows.

    `render` joins raw word texts, so a word holding a space reaches the model
    as two tokens, and an empty one as none.
    """
    return [(index, _key(token)) for index, text in enumerate(texts) for token in text.split()]


def _same_words(texts: Sequence[str], reply: Sequence[tuple[int | None, str]]) -> bool:
    """Whether the reply's words are `texts`' tokens, in order, as matching sees them."""
    return [key for _, key in _tokens(texts)] == [_key(word) for _, word in reply]


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
    tokens = _tokens(texts)
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


def prompts(
    texts: Sequence[str], ids: Sequence[int | None], spans: Sequence[tuple[int, int]]
) -> list[tuple[str, str]]:
    """Build the (system, user) prompt for each chunk.

    Context shows the input's labels, not an earlier chunk's corrected ones,
    so every chunk's call can run at once.
    """
    known = {speaker for speaker in ids if speaker is not None}
    listed = ", ".join(str(speaker) for speaker in sorted(known))
    system = _SYSTEM.format(
        ids=listed, unattributed=_UNATTRIBUTED if len(known) < len(set(ids)) else ""
    )
    built: list[tuple[str, str]] = []
    for start, end in spans:
        context_start = max(0, start - CONTEXT_WORDS)
        context = (
            render(texts[context_start:start], ids[context_start:start]) if start else _NO_CONTEXT
        )
        target = render(texts[start:end], ids[start:end])
        built.append((system, _USER.format(context=context, target=target)))
    return built


async def _ask_all(
    backend: SpeakerBackend, asked: Sequence[tuple[str, str]], concurrency: int
) -> list[Completion | Exception]:
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


def _warn_failed(index: int, reason: str, **logged: object) -> None:
    _logger().warning("speakers.chunk_failed", chunk=index, reason=reason, **logged)


def relabel(
    texts: Sequence[str],
    speakers: Sequence[int | None],
    backend: SpeakerBackend,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
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

    Returns:
        One id per word, drawn from `speakers`' ids, and each chunk's outcome.
        A chunk whose call failed or whose reply was unusable keeps its input
        ids, and a word whose id is None always keeps None.

    """
    if not needs_relabeling(speakers):
        return Relabeling(tuple(speakers), ())
    order = [speaker for speaker in dict.fromkeys(speakers) if speaker is not None]
    rank = {speaker: index for index, speaker in enumerate(order)}
    ids = [None if speaker is None else rank[speaker] for speaker in speakers]
    allowed = set(range(len(order)))
    spans = cut_points(texts)
    answers = anyio.run(_ask_all, backend, prompts(texts, ids, spans), concurrency)

    out = list(ids)
    chunks: list[ChunkOutcome] = []
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
        reply = parse_reply(answer.text)
        labels, aligned = align_labels(texts[start:end], ids[start:end], reply, allowed)
        reason = _unusable(answer, texts[start:end], reply, aligned)
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
        tuple(None if label is None else order[label] for label in out), tuple(chunks)
    )
