"""Gemini transcription of overlapping chunks, re-timed onto another engine's words."""

from __future__ import annotations

import base64
import bisect
import difflib
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import stamina
import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

from scribe.errors import AppError, ExternalServiceError, InputValidationError, SpendCapError
from scribe.schema import Engine, FiniteFloat, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from structlog.stdlib import BoundLogger

    from scribe.schema import Source

    Runner = Callable[..., subprocess.CompletedProcess[str]]

MODEL = "gemini-3.1-pro-preview"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
CHUNK_SECONDS = 600.0
STEP_SECONDS = 595.0
DEFAULT_MAX_USD = 3.0
RETRY_ATTEMPTS = 4
# Bounds visible output plus thinking tokens for one call.
MAX_OUT = 32768
# USD per 1M tokens for a prompt of at most 200k tokens; a 600 s chunk is about 15k.
PRICE_IN, PRICE_OUT = 2.00, 12.00
# Recorded usage of 17 chunk replies: 25 AUDIO tokens per second (15,000 per 600 s
# chunk) and 362 TEXT tokens of prompt per call.
AUDIO_TOKENS_PER_SECOND = 25
PROMPT_TOKENS = 362
# The same replies' mean output: 79,684 tokens over 9,381 s of chunk audio.
OUTPUT_TOKENS_PER_SECOND = 8.5
# Covers a 600 s chunk's 15,362 input tokens with room to spare.
_INPUT_BOUND_TOKENS = 25_000
# Spacing for a token past the first or last one aligned in its chunk.
_SECONDS_PER_UNALIGNED_TOKEN = 0.3
_EXCERPT_CHARS = 300

PROMPT = """This audio is a {dur:.0f}-second excerpt of a recorded business meeting (mono; several people, sometimes talking over each other).

Transcribe it verbatim. Include EVERY utterance, however short: one-word replies and backchannels such as "yeah", "okay", "mm-hm", "uh-huh", "right", "sure", "yes", "no", "got it", including ones spoken quietly or over another speaker. When a listener says "yeah" while someone else is talking, emit it as its own segment with its own start time and speaker, and continue the other speaker's speech in a new segment afterwards. Keep filler words (uh, um) and repetitions as spoken. Do not summarize, correct, or skip anything. Do not invent speech that is not in the audio.

Output JSON: {{"segments": [{{"start_seconds": <number>, "speaker": "<label>", "text": "<words>"}}, ...]}} in time order.
- "start_seconds" is the time the segment begins, as a count of plain seconds from the start of THIS audio excerpt (a decimal number). It is NOT minutes and seconds: 2 minutes 5.4 seconds is written 125.4 (never 205.4), and 9 minutes 5 seconds is 545.0 (never 905.0). Values range from 0 to {dur:.0f}.
- "speaker" is a consistent label for each distinct voice within this excerpt: "S1", "S2", "S3", ... A new segment starts whenever the speaker changes.
- Split long monologues into segments of at most about 30 words, each with its own start time."""  # noqa: E501  # the prompt is sent as written; wrapping it would change the request

SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_seconds": {"type": "NUMBER"},
                    "speaker": {"type": "STRING"},
                    "text": {"type": "STRING"},
                },
                "required": ["start_seconds", "speaker", "text"],
                "propertyOrdering": ["start_seconds", "speaker", "text"],
            },
        }
    },
    "required": ["segments"],
}

_APOSTROPHE = re.compile("\u2019")
_DASH_OR_SLASH = re.compile("[-\u2013\u2014/]")
_NOT_WORD = re.compile(r"[^\w\s']")

# DIVERGE: extra="ignore" on the response envelope, which carries fields this never
# reads (safety ratings, token details, model version) and gains new ones unannounced.
# The transcription inside it is held to its schema strictly below.
_ENVELOPE = ConfigDict(extra="ignore", frozen=True, alias_generator=to_camel)


class _Part(BaseModel):
    model_config = _ENVELOPE

    text: str = ""
    thought: bool = False


class _Content(BaseModel):
    model_config = _ENVELOPE

    parts: list[_Part] = Field(default_factory=list)


class _Candidate(BaseModel):
    model_config = _ENVELOPE

    content: _Content | None = None
    finish_reason: str | None = None


class _Usage(BaseModel):
    model_config = _ENVELOPE

    prompt_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int = 0


class _Reply(BaseModel):
    model_config = _ENVELOPE

    candidates: list[_Candidate] = Field(default_factory=list)
    usage_metadata: _Usage | None = None


class _Segment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    start_seconds: FiniteFloat
    speaker: str
    text: str


class _Segments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    segments: list[_Segment]


class _UnusableReplyError(ExternalServiceError):
    """A reply cut short, or not the JSON the schema asks for."""


@dataclass(frozen=True)
class Chunk:
    """One window of the audio, in seconds from its start."""

    offset: float
    duration: float


@dataclass(frozen=True)
class Preflight:
    """What a run will send, known before the first call."""

    duration: float
    chunks: tuple[Chunk, ...]
    expected_usd: float


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


def _usd(tokens_in: float, tokens_out: float) -> float:
    return tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT


WORST_CASE_USD = _usd(_INPUT_BOUND_TOKENS, MAX_OUT)


def resolve_api_key(env: Mapping[str, str] = os.environ) -> str:
    """Read the Gemini API key from `GEMINI_API_KEY`, stripped.

    Raises:
        InputValidationError: the variable is unset or empty.

    """
    key = env.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise InputValidationError("GEMINI_API_KEY is not set")
    return key


def plan_chunks(duration: float) -> list[Chunk]:
    """Split `duration` seconds into CHUNK_SECONDS windows starting every STEP_SECONDS.

    A tail no longer than the overlap gets no window of its own: the one before covers it.
    """
    chunks: list[Chunk] = []
    while len(chunks) * STEP_SECONDS < duration:
        offset = len(chunks) * STEP_SECONDS
        if chunks and duration - offset <= CHUNK_SECONDS - STEP_SECONDS:
            break
        chunks.append(Chunk(offset, min(CHUNK_SECONDS, duration - offset)))
    return chunks


def expected_usd(chunks: Sequence[Chunk]) -> float:
    """Estimate what transcribing `chunks` costs, from the recorded usage rates."""
    seconds = sum(chunk.duration for chunk in chunks)
    return _usd(
        seconds * AUDIO_TOKENS_PER_SECOND + len(chunks) * PROMPT_TOKENS,
        seconds * OUTPUT_TOKENS_PER_SECOND,
    )


class Spend:
    """Money committed to calls so far, checked against a cap before each one."""

    def __init__(self, cap_usd: float) -> None:
        """Start at nothing spent under `cap_usd`."""
        self.cap_usd = cap_usd
        self.committed_usd = 0.0

    def reserve(self, worst_usd: float) -> None:
        """Refuse an attempt whose worst case would take spending past the cap."""
        if self.committed_usd + worst_usd > self.cap_usd:
            raise SpendCapError(
                f"the next Gemini call's worst case of ${worst_usd:.2f} "
                f"would pass the --max-usd cap of ${self.cap_usd:.2f}"
            )

    def charge(self, usd: float) -> None:
        """Record what one attempt cost."""
        self.committed_usd += usd


def norm_tokens(text: str) -> list[str]:
    """Lowercase `text`, drop punctuation but apostrophes, and split it into tokens."""
    kept = _NOT_WORD.sub("", _DASH_OR_SLASH.sub(" ", _APOSTROPHE.sub("'", text.lower())))
    return [token.strip("'") for token in kept.split() if token.strip("'")]


def _token_time(
    index: int,
    pairs: Mapping[int, int],
    aligned: Sequence[int],
    starts: Sequence[float],
    offset: float,
) -> float:
    if index in pairs:
        return starts[pairs[index]]
    at = bisect.bisect_left(aligned, index)
    if 0 < at < len(aligned):
        before, after = aligned[at - 1], aligned[at]
        early, late = starts[pairs[before]], starts[pairs[after]]
        return early + (late - early) * (index - before) / (after - before)
    if at == 0 and aligned:
        return starts[pairs[aligned[0]]] - _SECONDS_PER_UNALIGNED_TOKEN * (aligned[0] - index)
    if aligned:
        return starts[pairs[aligned[-1]]] + _SECONDS_PER_UNALIGNED_TOKEN * (index - aligned[-1])
    return offset


def retime(
    chunks: Sequence[Chunk], replies: Sequence[Sequence[str]], anchor: Sequence[Word]
) -> list[Word]:
    """Time each chunk's reply by aligning its tokens to the anchor words in that window.

    A token aligned to an anchor token takes that anchor word's start; one between
    two aligned tokens is interpolated between them; one before the first or after
    the last is spaced 0.3 s per token outward; in a chunk with nothing aligned it
    takes the chunk's offset. Each chunk keeps the tokens timed inside its own half
    of the overlaps with its neighbors. Gemini's own segment times are not used.

    Args:
        chunks: The windows the replies answer, in order.
        replies: Each chunk's segment texts, in the order Gemini listed them.
        anchor: Word-timed words, in their own order, whose starts the tokens take.

    Returns:
        Words sorted by start, each a normalized lowercase token (`norm_tokens`)
        at a point in time (`end == start`), with no speaker.

    """
    anchor_tokens: list[str] = []
    anchor_starts: list[float] = []
    for word in anchor:
        for token in norm_tokens(word.text):
            anchor_tokens.append(token)
            anchor_starts.append(word.start)
    timed: list[tuple[float, str]] = []
    overlap = CHUNK_SECONDS - STEP_SECONDS
    for index, (chunk, texts) in enumerate(zip(chunks, replies, strict=True)):
        low = bisect.bisect_left(anchor_starts, chunk.offset - 1)
        high = bisect.bisect_right(anchor_starts, chunk.offset + chunk.duration + 1)
        starts = anchor_starts[low:high]
        tokens = [token for text in texts for token in norm_tokens(text)]
        matcher = difflib.SequenceMatcher(None, tokens, anchor_tokens[low:high], autojunk=False)
        pairs = {
            block.a + step: block.b + step
            for block in matcher.get_matching_blocks()
            for step in range(block.size)
        }
        aligned = sorted(pairs)
        first = chunk.offset + overlap / 2 if index else -math.inf
        # The last chunk has no neighbor to hand its tail to.
        last = chunk.offset + STEP_SECONDS + overlap / 2 if index < len(chunks) - 1 else math.inf
        for position, token in enumerate(tokens):
            start = _token_time(position, pairs, aligned, starts, chunk.offset)
            if first <= start < last:
                timed.append((start, token))
    timed.sort(key=lambda item: item[0])
    return [Word(text=token, start=start, end=start) for start, token in timed]


def _tool(run: Runner, argv: list[str]) -> str:
    try:
        done = run(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ExternalServiceError(f"cannot run {argv[0]}: {exc.strerror or exc}") from exc
    if done.returncode != 0:
        excerpt = " ".join(done.stderr[:_EXCERPT_CHARS].split())
        raise ExternalServiceError(f"{argv[0]} exited {done.returncode}: {excerpt}")
    return done.stdout


def probe_duration(audio: Path, run: Runner) -> float:
    """Return the audio's length in seconds, as ffprobe reports it."""
    argv = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]
    reported = _tool(run, [*argv, str(audio.absolute())]).strip()
    try:
        seconds = float(reported)
    except ValueError:
        seconds = math.nan
    # An infinite length would never finish being cut into chunks.
    if not (math.isfinite(seconds) and seconds > 0):
        raise ExternalServiceError(f"ffprobe reported no usable duration for {audio}: {reported!r}")
    return seconds


def _cut(audio: Path, chunks: Sequence[Chunk], directory: Path, run: Runner) -> list[Path]:
    paths: list[Path] = []
    for index, chunk in enumerate(chunks):
        path = directory / f"{index}.mp3"
        source = ["-ss", f"{chunk.offset:.3f}", "-t", f"{CHUNK_SECONDS:.3f}", "-i"]
        encoding = ["-ac", "1", "-ar", "16000", "-b:a", "64k"]
        # Absolute, so a colon in a relative name is not read as a protocol.
        argv = ["ffmpeg", "-nostdin", "-v", "error", "-y", *source, str(audio.absolute())]
        _tool(run, [*argv, *encoding, str(path)])
        paths.append(path)
    return paths


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == HTTPStatus.TOO_MANY_REQUESTS or code >= HTTPStatus.INTERNAL_SERVER_ERROR
    return isinstance(exc, httpx.TransportError)


def _texts(reply: _Reply) -> list[str]:
    candidate = reply.candidates[0] if reply.candidates else _Candidate()
    if candidate.finish_reason != "STOP":
        raise _UnusableReplyError(f"the reply ended with finishReason {candidate.finish_reason}")
    parts = [] if candidate.content is None else candidate.content.parts
    try:
        parsed = _Segments.model_validate_json("".join(p.text for p in parts if not p.thought))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise _UnusableReplyError(
            f"the reply is off its schema: {first['msg']} (at {where or 'top level'})"
        ) from exc
    return [segment.text for segment in parsed.segments]


class _Gemini:
    def __init__(self, api_key: str, spend: Spend, transport: httpx.BaseTransport | None) -> None:
        self._api_key = api_key
        self._spend = spend
        self._transport = transport

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[redacted]")

    def transcribe(self, paths: Sequence[Path], chunks: Sequence[Chunk]) -> list[list[str]]:
        replies: list[list[str]] = []
        try:
            with httpx.Client(
                base_url=BASE_URL,
                # A header, not the `key` query parameter, so no URL ever holds it.
                headers={"x-goog-api-key": self._api_key},
                timeout=httpx.Timeout(900.0, connect=30.0),
                transport=self._transport,
            ) as client:
                for index, (path, chunk) in enumerate(zip(paths, chunks, strict=True)):
                    label = f"chunk {index + 1} of {len(chunks)}"
                    replies.append(self._chunk(client, path.read_bytes(), chunk, label))
                    _logger().debug(
                        "gemini.chunk",
                        chunk=index + 1,
                        of=len(chunks),
                        spent_usd=round(self._spend.committed_usd, 4),
                    )
        except httpx.HTTPStatusError as exc:
            text = self._redact(exc.response.text)
            excerpt = " ".join(text[:_EXCERPT_CHARS].split())
            raise ExternalServiceError(
                f"Gemini failed with HTTP {exc.response.status_code}: {excerpt}"
            ) from exc
        except httpx.RequestError as exc:
            raise ExternalServiceError(self._redact(f"Gemini request failed: {exc}")) from exc
        except OSError as exc:
            raise ExternalServiceError(f"cannot read a cut chunk: {exc}") from exc
        return replies

    def _chunk(self, client: httpx.Client, audio: bytes, chunk: Chunk, label: str) -> list[str]:
        audio_part = {"mime_type": "audio/mpeg", "data": base64.b64encode(audio).decode()}
        body: dict[str, object] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": audio_part},
                        {"text": PROMPT.format(dur=chunk.duration)},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": SCHEMA,
                "maxOutputTokens": MAX_OUT,
                "thinkingConfig": {"thinkingLevel": "low"},
            },
        }
        try:
            return _texts(self._ask(client, body))
        except _UnusableReplyError as first:
            # Asked once more: the same request can come back whole.
            try:
                return _texts(self._ask(client, body))
            except _UnusableReplyError as second:
                raise ExternalServiceError(
                    f"Gemini's reply to {label} was unusable twice: {first}; then {second}"
                ) from second

    def _ask(self, client: httpx.Client, body: dict[str, object]) -> _Reply:
        # A context block, not a decorated function: stamina's retry hooks record a
        # decorated function's arguments, and these are the audio.
        for attempt in stamina.retry_context(
            on=_is_retryable, attempts=RETRY_ATTEMPTS, timeout=None, wait_initial=5.0, wait_max=60.0
        ):
            with attempt:
                return self._attempt(client, body)
        raise AssertionError("unreachable: stamina re-raises the last failure")  # pragma: no cover

    def _attempt(self, client: httpx.Client, body: dict[str, object]) -> _Reply:
        self._spend.reserve(WORST_CASE_USD)
        # Without a response or its usage, as on a timeout, the attempt may still be billed.
        cost = WORST_CASE_USD
        try:
            response = client.post(f"/models/{MODEL}:generateContent", json=body)
            if response.status_code != HTTPStatus.OK:
                # Google does not bill a request that failed with an HTTP error.
                cost = 0.0
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
            try:
                reply = _Reply.model_validate_json(response.content)
            except ValidationError as exc:
                raise ExternalServiceError("Gemini's response is not a reply envelope") from exc
            usage = reply.usage_metadata
            if usage is not None:
                cost = _usd(
                    usage.prompt_token_count,
                    usage.candidates_token_count + usage.thoughts_token_count,
                )
            return reply
        finally:
            self._spend.charge(cost)


def preflight(audio: Path, max_usd: float, run: Runner = subprocess.run) -> Preflight:
    """Plan the chunks of `audio` and refuse a run `max_usd` could not pay for.

    Runs ffprobe and nothing else, so a caller can learn this before any spending.

    Raises:
        InputValidationError: `max_usd` is not a finite amount above 0.
        SpendCapError: the expected cost, with room for the last call's worst
            case, passes `max_usd`.
        ExternalServiceError: ffprobe failed or reported no usable length.

    """
    # NaN and infinity pass every cap comparison, which would turn both guards off.
    if not (math.isfinite(max_usd) and max_usd > 0):
        raise InputValidationError(f"--max-usd must be a finite amount above 0, not {max_usd}")
    duration = probe_duration(audio, run)
    chunks = plan_chunks(duration)
    expected = expected_usd(chunks)
    # Every attempt needs room for its worst case, so a run that could spend its cap
    # on the other chunks would stop at the last one with nothing written.
    needed = expected - expected_usd(chunks[-1:]) + WORST_CASE_USD
    if needed > max_usd:
        raise SpendCapError(
            f"the expected cost of ${expected:.2f} plus room for the last call's worst case "
            f"needs ${needed:.2f}, over the --max-usd cap of ${max_usd:.2f}"
        )
    return Preflight(duration, tuple(chunks), expected)


def transcribe(
    audio: Path,
    anchor: Transcript,
    *,
    source: Source,
    api_key: str,
    max_usd: float,
    run: Runner = subprocess.run,
    transport: httpx.BaseTransport | None = None,
) -> Transcript:
    """Transcribe `audio` with Gemini in chunks, timed by `anchor`'s words.

    Args:
        audio: Audio file to cut into chunks with ffmpeg.
        anchor: Word-timed transcript of the same audio; only its words are used.
        source: Provenance to record for the audio.
        api_key: Gemini API key. Sent only in a request header; never logged.
        max_usd: Cap on the run's spending, checked before the first call against
            the expected cost and before every attempt against its worst case.
        run: Process runner for ffprobe and ffmpeg; tests inject a fake.
        transport: httpx transport override; tests inject `httpx.MockTransport`.

    Returns:
        A transcript of `retime`'s words, its cost in `engine.params`.

    Raises:
        InputValidationError: `max_usd` is not a finite amount above 0, or the
            anchor has no words.
        SpendCapError: the expected cost, or an attempt's worst case, passes `max_usd`.
        ExternalServiceError: ffprobe or ffmpeg failed, a request failed, or a
            chunk's reply was unusable twice.

    """
    if not anchor.words:
        raise InputValidationError("the --anchor transcript has no words to time against")
    planned = preflight(audio, max_usd, run)
    spend = Spend(max_usd)
    with tempfile.TemporaryDirectory(prefix="scribe-gemini-") as workdir:
        paths = _cut(audio, planned.chunks, Path(workdir), run)
        _logger().info(
            "gemini.preflight",
            chunks=len(planned.chunks),
            expected_usd=round(planned.expected_usd, 4),
        )
        try:
            replies = _Gemini(api_key, spend, transport).transcribe(paths, planned.chunks)
        except AppError as exc:
            raise type(exc)(f"{exc}; ${spend.committed_usd:.4f} spent") from exc
    words = retime(planned.chunks, replies, anchor.words)
    return Transcript(
        source=source,
        engine=Engine(
            name="gemini",
            model=MODEL,
            params={
                "chunk_seconds": CHUNK_SECONDS,
                "step_seconds": STEP_SECONDS,
                "cost_usd": round(spend.committed_usd, 4),
                "anchor_engine": anchor.engine.name,
            },
        ),
        duration=planned.duration,
        text=" ".join(word.text for word in words),
        words=words,
    )
