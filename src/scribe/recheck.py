"""Re-read each span a fill filled or flagged, as a clip of its own, by every engine that can run.

The fill puts the reference's words where the transcript's engine heard
nothing, and flags spans where words stayed missing; neither record holds a
second opinion on those words. Here each span, PAD_S each side, is cut from
the recording and heard again: by xAI, with the model and terms of the
transcript, and by the local recognizers where this host can run them. Each
engine's reading is one vote on the span, kept beside the others for the
reader to weigh.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Json, TypeAdapter, ValidationError

from scribe.curated import fill_ranges
from scribe.ear import RECOGNIZERS
from scribe.errors import (
    AppError,
    EarError,
    ExternalServiceError,
    InputValidationError,
    ToolMissingError,
)
from scribe.gaps import run_ffmpeg
from scribe.schema import FiniteFloat, XaiResponse
from scribe.third_ear import PAD_S
from scribe.xai_stt import DEFAULT_MODEL, check_keyterms, check_vad_threshold

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from scribe.ear import Heard, Recognizer
    from scribe.gaps import Runner
    from scribe.schema import Transcript
    from scribe.xai_stt import XaiStt

XAI = "xai-stt"
# Bounds one span's request and its retries, so a meeting's spans finish
# within the hour its caller allows even when several stall.
SPAN_DEADLINE_S = 120.0
_KINDS: tuple[tuple[Literal["fill", "unresolved"], str], ...] = (
    ("fill", "fill_ranges"),
    ("unresolved", "fill_unresolved_ranges"),
)
_KEYTERMS = TypeAdapter(Json[list[str]])
# Fast speech runs near 4 words a second at under 2 tokens a word, so a clip's
# words fit with room to spare; the recognizers' own default of 256 was sized
# for clips of a few seconds.
TOKENS_PER_S = 8
_MIN_TOKENS = 256
Status = Literal["heard", "failed", "not run", "out of range"]


class Ears(Protocol):
    """The local recognizers, as `ear.LocalEars` runs them."""

    def resolve(self, recognizers: Sequence[Recognizer] = RECOGNIZERS, /) -> object:
        """Raise EarError when this host cannot run each of `recognizers`."""
        ...

    def hear(
        self,
        audio: Path,
        clips: Sequence[tuple[float, float]],
        /,
        *,
        max_new_tokens: int,
        recognizers: Sequence[Recognizer],
    ) -> tuple[Heard, ...]:
        """Read each of `clips` of `audio` with each of `recognizers`."""
        ...


@dataclass(frozen=True)
class Span:
    """A span the fill filled or flagged, and the stretch of audio it is heard in."""

    kind: Literal["fill", "unresolved"]
    start: float
    end: float
    # None for a span that starts at or past the recording's end: any clip of
    # it would hold only audio from before it.
    clip: tuple[float, float] | None


class HeardWord(BaseModel):
    """A word an engine heard in a clip, timed on the recording's clock."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    start: FiniteFloat
    end: FiniteFloat


class Vote(BaseModel):
    """One engine's reading of one span's clip, or why there is none."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: str
    model: str
    status: Status
    text: str | None = None
    # Only an engine that times its words gives them.
    words: list[HeardWord] | None = None
    reason: str | None = None


class CheckedSpan(BaseModel):
    """A span, the engine whose words the fill took there, and every engine's vote on it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["fill", "unresolved"]
    start: FiniteFloat
    end: FiniteFloat
    clip: tuple[FiniteFloat, FiniteFloat] | None
    proposed_by: str
    votes: list[Vote]


class Recheck(BaseModel):
    """What `scribe recheck` writes: the votes on every span of one transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_sha256: str
    source_sha256: str
    model: str
    pad_s: FiniteFloat
    local_max_new_tokens: int
    spans: list[CheckedSpan]


def find_spans(transcript: Transcript, path: Path) -> list[Span]:
    """Return every span the fill filled or flagged, in time order, each with its clip.

    A clip runs PAD_S past each end of its span, from no earlier than 0 to no
    later than the transcript's duration. A span starting at or past that has none.

    Raises:
        InputValidationError: a range is malformed or starts before the
            recording, or there are ranges and the transcript records no duration.

    """
    found: list[Span] = []
    for kind, key in _KINDS:
        for index, (start, end) in enumerate(fill_ranges(transcript.engine, path, key=key)):
            if start < 0:
                raise InputValidationError(
                    f"{path} has a malformed {key}: range {index} starts before the recording"
                )
            duration = transcript.duration
            if duration is None:
                raise InputValidationError(
                    f"{path} records no duration, so its spans cannot be placed in the recording"
                )
            clip = (
                None if start >= duration else (max(0.0, start - PAD_S), min(end + PAD_S, duration))
            )
            found.append(Span(kind, start, end, clip))
    return sorted(found, key=lambda span: (span.start, span.end, span.kind))


def _check_model(transcript: Transcript, path: Path) -> None:
    # The request's model is fixed here; a value read from a file never picks it.
    model = transcript.engine.model
    if model != DEFAULT_MODEL:
        named = "no model" if model is None else model
        raise InputValidationError(
            f"{path} was transcribed with {named}; recheck re-reads with {DEFAULT_MODEL} only"
        )


def _keyterms(transcript: Transcript, path: Path) -> list[str]:
    recorded = transcript.engine.params.get("keyterms")
    if recorded is None:
        return []
    try:
        terms = _KEYTERMS.validate_python(recorded)
    except ValidationError as exc:
        raise InputValidationError(
            f"{path} has malformed keyterms: {exc.errors()[0]['msg']}"
        ) from exc
    check_keyterms(terms)
    return terms


@dataclass(frozen=True)
class _Settings:
    """What the request that made the transcript sent besides its model and keyterms."""

    language: str | None
    format_text: bool
    vad_threshold: float


def _settings(transcript: Transcript, path: Path) -> _Settings:
    # Every transcript `scribe transcribe` writes records all three, so a missing
    # one is a file it did not write, and a guess would vote on other settings.
    params = transcript.engine.params
    language, format_text, threshold = (
        params.get(key) for key in ("language", "format", "vad_threshold")
    )
    if not isinstance(language, str):
        raise InputValidationError(f"{path} records no language, as text, for the re-read")
    if not isinstance(format_text, bool):
        raise InputValidationError(f"{path} records no format, true or false, for the re-read")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise InputValidationError(f"{path} records no vad_threshold, a number, for the re-read")
    check_vad_threshold(threshold)
    # Stored as sent: an empty language sends none, and format is its effective value.
    return _Settings(language or None, format_text, float(threshold))


def _proposer(transcript: Transcript, path: Path) -> str:
    named = transcript.engine.params.get("fill_reference")
    if not isinstance(named, str) or not named.strip():
        raise InputValidationError(
            f"{path} has fill ranges but no fill_reference naming its engine"
        )
    return named


def _cut(audio: Path, window: tuple[float, float], clip: Path, run: Runner) -> None:
    start, end = window
    timing = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}"]
    # Absolute, so a colon in a relative name is not read as a protocol.
    argv = ["ffmpeg", "-nostdin", "-v", "error", *timing, "-i", str(audio.absolute()), "-vn"]
    run_ffmpeg(run, [*argv, "-ac", "1", "-ar", "16000", "-b:a", "64k", str(clip)])


def _xai_vote(
    client: XaiStt,
    audio: Path,
    window: tuple[float, float],
    clip: Path,
    keyterms: Sequence[str],
    settings: _Settings,
    run: Runner,
) -> Vote:
    try:
        _cut(audio, window, clip, run)
        payload = client.transcribe(
            clip,
            model=DEFAULT_MODEL,
            language=settings.language,
            format_text=settings.format_text,
            diarize=False,
            keyterms=keyterms,
            vad_threshold=settings.vad_threshold,
        )
        try:
            reply = XaiResponse.model_validate(payload)
        except ValidationError as exc:
            raise ExternalServiceError(f"malformed xAI transcription response: {exc}") from exc
    except AppError as exc:
        reason = " ".join(str(exc).split())
        return Vote(engine=XAI, model=DEFAULT_MODEL, status="failed", reason=reason)
    offset = window[0]
    words = [
        HeardWord(text=word.text, start=word.start + offset, end=word.end + offset)
        for word in reply.words
    ]
    return Vote(engine=XAI, model=DEFAULT_MODEL, status="heard", text=reply.text, words=words)


def _token_cap(clips: Sequence[tuple[float, float]]) -> int:
    longest = max((end - start for start, end in clips), default=0.0)
    return max(_MIN_TOKENS, math.ceil(TOKENS_PER_S * longest))


def _local_votes(
    audio: Path, clips: Sequence[tuple[float, float]], ears: Ears, cap: int
) -> list[list[Vote]]:
    """Each clip's vote from each recognizer, each run on its own so a failure is its own."""
    by_engine: list[list[Vote]] = []
    for spec in RECOGNIZERS:
        try:
            ears.resolve((spec,))
        # OSError too: the cache lookup can fail on a file it cannot read.
        except (EarError, OSError) as exc:
            reason = " ".join(str(exc).split())
            missing = Vote(engine=spec.name, model=spec.model, status="not run", reason=reason)
            by_engine.append([missing] * len(clips))
            continue
        try:
            (heard,) = ears.hear(audio, clips, max_new_tokens=cap, recognizers=(spec,))
        # OSError too: the run's temporary directory can fail to be made.
        except (EarError, OSError) as exc:
            failed = Vote(
                engine=spec.name,
                model=spec.model,
                status="failed",
                reason=" ".join(str(exc).split()),
            )
            by_engine.append([failed] * len(clips))
            continue
        by_engine.append(
            [
                Vote(engine=spec.name, model=spec.model, status="heard", text=text)
                for text in heard.texts
            ]
        )
    return [list(votes) for votes in zip(*by_engine, strict=True)]


def _out_of_range(span: Span, duration: float | None) -> list[Vote]:
    end = "" if duration is None else f" at {duration:g} s"
    reason = f"starts at {span.start:g} s, at or past the recording's end{end}"
    engines = [(XAI, DEFAULT_MODEL), *((spec.name, spec.model) for spec in RECOGNIZERS)]
    return [
        Vote(engine=engine, model=model, status="out of range", reason=reason)
        for engine, model in engines
    ]


def recheck(
    transcript: Transcript,
    path: Path,
    audio: Path,
    client: XaiStt,
    ears: Ears,
    *,
    transcript_sha256: str,
    source_sha256: str,
    run: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> Recheck:
    """Collect every engine's vote on each span `transcript`'s fill filled or flagged.

    A span whose clip cannot be cut or whose request fails gets a failed
    vote, and the rest go on. Local recognizers this host cannot run get a
    `not run` vote on every span, and one that fails a `failed` vote of its
    own. Every engine's vote on a span past the recording's end is `out of
    range`, and nothing is sent for it.

    Args:
        transcript: A filled transcript, read from `path`.
        path: Where it was read from, to name in an error.
        audio: The recording, already checked to be the transcript's.
        client: Sends each clip to xAI.
        ears: The local recognizers.
        transcript_sha256: The transcript file's hash, recorded with the votes.
        source_sha256: The recording's hash, recorded with the votes.
        run: Runs ffmpeg.
        which: Finds ffmpeg on PATH.

    Raises:
        InputValidationError: the transcript is not of DEFAULT_MODEL, its
            ranges, fill_reference or keyterms are malformed, or its language,
            format or vad_threshold is missing or malformed.
        ToolMissingError: ffmpeg is not on PATH.

    """
    _check_model(transcript, path)
    spans = find_spans(transcript, path)
    keyterms = _keyterms(transcript, path)
    settings = _settings(transcript, path)
    proposer = _proposer(transcript, path) if spans else ""
    if which("ffmpeg") is None:
        raise ToolMissingError("ffmpeg is not on PATH; the clips are cut with it")
    clips = [span.clip for span in spans if span.clip is not None]
    cap = _token_cap(clips)
    xai: list[Vote] = []
    if clips:
        try:
            # A directory left behind costs disk; failing on it would discard the votes in hand.
            scratch = tempfile.TemporaryDirectory(
                prefix="scribe-recheck-", ignore_cleanup_errors=True
            )
        except OSError as exc:
            raise InputValidationError(f"cannot make a working directory: {exc}") from exc
        with scratch as workdir:
            xai = [
                _xai_vote(
                    client, audio, clip, Path(workdir, f"{index}.mp3"), keyterms, settings, run
                )
                for index, clip in enumerate(clips)
            ]
    heard = iter(zip(xai, _local_votes(audio, clips, ears, cap) if clips else [], strict=True))
    checked: list[CheckedSpan] = []
    for span in spans:
        if span.clip is None:
            votes = _out_of_range(span, transcript.duration)
        else:
            grok, others = next(heard)
            votes = [grok, *others]
        checked.append(
            CheckedSpan(
                kind=span.kind,
                start=span.start,
                end=span.end,
                clip=span.clip,
                proposed_by=proposer,
                votes=votes,
            )
        )
    return Recheck(
        transcript_sha256=transcript_sha256,
        source_sha256=source_sha256,
        model=DEFAULT_MODEL,
        pad_s=PAD_S,
        local_max_new_tokens=cap,
        spans=checked,
    )
