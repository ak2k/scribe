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

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
from scribe.xai_stt import DEFAULT_MODEL, check_keyterms

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from scribe.ear import Heard
    from scribe.gaps import Runner
    from scribe.schema import Transcript
    from scribe.xai_stt import XaiStt

    Hear = Callable[[Path, Sequence[tuple[float, float]]], tuple[Heard, ...]]

XAI = "xai-stt"
# Bounds one span's request and its retries, so a meeting's spans finish
# within the hour its caller allows even when several stall.
SPAN_DEADLINE_S = 120.0
_KINDS: tuple[tuple[Literal["fill", "unresolved"], str], ...] = (
    ("fill", "fill_ranges"),
    ("unresolved", "fill_unresolved_ranges"),
)
_KEYTERMS = TypeAdapter(Json[list[str]])


@dataclass(frozen=True)
class Span:
    """A span the fill filled or flagged, and the stretch of audio it is heard in."""

    kind: Literal["fill", "unresolved"]
    start: float
    end: float
    clip: tuple[float, float]


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
    status: Literal["heard", "failed", "not run"]
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
    clip: tuple[FiniteFloat, FiniteFloat]
    proposed_by: str
    votes: list[Vote]


class Recheck(BaseModel):
    """What `scribe recheck` writes: the votes on every span of one transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_sha256: str
    source_sha256: str
    model: str
    pad_s: FiniteFloat
    spans: list[CheckedSpan]


def find_spans(transcript: Transcript, path: Path) -> list[Span]:
    """Return every span the fill filled or flagged, in time order, each with its clip.

    A clip runs PAD_S past each end of its span, no earlier than 0 and, when
    the transcript's duration is known, no later than it.

    Raises:
        InputValidationError: a range is malformed or starts before the recording.

    """
    found: list[Span] = []
    for kind, key in _KINDS:
        for index, (start, end) in enumerate(fill_ranges(transcript.engine, path, key=key)):
            if start < 0:
                raise InputValidationError(
                    f"{path} has a malformed {key}: range {index} starts before the recording"
                )
            high = end + PAD_S
            if transcript.duration is not None:
                high = min(high, transcript.duration)
            found.append(Span(kind, start, end, (max(0.0, start - PAD_S), high)))
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


def _proposer(transcript: Transcript, path: Path) -> str:
    named = transcript.engine.params.get("fill_reference")
    if not isinstance(named, str) or not named.strip():
        raise InputValidationError(
            f"{path} has fill ranges but no fill_reference naming its engine"
        )
    return named


def _cut(audio: Path, span: Span, clip: Path, run: Runner) -> None:
    start, end = span.clip
    if end <= start:
        raise InputValidationError(f"span at {span.start:g} s is past the recording's end")
    timing = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}"]
    # Absolute, so a colon in a relative name is not read as a protocol.
    argv = ["ffmpeg", "-nostdin", "-v", "error", *timing, "-i", str(audio.absolute()), "-vn"]
    run_ffmpeg(run, [*argv, "-ac", "1", "-ar", "16000", "-b:a", "64k", str(clip)])


def _xai_vote(
    client: XaiStt, audio: Path, span: Span, clip: Path, keyterms: Sequence[str], run: Runner
) -> Vote:
    try:
        _cut(audio, span, clip, run)
        payload = client.transcribe(clip, model=DEFAULT_MODEL, diarize=False, keyterms=keyterms)
        try:
            reply = XaiResponse.model_validate(payload)
        except ValidationError as exc:
            raise ExternalServiceError(f"malformed xAI transcription response: {exc}") from exc
    except AppError as exc:
        reason = " ".join(str(exc).split())
        return Vote(engine=XAI, model=DEFAULT_MODEL, status="failed", reason=reason)
    offset = span.clip[0]
    words = [
        HeardWord(text=word.text, start=word.start + offset, end=word.end + offset)
        for word in reply.words
    ]
    return Vote(engine=XAI, model=DEFAULT_MODEL, status="heard", text=reply.text, words=words)


def _local_votes(audio: Path, spans: Sequence[Span], hear: Hear) -> list[list[Vote]]:
    try:
        heard = hear(audio, [span.clip for span in spans])
    # OSError too: the run's temporary directory can fail to be made.
    except (EarError, OSError) as exc:
        reason = " ".join(str(exc).split())
        missing = [
            Vote(engine=spec.name, model=spec.model, status="not run", reason=reason)
            for spec in RECOGNIZERS
        ]
        return [missing for _ in spans]
    return [
        [
            Vote(engine=ear.recognizer.name, model=ear.recognizer.model, status="heard", text=text)
            for ear, text in ((ear, ear.texts[index]) for ear in heard)
        ]
        for index in range(len(spans))
    ]


def recheck(
    transcript: Transcript,
    path: Path,
    audio: Path,
    client: XaiStt,
    hear: Hear,
    *,
    transcript_sha256: str,
    source_sha256: str,
    run: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> Recheck:
    """Collect every engine's vote on each span `transcript`'s fill filled or flagged.

    A span whose clip cannot be cut or whose request fails gets a failed
    vote, and the rest go on; local recognizers this host cannot run get a
    vote saying so on every span.

    Args:
        transcript: A filled transcript, read from `path`.
        path: Where it was read from, to name in an error.
        audio: The recording, already checked to be the transcript's.
        client: Sends each clip to xAI.
        hear: Reads all the clips of `audio` with the local recognizers.
        transcript_sha256: The transcript file's hash, recorded with the votes.
        source_sha256: The recording's hash, recorded with the votes.
        run: Runs ffmpeg.
        which: Finds ffmpeg on PATH.

    Raises:
        InputValidationError: the transcript is not of DEFAULT_MODEL, or its
            ranges, fill_reference or keyterms are malformed.
        ToolMissingError: ffmpeg is not on PATH.

    """
    _check_model(transcript, path)
    spans = find_spans(transcript, path)
    keyterms = _keyterms(transcript, path)
    proposer = _proposer(transcript, path) if spans else ""
    if which("ffmpeg") is None:
        raise ToolMissingError("ffmpeg is not on PATH; the clips are cut with it")
    checked: list[CheckedSpan] = []
    if spans:
        try:
            # A directory left behind costs disk; failing on it would discard the votes in hand.
            scratch = tempfile.TemporaryDirectory(
                prefix="scribe-recheck-", ignore_cleanup_errors=True
            )
        except OSError as exc:
            raise InputValidationError(f"cannot make a working directory: {exc}") from exc
        with scratch as workdir:
            xai = [
                _xai_vote(client, audio, span, Path(workdir, f"{index}.mp3"), keyterms, run)
                for index, span in enumerate(spans)
            ]
        local = _local_votes(audio, spans, hear)
        checked = [
            CheckedSpan(
                kind=span.kind,
                start=span.start,
                end=span.end,
                clip=span.clip,
                proposed_by=proposer,
                votes=[grok, *others],
            )
            for span, grok, others in zip(spans, xai, local, strict=True)
        ]
    return Recheck(
        transcript_sha256=transcript_sha256,
        source_sha256=source_sha256,
        model=DEFAULT_MODEL,
        pad_s=PAD_S,
        spans=checked,
    )
