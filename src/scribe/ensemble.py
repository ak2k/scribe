"""Three engines' transcripts of one recording, voted into one.

xAI's words are the backbone. Parakeet runs first because it is local and free:
a failed first-run download, or audio it hears nothing in, then costs nothing,
and its words both vote and time Gemini's.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import TYPE_CHECKING

from scribe import gemini_stt
from scribe.coverage import describe, fill_holes, moved, possible_drop
from scribe.errors import AppError, ExternalServiceError, InputValidationError
from scribe.outputs import plan_outputs, prove_writable, sibling
from scribe.vote import vote_transcripts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import httpx

    from scribe.gemini_stt import Runner
    from scribe.parakeet import ParakeetMlx
    from scribe.schema import Source, Transcript

_OUT = "--out"
_XAI = "the xAI transcript"
_PARAKEET = "the Parakeet transcript"
_GEMINI = "the Gemini transcript"
_SUFFIXES = {_XAI: ".xai.json", _PARAKEET: ".parakeet.json", _GEMINI: ".gemini.json"}


def _write(transcript: Transcript, path: Path) -> None:
    try:
        transcript.dump(path)
    # ValueError too: `model_dump_json` raises PydanticSerializationError, a
    # ValueError, on text that UTF-8 cannot encode.
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"cannot write transcript to {path}: {exc}") from exc


def _timed(step: Callable[[], Transcript], path: Path, kept: list[Path]) -> tuple[Transcript, str]:
    """Run one engine, write its transcript to `path`, and describe the run."""
    started = time.monotonic()
    transcript = step()
    _write(transcript, path)
    kept.append(path)
    return transcript, f"{len(transcript.words)} words in {time.monotonic() - started:.1f} s"


def transcribe_voted(
    audio: Path,
    source: Source,
    out: Path,
    inputs: Mapping[str, Path],
    *,
    xai: Callable[[Source], Transcript],
    parakeet: ParakeetMlx,
    max_usd: float,
    report: Callable[[str], None],
    run: Runner = subprocess.run,
    which: Callable[[str], str | None] | None = None,
    transport: httpx.BaseTransport | None = None,
    fill: bool = True,
) -> Path:
    """Transcribe `audio` with Parakeet, xAI and Gemini, and vote their words into `out`.

    Every check that can fail without spending runs first. Each engine's
    transcript is written beside `out` as soon as it finishes, named as
    `outputs.sibling` names it, so a failure later keeps what was paid for.

    Args:
        audio: The recording, already checked to be a regular file.
        source: Provenance recorded in every transcript written.
        out: Where the voted transcript goes.
        inputs: Files the run reads, by how a refusal describes them; no
            output may be one of them.
        xai: Runs xAI's transcription, whose words are the vote's backbone.
        parakeet: Local backend whose words vote and time Gemini's.
        max_usd: Cap on Gemini's spending.
        report: Takes one progress line per engine, then one for the vote;
            Gemini's also carries the cost its pre-flight expected, and the
            vote's what the fill did. Then one line per run of words the fill
            moved to Parakeet's times, and one per span it left unresolved,
            to listen to.
        run: Process runner for ffprobe and ffmpeg; tests inject a fake.
        which: PATH lookup. Default: `shutil.which`.
        transport: httpx transport for Gemini; tests inject `httpx.MockTransport`.
        fill: Fill the voted words' holes from Parakeet's, as `coverage.fill_holes` does.

    Returns:
        The path the voted transcript was written to.

    Raises:
        AppError: a check failed before any engine ran, or an engine failed;
            then the message names the transcripts already written.

    """
    gemini_key = gemini_stt.resolve_api_key()
    parakeet.resolve()
    for tool in ("ffprobe", "ffmpeg"):
        # Resolved at call time, so a patched `shutil.which` is used.
        if (which or shutil.which)(tool) is None:
            raise ExternalServiceError(f"{tool} is not on PATH; Gemini's audio is cut with it")
    outputs = {_OUT: out} | {label: sibling(out, suffix) for label, suffix in _SUFFIXES.items()}
    plan = plan_outputs(outputs, inputs)
    prove_writable(plan)
    planned = gemini_stt.preflight(audio, max_usd, run)
    kept: list[Path] = []
    try:
        anchor, done = _timed(
            lambda: parakeet.transcribe(audio, source=source), plan[_PARAKEET], kept
        )
        report(f"Parakeet: {done}")
        if not anchor.words:
            raise ExternalServiceError(
                f"Parakeet heard no words in {audio}, so Gemini's have nothing to be timed by"
            )
        backbone, done = _timed(lambda: xai(source), plan[_XAI], kept)
        report(f"xAI: {done}")
        gemini, done = _timed(
            lambda: gemini_stt.transcribe(
                audio,
                anchor,
                source=source,
                api_key=gemini_key,
                max_usd=max_usd,
                run=run,
                transport=transport,
            ),
            plan[_GEMINI],
            kept,
        )
        expected = round(planned.expected_usd, 4)
        report(f"Gemini: {done} for ${gemini.engine.params['cost_usd']} (expected ${expected})")
        voted = vote_transcripts(backbone, anchor, gemini)
        params = voted.engine.params
        filled = fill_holes(voted, anchor) if fill else None
        _write(voted if filled is None else filled[0], plan[_OUT])
    except AppError as exc:
        if not kept:
            raise
        raise type(exc)(f"{exc}; kept {', '.join(map(str, kept))}") from exc
    report(
        f"vote: {params['words_inserted']} words inserted "
        f"({params['words_inserted_unattributed']} with no speaker), "
        f"{params['words_substituted']} substituted"
        + ("" if filled is None else f"; {describe(filled[1])}")
    )
    for retimed in () if filled is None else filled[1].retimed:
        report(f"scribe: {moved(retimed)}")
    for span in () if filled is None else filled[1].unresolved:
        report(f"scribe: {possible_drop(span.start, span.end)}")
    return plan[_OUT]
