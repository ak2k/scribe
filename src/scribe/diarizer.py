"""A local pyannote Community-1 diarization, and embeddings of stretches of the same audio.

Run through `uvx` and `diarize_worker.py` rather than imported: torch and
pyannote would join the project's own dependencies, which have to build on
every platform the flake targets.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from scribe.attribution import Speech
from scribe.errors import ExternalServiceError, ToolMissingError
from scribe.parakeet import TOKENLESS, child_env, run_in_own_group
from scribe.schema import FiniteFloat

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PACKAGE = "pyannote.audio"
VERSION = "4.0.7"
MODEL = "pyannote/speaker-diarization-community-1"
# The snapshot the rule was measured with; a moved `main` could change the clusters.
REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
# The measured environment's own pins; pyannote.audio 4.0.7 allows newer ones.
WITH = ("torch==2.14.0", "numpy==2.5.3", "soundfile==0.14.0")
# Its lock needs >=3.12,<3.14, and ran on 3.13.
PYTHON = "3.13"
# Dependencies as published by then: uvx otherwise resolves them afresh, and a
# dependency that moves can change the clusters.
EXCLUDE_NEWER = "2026-09-24T00:00:00Z"
# About 85 s per audio hour on Apple silicon once cached; a first run also
# downloads ~960 MB of packages and weights, which a slow link stretches.
DEFAULT_TIMEOUT_S = 3600.0
DIMENSION = 256
# Where the worker's Hugging Face client looks for the token the gated model
# needs. Only a worker loading a gated model is given these; a token saved on
# disk by `hf auth login` stays readable to any child.
TOKEN_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH")
# How the worker marks the one line that names its own failure.
FAILED = "scribe-diarize: "
_RATE = "16000"
# Kept from each end of a long report: uvx names the package first and ends with the cause.
_EXCERPT_CHARS = 150


# Extra fields are refused: the worker ships with this module, so a field it
# writes and this does not read is a mismatch between the two.
class _Segment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: FiniteFloat
    end: FiniteFloat
    speaker: str


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    versions: dict[str, str]
    device: str
    runtime_s: FiniteFloat
    exclusive: list[_Segment]
    embeddings: list[list[FiniteFloat] | None]


@dataclass(frozen=True)
class Diarization:
    """The worker's answer: the exclusive timeline, and one embedding or None per interval."""

    device: str
    runtime_s: float
    speech: tuple[Speech, ...]
    embeddings: tuple[tuple[float, ...] | None, ...]


def parse_output(raw: bytes, intervals: int) -> Diarization:
    """Read the worker's OUT file, which must answer `intervals` intervals.

    Raises:
        ExternalServiceError: `raw` is not the worker's shape, or its embeddings
            are not one per interval of DIMENSION values each.

    """
    try:
        parsed = _Output.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        # A key holding a line break would otherwise split this one-line cause.
        where = " ".join(".".join(str(part) for part in first["loc"]).split())
        raise ExternalServiceError(
            f"the diarizer wrote an unexpected answer: {first['msg']} (at {where or 'top level'})"
        ) from exc
    if len(parsed.embeddings) != intervals:
        raise ExternalServiceError(
            f"the diarizer embedded {len(parsed.embeddings)} of {intervals} intervals"
        )
    for index, vector in enumerate(parsed.embeddings):
        if vector is not None and len(vector) != DIMENSION:
            raise ExternalServiceError(
                f"the diarizer's embedding {index} has {len(vector)} values, not {DIMENSION}"
            )
    return Diarization(
        device=parsed.device,
        runtime_s=parsed.runtime_s,
        speech=tuple(Speech(part.start, part.end, part.speaker) for part in parsed.exclusive),
        embeddings=tuple(None if vector is None else tuple(vector) for vector in parsed.embeddings),
    )


def _host() -> tuple[str, str]:
    return platform.system(), platform.machine()


def _excerpt(completed: subprocess.CompletedProcess[str]) -> str:
    # The worker's own line names the cause; pyannote's advice on stdout, and
    # warnings on stderr, would bury it.
    own = [line for line in completed.stderr.splitlines() if line.startswith(FAILED)]
    if own:
        return f": {own[-1].removeprefix(FAILED)}"
    shown = completed.stderr if completed.stderr.strip() else completed.stdout
    excerpt = " ".join(shown.split())
    if len(excerpt) > 2 * _EXCERPT_CHARS:
        excerpt = f"{excerpt[:_EXCERPT_CHARS]} ... {excerpt[-_EXCERPT_CHARS:]}"
    return f": {excerpt}" if excerpt else ""


@dataclass(frozen=True)
class Tools:
    """The executables a run goes through."""

    uvx: str
    ffmpeg: str


class PyannoteDiarizer:
    """Diarization and embeddings by pyannote on this machine, nothing uploaded."""

    def __init__(
        self,
        run: Callable[..., subprocess.CompletedProcess[str]] = run_in_own_group,
        *,
        which: Callable[[str], str | None] | None = None,
        host: Callable[[], tuple[str, str]] = _host,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Hold the run settings; nothing is probed or spawned until called.

        Args:
            run: Process runner for ffmpeg and uvx, injectable so tests never spawn anything.
            which: PATH lookup. Default: `shutil.which`.
            host: Operating system and machine names, as `platform` reports them.
            timeout_s: Ceiling for each of the decode and the worker, first-run downloads included.

        """
        self._run = run
        self._which = which
        self._host = host
        self.timeout_s = timeout_s

    def resolve(self) -> Tools:
        """Check this machine can run the worker, before anything is spawned.

        Raises:
            ExternalServiceError: this is not macOS on arm64, or `uvx` is not on PATH.
            ToolMissingError: ffmpeg is not on PATH.

        """
        system, machine = self._host()
        # Only Apple silicon's GPU was measured; elsewhere the clusters are untested.
        if (system, machine) != ("Darwin", "arm64"):
            raise ExternalServiceError(
                f"the diarizer needs Apple silicon (macOS on arm64); this is {system} {machine}"
            )
        # Resolved at call time, so a patched `shutil.which` is used.
        which = self._which or shutil.which
        uvx = which("uvx")
        if uvx is None:
            raise ExternalServiceError("uvx is not on PATH; pyannote runs through it")
        ffmpeg = which("ffmpeg")
        if ffmpeg is None:
            raise ToolMissingError("ffmpeg is not on PATH; the diarizer reads the audio with it")
        return Tools(uvx, ffmpeg)

    def _spawn(
        self, argv: list[str], what: str, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = self._run(argv, env=env, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ExternalServiceError(
                f"{what} did not finish within {self.timeout_s:g} s"
            ) from exc
        except OSError as exc:
            raise ExternalServiceError(f"cannot run {argv[0]}: {exc}") from exc
        if completed.returncode != 0:
            raise ExternalServiceError(f"{what} exited {completed.returncode}{_excerpt(completed)}")
        return completed

    def diarize(self, audio: Path, intervals: Sequence[tuple[float, float]]) -> Diarization:
        """Diarize `audio` and embed each of `intervals` of it, in seconds.

        Raises:
            ExternalServiceError: `resolve` failed, the request could not be written,
                or the decode or the worker failed, timed out, or left no readable answer.

        """
        tools = self.resolve()
        # A directory left behind costs disk; failing on it would discard an answer in hand.
        with tempfile.TemporaryDirectory(
            prefix="scribe-diarize-", ignore_cleanup_errors=True
        ) as workdir:
            wav, request, out = (
                Path(workdir) / name for name in ("audio.wav", "in.json", "out.json")
            )
            # Absolute, so a colon in a relative name is not read as a protocol.
            decode = [tools.ffmpeg, "-nostdin", "-v", "error", "-i", str(audio.absolute()), "-vn"]
            self._spawn(
                [*decode, "-ac", "1", "-ar", _RATE, str(wav)], "ffmpeg", child_env(*TOKENLESS)
            )
            body = {
                "audio": str(wav),
                "model": MODEL,
                "revision": REVISION,
                "intervals": [list(interval) for interval in intervals],
            }
            try:
                request.write_text(json.dumps(body), encoding="utf-8")
            except OSError as exc:
                raise ExternalServiceError(f"cannot write the diarizer's request: {exc}") from exc
            pins = [tools.uvx, "--python", PYTHON, "--exclude-newer", EXCLUDE_NEWER]
            pins += ["--from", f"{PACKAGE}=={VERSION}"]
            pins += [option for pin in WITH for option in ("--with", pin)]
            with resources.as_file(resources.files("scribe") / "diarize_worker.py") as worker:
                # -P keeps the worker's own directory, which holds scribe's modules, off its path.
                argv = [*pins, "python", "-P", str(worker), str(request), str(out)]
                completed = self._spawn(argv, "the diarizer", child_env(*TOKEN_NAMES))
            try:
                raw = out.read_bytes()
            except OSError:
                raw = b""
            if not raw:
                raise ExternalServiceError(
                    f"the diarizer exited 0 without writing an answer{_excerpt(completed)}"
                )
            return parse_output(raw, len(intervals))
