"""Local recognizers' readings of clips of one recording, for a vote at the pick's spots.

Run through `uvx` and `ear_worker.py` rather than imported, as the diarizer is:
torch and transformers would join the project's own dependencies, which have to
build on every platform the flake targets.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from scribe.diarizer import TOKEN_NAMES, Tools
from scribe.errors import EarError
from scribe.parakeet import TOKENLESS, child_env, run_in_own_group
from scribe.schema import FiniteFloat

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PACKAGE = "transformers"
VERSION = "5.18.0"
# The measured environment's own pins; Cohere's feature extractor needs librosa.
WITH = ("torch==2.14.1", "numpy==2.5.3", "librosa==1.0.0")
PYTHON = "3.12"
# Dependencies as published by then: uvx otherwise resolves them afresh, and a
# dependency that moves can change the words. The measured environment resolved
# at about this time, after transformers 5.18.0 was published at 16:46Z.
EXCLUDE_NEWER = "2026-09-30T21:00:00Z"
# About 1 s per clip per recognizer on Apple silicon once cached; a run with
# downloads allowed may first fetch ~8 GB of weights and packages.
DEFAULT_TIMEOUT_S = 3600.0
# How the worker marks the one line that names its own failure.
FAILED = "scribe-ear: "
# What `resolve` finds in each pinned snapshot before anything is spawned.
_WEIGHTS = ("config.json", "model.safetensors")
_OFFLINE = {"UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
_RATE = "16000"
# Kept from each end of a long report: uvx names the package first and ends with the cause.
_EXCERPT_CHARS = 150


@dataclass(frozen=True)
class Recognizer:
    """A local recognizer, pinned to the Hub snapshot the vote was measured with."""

    name: str
    model: str
    revision: str
    # Its worker alone is given the Hugging Face token; no other child needs it.
    gated: bool


RECOGNIZERS = (
    Recognizer(
        "cohere-transcribe",
        "CohereLabs/cohere-transcribe-03-2026",
        "b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
        gated=True,
    ),
    Recognizer(
        "qwen3-asr",
        "Qwen/Qwen3-ASR-1.7B-hf",
        "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        gated=False,
    ),
)


# Extra fields are refused: the worker ships with this module, so a field it
# writes and this does not read is a mismatch between the two.
class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    versions: dict[str, str]
    device: str
    dtype: str
    runtime_s: FiniteFloat
    texts: list[str]


@dataclass(frozen=True)
class Heard:
    """One recognizer's text for each clip, and what it ran on."""

    recognizer: Recognizer
    versions: dict[str, str]
    device: str
    dtype: str
    runtime_s: float
    texts: tuple[str, ...]


def parse_output(recognizer: Recognizer, raw: bytes, clips: int) -> Heard:
    """Read `recognizer`'s worker's OUT file, which must hold one text for each of `clips`.

    Raises:
        EarError: `raw` is not the worker's shape, or holds another number of texts.

    """
    try:
        parsed = _Output.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        # A key holding a line break would otherwise split this one-line cause.
        where = " ".join(".".join(str(part) for part in first["loc"]).split())
        raise EarError(
            f"{recognizer.name} wrote an unexpected answer: {first['msg']} "
            f"(at {where or 'top level'})"
        ) from exc
    if len(parsed.texts) != clips:
        raise EarError(f"{recognizer.name} heard {len(parsed.texts)} of {clips} clips")
    return Heard(
        recognizer,
        parsed.versions,
        parsed.device,
        parsed.dtype,
        parsed.runtime_s,
        tuple(parsed.texts),
    )


def _host() -> tuple[str, str]:
    return platform.system(), platform.machine()


def _hub_cache() -> Path:
    # Hugging Face's own lookup order, so this checks the cache the worker will read.
    home = os.getenv("HF_HOME", str(Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "huggingface"))
    hub = os.getenv("HF_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", str(Path(home) / "hub")))
    return Path(hub).expanduser()


def _excerpt(completed: subprocess.CompletedProcess[str]) -> str:
    # The worker's own line names the cause; warnings and progress bars would bury it.
    own = [line for line in completed.stderr.splitlines() if line.startswith(FAILED)]
    if own:
        return f": {own[-1].removeprefix(FAILED)}"
    shown = completed.stderr if completed.stderr.strip() else completed.stdout
    excerpt = " ".join(shown.split())
    if len(excerpt) > 2 * _EXCERPT_CHARS:
        excerpt = f"{excerpt[:_EXCERPT_CHARS]} ... {excerpt[-_EXCERPT_CHARS:]}"
    return f": {excerpt}" if excerpt else ""


class LocalEars:
    """Readings by each of RECOGNIZERS on this machine, nothing uploaded."""

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
            timeout_s: Ceiling for the decode and for each worker.

        """
        self._run = run
        self._which = which
        self._host = host
        self.timeout_s = timeout_s

    def resolve(self) -> Tools:
        """Check this machine can run every recognizer, spawning nothing.

        Raises:
            EarError: this is not macOS on arm64, `uvx` or ffmpeg is not on PATH,
                or a pinned snapshot is not in the Hugging Face cache or cannot be read.

        """
        system, machine = self._host()
        # Only Apple silicon's GPU was measured; elsewhere the words are untested.
        if (system, machine) != ("Darwin", "arm64"):
            raise EarError(
                f"the recognizers need Apple silicon (macOS on arm64); this is {system} {machine}"
            )
        # Resolved at call time, so a patched `shutil.which` is used.
        which = self._which or shutil.which
        uvx, ffmpeg = which("uvx"), which("ffmpeg")
        if uvx is None:
            raise EarError("uvx is not on PATH; the recognizers run through it")
        if ffmpeg is None:
            raise EarError("ffmpeg is not on PATH; the recognizers read the audio with it")
        hub = _hub_cache()
        for spec in RECOGNIZERS:
            snapshot = (
                hub / f"models--{spec.model.replace('/', '--')}" / "snapshots" / spec.revision
            )
            for name in _WEIGHTS:
                try:
                    cached = (snapshot / name).is_file()
                except OSError as exc:
                    raise EarError(
                        f"cannot look for {spec.model} at {spec.revision} in the Hugging Face "
                        f"cache: {exc}"
                    ) from exc
                if not cached:
                    raise EarError(
                        f"{spec.model} at {spec.revision} is not in the Hugging Face cache: "
                        f"no {name} under {snapshot}"
                    )
        return Tools(uvx, ffmpeg)

    def _spawn(
        self, argv: list[str], what: str, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = self._run(argv, env=env, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise EarError(f"{what} did not finish within {self.timeout_s:g} s") from exc
        except OSError as exc:
            raise EarError(f"cannot run {argv[0]}: {exc}") from exc
        if completed.returncode != 0:
            raise EarError(f"{what} exited {completed.returncode}{_excerpt(completed)}")
        return completed

    def hear(
        self,
        audio: Path,
        clips: Sequence[tuple[float, float]],
        *,
        max_new_tokens: int = 256,
        offline: bool = True,
    ) -> tuple[Heard, ...]:
        """Transcribe each of `clips` of `audio`, in seconds, by each recognizer in turn.

        Args:
            audio: The recording.
            clips: Start and end of each stretch, each transcribed on its own.
            max_new_tokens: Ceiling on the tokens each recognizer writes for one clip.
            offline: Use only what uv and Hugging Face have cached; download nothing.

        Raises:
            EarError: `resolve` failed, a working file could not be written, or the
                decode or a worker failed, timed out, or left no readable answer.

        """
        tools = self.resolve()
        try:
            # A directory left behind costs disk; failing on it would discard an answer in hand.
            scratch = tempfile.TemporaryDirectory(prefix="scribe-ear-", ignore_cleanup_errors=True)
        except OSError as exc:
            raise EarError(f"cannot make a working directory: {exc}") from exc
        with scratch as workdir:
            samples = Path(workdir) / "audio.f32"
            # Float samples, not the diarizer's 16-bit wav: the vote was measured on these.
            decode = [tools.ffmpeg, "-nostdin", "-v", "error", "-i", str(audio.absolute()), "-vn"]
            decode += ["-ac", "1", "-ar", _RATE, "-f", "f32le", str(samples)]
            self._spawn(decode, "ffmpeg", child_env(*TOKENLESS))
            return tuple(
                self._worker(tools.uvx, spec, samples, clips, max_new_tokens, offline=offline)
                for spec in RECOGNIZERS
            )

    def _worker(
        self,
        uvx: str,
        spec: Recognizer,
        samples: Path,
        clips: Sequence[tuple[float, float]],
        max_new_tokens: int,
        *,
        offline: bool,
    ) -> Heard:
        path, out = samples.with_name(f"{spec.name}.json"), samples.with_name(f"{spec.name}.out")
        request = {
            "audio": str(samples),
            "name": spec.name,
            "model": spec.model,
            "revision": spec.revision,
            "intervals": [list(clip) for clip in clips],
            "max_new_tokens": max_new_tokens,
        }
        try:
            path.write_text(json.dumps(request), encoding="utf-8")
        except OSError as exc:
            raise EarError(f"cannot write {spec.name}'s request: {exc}") from exc
        env = child_env(*(TOKEN_NAMES if spec.gated else TOKENLESS))
        if offline:
            env.update(_OFFLINE)
        pins = [uvx, "--python", PYTHON, "--exclude-newer", EXCLUDE_NEWER]
        pins += ["--from", f"{PACKAGE}=={VERSION}"]
        pins += [option for pin in WITH for option in ("--with", pin)]
        with resources.as_file(resources.files("scribe") / "ear_worker.py") as worker:
            # -P keeps the worker's own directory, which holds scribe's modules, off its path.
            argv = [*pins, "python", "-P", str(worker), str(path), str(out)]
            completed = self._spawn(argv, spec.name, env)
        try:
            raw = out.read_bytes()
        except OSError:
            raw = b""
        if not raw:
            raise EarError(f"{spec.name} exited 0 without writing an answer{_excerpt(completed)}")
        return parse_output(spec, raw, len(clips))
