"""A local Parakeet TDT v3 transcript, from the pinned `parakeet-mlx` CLI.

Run through `uvx` rather than imported: its ML stack (mlx, numba, scipy,
scikit-learn) would join the project's own dependencies, which have to build
on every platform the flake targets, Linux included.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from scribe.errors import ExternalServiceError
from scribe.schema import Engine, FiniteFloat, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from scribe.schema import Source

PACKAGE = "parakeet-mlx"
VERSION = "0.5.2"
MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
# Dependencies as published by then: uvx otherwise resolves them afresh, and a
# dependency that moves can change the words.
EXCLUDE_NEWER = "2026-09-24T00:00:00Z"
# About a minute for a 45-minute meeting once cached; the first run also
# downloads the package and a ~1.2 GB model, which a slow link stretches.
DEFAULT_TIMEOUT_S = 3600.0
_OUTPUT_NAME = "transcript"
# Kept from each end of a long report: it names the file first and ends with the cause.
_EXCERPT_CHARS = 150
# Wide enough that the tool's console does not wrap a report line mid-word, and
# narrow enough that a progress bar drawn to this width stays small.
_COLUMNS = "1000"


# These three forbid extra fields though the schema is third-party: the version
# is pinned, so they are all it writes, and a new field means the pin moved
# without anyone checking this parse.
class _Token(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    start: FiniteFloat
    end: FiniteFloat
    duration: float
    confidence: float


class _Sentence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    start: float
    end: float
    duration: float
    confidence: float
    tokens: list[_Token]


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    sentences: list[_Sentence]


def _sentence_words(tokens: Sequence[_Token]) -> list[Word]:
    spans: list[tuple[str, float, float]] = []
    for token in tokens:
        # Tokens are sub-word pieces; a leading space marks a new word, and no
        # word spans two sentences.
        if token.text.startswith(" ") or not spans:
            spans.append((token.text.strip(), token.start, token.end))
        else:
            text, start, _ = spans[-1]
            spans[-1] = (text + token.text, start, token.end)
    return [Word(text=text, start=start, end=end) for text, start, end in spans if text]


def parse_output(raw: bytes) -> list[Word]:
    """Rebuild words from the JSON `parakeet-mlx --output-format json` writes.

    Raises:
        ExternalServiceError: `raw` is not the shape the pinned version writes.

    """
    try:
        parsed = _Output.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise ExternalServiceError(
            f"parakeet-mlx wrote an unexpected transcript: {first['msg']} "
            f"(at {where or 'top level'})"
        ) from exc
    clamped: list[Word] = []
    for sentence in parsed.sentences:
        for word in _sentence_words(sentence.tokens):
            # Overlapping sentences time a few words too early, in the right order: clamp, not sort.
            start = max(word.start, clamped[-1].start) if clamped else word.start
            clamped.append(Word(text=word.text, start=start, end=max(word.end, start)))
    return clamped


def _host() -> tuple[str, str]:
    return platform.system(), platform.machine()


def run_in_own_group(
    argv: list[str], *, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run `argv` to completion, killing its whole process tree on a timeout or interrupt."""
    # uvx runs the tool as its own child, so killing uvx alone would leave the
    # model running; a new session puts the whole tree in one process group.
    with subprocess.Popen(  # noqa: S603  # argv is built here from pinned values and a path
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        # BaseException: a Ctrl-C at the terminal no longer reaches a new session.
        except BaseException:
            with contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
            raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


# The part of scribe's environment a uvx child reads and needs. The child runs
# third-party code, so the keys scribe's own backends read stay out of it.
_CHILD_NAMES = frozenset(
    {
        # Where executables are (uv's interpreters, the tool's ffmpeg), the user, the locale.
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        # Temporary files, in the order Python's tempfile tries them.
        "TMPDIR",
        "TEMP",
        "TMP",
        # The roots uv and Hugging Face put their caches under, and uv its
        # interpreters and config: dropping one that is set downloads it all again.
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CONFIG_DIRS",
        # The CAs a download trusts.
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        # Older names uv and Hugging Face still read for a timeout and for offline.
        "HTTP_TIMEOUT",
        "TRANSFORMERS_OFFLINE",
        # Stops Hugging Face sending a saved token where none is needed; it holds no token.
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        # The tool's model cache. Its other PARAKEET_* settings override the
        # decoding and sentence defaults, which the recorded engine would then misdescribe.
        "PARAKEET_CACHE_DIR",
    }
)
# Downloads go through these; uv reads each name in either case.
_PROXIES = frozenset({"http_proxy", "https_proxy", "all_proxy", "no_proxy"})
# Settings under the tools' own prefixes: the locale's categories, uv's, Hugging
# Face's, and the GPU runtimes' (memory limits, CPU fallback).
_CHILD_PREFIXES = ("LC_", "UV_", "HF_", "HUGGINGFACE_", "MLX_", "PYTORCH_")
# A name holding one of these words holds a credential or says where one is,
# whatever its prefix.
_SECRET_WORDS = frozenset({"TOKEN", "PASSWORD", "SECRET", "KEY", "CREDENTIAL", "CREDENTIALS"})


def _needed(name: str) -> bool:
    if name in _CHILD_NAMES:
        return True
    if _SECRET_WORDS.intersection(name.upper().split("_")):
        return False
    return name.lower() in _PROXIES or name.startswith(_CHILD_PREFIXES)


def child_env(*granted: str) -> dict[str, str]:
    """Return what of this process's environment a uvx child needs, and the `granted` names."""
    return {key: value for key, value in os.environ.items() if key in granted or _needed(key)}


def _env() -> dict[str, str]:
    env = child_env()
    env["COLUMNS"] = _COLUMNS
    return env


def _excerpt(completed: subprocess.CompletedProcess[str]) -> str:
    # The tool reports its own failures on stdout, uvx its failures on stderr.
    shown = completed.stdout if completed.stdout.strip() else completed.stderr
    # Its closing line reports completion even after a failure.
    kept = [line for line in shown.splitlines() if " transcription complete. " not in line]
    excerpt = " ".join(" ".join(kept).split())
    if len(excerpt) > 2 * _EXCERPT_CHARS:
        excerpt = f"{excerpt[:_EXCERPT_CHARS]} ... {excerpt[-_EXCERPT_CHARS:]}"
    return f": {excerpt}" if excerpt else ""


class ParakeetMlx:
    """Transcription by `parakeet-mlx` on this machine, nothing uploaded."""

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
            run: Process runner, injectable so tests never spawn anything.
            which: PATH lookup. Default: `shutil.which`.
            host: Operating system and machine names, as `platform` reports them.
            timeout_s: Ceiling for the whole run, first-run downloads included.

        """
        self._run = run
        self._which = which
        self._host = host
        self.timeout_s = timeout_s

    def resolve(self) -> str:
        """Check this machine can run the tool, before anything is spawned.

        Returns:
            The `uvx` executable the run goes through.

        Raises:
            ExternalServiceError: this is not macOS on arm64, or `uvx` is not on PATH.

        """
        system, machine = self._host()
        if (system, machine) != ("Darwin", "arm64"):
            raise ExternalServiceError(
                f"parakeet-mlx needs Apple silicon (macOS on arm64); this is {system} {machine}"
            )
        # Resolved at call time, so a patched `shutil.which` is used.
        found = (self._which or shutil.which)("uvx")
        if found is None:
            raise ExternalServiceError("uvx is not on PATH; parakeet-mlx runs through it")
        return found

    def transcribe(self, audio: Path, *, source: Source) -> Transcript:
        """Transcribe `audio` and return its words as a transcript.

        Raises:
            ExternalServiceError: `resolve` failed, or the run failed, timed out,
                or left no readable transcript.

        """
        executable = self.resolve()
        with tempfile.TemporaryDirectory(prefix="scribe-parakeet-") as workdir:
            argv = [executable, "--exclude-newer", EXCLUDE_NEWER, "--from", f"{PACKAGE}=={VERSION}"]
            # Absolute, so an audio name that begins with "-" is not read as an option.
            argv += [PACKAGE, str(audio.absolute()), "--model", MODEL, "--output-format", "json"]
            argv += ["--output-dir", workdir, "--output-template", _OUTPUT_NAME]
            try:
                completed = self._run(argv, env=_env(), timeout=self.timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise ExternalServiceError(
                    f"parakeet-mlx did not finish within {self.timeout_s:g} s"
                ) from exc
            except OSError as exc:
                raise ExternalServiceError(f"cannot run uvx at {executable}: {exc}") from exc
            if completed.returncode != 0:
                raise ExternalServiceError(
                    f"parakeet-mlx exited {completed.returncode}{_excerpt(completed)}"
                )
            try:
                raw = (Path(workdir) / f"{_OUTPUT_NAME}.json").read_bytes()
            # It exits 0 when the transcription or the write failed, too.
            except OSError:
                raw = b""
            if not raw:
                raise ExternalServiceError(
                    f"parakeet-mlx exited 0 without writing a transcript{_excerpt(completed)}"
                )
        words = parse_output(raw)
        return Transcript(
            source=source,
            engine=Engine(
                name=PACKAGE,
                model=MODEL,
                params={"package_version": VERSION, "exclude_newer": EXCLUDE_NEWER},
            ),
            text=" ".join(word.text for word in words),
            words=words,
        )
