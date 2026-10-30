"""A local Parakeet TDT v3 transcript: `parakeet-mlx` on Apple silicon, `parakeet-cli` elsewhere.

parakeet-mlx runs through `uvx` rather than imported: its ML stack (mlx,
numba, scipy, scikit-learn) would join the project's own dependencies, which
have to build on every platform the flake targets, Linux included. Elsewhere
whisper.cpp's `parakeet-cli` runs the same model converted to ggml, which the
host provides, as it does the program.
"""

from __future__ import annotations

import contextlib
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from scribe.errors import ExternalServiceError, ToolMissingError
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
        # A key holding a line break would otherwise split this one-line cause.
        where = " ".join(".".join(str(part) for part in first["loc"]).split())
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
        # Opt-outs a user set on purpose, none holding a secret: of uv sending a
        # Hugging Face token, of telemetry.
        "UV_NO_HF_TOKEN",
        "DO_NOT_TRACK",
        "DISABLE_TELEMETRY",
        # Caps a user set on the CPU threads the numeric libraries start.
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        # The tool's model cache. Its other PARAKEET_* settings override the
        # decoding and sentence defaults, which the recorded engine would then misdescribe.
        "PARAKEET_CACHE_DIR",
    }
)
# Downloads go through these; uv reads each name in either case.
_PROXIES = frozenset({"http_proxy", "https_proxy", "all_proxy", "no_proxy"})
# Settings under the tools' own prefixes: the locale's categories, uv's, Hugging
# Face's, and the GPU runtimes' (memory limits, CPU fallback).
# UV_INDEX_URL passes with any password in it, as uv needs it; UV_INDEX_<NAME>_PASSWORD does not.
_CHILD_PREFIXES = ("LC_", "UV_", "HF_", "HUGGINGFACE_", "MLX_", "PYTORCH_")
# uv settings neither child uses that could hand the tool a secret: a file uv
# loads into the tool's environment, and where and as whom to upload a package.
_WITHHELD = ("UV_ENV_FILE", "UV_PUBLISH_")
# A name holding one of these words holds a credential, or says where or how to
# get one, whatever its prefix. An OIDC setting without the ID token it goes
# with makes Hugging Face's client fail every call.
_SECRET_WORDS = frozenset(
    {"TOKEN", "PASSWORD", "SECRET", "KEY", "CREDENTIAL", "CREDENTIALS", "OIDC"}
)
# Granted to each child that needs no Hugging Face token: the user's opt-out of
# sending a token on a call that names none. The TOKEN in its name keeps it from
# the diarizer's worker, which names none for its gated model and so, given
# this, would send none and be refused.
TOKENLESS = ("HF_HUB_DISABLE_IMPLICIT_TOKEN",)


def _needed(name: str) -> bool:
    if name in _CHILD_NAMES:
        return True
    if name.startswith(_WITHHELD) or _SECRET_WORDS.intersection(name.upper().split("_")):
        return False
    return name.lower() in _PROXIES or name.startswith(_CHILD_PREFIXES)


def child_env(*granted: str) -> dict[str, str]:
    """Return what of this process's environment a uvx child needs, and the `granted` names."""
    return {key: value for key, value in os.environ.items() if key in granted or _needed(key)}


def _env() -> dict[str, str]:
    env = child_env(*TOKENLESS)
    env["COLUMNS"] = _COLUMNS
    return env


def _excerpt(completed: subprocess.CompletedProcess[str]) -> str:
    # The tool reports its own failures on stdout, uvx its failures on stderr.
    shown = completed.stdout if completed.stdout.strip() else completed.stderr
    # Its closing line reports completion even after a failure.
    kept = [line for line in shown.splitlines() if " transcription complete. " not in line]
    return _shorten(" ".join(kept))


def _shorten(text: str) -> str:
    excerpt = " ".join(text.split())
    if len(excerpt) > 2 * _EXCERPT_CHARS:
        excerpt = f"{excerpt[:_EXCERPT_CHARS]} ... {excerpt[-_EXCERPT_CHARS:]}"
    return f": {excerpt}" if excerpt else ""


class ParakeetBackend(Protocol):
    """A local Parakeet run, as the commands use it, whichever program makes it."""

    @property
    def pace(self) -> str:
        """How long a run takes and what a first run fetches, for a progress line."""
        ...

    @property
    def notice(self) -> str:
        """The line `scribe parakeet` prints before the run."""
        ...

    def resolve(self) -> object:
        """Check this machine can run the backend, before anything is spawned."""
        ...

    def inputs(self) -> dict[str, Path]:
        """Files the run reads besides the audio, keyed by how a refusal describes them."""
        ...

    def transcribe(self, audio: Path, *, source: Source) -> Transcript:
        """Transcribe `audio` and return its words as a transcript."""
        ...


class ParakeetMlx:
    """Transcription by `parakeet-mlx` on this machine, nothing uploaded."""

    pace = "about 45 s per audio hour; a first run downloads a ~1.2 GB model"
    notice = (
        "running parakeet-mlx locally; a first run downloads a ~1.2 GB model and can take minutes"
    )

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

    def inputs(self) -> dict[str, Path]:
        """None: the model lives in Hugging Face's cache, which no output names."""
        return {}

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


CPP_NAME = "parakeet-cpp"
CPP_TOOL = "parakeet-cli"
CLI_SETTING = "SCRIBE_PARAKEET_CLI"
MODEL_SETTING = "SCRIBE_PARAKEET_MODEL"
THREADS_SETTING = "SCRIBE_PARAKEET_THREADS"
# The tool holds all the audio it is given in memory: with the f16 model it
# peaked at 15 GB for 31 min and 29 GB for 62 min, and on the 62 min it printed
# no words and exited 0. So nothing past the length seen to work reaches it.
_WHOLE_SECONDS = 31 * 60
_PIECE_SECONDS = 30 * 60
# Heard by both pieces. The boundary sits in its middle, so each piece's cut is
# 7.5 s from any word it owns: a word beside a cut is heard with one side missing.
_OVERLAP_SECONDS = 15
# The slow end of the measured pace, in minutes of running per minute of audio.
_SLOW_PACE = 17 / 60
# What 16 kHz mono 16-bit audio takes per second.
_WAV_BYTES_PER_SECOND = 32_000
# The ggml runtime's own settings, the GPUs it may use, and where its backend
# libraries load from; the uvx children need none of them.
_GGML_PREFIX = "GGML_"
_GGML_NAMES = frozenset({"CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"})
# The hosts it runs on are shared.
_MAX_THREADS = 8
# Token times count the model's 10 ms mel frames, from the start of the audio.
_TICKS_PER_SECOND = 100
_WORD_START = "\N{LOWER ONE EIGHTH BLOCK}"

_SEGMENTS = re.compile(r"Segments \((\d+)\):")
_SEGMENT = re.compile(r'Segment \d+: \[\d+ -> \d+\] ".*"')
_TOKENS = re.compile(r"Tokens \[(\d+)\]:")
# The text is the last field and quoted unescaped, so it runs to the line's final quote.
_TOKEN = re.compile(
    r" *\[ *\d+\] id= *\d+ frame= *\d+ dur_idx= *\d+ dur_val= *\d+ p=\S+ plog=\S+ "
    r't0= *(\d+) t1= *(\d+) word_start=(true|false) "(.*)"'
)


def _report_excerpt(report: str) -> str:
    # The model's parameters and timings go to stderr as well; its own error lines name the cause.
    lines = [line for line in report.split("\n") if line.strip()]
    errors = [line for line in lines if line.startswith("error:")]
    return _shorten(" ".join(errors or lines[-1:]))


def _unexpected(line: str) -> ExternalServiceError:
    return ExternalServiceError(f"parakeet-cli printed an unexpected line{_shorten(repr(line))}")


def _segment_words(rows: list[tuple[int, int, bool, str]]) -> list[Word]:
    spans: list[tuple[str, int, int]] = []
    for t0, t1, starts, text in rows:
        # A piece that opens no word, punctuation included, joins the word before it.
        if starts or not spans:
            spans.append((text.removeprefix(_WORD_START), t0, t1))
        else:
            joined, start, _ = spans[-1]
            spans[-1] = (joined + text, start, t1)
    return [
        Word(text=text, start=t0 / _TICKS_PER_SECOND, end=t1 / _TICKS_PER_SECOND)
        for text, t0, t1 in spans
        if text
    ]


def parse_segments(report: str) -> list[Word]:
    """Rebuild words from the segments `parakeet-cli --print-segments` writes to stderr.

    Raises:
        ExternalServiceError: `report` holds no segments, or they are not the
            shape the tool prints.

    """
    # Split on newlines alone: a token's text may hold any other line separator.
    lines = report.split("\n")
    rows = iter(lines)
    opening = next((found for line in rows if (found := _SEGMENTS.fullmatch(line))), None)
    if opening is None:
        # It exits 0 after failing to read the audio, too.
        raise ExternalServiceError(
            f"parakeet-cli finished without printing its segments{_report_excerpt(report)}"
        )
    segments = int(opening.group(1))
    words: list[Word] = []
    for number in range(segments):
        header = next(rows, "")
        if not header.strip():
            raise ExternalServiceError(f"parakeet-cli printed {number} of {segments} segments")
        if not _SEGMENT.fullmatch(header):
            raise _unexpected(header)
        line = next(rows, "")
        counted = _TOKENS.fullmatch(line)
        if counted is None:
            raise _unexpected(line)
        tokens = int(counted.group(1))
        pieces: list[tuple[int, int, bool, str]] = []
        for index in range(tokens):
            line = next(rows, "")
            if not line.strip():
                raise ExternalServiceError(f"parakeet-cli printed {index} of {tokens} tokens")
            token = _TOKEN.fullmatch(line)
            if token is None:
                raise _unexpected(line)
            t0, t1, starts, text = token.groups()
            pieces.append((int(t0), int(t1), starts == "true", text))
        words += _segment_words(pieces)
    clamped: list[Word] = []
    for word in words:
        # As in parse_output: a word timed early keeps its place, its start clamped.
        start = max(word.start, clamped[-1].start) if clamped else word.start
        clamped.append(Word(text=word.text, start=start, end=max(word.end, start)))
    return clamped


@dataclass(frozen=True)
class _Tools:
    cli: str
    model: Path
    ffmpeg: str
    threads: int


def _piece_starts(seconds: int) -> list[int]:
    """Where each piece opens, in seconds: one piece for audio that can go whole.

    The pieces are equal rather than full ones and a remainder: a short last
    piece would cost the most per hour, or be too short to hear anything in.
    """
    if seconds <= _WHOLE_SECONDS:
        return [0]
    # Past the first piece, each adds only what it does not share with the one before.
    span = seconds - _OVERLAP_SECONDS
    count = math.ceil(span / (_PIECE_SECONDS - _OVERLAP_SECONDS))
    step = math.ceil(span / count)
    return [index * step for index in range(count)]


def _cpp_env() -> dict[str, str]:
    granted = [name for name in os.environ if name.startswith(_GGML_PREFIX) or name in _GGML_NAMES]
    return child_env(*granted)


def _threads() -> int:
    setting = os.environ.get(THREADS_SETTING)
    if setting is None:
        return min(_MAX_THREADS, os.cpu_count() or 1)
    try:
        threads = int(setting)
    except ValueError:
        threads = 0
    if threads < 1:
        raise ExternalServiceError(
            f"{THREADS_SETTING} must be a whole number above 0, not {setting!r}"
        )
    return threads


def _model() -> Path:
    setting = os.environ.get(MODEL_SETTING)
    if not setting:
        raise ExternalServiceError(
            f"{MODEL_SETTING} is not set; it names the ggml Parakeet TDT v3 model {CPP_TOOL} reads"
        )
    model = Path(setting)
    if not model.is_absolute():
        raise ExternalServiceError(f"{MODEL_SETTING} must be an absolute path, not {setting}")
    try:
        found = model.is_file()
    # A name too long to look up.
    except OSError:
        found = False
    if not found:
        raise ExternalServiceError(f"{MODEL_SETTING} names {setting}, which is not a file")
    return model


class ParakeetCpp:
    """Transcription by whisper.cpp's `parakeet-cli` on this machine, nothing uploaded."""

    # Measured on a busy shared 32-thread host: 31 min of audio in 3 to 9 min.
    # Under about 11 min of audio the tool attends over all of it at once, which
    # costs more per hour: 5 min of audio took 1.5 to 5 min.
    pace = (
        "with parakeet-cli, 6 to 17 min per audio hour on 8 CPU threads and more per hour"
        " under 11 min of audio; nothing is downloaded"
    )
    notice = (
        "running parakeet-cli locally, 6 to 17 min per audio hour on 8 CPU threads"
        " and more per hour under 11 min of audio"
    )

    def __init__(
        self,
        run: Callable[..., subprocess.CompletedProcess[str]] = run_in_own_group,
        *,
        which: Callable[[str], str | None] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Hold the run settings; nothing is probed or spawned until called.

        Args:
            run: Process runner, injectable so tests never spawn anything.
            which: PATH lookup. Default: `shutil.which`.
            timeout_s: Ceiling for the whole run, the audio's conversion included.

        """
        self._run = run
        self._which = which
        self.timeout_s = timeout_s

    def resolve(self) -> _Tools:
        """Check this machine has all the run needs, before anything is spawned.

        Returns:
            The programs, model and thread count the run uses.

        Raises:
            ExternalServiceError: `parakeet-cli`, the model or `ffmpeg` cannot be
                found, or the thread count set is not a number above 0.

        """
        # Resolved at call time, so a patched `shutil.which` is used.
        which = self._which or shutil.which
        named = os.environ.get(CLI_SETTING)
        cli = which(named or CPP_TOOL)
        if cli is None and named:
            raise ToolMissingError(f"{CLI_SETTING} names {named}, which is not a program")
        if cli is None:
            raise ToolMissingError(
                f"{CPP_TOOL} is not on PATH; it comes with whisper.cpp, or set {CLI_SETTING}"
            )
        model = _model()
        ffmpeg = which("ffmpeg")
        if ffmpeg is None:
            raise ToolMissingError("ffmpeg is not on PATH; the audio is converted with it")
        return _Tools(cli=cli, model=model, ffmpeg=ffmpeg, threads=_threads())

    def inputs(self) -> dict[str, Path]:
        """The model file, once `resolve` has found it."""
        setting = os.environ.get(MODEL_SETTING)
        return {"the Parakeet model": Path(setting)} if setting else {}

    def _check_length(self, seconds: int, pieces: int) -> None:
        heard = seconds + (pieces - 1) * _OVERLAP_SECONDS
        if heard * _SLOW_PACE > self.timeout_s:
            raise ExternalServiceError(
                f"the audio is {seconds / 60:.1f} min long; at up to 17 min per audio hour,"
                f" parakeet-cli would not finish within {self.timeout_s:g} s"
            )

    def _call(self, argv: list[str], deadline: float) -> subprocess.CompletedProcess[str]:
        name = Path(argv[0]).name
        try:
            completed = self._run(
                argv, env=_cpp_env(), timeout=max(deadline - time.monotonic(), 0.0)
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalServiceError(
                f"{name} did not finish within {self.timeout_s:g} s"
            ) from exc
        except OSError as exc:
            raise ExternalServiceError(f"cannot run {name} at {argv[0]}: {exc}") from exc
        if completed.returncode != 0:
            raise ExternalServiceError(
                f"{name} exited {completed.returncode}{_report_excerpt(completed.stderr)}"
            )
        return completed

    def _hear(self, wav: Path, tools: _Tools, deadline: float) -> list[Word]:
        argv = [tools.cli, "-m", str(tools.model), "-t", str(tools.threads)]
        completed = self._call([*argv, "--print-segments", "-f", str(wav)], deadline)
        return parse_segments(completed.stderr)

    def _hear_pieces(
        self, wav: Path, starts: list[int], tools: _Tools, deadline: float
    ) -> list[Word]:
        span = starts[1] + _OVERLAP_SECONDS
        boundaries = [start + _OVERLAP_SECONDS / 2 for start in starts[1:]]
        owned_from = [-math.inf, *boundaries]
        owned_until = [*boundaries, math.inf]
        words: list[Word] = []
        for index, start in enumerate(starts):
            piece = wav.with_name(f"piece{index}.wav")
            argv = [tools.ffmpeg, "-nostdin", "-v", "error", "-ss", str(start)]
            # The last runs to the end, which the file's size gives only to the second.
            argv += ["-t", str(span)] if index < len(starts) - 1 else []
            self._call([*argv, "-i", str(wav), "-c:a", "pcm_s16le", str(piece)], deadline)
            heard = self._hear(piece, tools, deadline)
            # Temporary space can be memory, and the pieces together outweigh the audio.
            piece.unlink()
            for word in heard:
                # Rounded to the tool's ticks, so the sum carries no float noise.
                at = round(word.start + start, 2)
                if owned_from[index] <= at < owned_until[index]:
                    words.append(Word(text=word.text, start=at, end=round(word.end + start, 2)))
        return words

    def transcribe(self, audio: Path, *, source: Source) -> Transcript:
        """Transcribe `audio` and return its words as a transcript.

        Audio too long for the tool to hear whole goes to it in overlapping
        pieces, one at a time, and any piece that fails fails the whole run.

        Raises:
            ExternalServiceError: `resolve` failed, the audio could not finish
                within `timeout_s`, or a conversion, cut or run failed, timed
                out, or printed no readable segments.

        """
        tools = self.resolve()
        deadline = time.monotonic() + self.timeout_s
        with tempfile.TemporaryDirectory(prefix="scribe-parakeet-") as workdir:
            wav = Path(workdir) / "audio.wav"
            # The tool reads flac, mp3, ogg and wav only; every input is converted,
            # so all of them reach it the same way, as the 16 kHz mono it hears.
            # Absolute, so an audio name that begins with "-" is not read as an option.
            argv = [tools.ffmpeg, "-nostdin", "-v", "error", "-i", str(audio.absolute())]
            argv += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
            self._call(argv, deadline)
            # Whole seconds: the wav's header, tags included, is under one.
            seconds = wav.stat().st_size // _WAV_BYTES_PER_SECOND
            starts = _piece_starts(seconds)
            self._check_length(seconds, len(starts))
            if len(starts) == 1:
                words = self._hear(wav, tools, deadline)
            else:
                words = self._hear_pieces(wav, starts, tools, deadline)
        return Transcript(
            source=source,
            engine=Engine(name=CPP_NAME, model=tools.model.name),
            text=" ".join(word.text for word in words),
            words=words,
        )


def parakeet_backend(host: Callable[[], tuple[str, str]] = _host) -> ParakeetBackend:
    """Return this machine's backend: parakeet-mlx on Apple silicon, parakeet-cli elsewhere."""
    if host() == ("Darwin", "arm64"):
        return ParakeetMlx()
    return ParakeetCpp()
