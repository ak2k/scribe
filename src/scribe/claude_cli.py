"""One `claude -p` call per completion, shared by every stage that asks a model.

The caller supplies the system prompt; this module owns how the CLI is found,
invoked and read back: `claude` from PATH, on the user's own login.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import stamina
import structlog
from pydantic import BaseModel, ConfigDict

from scribe.errors import ExternalServiceError

if TYPE_CHECKING:
    from collections.abc import Callable

    from structlog.stdlib import BoundLogger

DEFAULT_MODEL = "opus"
# Per CALL, not per run: a stage that chunks makes one call per chunk. The
# harness adds about 23,000 cache-creation tokens to every call, so even a
# one-word reply costs about $0.10 and a ceiling below that would fail before
# the model ever answered.
DEFAULT_MAX_BUDGET_USD = 6.0

# A generous ceiling for one ~700-word opus chunk; a hung call still ends.
DEFAULT_TIMEOUT_S = 300.0
RETRY_ATTEMPTS = 3

# An exported API key silently moves the CLI to per-token billing, and a
# CLAUDECODE inherited from a parent session trips the CLI's nesting guard.
_ALWAYS_DROPPED = frozenset({"ANTHROPIC_API_KEY", "CLAUDECODE"})
# Overload, rate limit and server errors. A budget error also exits 1, so a
# non-zero exit alone is not evidence a retry could succeed.
_TRANSIENT_STATUSES = frozenset({429, 529, *range(500, 600)})
# A leaked `--no-session-persistence` stub is about 110 bytes; anything this
# large is a real transcript that happens to share the name.
_STUB_MAX_BYTES = 1024
_STDERR_EXCERPT_CHARS = 300

_CLAUDE_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_CRON": "1",
}


class Completion(BaseModel):
    """One backend reply, with the metadata the truncation check needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    # The model the backend was configured with. The CLI's JSON carries no model
    # key, so reading it back from a reply is not an option.
    model: str
    output_tokens: int | None = None
    stop_reason: str | None = None
    is_error: bool = False


class _TransientError(ExternalServiceError):
    """A failure another attempt could get past: a timeout, or an overload."""


def _logger() -> BoundLogger:
    # Per use, not once at import: the CLI's configuration caches a logger on
    # first use, which would then outlive any later reconfiguration.
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


class ClaudeCliBackend:
    """Completions through the Claude Code CLI, on subscription auth.

    No API key and no SDK: the CLI is already authenticated, and `-p` with
    `--output-format json` reports the error and truncation signals callers
    gate on.
    """

    name: ClassVar[str] = "claude-cli"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        *,
        which: Callable[[str], str | None] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        stub_root: Path | None = None,
        disable_tools: bool = False,
    ) -> None:
        """Hold the call settings; nothing is probed or spawned until `complete`.

        Args:
            model: Value for `--model`.
            max_budget_usd: Ceiling for one call, not for the whole run.
            run: Process runner, injectable so tests never spawn anything.
            which: PATH lookup. Default: `shutil.which`.
            timeout_s: Ceiling for one attempt, in seconds.
            stub_root: The `projects` directory leaked session stubs are
                removed from. Default: the one under
                `CLAUDE_CONFIG_DIR`, else under `~/.claude`.
            disable_tools: Pass `--tools ""`, so the model can only reply.

        """
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.timeout_s = timeout_s
        self._run = run
        self._which = which
        self._stub_root = stub_root
        self.disable_tools = disable_tools
        self._executable: str | None = None
        # `complete` may run on several threads at once; this keeps the
        # executable looked up once.
        self._lock = threading.Lock()

    def _argv(self, executable: str, system_prompt_file: Path) -> list[str]:
        # No positional prompt: `-p` then reads it from stdin, which has no
        # per-argument size limit (Linux caps one argv element at 128 KiB).
        tools = ["--tools", ""] if self.disable_tools else []
        return [
            executable,
            "-p",
            "--model",
            self.model,
            "--no-session-persistence",
            "--disable-slash-commands",
            # An empty value, not an omitted flag: it is what stops the CLI from
            # reading this machine's settings into the run.
            "--setting-sources",
            "",
            "--system-prompt-file",
            str(system_prompt_file),
            "--max-budget-usd",
            str(self.max_budget_usd),
            "--output-format",
            "json",
            *tools,
        ]

    def complete(self, system: str, user: str) -> Completion:
        """Send one prompt to `claude -p` and parse its reply.

        Args:
            system: System prompt; written to a file the CLI is pointed at.
            user: User prompt; written to the CLI's stdin.

        Returns:
            The completion, carrying the configured model.

        Raises:
            ExternalServiceError: `claude` is not on PATH, the CLI
                cannot be run, it timed out or was overloaded on every attempt,
                exited non-zero, reported an error, or wrote something other
                than one JSON object carrying a string `result`.

        """
        executable = self.resolve()
        # A context block, not a decorated function: stamina's retry hooks
        # record a decorated function's arguments, and these are the prompt.
        for attempt in stamina.retry_context(
            on=_TransientError, attempts=RETRY_ATTEMPTS, timeout=None
        ):
            with attempt:
                return self._attempt(executable, system, user)
        raise AssertionError("unreachable: stamina re-raises the last failure")  # pragma: no cover

    def resolve(self) -> str:
        """Find `claude` now, before any call is paid for.

        Returns:
            The executable every later call runs.

        Raises:
            ExternalServiceError: `claude` is not on PATH.

        """
        with self._lock:
            if self._executable is None:
                # Resolved at call time, so a patched `shutil.which` is used.
                found = (self._which or shutil.which)("claude")
                if found is None:
                    raise ExternalServiceError(
                        "claude is not on PATH; the backend is the Claude Code CLI"
                    )
                self._executable = found
                _logger().info("claude_cli.executable", executable=found)
            return self._executable

    @staticmethod
    def _env() -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key not in _ALWAYS_DROPPED}
        env.update(_CLAUDE_ENV)
        return env

    def _attempt(self, executable: str, system: str, user: str) -> Completion:
        with tempfile.TemporaryDirectory() as workdir:
            # Absolute, and inside the cwd the call runs in: `claude -p` resolves
            # --system-prompt-file against its own cwd, not the caller's.
            prompt_file = Path(workdir) / "system_prompt.txt"
            try:
                prompt_file.write_text(system, encoding="utf-8")
                completed = self._run(
                    self._argv(executable, prompt_file),
                    input=user,
                    cwd=workdir,
                    env=self._env(),
                    # Not the locale's: under a non-UTF-8 one, encoding the
                    # prompt fails on the first character it lacks.
                    encoding="utf-8",
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                error = _TransientError(f"claude -p timed out after {self.timeout_s:g} s")
                _logger().warning("claude_cli.transient_failure", error=str(error))
                raise error from exc
            # A `claude` on PATH that is not executable, or gone by exec time.
            except OSError as exc:
                raise ExternalServiceError(f"cannot run claude at {executable}: {exc}") from exc
        return self._read(completed)

    def _read(self, completed: subprocess.CompletedProcess[str]) -> Completion:
        # Never the prompt: an excerpt of the transcript in an error message
        # would leak the recording into a log.
        excerpt = " ".join(completed.stderr[:_STDERR_EXCERPT_CHARS].split())
        try:
            decoded: object = json.loads(completed.stdout)  # pyright: ignore[reportAny]  # json.loads is Any
        except ValueError as exc:
            raise ExternalServiceError(
                f"claude -p exited {completed.returncode} without parseable JSON: {excerpt}"
            ) from exc
        if not isinstance(decoded, dict):
            raise ExternalServiceError(
                f"claude -p exited {completed.returncode} with a JSON "
                f"{type(decoded).__name__}, not an object: {excerpt}"
            )
        # Explicit key extraction, not a model: the CLI's reply carries about
        # thirty undocumented keys, so extra="forbid" would reject every real one.
        payload = cast("dict[str, object]", decoded)
        self._remove_stub(payload.get("session_id"))
        result = payload.get("result")
        status = payload.get("api_error_status")
        if (
            payload.get("is_error") is True
            and type(status) is int
            and status in _TRANSIENT_STATUSES
        ):
            error = _TransientError(
                f"claude -p exited {completed.returncode} with api_error_status {status}: {excerpt}"
            )
            _logger().warning("claude_cli.transient_failure", error=str(error))
            raise error
        # `stop_reason` reads "end_turn" even on a budget error, so it cannot
        # stand in for any of these three.
        if (
            completed.returncode != 0
            or payload.get("is_error") is True
            or not isinstance(result, str)
        ):
            raise ExternalServiceError(
                f"claude -p exited {completed.returncode} with subtype "
                f"{payload.get('subtype')!r}: {excerpt}"
            )
        stop_reason = payload.get("stop_reason")
        return Completion(
            text=result,
            model=self.model,
            output_tokens=_output_tokens(payload.get("usage")),
            stop_reason=stop_reason if isinstance(stop_reason, str) else None,
            is_error=False,
        )

    def _remove_stub(self, session_id: object) -> None:
        # `--no-session-persistence` still leaves a stub per call under the
        # user's projects. A UUID check keeps the glob to one exact name.
        if not isinstance(session_id, str) or not _is_uuid(session_id):
            return
        root = self._stub_root or _default_stub_root()
        try:
            real_root = root.resolve(strict=True)
        except OSError:
            return
        for stub in root.glob(f"*/{session_id}.jsonl"):
            try:
                info = stub.lstat()
                # Neither a linked stub nor a linked project dir: either could
                # make the unlink land outside the projects root.
                if (
                    stub.parent.is_symlink()
                    or stub.resolve().parent.parent != real_root
                    or not stat.S_ISREG(info.st_mode)
                    or info.st_size >= _STUB_MAX_BYTES
                ):
                    continue
                stub.unlink()
            except OSError as exc:
                _logger().warning("claude_cli.stub_kept", path=str(stub), error=str(exc))
                continue
            # The per-cwd project dir is the call's own temp cwd's, so it is
            # empty now unless the name collided with a real project.
            with contextlib.suppress(OSError):
                stub.parent.rmdir()


def _default_stub_root() -> Path:
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(config).expanduser() if config else Path.home() / ".claude"
    return base / "projects"


def _is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _output_tokens(usage: object) -> int | None:
    if not isinstance(usage, dict):
        return None
    tokens = cast("dict[str, object]", usage).get("output_tokens")
    return tokens if isinstance(tokens, int) else None
