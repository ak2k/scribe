"""Session terms for `scribe serve`: hook files polled off the request path, merged per dictation.

Each source (the local hook file, or one host's, read over ssh) is refreshed in
the background and keeps its last good blocks; a dictation only merges what is
already in memory, so no request waits on ssh or a slow disk.
"""

from __future__ import annotations

import subprocess
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread
import structlog

from scribe.errors import AppError, ExternalServiceError, InputValidationError
from scribe.session_terms import parse_blocks
from scribe.vocab import Vocab
from scribe.xai_stt import MAX_KEYTERMS, check_keyterms

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from structlog.stdlib import BoundLogger

    from scribe.session_terms import SessionBlock

    Runner = Callable[[Sequence[str], float], str]
    Clock = Callable[[], datetime]

LOCAL_POLL_SECONDS = 1.0
REMOTE_POLL_SECONDS = 3.0
FETCH_TIMEOUT_SECONDS = 4.0
# The hook writes about 1 KiB per session; far more is not a hook file.
MAX_FETCH_BYTES = 2**20
# Relative to the remote home: the hosts set no XDG_STATE_HOME.
REMOTE_FILE = ".local/state/scribe/terms/current.txt"
# Holds a space, so no valid host can share its name.
LOCAL_NAME = "local file"
_STDERR_CHARS = 200


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


def run_program(argv: Sequence[str], timeout: float) -> str:
    """Run `argv` and return its standard output.

    Raises:
        ExternalServiceError: the program is missing, exits non-zero, outlives
            `timeout` (it is killed), or prints something other than UTF-8.

    """
    try:
        done = subprocess.run(  # noqa: S603  # argv is ssh and a host checked at startup
            list(argv), capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalServiceError(f"no answer within {timeout:g} s") from exc
    except OSError as exc:
        raise ExternalServiceError(f"cannot run {argv[0]}: {exc.strerror}") from exc
    if done.returncode != 0:
        lines = done.stderr.decode(errors="replace").strip().splitlines()
        detail = lines[-1][:_STDERR_CHARS] if lines else "no error output"
        raise ExternalServiceError(f"exit {done.returncode}: {detail}")
    try:
        return done.stdout.decode()
    except UnicodeDecodeError as exc:
        raise ExternalServiceError("the output is not UTF-8") from exc


def check_host(host: str) -> None:
    """Refuse a host value ssh could read as an option or as more than one argument.

    Raises:
        InputValidationError: it is empty, starts with "-", or holds whitespace
            or a control character.

    """
    if (
        not host
        or host.startswith("-")
        or any(char.isspace() or unicodedata.category(char).startswith("C") for char in host)
    ):
        raise InputValidationError(f"--session-terms-host {host!r} is not a host name")


@dataclass(frozen=True)
class _Good:
    blocks: tuple[SessionBlock, ...]
    at: datetime


def _live(good: _Good | None, now: datetime) -> tuple[SessionBlock, ...]:
    return () if good is None else tuple(block for block in good.blocks if block.expires > now)


class Source:
    """One hook file, with the blocks of its last good read."""

    def __init__(self, name: str, fetch: Callable[[], str | None], *, interval: float) -> None:
        """Hold a source that `fetch` reads; None from it means unchanged since the last read."""
        self.name = name
        self.interval = interval
        self._fetch = fetch
        self._good: _Good | None = None
        self._failure: str | None = None

    def refresh(self, clock: Clock) -> None:
        """Read the source; a failure keeps the last good blocks and logs once per new cause."""
        try:
            text = self._fetch()
            if text is None:
                # Unchanged since the last read, which stands, good or bad.
                if self._failure is None and self._good is not None:
                    self._good = _Good(self._good.blocks, clock())
                return
            blocks = tuple(parse_blocks(text))
        except AppError as exc:
            self._failed(" ".join(str(exc).split()))
            return
        except Exception as exc:  # noqa: BLE001  # a poll never stops the server; the type is the cause
            self._failed(type(exc).__name__)
            return
        if self._failure is not None:
            _logger().info("serve.session_terms_restored", source=self.name)
            self._failure = None
        self._good = _Good(blocks, clock())

    def _failed(self, cause: str) -> None:
        if cause != self._failure:
            _logger().warning("serve.session_terms_failed", source=self.name, error=cause)
        self._failure = cause

    def live(self, now: datetime) -> tuple[SessionBlock, ...]:
        """The blocks of the last good read that have not expired by `now`."""
        return _live(self._good, now)

    def health(self, now: datetime) -> dict[str, object]:
        """Counts and the last good read's age; no term or session id."""
        # One read: the poller may swap in a newer read meanwhile.
        good = self._good
        live = _live(good, now)
        return {
            "source": self.name,
            "blocks": len(live),
            "terms": sum(len(block.terms) for block in live),
            "age_seconds": None if good is None else round((now - good.at).total_seconds()),
        }


class _LocalFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._signature: tuple[int, int] | None = None

    def __call__(self) -> str | None:
        try:
            info = self.path.stat()
            signature = (info.st_mtime_ns, info.st_size)
            if signature == self._signature:
                return None
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ExternalServiceError(f"cannot read {self.path}: {exc.strerror}") from exc
        # A non-UTF-8 file raises UnicodeDecodeError, not OSError.
        except ValueError as exc:
            raise ExternalServiceError(f"cannot read {self.path}: {exc}") from exc
        self._signature = signature
        return text


def local_source(path: Path, *, interval: float = LOCAL_POLL_SECONDS) -> Source:
    """The hook file on this machine, read again whenever its time or size changes."""
    return Source(LOCAL_NAME, _LocalFile(path), interval=interval)


def remote_source(host: str, *, runner: Runner, interval: float = REMOTE_POLL_SECONDS) -> Source:
    """The hook file on `host`, fetched with `runner` over ssh."""
    argv = ["ssh", "-o", "ConnectTimeout=2", "-o", "BatchMode=yes", host, "cat", REMOTE_FILE]

    def fetch() -> str:
        text = runner(argv, FETCH_TIMEOUT_SECONDS)
        if len(text.encode()) > MAX_FETCH_BYTES:
            raise ExternalServiceError(f"more than {MAX_FETCH_BYTES} bytes of output")
        return text

    return Source(host, fetch, interval=interval)


@dataclass(frozen=True)
class Merged:
    """The keyterms of one dictation, and how many each source gave."""

    terms: tuple[str, ...]
    counts: dict[str, int]


def _keyterm(term: str) -> bool:
    try:
        check_keyterms([term])
    except InputValidationError:
        return False
    return True


def merge(static: Sequence[str], sources: Sequence[tuple[str, Sequence[SessionBlock]]]) -> Merged:
    """Static terms whole, then every source's blocks, newest rank first, taking turns a term each.

    `static` is a parsed terms file, so at most `MAX_KEYTERMS`, and each source's
    blocks are its live ones. Repeats and terms xAI would refuse are skipped; the
    list ends at `MAX_KEYTERMS`.
    """
    taken = dict.fromkeys(static)
    counts = dict.fromkeys((name for name, _ in sources), 0)
    live = sorted(
        ((name, block) for name, blocks in sources for block in blocks),
        key=lambda pair: pair[1].ranked,
        reverse=True,
    )
    # A turn per block, so one busy session cannot crowd out another.
    queues = [(name, iter(block.terms)) for name, block in live]
    while queues and len(taken) < MAX_KEYTERMS:
        for entry in list(queues):
            name, terms = entry
            term = next((t for t in terms if t not in taken and _keyterm(t)), None)
            if term is None:
                queues.remove(entry)
            elif len(taken) < MAX_KEYTERMS:
                taken[term] = None
                counts[name] += 1
    return Merged(tuple(taken), counts)


class SessionTerms:
    """Every source of session terms, polled in the background and merged per dictation."""

    def __init__(self, sources: Sequence[Source], *, clock: Clock = _utcnow) -> None:
        """Hold `sources`; `clock` decides what has expired."""
        self.sources = tuple(sources)
        self._clock = clock

    def vocab(self, static: Vocab) -> tuple[Vocab, dict[str, int]]:
        """The static vocabulary with the live session terms merged in, and per-source counts."""
        now = self._clock()
        merged = merge(static.terms, [(source.name, source.live(now)) for source in self.sources])
        return Vocab(terms=merged.terms, aliases=static.aliases), merged.counts

    async def poll(self) -> None:
        """Refresh each source on its interval until cancelled."""
        async with anyio.create_task_group() as group:
            for source in self.sources:
                group.start_soon(self._poll, source)

    async def _poll(self, source: Source) -> None:
        while True:
            # Abandoned on shutdown: the fetch's own timeout ends it soon after.
            await anyio.to_thread.run_sync(source.refresh, self._clock, abandon_on_cancel=True)
            await anyio.sleep(source.interval)

    def health(self) -> list[dict[str, object]]:
        """Per source: live blocks and terms, and the age of its last good read."""
        now = self._clock()
        return [source.health(now) for source in self.sources]
