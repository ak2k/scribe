"""VoiceInk's Dictionary, read straight from the app's store on every dictation."""

from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
from contextlib import closing
from typing import TYPE_CHECKING, Literal

import anyio.to_thread
import structlog

from scribe.errors import AppError, InputValidationError
from scribe.xai_stt import MAX_KEYTERMS, check_keyterms

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from structlog.stdlib import BoundLogger

# A dictation waits on this read, so a store VoiceInk holds locked is given up on quickly.
BUSY_TIMEOUT_SECONDS = 0.25

State = Literal["never", "ok", "missing", "unreadable"]


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


def default_path(home: Path) -> Path:
    """Where VoiceInk keeps its Dictionary for the user whose home is `home`."""
    support = home / "Library" / "Application Support" / "com.prakashjoshipax.VoiceInk"
    return support / "dictionary.store"


class _UnindexedWalError(AppError):
    """Words sit in a -wal whose -shm is gone, where no read-only open can reach them."""


def _uri(path: Path) -> str:
    # SQLite keeps the -wal and -shm beside a symlink's target, not beside the link.
    path = path.resolve()
    wal, shm = (path.with_name(path.name + suffix) for suffix in ("-wal", "-shm"))
    # A read-only open creates a missing -wal or -shm, and deletes a -wal beside an
    # empty main file, so it is used only where neither can happen.
    if wal.exists() and shm.exists() and path.stat().st_size > 0:
        return f"{path.as_uri()}?mode=ro"
    # Without the -shm, an immutable open would read the main file alone and miss the
    # -wal's words without a sign; an empty -wal holds none.
    if wal.exists() and not shm.exists() and wal.stat().st_size > 0:
        raise _UnindexedWalError(f"{wal.name} holds changes but {shm.name} is missing")
    return f"{path.as_uri()}?mode=ro&immutable=1"


def _digits(run: str) -> str:
    return "".join(str(unicodedata.decimal(char)) for char in run)


def _order(word: str) -> tuple[list[str | tuple[int, str]], list[str], list[tuple[int, str]]]:
    # VoiceInk sorts with localizedStandardCompare: letters ignoring case and accents with
    # digit runs by value, then accents, then case and leading zeros, whichever differs
    # first from the left, the lowercase twin and fewer zeros first. A run is compared by
    # its digits, as int() refuses a long one.
    folded = unicodedata.normalize("NFKD", word.casefold())
    bare = "".join(char for char in folded if not unicodedata.combining(char))
    parts = re.split(r"(\d+)", bare)
    primary: list[str | tuple[int, str]] = [parts[0]]
    for run, text in zip(parts[1::2], parts[2::2], strict=True):
        value = _digits(run).lstrip("0")
        primary += [(len(value), value), text]
    accents: list[str] = []
    for char in folded:
        if unicodedata.combining(char) and accents:
            accents[-1] += char
        elif not char.isdecimal():
            accents.append("")
    tokens = (match.group() for match in re.finditer(r"\d+|.", word, re.DOTALL))
    tertiary = [
        (len(token) - len(_digits(token).lstrip("0")), "")
        if token.isdecimal()
        else (0, token.swapcase())
        for token in tokens
    ]
    return primary, accents, tertiary


def _select(path: Path) -> list[str]:
    with closing(sqlite3.connect(_uri(path), uri=True, timeout=BUSY_TIMEOUT_SECONDS)) as db:
        rows: list[tuple[object]] = db.execute("SELECT ZWORD FROM ZVOCABULARYWORD").fetchall()
    return [value for (value,) in rows if isinstance(value, str)]


def join(file_terms: Sequence[str], words: Sequence[str]) -> tuple[str, ...]:
    """The file's terms, then the words the file lacks, ignoring case, up to `MAX_KEYTERMS`."""
    held = {term.lower() for term in file_terms}
    room = MAX_KEYTERMS - len(file_terms)
    added = [word for word in words if word.lower() not in held][:room]
    return (*file_terms, *added)


class Dictionary:
    """VoiceInk's Dictionary store, never written; a bad read keeps the last good words."""

    def __init__(self, path: Path) -> None:
        """Hold `path`; nothing is read until `read`."""
        self.path = path
        self._words: tuple[str, ...] = ()
        self._state: State = "never"
        self._failure: str | None = None
        self._refused: frozenset[str] = frozenset()
        # Dictations and health checks read from worker threads at once.
        self._lock = threading.Lock()

    def read(self) -> tuple[str, ...]:
        """Read the words now, trimmed and deduplicated ignoring case as VoiceInk does."""
        with self._lock:
            try:
                values = _select(self.path) if self.path.exists() else None
            # The type alone is logged, as the text can quote the store's contents.
            except Exception as exc:  # noqa: BLE001  # no read may fail a dictation
                self._settle("unreadable", type(exc).__name__)
                return self._words
            if values is None:
                self._settle("missing", None)
                self._words = ()
                return ()
            self._settle("ok", None)
            self._words = self._accept(values)
            return self._words

    async def words(self) -> tuple[str, ...]:
        """`read`, off the event loop."""
        return await anyio.to_thread.run_sync(self.read)

    def health(self) -> dict[str, object]:
        """The store's path, the last read's state and the words it held."""
        return {"path": str(self.path), "state": self._state, "words": len(self._words)}

    def _settle(self, state: State, failure: str | None) -> None:
        if (state, failure) != (self._state, self._failure):
            if state == "unreadable":
                _logger().warning("serve.dictionary_unreadable", path=str(self.path), error=failure)
            elif state == "missing":
                _logger().info("serve.dictionary_missing", path=str(self.path))
            elif self._state != "never":
                _logger().info("serve.dictionary_restored", path=str(self.path))
        self._state, self._failure = state, failure

    def _accept(self, values: Sequence[str]) -> tuple[str, ...]:
        # Each word is checked alone before the sort, so no entry can fail the whole read.
        kept: list[str] = []
        refused: set[str] = set()
        for value in values:
            word = value.strip()
            if not word:
                continue
            try:
                check_keyterms([word])
            except InputValidationError:
                refused.add(word)
                # Its length, not the word: a log line outlives the Dictionary entry.
                if word not in self._refused:
                    _logger().warning("serve.dictionary_word_refused", characters=len(word))
                continue
            kept.append(value)
        self._refused = frozenset(refused)
        seen: dict[str, str] = {}
        for value in sorted(kept, key=_order):
            word = value.strip()
            _ = seen.setdefault(word.lower(), word)
        return tuple(seen.values())
