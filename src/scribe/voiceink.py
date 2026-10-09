"""VoiceInk's Dictionary, read straight from the app's store on every dictation."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from typing import TYPE_CHECKING, Literal

import anyio.to_thread
import structlog

from scribe.errors import InputValidationError
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


def _uri(path: Path) -> str:
    # A read-only open of a WAL database creates a missing -wal or -shm beside it;
    # without both, VoiceInk is not writing, so the main file alone is the whole store.
    live = all(path.with_name(path.name + suffix).exists() for suffix in ("-wal", "-shm"))
    return f"{path.absolute().as_uri()}?mode=ro" + ("" if live else "&immutable=1")


def _select(path: Path) -> list[object]:
    with closing(sqlite3.connect(_uri(path), uri=True, timeout=BUSY_TIMEOUT_SECONDS)) as db:
        rows: list[tuple[object]] = db.execute(
            "SELECT ZWORD FROM ZVOCABULARYWORD ORDER BY ZWORD"
        ).fetchall()
    return [value for (value,) in rows]


def join(file_terms: Sequence[str], words: Sequence[str]) -> tuple[str, ...]:
    """The file's terms, then the words the file lacks, ignoring case, up to `MAX_KEYTERMS`."""
    held = {term.lower() for term in file_terms}
    room = max(MAX_KEYTERMS - len(file_terms), 0)
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

    def _accept(self, values: Sequence[object]) -> tuple[str, ...]:
        seen: dict[str, str] = {}
        refused: set[str] = set()
        for value in values:
            word = value.strip() if isinstance(value, str) else ""
            if not word or word.lower() in seen:
                continue
            try:
                check_keyterms([word])
            except InputValidationError:
                refused.add(word)
                # Its length, not the word: a log line outlives the Dictionary entry.
                if word not in self._refused:
                    _logger().warning("serve.dictionary_word_refused", characters=len(word))
                continue
            seen[word.lower()] = word
        self._refused = frozenset(refused)
        return tuple(seen.values())
