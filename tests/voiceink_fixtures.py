"""A VoiceInk Dictionary store, written as VoiceInk writes it."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = (
    "CREATE TABLE ZVOCABULARYWORD ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER,"
    " Z_OPT INTEGER, ZDATEADDED TIMESTAMP, ZWORD VARCHAR )"
)


def open_store(path: Path, *words: str) -> sqlite3.Connection:
    """Create the store in WAL mode holding `words`, and return the open writer."""
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=wal")
    connection.execute(SCHEMA)
    add_words(connection, *words)
    return connection


def add_words(connection: sqlite3.Connection, *words: str) -> None:
    connection.executemany(
        "INSERT INTO ZVOCABULARYWORD (Z_ENT, Z_OPT, ZDATEADDED, ZWORD) VALUES (1, 1, 0, ?)",
        [(word,) for word in words],
    )
    connection.commit()


def closed_store(path: Path, *words: str) -> Path:
    """Create the store holding `words` and close it, which leaves no -wal or -shm."""
    open_store(path, *words).close()
    return path
