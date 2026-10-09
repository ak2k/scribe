"""A VoiceInk Dictionary store, written as VoiceInk writes it."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
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


_CRASHING_WRITER = """
import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1])
connection.execute("PRAGMA journal_mode=wal")
connection.execute("PRAGMA wal_autocheckpoint=0")
connection.execute(sys.argv[2])
connection.commit()
connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
connection.executemany(
    "INSERT INTO ZVOCABULARYWORD (Z_ENT, Z_OPT, ZDATEADDED, ZWORD) VALUES (1, 1, 0, ?)",
    [(word,) for word in sys.argv[3:]],
)
connection.commit()
os._exit(0)
"""


def crashed_store(path: Path, *words: str) -> Path:
    """Leave the store as a killed VoiceInk does: `words` only in -wal frames, -shm beside it."""
    argv = [sys.executable, "-c", _CRASHING_WRITER, str(path), SCHEMA, *words]
    subprocess.run(argv, check=True)  # noqa: S603  # this interpreter, fixed code, test-chosen args
    return path
