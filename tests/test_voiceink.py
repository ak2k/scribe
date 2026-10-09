from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from typing import TYPE_CHECKING

import pytest
from structlog.testing import capture_logs

from scribe.voiceink import Dictionary, default_path, join
from scribe.xai_stt import MAX_KEYTERMS
from tests.voiceink_fixtures import add_words, closed_store, open_store

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "dictionary.store"


@pytest.fixture
def open_writer(store: Path) -> Iterator[sqlite3.Connection]:
    with closing(open_store(store, "Zorblatt")) as connection:
        yield connection


def _listing(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


def test_the_default_path_is_voiceinks_store_under_home(tmp_path: Path) -> None:
    assert default_path(tmp_path) == (
        tmp_path
        / "Library"
        / "Application Support"
        / "com.prakashjoshipax.VoiceInk"
        / "dictionary.store"
    )


def test_words_are_trimmed_deduped_ignoring_case_and_sorted(store: Path) -> None:
    closed_store(store, "Quindle", "  Zorblatt ", "", "  ", "zorblatt", "Aplix")

    dictionary = Dictionary(store)

    assert dictionary.read() == ("Aplix", "Quindle", "Zorblatt")
    assert dictionary.health() == {"path": str(store), "state": "ok", "words": 3}


def test_a_word_xai_would_refuse_is_skipped_and_logged_once(store: Path) -> None:
    too_long = "x" * 51
    closed_store(store, "Quindle", too_long, "Zorb\x07latt")
    dictionary = Dictionary(store)

    with capture_logs() as logs:
        first = dictionary.read()
        second = dictionary.read()

    assert first == second == ("Quindle",)
    refused = [entry for entry in logs if entry["event"] == "serve.dictionary_word_refused"]
    assert len(refused) == 2
    assert too_long not in repr(logs)


def test_a_missing_store_gives_no_words_and_creates_nothing(tmp_path: Path) -> None:
    store = tmp_path / "absent" / "dictionary.store"
    dictionary = Dictionary(store)

    assert dictionary.read() == ()
    assert dictionary.health() == {"path": str(store), "state": "missing", "words": 0}
    assert not store.parent.exists()


def test_a_word_added_while_voiceink_holds_the_store_open_is_read_next_time(
    store: Path, open_writer: sqlite3.Connection
) -> None:
    dictionary = Dictionary(store)
    assert dictionary.read() == ("Zorblatt",)
    before = store.stat()

    add_words(open_writer, "Quindle")

    assert dictionary.read() == ("Quindle", "Zorblatt")
    after = store.stat()
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


def test_a_closed_store_is_read_without_creating_a_file(store: Path) -> None:
    closed_store(store, "Zorblatt")
    before = _listing(store.parent)
    assert set(before) == {"dictionary.store"}

    assert Dictionary(store).read() == ("Zorblatt",)

    assert _listing(store.parent) == before


def test_an_open_store_is_read_without_writing_the_store_or_its_wal(
    store: Path, open_writer: sqlite3.Connection
) -> None:
    add_words(open_writer, "Quindle")
    names = {path.name for path in store.parent.iterdir()}
    assert names == {"dictionary.store", "dictionary.store-wal", "dictionary.store-shm"}
    store_bytes = store.read_bytes()
    wal_bytes = (store.parent / "dictionary.store-wal").read_bytes()

    assert Dictionary(store).read() == ("Quindle", "Zorblatt")

    assert {path.name for path in store.parent.iterdir()} == names
    assert store.read_bytes() == store_bytes
    assert (store.parent / "dictionary.store-wal").read_bytes() == wal_bytes


def _spoil(path: Path, how: str) -> None:
    path.unlink()
    if how == "not-sqlite":
        path.write_bytes(b"not a database, only some bytes" * 64)
        return
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE ZVOCABULARYWORD ( Z_PK INTEGER PRIMARY KEY, ZTEXT )")


@pytest.mark.parametrize("how", ["not-sqlite", "no-zword"])
def test_an_unreadable_store_keeps_the_last_good_words_and_warns_once(
    store: Path, how: str
) -> None:
    closed_store(store, "Zorblatt")
    dictionary = Dictionary(store)
    assert dictionary.read() == ("Zorblatt",)
    _spoil(store, how)

    with capture_logs() as logs:
        first = dictionary.read()
        second = dictionary.read()

    assert first == second == ("Zorblatt",)
    assert dictionary.health() == {"path": str(store), "state": "unreadable", "words": 1}
    warnings = [entry for entry in logs if entry["event"] == "serve.dictionary_unreadable"]
    assert len(warnings) == 1
    store.unlink()
    closed_store(store, "Quindle")
    with capture_logs() as logs:
        assert dictionary.read() == ("Quindle",)
    assert [entry["event"] for entry in logs] == ["serve.dictionary_restored"]


def test_an_unreadable_store_never_read_well_gives_no_words(store: Path) -> None:
    store.write_bytes(b"not a database, only some bytes" * 64)

    assert Dictionary(store).read() == ()


def test_a_read_is_quick(store: Path) -> None:
    closed_store(store, *(f"Word{n}" for n in range(200)))
    dictionary = Dictionary(store)

    started = time.monotonic()
    dictionary.read()

    assert time.monotonic() - started < 0.5


def test_join_puts_the_words_after_the_file_terms_and_never_displaces_them() -> None:
    file_terms = [f"file{n}" for n in range(95)]
    words = [f"Word{n}" for n in range(10)]

    joined = join(file_terms, words)

    assert len(joined) == MAX_KEYTERMS
    assert joined[:95] == tuple(file_terms)
    assert joined[95:] == ("Word0", "Word1", "Word2", "Word3", "Word4")


def test_join_drops_a_word_the_file_already_holds_in_any_case() -> None:
    assert join(["Zorblatt", "herdr"], ["zorblatt", "Quindle", "HERDR"]) == (
        "Zorblatt",
        "herdr",
        "Quindle",
    )


def test_join_with_a_full_file_adds_nothing() -> None:
    file_terms = [f"file{n}" for n in range(MAX_KEYTERMS)]

    assert join(file_terms, ["Zorblatt"]) == tuple(file_terms)
