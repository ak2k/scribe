from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from structlog.testing import capture_logs

from scribe.voiceink import Dictionary, join
from scribe.xai_stt import MAX_KEYTERMS
from tests.voiceink_fixtures import add_words, closed_store, crashed_store, open_store

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "dictionary.store"


@pytest.fixture
def open_writer(store: Path) -> Iterator[sqlite3.Connection]:
    with closing(open_store(store, "Zorblatt")) as connection:
        yield connection


def _listing(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


def test_words_are_trimmed_and_deduped_keeping_the_twin_voiceink_sorts_first(
    store: Path,
) -> None:
    closed_store(store, "Quindle", "Zorblatt  ", "", "  ", "zorblatt", "Aplix")

    dictionary = Dictionary(store)

    assert dictionary.read() == ("Aplix", "Quindle", "zorblatt")
    assert dictionary.health() == {"path": str(store), "state": "ok", "words": 3}


def test_words_sort_as_voiceink_sorts_them_ignoring_case_with_numbers_by_value(
    store: Path,
) -> None:
    closed_store(store, "zeta", "Word10", "Beta", "_x", "Word2", "aardvark")

    assert Dictionary(store).read() == ("_x", "aardvark", "Beta", "Word2", "Word10", "zeta")


def test_a_padded_word_sorts_by_its_stored_padding_as_voiceink_fetches_it(store: Path) -> None:
    # VoiceInk sorts the stored values before it trims them, so leading padding moves a word up.
    closed_store(store, "Mango", " Zorblatt", "Aplix", "zorblatt")

    assert Dictionary(store).read() == ("Zorblatt", "Aplix", "Mango")


def test_of_exact_case_twins_the_lowercase_one_is_kept(store: Path) -> None:
    closed_store(store, "Zorblatt", "zorblatt")

    assert Dictionary(store).read() == ("zorblatt",)


def test_with_one_free_slot_the_word_with_fewer_leading_zeros_is_sent(store: Path) -> None:
    closed_store(store, "Word002", "Word02", "Word2")
    words = Dictionary(store).read()
    file_terms = [f"file{n}" for n in range(MAX_KEYTERMS - 1)]

    assert words == ("Word2", "Word02", "Word002")
    assert join(file_terms, words)[-1] == "Word2"


def test_ties_go_to_accents_then_to_case_or_leading_zeros_from_the_left(store: Path) -> None:
    closed_store(store, "Word2", "word02", "02quindle", "2Quindle", "zorblätt", "Zorblatt")

    assert Dictionary(store).read() == (
        "2Quindle",
        "02quindle",
        "word02",
        "Word2",
        "Zorblatt",
        "zorblätt",
    )


def test_a_number_too_long_to_convert_is_refused_alone(store: Path) -> None:
    closed_store(store, "Quindle", "9" * 5000)
    dictionary = Dictionary(store)

    with capture_logs() as logs:
        assert dictionary.read() == ("Quindle",)

    assert dictionary.health() == {"path": str(store), "state": "ok", "words": 1}
    assert [entry["event"] for entry in logs] == ["serve.dictionary_word_refused"]


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


def test_join_drops_a_word_the_file_already_holds_in_any_case() -> None:
    assert join(["Zorblatt", "herdr"], ["zorblatt", "Quindle", "HERDR"]) == (
        "Zorblatt",
        "herdr",
        "Quindle",
    )


def test_join_counts_file_case_twins_against_the_cap() -> None:
    file_terms = ["Zorblatt", "zorblatt", *(f"file{n}" for n in range(96))]

    assert len(join(file_terms, [f"Word{n}" for n in range(10)])) == MAX_KEYTERMS


def test_join_with_a_full_file_adds_nothing() -> None:
    file_terms = [f"file{n}" for n in range(MAX_KEYTERMS)]

    assert join(file_terms, ["Zorblatt"]) == tuple(file_terms)


def test_a_relative_path_is_read_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed_store(tmp_path / "dictionary.store", "Zorblatt")
    monkeypatch.chdir(tmp_path)

    assert Dictionary(Path("dictionary.store")).read() == ("Zorblatt",)


def test_a_store_that_cannot_be_looked_at_is_unreadable_not_an_error(tmp_path: Path) -> None:
    folder = tmp_path / "locked"
    folder.mkdir()
    closed_store(folder / "dictionary.store", "Zorblatt")
    folder.chmod(0)
    try:
        dictionary = Dictionary(folder / "dictionary.store")
        words = dictionary.read()
    finally:
        folder.chmod(0o700)

    assert words == ()
    assert dictionary.health()["state"] == "unreadable"


def test_a_store_gone_after_a_good_read_gives_no_words(store: Path) -> None:
    closed_store(store, "Zorblatt")
    dictionary = Dictionary(store)
    assert dictionary.read() == ("Zorblatt",)

    store.unlink()

    assert dictionary.read() == ()
    assert dictionary.health()["state"] == "missing"


def test_a_crashed_writers_wal_is_read_and_left_unwritten(store: Path) -> None:
    crashed_store(store, "Zorblatt", "Quindle")
    before = _listing(store.parent)

    assert Dictionary(store).read() == ("Quindle", "Zorblatt")

    after = _listing(store.parent)
    assert set(after) == set(before)
    assert after["dictionary.store"] == before["dictionary.store"]
    assert after["dictionary.store-wal"] == before["dictionary.store-wal"]


@pytest.mark.parametrize("gone", ["-wal", "-shm"])
def test_a_store_with_one_companion_file_gains_no_file(store: Path, gone: str) -> None:
    crashed_store(store, "Zorblatt")
    store.with_name(store.name + gone).unlink()
    before = _listing(store.parent)

    Dictionary(store).read()

    after = _listing(store.parent)
    assert set(after) == set(before)
    assert after["dictionary.store"] == before["dictionary.store"]


def test_words_in_a_wal_without_its_shm_are_unreadable_and_the_last_words_kept(
    store: Path,
) -> None:
    crashed_store(store, "Zorblatt")
    dictionary = Dictionary(store)
    assert dictionary.read() == ("Zorblatt",)
    store.with_name(store.name + "-shm").unlink()

    with capture_logs() as logs:
        first = dictionary.read()
        second = dictionary.read()

    assert first == second == ("Zorblatt",)
    assert dictionary.health()["state"] == "unreadable"
    warnings = [entry for entry in logs if entry["event"] == "serve.dictionary_unreadable"]
    assert len(warnings) == 1


def test_an_empty_store_beside_a_wal_keeps_its_wal(store: Path) -> None:
    crashed_store(store, "Zorblatt")
    store.write_bytes(b"")
    before = _listing(store.parent)

    Dictionary(store).read()

    assert _listing(store.parent) == before


def test_a_symlinked_store_reads_the_wal_beside_its_target(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "dictionary.store"
    with closing(open_store(target / "dictionary.store", "Zorblatt")):
        link.symlink_to(target / "dictionary.store")

        assert Dictionary(link).read() == ("Zorblatt",)
