from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from structlog.testing import capture_logs

from scribe.errors import InputValidationError
from scribe.vocab import Alias, Edit, TermsFile, Vocab, deliver, parse_terms, snap

if TYPE_CHECKING:
    from pathlib import Path

TERMS = ["herdr", "akms25", "VoiceInk", "MacWhisper", "modules/darwin/base.nix"]


def _deliver(text: str, vocab: Vocab) -> str:
    words, _ = deliver(text.split(), vocab)
    return " ".join(words)


def _key(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def test_terms_comments_blanks_and_aliases_are_parsed() -> None:
    vocab = parse_terms("# my terms\n\nherdr\n  VoiceInk  \nherder => herdr\nherdr\n", where="t")

    assert vocab.terms == ("herdr", "VoiceInk")
    assert vocab.aliases == (Alias(heard="herder", written="herdr"),)


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("ok\n => herdr\n", "t:2"),
        ("ok\nherder =>\n", "t:2"),
        ("one two three four five => x\n", "t:1"),
        ("a => b => c\n", "t:1"),
        ("x" * 51 + "\n", "t:1"),
        ("".join(f"term{index}\n" for index in range(101)), "at most 100"),
    ],
)
def test_a_malformed_line_names_where_it_is(text: str, why: str) -> None:
    with pytest.raises(InputValidationError, match=why):
        parse_terms(text, where="t")


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("so herder works", "so herdr works"),
        ("so Herder, works", "so herdr, works"),
        ("(herder)", "(herdr)"),
        ("a herders b", "a herders b"),
    ],
)
def test_an_alias_rewrites_whole_words_case_insensitively_keeping_edge_punctuation(
    heard: str, expected: str
) -> None:
    vocab = Vocab(terms=(), aliases=(Alias(heard="herder", written="herdr"),))

    assert _deliver(heard, vocab) == expected


def test_a_multi_word_alias_takes_the_longest_leftmost_run() -> None:
    vocab = Vocab(
        terms=(),
        aliases=(
            Alias(heard="voice", written="VOICE"),
            Alias(heard="voice ink app", written="VoiceInk"),
        ),
    )

    words, edits = deliver(["open", "voice", "ink", "app.", "voice"], vocab)

    assert words == ["open", "VoiceInk.", "VOICE"]
    assert edits == [
        Edit(rule="alias", before="voice ink app.", after="VoiceInk."),
        Edit(rule="alias", before="voice", after="VOICE"),
    ]


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("open voice ink now", "open VoiceInk now"),
        ("open Voiceink.", "open VoiceInk."),
        ("edit modules darwinbase.nix, then", "edit modules/darwin/base.nix, then"),
        ("host akms 25.", "host akms25."),
        ("mac whisper", "MacWhisper"),
    ],
)
def test_snapping_respells_a_run_whose_letters_match_an_identifier(
    heard: str, expected: str
) -> None:
    assert _deliver(heard, Vocab(terms=tuple(TERMS), aliases=())) == expected


def test_a_plain_word_term_never_recases_or_respaces_speech() -> None:
    vocab = Vocab(terms=("check", "Makefile", "herdr"), aliases=())

    assert _deliver("Check the make file and her dr", vocab) == "Check the make file and her dr"


def test_aliases_run_before_snapping() -> None:
    vocab = Vocab(terms=("herdr-ui",), aliases=(Alias(heard="herder", written="herdr"),))

    words, edits = deliver(["herder", "ui"], vocab)

    assert words == ["herdr-ui"]
    assert [edit.rule for edit in edits] == ["alias", "snap"]


def test_snapping_records_no_edit_where_the_words_already_match() -> None:
    words, edits = snap(["VoiceInk"], ("VoiceInk",))

    assert words == ["VoiceInk"]
    assert edits == []


_WORD = st.text(alphabet="abAB1. -/,", min_size=1, max_size=6).filter(lambda w: " " not in w)


@given(words=st.lists(_WORD, max_size=8), terms=st.lists(_WORD, max_size=4))
def test_snapping_changes_nothing_when_no_run_spells_a_term(
    words: list[str], terms: list[str]
) -> None:
    term_keys = {_key(term) for term in terms}
    run_keys = {
        _key("".join(words[start : start + size]))
        for start in range(len(words))
        for size in range(1, 5)
    }
    assume(not term_keys & run_keys)

    assert snap(words, tuple(terms)) == (words, [])


@given(words=st.lists(_WORD, max_size=8), terms=st.lists(_WORD, max_size=4))
def test_snapping_never_changes_the_letters_heard(words: list[str], terms: list[str]) -> None:
    snapped, _ = snap(words, tuple(terms))

    assert _key("".join(snapped)) == _key("".join(words))


def test_a_missing_default_terms_file_means_no_terms(tmp_path: Path) -> None:
    assert TermsFile(tmp_path / "absent.txt", required=False).current() == Vocab((), ())


def test_a_missing_required_terms_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match=r"absent\.txt"):
        TermsFile(tmp_path / "absent.txt", required=True)


def test_a_changed_file_is_reread_and_a_bad_reread_keeps_the_previous_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("herdr\n", encoding="utf-8")
    terms = TermsFile(path, required=True)
    assert terms.current().terms == ("herdr",)

    path.write_text("herdr\nVoiceInk\n", encoding="utf-8")
    os.utime(path, ns=(1, 1))
    assert terms.current().terms == ("herdr", "VoiceInk")

    path.write_text("herdr\n => nothing heard\n", encoding="utf-8")
    with capture_logs() as logs:
        assert terms.current().terms == ("herdr", "VoiceInk")
        assert terms.current().terms == ("herdr", "VoiceInk")

    assert [entry["log_level"] for entry in logs] == ["error"]
    error: object = logs[0]["error"]  # pyright: ignore[reportAny]  # captured log values are Any
    assert "terms.txt:2" in str(error)


def test_a_bad_file_at_startup_is_an_error_naming_the_line(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("ok\nbad =>\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match=r"terms\.txt:2"):
        TermsFile(path, required=False)
