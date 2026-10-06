from __future__ import annotations

import os
import string
import unicodedata
from itertools import pairwise
from typing import TYPE_CHECKING

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from structlog.testing import capture_logs

from scribe import vocab as vocab_module
from scribe.errors import InputValidationError
from scribe.vocab import (
    MAX_RUN_WORDS,
    MIN_SNAP_KEY,
    Alias,
    Edit,
    TermsFile,
    Vocab,
    deliver,
    parse_terms,
    snap,
)

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


def test_an_alias_heard_with_punctuation_needs_that_punctuation() -> None:
    vocab = Vocab(terms=(), aliases=(Alias(heard="ok.", written="okay"),))

    assert _deliver("ok then, ok. done", vocab) == "ok then, okay done"


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


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("open scope pkg,", "open @scope/pkg,"),
        ("open @scope pkg", "open @scope/pkg"),
        ("see c++", "see c++"),
    ],
)
def test_a_terms_own_punctuation_is_written_once(heard: str, expected: str) -> None:
    assert _deliver(heard, Vocab(terms=("@scope/pkg", "c++"), aliases=())) == expected


def test_an_alias_heard_as_punctuation_alone_never_matches() -> None:
    vocab = Vocab(terms=(), aliases=(Alias(heard="??", written="huh"),))

    assert _deliver("what ?? now", vocab) == "what ?? now"


def test_an_unreadable_terms_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="cannot read terms file"):
        TermsFile(tmp_path, required=False)


@pytest.mark.parametrize(
    ("term", "heard"),
    [
        ("c++", "plan c is fine"),
        ("C#", "vitamin C works"),
        (".NET", "net income rose"),
        ("GET", "get the file"),
        ("TODO", "what is left to do."),
        ("README", "can you read me the file"),
        ("__init__", "put a print in it"),
        ("--force", "force push it"),
        ("IT", "it works"),
        ("CD", "cd into src"),
        ("U.S.", "tell us more"),
        ("2.0", "wait 20 minutes"),
        ("pull request", "Pull request is open"),
        (".bashrc", "open bashrc,"),
    ],
)
def test_a_term_that_spells_ordinary_speech_never_rewrites_it(term: str, heard: str) -> None:
    assert _deliver(heard, Vocab(terms=(term,), aliases=())) == heard


@pytest.mark.parametrize(
    ("heard", "term"),
    [
        ("my voice. Ink is new", "VoiceInk"),
        ("is it voice? Ink later", "VoiceInk"),
        ("akms, 25 of them", "akms25"),
        ("akms 2.5 now", "akms25"),
    ],
)
def test_punctuation_inside_a_run_that_the_term_lacks_keeps_it_apart(heard: str, term: str) -> None:
    assert _deliver(heard, Vocab(terms=(term,), aliases=())) == heard


@pytest.mark.parametrize(
    ("heard", "term", "expected"),
    [
        (unicodedata.normalize("NFD", "the café2 file"), "cafe2", None),
        ("İstanbul5 x", "istanbul5", None),
        (unicodedata.normalize("NFD", "the café2 file"), "café2", "the café2 file"),
    ],
)
def test_letters_that_differ_by_a_mark_never_snap_and_normal_forms_do(
    heard: str, term: str, expected: str | None
) -> None:
    assert _deliver(heard, Vocab(terms=(term,), aliases=())) == (expected or heard)


def test_a_comment_can_follow_a_line() -> None:
    vocab = parse_terms("herdr # my tool\nC#\nherder => herdr\t# heard often\n", where="t")

    assert vocab.terms == ("herdr", "C#")
    assert vocab.aliases == (Alias(heard="herder", written="herdr"),)


def test_each_alias_is_read_once_per_delivery_not_once_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    edges = vocab_module._edges  # pyright: ignore[reportPrivateUsage]  # counting its calls measures the work

    def counted(text: str) -> tuple[str, str, str]:
        calls.append(text)
        return edges(text)

    monkeypatch.setattr(vocab_module, "_edges", counted)
    words = ("so I was thinking " * 10).split()
    aliases = tuple(Alias(heard=f"heard{index} word", written=f"W{index}") for index in range(50))

    deliver(words, Vocab(terms=(), aliases=aliases))

    assert len(calls) <= len(aliases) + MAX_RUN_WORDS * len(words)


_SPEECH_LEAD = ["", "(", '"', "\u201c"]
_SPEECH_TRAIL = ["", ",", ".", ")", "?", "\u201d"]


@st.composite
def _identifiers(draw: st.DrawFn) -> tuple[str, str, str]:
    """A term snapping targets, as (leading punctuation, core, trailing punctuation).

    The core has a joint inside it, a digit, a capital after a lowercase letter
    or punctuation, and at least 3 letters and digits; its edges share no
    character with `_SPEECH_LEAD` or `_SPEECH_TRAIL`.
    """
    first = draw(st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=4))
    joint = draw(st.sampled_from(["", " ", ".", "/", "-", "_"]))
    if joint == "":
        second = draw(st.sampled_from(string.ascii_uppercase))
    elif joint == " ":
        second = draw(st.sampled_from(string.digits))
    else:
        second = draw(st.sampled_from(string.ascii_letters + string.digits))
    rest = draw(st.text(alphabet=string.ascii_letters + string.digits, max_size=3))
    lead = draw(st.sampled_from(["", ".", "@", "--"]))
    trail = draw(st.sampled_from(["", "+", "#", "++"]))
    return lead, first + joint + second + rest, trail


def _spoken(data: st.DataObject, key: str, words: int = 4) -> list[str]:
    """`key` cut into 1 to `words` words, each in some case, as a speaker might be heard."""
    # 4 written out, not the module's cap, so a lowered cap fails here.
    cuts = data.draw(st.sets(st.integers(1, len(key) - 1), max_size=words - 1))
    pieces = [key[start:end] for start, end in pairwise([0, *sorted(cuts), len(key)])]
    return [data.draw(st.sampled_from([piece, piece.upper(), piece.title()])) for piece in pieces]


@given(data=st.data())
def test_an_identifier_heard_in_pieces_snaps_back_to_the_first_term_spelling_it(
    data: st.DataObject,
) -> None:
    lead, core, trail = data.draw(_identifiers())
    term = lead + core + trail
    words = _spoken(data, _key(term))
    before = data.draw(st.sampled_from(_SPEECH_LEAD))
    after = data.draw(st.sampled_from(_SPEECH_TRAIL))
    words[0] = before + (lead if data.draw(st.booleans()) else "") + words[0]
    words[-1] += (trail if data.draw(st.booleans()) else "") + after
    terms = (term, f"~{term}~")
    expected = (before + term + after).split()

    assert snap(words, terms)[0] == expected
    assert snap(expected, terms) == (expected, [])


@given(data=st.data())
def test_the_longest_run_spelling_a_term_wins(data: st.DataObject) -> None:
    _, short, _ = data.draw(_identifiers())
    _, tail, _ = data.draw(_identifiers())
    long = f"{short}/{tail}"
    words = _spoken(data, _key(short), 2) + _spoken(data, _key(tail), 2)

    assert snap(words, (short, long))[0] == long.split()


@given(data=st.data())
def test_a_run_broken_by_punctuation_the_term_lacks_is_left_alone(data: st.DataObject) -> None:
    lead, core, trail = data.draw(_identifiers())
    words = _spoken(data, _key(core))
    assume(len(words) > 1)
    index = data.draw(st.integers(0, len(words) - 2))
    words[index] += data.draw(st.sampled_from([",", "?", "!", ";", ":"]))

    assert snap(words, (lead + core + trail,)) == (words, [])


@st.composite
def _ordinary(draw: st.DrawFn) -> str:
    """A term spelled like ordinary speech: no joint inside it, or too short to name."""
    kind = draw(st.sampled_from(["words", "short", "number"]))
    if kind == "short":
        return draw(
            st.text(alphabet="abAB1.+#-", min_size=1, max_size=4).filter(
                lambda text: 0 < len(_key(text)) < MIN_SNAP_KEY
            )
        )
    if kind == "number":
        return draw(st.from_regex(r"[0-9]{1,3}(\.[0-9]{1,2})?", fullmatch=True))
    words = draw(
        st.lists(
            st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5), min_size=1, max_size=2
        )
    )
    cased = [draw(st.sampled_from([word, word.upper(), word.title()])) for word in words]
    lead = draw(st.sampled_from(["", ".", "--", "__"]))
    trail = draw(st.sampled_from(["", "++", "#", "__"]))
    return lead + " ".join(cased) + trail


@given(data=st.data())
def test_a_term_spelled_like_ordinary_speech_is_never_a_target(data: st.DataObject) -> None:
    term = data.draw(_ordinary())
    words = _spoken(data, _key(term)) if len(_key(term)) > 1 else [_key(term)]

    assert snap(words, (term,)) == (words, [])


@given(
    heard=st.lists(
        st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=5), min_size=1, max_size=3
    ),
    data=st.data(),
)
def test_an_alias_matches_its_heard_words_in_any_case_and_runs_before_snapping(
    heard: list[str], data: st.DataObject
) -> None:
    assume("ui" not in heard and heard != ["zq"])
    said = [data.draw(st.sampled_from([word, word.upper(), word.title()])) for word in heard]
    vocab = Vocab(terms=("zq-ui",), aliases=(Alias(heard=" ".join(heard), written="zq"),))

    assert deliver([*said, "ui"], vocab)[0] == ["zq-ui"]
