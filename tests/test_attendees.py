from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scribe.attendees import MIN_POINTERS, RIVAL_FACTOR, name_speakers, parse_attendees
from scribe.errors import InputValidationError
from scribe.schema import Word
from scribe.speakers import Claim, relabel
from scribe.turns import turns_from_speakers
from tests.speakers_fakes import FakeSpeakerBackend

if TYPE_CHECKING:
    from scribe.attendees import SpeakerNames


def test_attendees_are_read_in_order_with_their_spacing_trimmed() -> None:
    assert parse_attendees("Connor, Jose,Adam ,  Keigo") == ("Connor", "Jose", "Adam", "Keigo")


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        pytest.param("Connor,,Jose", "empty name", id="empty-item"),
        pytest.param("Connor, Jose,", "empty name", id="trailing-comma"),
        pytest.param("", "names nobody", id="empty-list"),
        pytest.param("  ", "names nobody", id="blank-list"),
        pytest.param("Jose, Connor, jose", "twice", id="duplicate"),
        pytest.param("Connor | Jose", "may not hold", id="pipe"),
        pytest.param("Con\nnor, Jose", "may not hold", id="newline"),
        pytest.param("Connor <spk:1>", "may not hold", id="angle-bracket"),
        pytest.param("Connor >, Jose", "may not hold", id="closing-bracket"),
        pytest.param("Jose, Speaker 2", "looks like a speaker label", id="numbered-label"),
        pytest.param("Speaker ?, Jose", "looks like a speaker label", id="unattributed-label"),
        pytest.param("Jose, sPEAKER  12", "looks like a speaker label", id="label-any-case"),
    ],
)
def test_a_malformed_attendee_list_is_refused(text: str, complaint: str) -> None:
    with pytest.raises(InputValidationError, match=complaint):
        parse_attendees(text)


ATTENDEES = ("Connor", "Jose", "Adam", "Keigo")
Run = tuple[int | None, str]

MEETING: tuple[Run, ...] = (
    (7, "Okay, let's start. Connor, can you share the deck?"),
    (3, "Sure, sharing it now."),
    (7, "Thanks. Jose, what do you think of it?"),
    (5, "Looks good to me."),
    (7, "Great. Connor, one more thing about pricing."),
    (3, "Yes, pricing is settled."),
    (7, "And Jose, the timeline?"),
    (5, "End of month. Keigo sent the notes yesterday."),
)
# One NAME in another case: it still names the attendee, as listed.
ANSWERED = """\
Connor | Connor | next | Connor, can you share the deck?
Jose | Jose | next | Jose, what do you think of it?
connor | Connor | next | Connor, one more thing about pricing.
Jose | Jose | next | And Jose, the timeline?"""


def _words(*runs: Run) -> list[Word]:
    """Words one second apart from (speaker, "words of one run") pairs."""
    said = [(speaker, text) for speaker, line in runs for text in line.split()]
    return [
        Word(text=text, start=float(index), end=index + 0.9, speaker=speaker)
        for index, (speaker, text) in enumerate(said)
    ]


def _named_from_replies(runs: tuple[Run, ...], block: str) -> tuple[SpeakerNames, list[Word]]:
    """Run the pass with every reply echoing its chunk and carrying `block` as its names."""
    words = _words(*runs)

    def trailer(_target: str) -> str:
        return f"<names>\n{block}\n</names>"

    result = relabel(
        [word.text for word in words],
        [word.speaker for word in words],
        FakeSpeakerBackend(trailer=trailer),
        attendees=ATTENDEES,
    )
    return name_speakers(words, result.speakers, result.claims, ATTENDEES), words


def _labels(named: SpeakerNames, words: list[Word]) -> list[str]:
    turns = turns_from_speakers(words, [word.speaker for word in words], named.names)
    return [turn.speaker for turn in turns]


def test_two_answers_each_name_the_labels_that_gave_them() -> None:
    named, words = _named_from_replies(MEETING, ANSWERED)

    assert named.names == {3: "Connor", 5: "Jose"}
    assert named.unassigned == ("Adam", "Keigo")
    assert _labels(named, words) == [
        "Speaker 1",
        "Connor",
        "Speaker 1",
        "Jose",
        "Speaker 1",
        "Connor",
        "Speaker 1",
        "Jose",
    ]
    assert [
        (mention.name, mention.kind, mention.by, mention.points_to, mention.status)
        for mention in named.evidence
    ] == [
        ("Connor", "next", 7, 3, "counted"),
        ("Jose", "next", 7, 5, "counted"),
        ("Connor", "next", 7, 3, "counted"),
        ("Jose", "next", 7, 5, "counted"),
    ]
    first = named.evidence[0]
    assert (first.word, first.time, first.said) == (3, 3.0, "Connor")
    assert named.pointed == {"Connor": {3: 2}, "Jose": {5: 2}, "Adam": {}, "Keigo": {}}
    assert named.says == {"Connor": {7: 2}, "Jose": {7: 2}, "Adam": {}, "Keigo": {}}


def test_an_attendee_only_talked_about_names_no_label() -> None:
    named, words = _named_from_replies(
        MEETING, ANSWERED + "\nKeigo | Keigo | about | Keigo sent the notes yesterday."
    )

    assert named.unassigned == ("Adam", "Keigo")
    assert "Keigo" not in _labels(named, words)
    assert (named.evidence[-1].status, named.evidence[-1].by) == ("counted", 5)
    assert named.says["Keigo"] == {5: 1}


CONTESTED: tuple[Run, ...] = (
    (7, "Connor, can you share the deck?"),
    (3, "Sure, sharing it now."),
    (7, "Connor, are you still with us?"),
    (5, "I think he dropped."),
    (7, "Connor, one more thing about pricing."),
    (3, "Yes, pricing is settled."),
    (7, "Connor, did you hear that one?"),
    (5, "He dropped again, sorry."),
)
SAID_BY_THE_ANSWERER: tuple[Run, ...] = (
    (7, "Connor, can you share the deck?"),
    (3, "Sure, sharing it now."),
    (7, "Connor, one more thing about pricing."),
    (3, "Pricing is settled, Connor told me so yesterday."),
)


@pytest.mark.parametrize(
    ("runs", "block"),
    [
        pytest.param(
            CONTESTED,
            "Connor | Connor | next | Connor, can you share the deck?\n"
            "Connor | Connor | next | Connor, are you still with us?\n"
            "Connor | Connor | next | Connor, one more thing about pricing.\n"
            "Connor | Connor | next | Connor, did you hear that one?",
            id="two-labels-answer-twice-each",
        ),
        pytest.param(
            SAID_BY_THE_ANSWERER,
            "Connor | Connor | next | Connor, can you share the deck?\n"
            "Connor | Connor | next | Connor, one more thing about pricing.\n"
            "Connor | Connor | about | settled, Connor told me so",
            id="the-answerer-says-the-name",
        ),
    ],
)
def test_contradicted_evidence_names_no_label(runs: tuple[Run, ...], block: str) -> None:
    named, words = _named_from_replies(runs, block)

    assert named.names == {}
    assert "Connor" in named.unassigned
    assert all(mention.status == "counted" for mention in named.evidence)
    assert "Connor" not in _labels(named, words)


def test_a_label_two_names_win_stays_unnamed() -> None:
    runs: tuple[Run, ...] = (
        (7, "Connor, can you share the deck?"),
        (3, "Sure, sharing it now."),
        (7, "Jose, can you share the notes?"),
        (3, "Sharing those as well."),
        (7, "Connor, one more thing about pricing."),
        (3, "Yes, pricing is settled."),
        (7, "Jose, and the timeline?"),
        (3, "End of month."),
    )
    block = (
        "Connor | Connor | next | Connor, can you share the deck?\n"
        "Jose | Jose | next | Jose, can you share the notes?\n"
        "Connor | Connor | next | Connor, one more thing about pricing.\n"
        "Jose | Jose | next | Jose, and the timeline?"
    )

    named, _ = _named_from_replies(runs, block)

    assert named.pointed["Connor"] == named.pointed["Jose"] == {3: 2}
    assert named.names == {}
    assert named.unassigned == ATTENDEES


_TOPICS = ("pricing", "hiring", "travel", "budget", "timing")


def _asked(answerers: tuple[int, ...]) -> tuple[tuple[Run, ...], str]:
    """Connor asked once per answerer, each answer given by that id, and the model's lines."""
    runs: list[Run] = []
    lines: list[str] = []
    for answerer, topic in zip(answerers, _TOPICS, strict=False):
        runs += [(7, f"Connor, what about {topic}?"), (answerer, f"The {topic} is fine.")]
        lines.append(f"Connor | Connor | next | Connor, what about {topic}?")
    return tuple(runs), "\n".join(lines)


# Literal counts rather than the constants, so a changed constant fails here.
@pytest.mark.parametrize(
    ("answerers", "names"),
    [
        pytest.param((3,), {}, id="one-pointer-is-not-enough"),
        pytest.param((3, 3), {3: "Connor"}, id="two-pointers-are"),
        pytest.param((3, 3, 5), {3: "Connor"}, id="twice-the-rival"),
        pytest.param((3, 3, 3, 5, 5), {}, id="less-than-twice-the-rival"),
    ],
)
def test_a_label_needs_two_pointers_and_twice_any_rivals(
    answerers: tuple[int, ...], names: dict[int, str]
) -> None:
    named, _ = _named_from_replies(*_asked(answerers))

    assert named.names == names


def test_a_speaker_naming_themself_twice_is_named() -> None:
    runs: tuple[Run, ...] = (
        (7, "Morning, everyone, quick update first."),
        (3, "This is Connor, the deck is ready."),
        (7, "Great, thanks for that."),
        (3, "Connor again, one more thing on pricing."),
    )
    block = (
        "Connor | Connor | self | This is Connor, the deck\n"
        "Connor | Connor | self | Connor again, one more thing"
    )

    named, _ = _named_from_replies(runs, block)

    assert named.names == {3: "Connor"}
    assert named.says["Connor"] == {}


@pytest.mark.parametrize(
    "block",
    [
        pytest.param("", id="no-one-named"),
        pytest.param(
            "Maria | Connor | next | Connor, can you share the deck?\n"
            "Maria | Connor | next | Connor, one more thing about pricing.",
            id="off-the-list",
        ),
        pytest.param(
            "Connor | Connor | next | Connor, will you share the slides?\n"
            "Connor | Connor | next | Connor, two more things about pricing.",
            id="not-in-the-words",
        ),
    ],
)
def test_an_attendee_list_the_words_do_not_bear_out_names_nobody(block: str) -> None:
    named, words = _named_from_replies(MEETING, block)
    unnamed = turns_from_speakers(words, [word.speaker for word in words])

    assert named.names == {}
    assert named.unassigned == ATTENDEES
    assert turns_from_speakers(words, [word.speaker for word in words], named.names) == unnamed
    assert {mention.reason for mention in named.evidence} <= {"not_attendee", "unlocated"}


def _claim(words: list[Word], line: str) -> Claim:
    """A names line as the pass reads it, over a single chunk holding every word."""
    fields = [field.strip() for field in line.split("|", 3)]
    name, said, kind, quote = fields + [""] * (4 - len(fields))
    return Claim(0, 0, len(words), name, said, kind, quote)


def _reasons(runs: tuple[Run, ...], *lines: str) -> list[str | None]:
    words = _words(*runs)
    claims = [_claim(words, line) for line in lines]
    named = name_speakers(words, [word.speaker for word in words], claims, ATTENDEES)
    return [mention.reason for mention in named.evidence]


def test_a_quote_across_a_run_boundary_is_found_without_its_tag() -> None:
    # The tag the prompt shows between runs, glued to the words around it.
    assert _reasons(
        MEETING,
        "Jose | Jose | next | pricing is settled.<spk:0>And Jose, the timeline?",
    ) == [None]


def test_a_quote_found_twice_is_ambiguous() -> None:
    runs: tuple[Run, ...] = (
        (7, "Thanks, Connor, see you."),
        (3, "Bye."),
        (7, "Thanks, Connor, see you."),
    )

    assert _reasons(runs, "Connor | Connor | about | Thanks, Connor, see you.") == ["ambiguous"]


def test_a_name_said_outside_its_quote_is_dropped() -> None:
    assert _reasons(MEETING, "Jose | Jose | next | Connor, can you share the deck?") == [
        "said_outside_quote"
    ]


def test_a_pointer_at_unattributed_words_is_dropped() -> None:
    runs: tuple[Run, ...] = (
        (7, "Okay. Connor, can you share the deck?"),
        (None, "Sure, sharing it."),
        (3, "There it is."),
    )

    assert _reasons(runs, "Connor | Connor | next | Connor, can you share the deck?") == [
        "unattributed"
    ]


def test_a_pointer_from_unattributed_words_is_dropped() -> None:
    # The asker's own words lost their speaker: counted, the answers would name
    # the label beside them, and no check could see whose words they were.
    runs: tuple[Run, ...] = (
        (1, "Okay then, next item."),
        (None, "Adam, what do you think?"),
        (2, "Looks fine to me."),
        (1, "Right, and pricing."),
        (None, "Adam, what do you think about pricing?"),
        (2, "Also fine."),
    )
    lines = (
        "Adam | Adam | next | next item. Adam, what do you think?",
        "Adam | Adam | previous | Adam, what do you think about pricing?",
    )

    assert _reasons(runs, *lines) == ["unattributed", "unattributed"]

    assert _reasons(
        MEETING,
        "Keigo | Keigo | next | Keigo sent the notes yesterday.",
        "Connor | Connor | previous | Okay, let's start. Connor, can",
    ) == ["no_turn", "no_turn"]


def test_a_name_counts_once_per_turn() -> None:
    runs: tuple[Run, ...] = (
        (7, "Connor, can you share the deck? Connor, are you there?"),
        (3, "Sorry, sharing it now."),
    )

    # The same place listed twice, and a second place in the same turn.
    assert _reasons(
        runs,
        "Connor | Connor | next | Connor, can you share the deck?",
        "Connor | Connor | next | Connor, can you share the",
        "Connor | Connor | next | deck? Connor, are you there?",
    ) == [None, "repeat", "repeat"]


def test_an_unknown_kind_and_a_short_line_are_dropped() -> None:
    assert _reasons(
        MEETING,
        "Connor | Connor | addressed | Connor, can you share the deck?",
        "Connor | Connor | next",
        "Jose | <spk:1> | next | Jose, what do you think of it?",
    ) == ["bad_kind", "bad_line", "bad_line"]


# Few runs and many lines, two names and one stranger: sparser draws almost
# never put two counted pointers on one label, and so never name anyone.
_RUNS = st.lists(
    st.tuples(st.sampled_from([None, 0, 1, 2]), st.integers(min_value=1, max_value=5)),
    min_size=2,
    max_size=6,
)
_LINES = st.lists(
    st.tuples(
        st.sampled_from(["Connor", "Jose", "Maria"]),
        st.sampled_from(["next", "previous", "self", "about", "later"]),
        st.integers(min_value=0, max_value=29),
        st.integers(min_value=1, max_value=4),
        # SAID's offset into the quote; past its end, SAID lies outside it.
        st.integers(min_value=0, max_value=3),
    ),
    min_size=16,
    max_size=40,
)


@given(runs=_RUNS, lines=_LINES)
def test_no_name_lands_on_two_labels_or_without_its_pointers(
    runs: list[tuple[int | None, int]], lines: list[tuple[str, str, int, int, int]]
) -> None:
    spoken = [speaker for speaker, count in runs for _ in range(count)]
    words = [
        Word(text=f"w{index}", start=float(index), end=index + 0.9, speaker=speaker)
        for index, speaker in enumerate(spoken)
    ]
    texts = [word.text for word in words]
    claims = [
        Claim(
            0,
            0,
            len(texts),
            name,
            texts[(at + said) % len(texts)],
            kind,
            " ".join(texts[at % len(texts) : at % len(texts) + size]),
        )
        for name, kind, at, size, said in lines
    ]

    named = name_speakers(words, spoken, claims, ATTENDEES)

    assert None not in named.names
    assert len(set(named.names.values())) == len(named.names)
    assert set(named.names.values()) <= set(ATTENDEES)
    assert all(
        mention.by is not None
        for mention in named.evidence
        if mention.status == "counted" and mention.points_to is not None
    )
    for speaker, name in named.names.items():
        counted = [
            mention
            for mention in named.evidence
            if mention.status == "counted" and mention.name == name
        ]
        pointers = Counter(
            mention.points_to for mention in counted if mention.points_to is not None
        )
        assert pointers[speaker] >= MIN_POINTERS
        assert all(
            pointers[speaker] >= RIVAL_FACTOR * count
            for other, count in pointers.items()
            if other != speaker
        )
        assert not [
            mention for mention in counted if mention.by == speaker and mention.kind != "self"
        ]
    unnamed = turns_from_speakers(words, spoken)
    renamed = turns_from_speakers(words, spoken, named.names)
    assert [turn.model_copy(update={"speaker": ""}) for turn in renamed] == [
        turn.model_copy(update={"speaker": ""}) for turn in unnamed
    ]
