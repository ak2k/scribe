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
from scribe.turns import turns_from_speakers, word_speakers
from tests.speakers_fakes import FakeSpeakerBackend

if TYPE_CHECKING:
    from scribe.attendees import SpeakerNames


def test_attendees_are_read_in_order_with_their_spacing_trimmed() -> None:
    assert parse_attendees("Alice, Bruno,Carol ,  Dmitri") == ("Alice", "Bruno", "Carol", "Dmitri")


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        pytest.param("Alice,,Bruno", "empty name", id="empty-item"),
        pytest.param("Alice, Bruno,", "empty name", id="trailing-comma"),
        pytest.param("", "names nobody", id="empty-list"),
        pytest.param("  ", "names nobody", id="blank-list"),
        pytest.param("Bruno, Alice, bruno", "twice", id="duplicate"),
        pytest.param(
            "Zo\N{LATIN SMALL LETTER E WITH ACUTE}, Zoe\N{COMBINING ACUTE ACCENT}",
            "twice",
            id="composed-and-decomposed",
        ),
        pytest.param("Alice | Bruno", "may not hold", id="pipe"),
        pytest.param("Ali\nce, Bruno", "may not hold", id="newline"),
        pytest.param("Alice <spk:1>", "may not hold", id="angle-bracket"),
        pytest.param("Alice >, Bruno", "may not hold", id="closing-bracket"),
        pytest.param("Bruno, Speaker 2", "looks like a speaker label", id="numbered-label"),
        pytest.param("Speaker ?, Bruno", "looks like a speaker label", id="unattributed-label"),
        pytest.param("Bruno, sPEAKER  12", "looks like a speaker label", id="label-any-case"),
    ],
)
def test_a_malformed_attendee_list_is_refused(text: str, complaint: str) -> None:
    with pytest.raises(InputValidationError, match=complaint):
        parse_attendees(text)


ATTENDEES = ("Alice", "Bruno", "Carol", "Dmitri")
Run = tuple[int | None, str]

MEETING: tuple[Run, ...] = (
    (7, "Okay, let's start. Alice, can you share the deck?"),
    (3, "Sure, sharing it now."),
    (7, "Thanks. Bruno, what do you think of it?"),
    (5, "Looks good to me."),
    (7, "Great. Alice, one more thing about pricing."),
    (3, "Yes, pricing is settled."),
    (7, "And Bruno, the timeline?"),
    (5, "End of month. Dmitri sent the notes yesterday."),
)
# One NAME in another case: it still names the attendee, as listed.
ANSWERED = """\
Alice | Alice | next | Alice, can you share the deck?
Bruno | Bruno | next | Bruno, what do you think of it?
alice | Alice | next | Alice, one more thing about pricing.
Bruno | Bruno | next | And Bruno, the timeline?"""


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

    assert named.names == {3: "Alice", 5: "Bruno"}
    assert named.unassigned == ("Carol", "Dmitri")
    assert _labels(named, words) == [
        "Speaker 1",
        "Alice",
        "Speaker 1",
        "Bruno",
        "Speaker 1",
        "Alice",
        "Speaker 1",
        "Bruno",
    ]
    assert [
        (mention.name, mention.kind, mention.by, mention.points_to, mention.status)
        for mention in named.evidence
    ] == [
        ("Alice", "next", 7, 3, "counted"),
        ("Bruno", "next", 7, 5, "counted"),
        ("Alice", "next", 7, 3, "counted"),
        ("Bruno", "next", 7, 5, "counted"),
    ]
    first = named.evidence[0]
    assert (first.word, first.time, first.said) == (3, 3.0, "Alice")
    assert named.pointed == {"Alice": {3: 2}, "Bruno": {5: 2}, "Carol": {}, "Dmitri": {}}
    assert named.says == {"Alice": {7: 2}, "Bruno": {7: 2}, "Carol": {}, "Dmitri": {}}


def test_an_attendee_only_talked_about_names_no_label() -> None:
    named, words = _named_from_replies(
        MEETING, ANSWERED + "\nDmitri | Dmitri | about | Dmitri sent the notes yesterday."
    )

    assert named.unassigned == ("Carol", "Dmitri")
    assert "Dmitri" not in _labels(named, words)
    assert (named.evidence[-1].status, named.evidence[-1].by) == ("counted", 5)
    assert named.says["Dmitri"] == {5: 1}


CONTESTED: tuple[Run, ...] = (
    (7, "Alice, can you share the deck?"),
    (3, "Sure, sharing it now."),
    (7, "Alice, are you still with us?"),
    (5, "I think he dropped."),
    (7, "Alice, one more thing about pricing."),
    (3, "Yes, pricing is settled."),
    (7, "Alice, did you hear that one?"),
    (5, "He dropped again, sorry."),
)
SAID_BY_THE_ANSWERER: tuple[Run, ...] = (
    (7, "Alice, can you share the deck?"),
    (3, "Sure, sharing it now."),
    (7, "Alice, one more thing about pricing."),
    (3, "Pricing is settled, Alice told me so yesterday."),
)


@pytest.mark.parametrize(
    ("runs", "block"),
    [
        pytest.param(
            CONTESTED,
            "Alice | Alice | next | Alice, can you share the deck?\n"
            "Alice | Alice | next | Alice, are you still with us?\n"
            "Alice | Alice | next | Alice, one more thing about pricing.\n"
            "Alice | Alice | next | Alice, did you hear that one?",
            id="two-labels-answer-twice-each",
        ),
        pytest.param(
            SAID_BY_THE_ANSWERER,
            "Alice | Alice | next | Alice, can you share the deck?\n"
            "Alice | Alice | next | Alice, one more thing about pricing.\n"
            "Alice | Alice | about | settled, Alice told me so",
            id="the-answerer-says-the-name",
        ),
    ],
)
def test_contradicted_evidence_names_no_label(runs: tuple[Run, ...], block: str) -> None:
    named, words = _named_from_replies(runs, block)

    assert named.names == {}
    assert "Alice" in named.unassigned
    assert all(mention.status == "counted" for mention in named.evidence)
    assert "Alice" not in _labels(named, words)


@pytest.mark.parametrize(
    ("runs", "block"),
    [
        pytest.param(
            (
                (7, "I agree with Alice completely today."),
                (3, "Alice please start the update now."),
                (7, "I agree with Alice completely again."),
                (3, "Alice please cover the budget now."),
            ),
            "Alice | Alice | next | with Alice completely today. Alice please start\n"
            "Alice | Alice | next | with Alice completely again. Alice please cover",
            id="said-twice-in-a-quote-across-a-handoff",
        ),
        pytest.param(
            (
                *SAID_BY_THE_ANSWERER[:3],
                (3, "Yes, pricing is settled."),
                (7, "Thanks Alice."),
                (3, "Alice will join later."),
            ),
            "Alice | Alice | next | Alice, can you share the deck?\n"
            "Alice | Alice | next | Alice, one more thing about pricing.\n"
            "Alice | Alice | about | Thanks Alice. Alice will join later.",
            id="said-twice-in-a-quote-by-the-answerer",
        ),
        pytest.param(
            (
                *SAID_BY_THE_ANSWERER[:3],
                (3, "Thanks, Alice, see you."),
                (7, "Thanks, Alice, see you."),
            ),
            "Alice | Alice | next | Alice, can you share the deck?\n"
            "Alice | Alice | next | Alice, one more thing about pricing.\n"
            "Alice | Alice | about | Thanks, Alice, see you.",
            id="quote-found-twice-once-by-the-answerer",
        ),
    ],
)
def test_a_mention_that_fits_two_places_counts_against_both_speakers(
    runs: tuple[Run, ...], block: str
) -> None:
    named, _ = _named_from_replies(runs, block)

    assert named.names == {}
    assert "ambiguous" in [mention.reason for mention in named.evidence]
    assert named.says["Alice"].get(3)


@pytest.mark.parametrize(
    ("after", "reason"),
    [
        pytest.param((), "no_turn", id="nobody-answers"),
        pytest.param(((None, "Sure, on it."),), "unattributed", id="nobody-known-answers"),
    ],
)
def test_a_pointer_that_cannot_count_still_counts_against_its_speaker(
    after: tuple[Run, ...], reason: str
) -> None:
    runs: tuple[Run, ...] = (
        *SAID_BY_THE_ANSWERER[:3],
        (3, "Pricing is settled. Alice, can you check the budget?"),
        *after,
    )
    block = (
        "Alice | Alice | next | Alice, can you share the deck?\n"
        "Alice | Alice | next | Alice, one more thing about pricing.\n"
        "Alice | Alice | next | Alice, can you check the budget?"
    )

    named, _ = _named_from_replies(runs, block)

    assert named.names == {}
    assert [mention.reason for mention in named.evidence] == [None, None, reason]
    assert named.says["Alice"] == {7: 2, 3: 1}


def test_a_label_two_names_win_stays_unnamed() -> None:
    runs: tuple[Run, ...] = (
        (7, "Alice, can you share the deck?"),
        (3, "Sure, sharing it now."),
        (7, "Bruno, can you share the notes?"),
        (3, "Sharing those as well."),
        (7, "Alice, one more thing about pricing."),
        (3, "Yes, pricing is settled."),
        (7, "Bruno, and the timeline?"),
        (3, "End of month."),
    )
    block = (
        "Alice | Alice | next | Alice, can you share the deck?\n"
        "Bruno | Bruno | next | Bruno, can you share the notes?\n"
        "Alice | Alice | next | Alice, one more thing about pricing.\n"
        "Bruno | Bruno | next | Bruno, and the timeline?"
    )

    named, _ = _named_from_replies(runs, block)

    assert named.pointed["Alice"] == named.pointed["Bruno"] == {3: 2}
    assert named.names == {}
    assert named.unassigned == ATTENDEES


def test_a_mention_does_not_hide_a_pointer_later_in_its_turn() -> None:
    # As CONTESTED, with Alice talked about just before he is asked in the
    # second turn: listing that mention must not decide the contest.
    runs: tuple[Run, ...] = (
        (7, "Alice, can you share the deck?"),
        (3, "Sure, sharing it now."),
        (7, "Alice's deck looks great. Alice, are you still with us?"),
        (5, "I think he dropped."),
        (7, "Alice, one more thing about pricing."),
        (3, "Yes, pricing is settled."),
        (7, "Alice, did you hear that one?"),
        (5, "He dropped again, sorry."),
    )
    block = (
        "Alice | Alice's | about | Alice's deck looks great.\n"
        "Alice | Alice | next | Alice, can you share the deck?\n"
        "Alice | Alice | next | Alice, are you still with us?\n"
        "Alice | Alice | next | Alice, one more thing about pricing.\n"
        "Alice | Alice | next | Alice, did you hear that one?"
    )

    named, _ = _named_from_replies(runs, block)

    assert [mention.reason for mention in named.evidence] == [None] * 5
    assert named.pointed["Alice"] == {3: 2, 5: 2}
    assert named.names == {}


def test_a_pointer_does_not_hide_one_at_another_label_later_in_its_turn() -> None:
    runs: tuple[Run, ...] = (
        (5, "Ok sure, sounds good."),
        (7, "Thanks Alice for that. Now go ahead please, Alice."),
        (3, "Here is my update."),
        (5, "Ok sure, sounds fine."),
        (7, "Thanks Alice for this. Now go on please, Alice."),
        (3, "Another update from me."),
    )
    block = (
        "Alice | Alice | previous | Thanks Alice for that.\n"
        "Alice | Alice | next | Now go ahead please, Alice.\n"
        "Alice | Alice | previous | Thanks Alice for this.\n"
        "Alice | Alice | next | Now go on please, Alice."
    )

    named, _ = _named_from_replies(runs, block)

    assert named.names == {}
    assert named.pointed["Alice"] == {5: 2, 3: 2}
    assert [mention.reason for mention in named.evidence] == [None] * 4


@pytest.mark.parametrize(
    ("runs", "block", "names"),
    [
        pytest.param(
            (
                (7, "Alice, can you share the deck?"),
                (9, "Yeah."),
                (3, "Sure, sharing it now."),
                (7, "Alice, is pricing settled?"),
                (9, "Right."),
                (3, "Yes, pricing is settled."),
            ),
            "Alice | Alice | next | Alice, can you share the deck?\n"
            "Alice | Alice | next | Alice, is pricing settled?",
            {},
            id="a-third-speaker-answers-after-it",
        ),
        pytest.param(
            (
                (3, "Here is my update on pricing."),
                (9, "Yeah."),
                (7, "Thanks, Alice, that helps."),
                (3, "And one more on hiring."),
                (9, "Right."),
                (7, "Thanks again, Alice, noted."),
            ),
            "Alice | Alice | previous | Thanks, Alice, that helps.\n"
            "Alice | Alice | previous | Thanks again, Alice, noted.",
            {},
            id="a-third-speaker-spoke-before-it",
        ),
        pytest.param(
            (
                (7, "Alice, are you ready?"),
                (3, "Yes."),
                (7, "Great. Alice, is pricing settled?"),
                (3, "Yes."),
                (7, "Good, moving on."),
            ),
            "Alice | Alice | next | Alice, are you ready?\n"
            "Alice | Alice | next | Alice, is pricing settled?",
            {3: "Alice"},
            id="the-speaker-comes-back-after-it",
        ),
    ],
)
def test_a_pointer_at_a_backchannel_counts_only_where_no_third_speaker_is_past_it(
    runs: tuple[Run, ...], block: str, names: dict[int, str]
) -> None:
    words = _words(*runs)
    # The turns keep each short reply after a sentence end as a turn of its own.
    assert word_speakers(words) == [word.speaker for word in words]

    named, _ = _named_from_replies(runs, block)

    assert named.names == names
    assert named.says["Alice"] == {7: 2}


_TOPICS = ("pricing", "hiring", "travel", "budget", "timing")


def _asked(answerers: tuple[int, ...]) -> tuple[tuple[Run, ...], str]:
    """Alice asked once per answerer, each answer given by that id, and the model's lines."""
    runs: list[Run] = []
    lines: list[str] = []
    for answerer, topic in zip(answerers, _TOPICS, strict=False):
        runs += [(7, f"Alice, what about {topic}?"), (answerer, f"The {topic} is fine.")]
        lines.append(f"Alice | Alice | next | Alice, what about {topic}?")
    return tuple(runs), "\n".join(lines)


# Literal counts rather than the constants, so a changed constant fails here.
@pytest.mark.parametrize(
    ("answerers", "names"),
    [
        pytest.param((3,), {}, id="one-pointer-is-not-enough"),
        pytest.param((3, 3), {3: "Alice"}, id="two-pointers-are"),
        pytest.param((3, 3, 5), {3: "Alice"}, id="twice-the-rival"),
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
        (3, "This is Alice, the deck is ready."),
        (7, "Great, thanks for that."),
        (3, "Alice again, one more thing on pricing."),
    )
    block = (
        "Alice | Alice | self | This is Alice, the deck\n"
        "Alice | Alice | self | Alice again, one more thing"
    )

    named, _ = _named_from_replies(runs, block)

    assert named.names == {3: "Alice"}
    assert named.says["Alice"] == {}


@pytest.mark.parametrize(
    "block",
    [
        pytest.param("", id="no-one-named"),
        pytest.param(
            "Maria | Alice | next | Alice, can you share the deck?\n"
            "Maria | Alice | next | Alice, one more thing about pricing.",
            id="off-the-list",
        ),
        pytest.param(
            "Alice | Alice | next | Alice, will you share the slides?\n"
            "Alice | Alice | next | Alice, two more things about pricing.",
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
        "Bruno | Bruno | next | pricing is settled.<spk:0>And Bruno, the timeline?",
    ) == [None]


def test_a_quote_found_twice_is_ambiguous() -> None:
    runs: tuple[Run, ...] = (
        (7, "Thanks, Alice, see you."),
        (3, "Bye."),
        (7, "Thanks, Alice, see you."),
    )

    assert _reasons(runs, "Alice | Alice | about | Thanks, Alice, see you.") == ["ambiguous"]


_COMPOSED = "Zo\N{LATIN SMALL LETTER E WITH ACUTE}"
_DECOMPOSED = "Zoe\N{COMBINING ACUTE ACCENT}"


def test_a_name_is_one_attendee_however_its_accents_are_encoded() -> None:
    words = _words((7, f"So, {_COMPOSED}, what do you think?"), (5, "Looks good to me."))
    claim = _claim(words, f"{_COMPOSED} | {_COMPOSED} | next | So, {_COMPOSED}, what do you")

    named = name_speakers(words, [word.speaker for word in words], [claim], (_DECOMPOSED,))

    assert parse_attendees(_DECOMPOSED) == (_COMPOSED,)
    assert [(mention.reason, mention.points_to) for mention in named.evidence] == [(None, 5)]


def test_a_name_said_outside_its_quote_is_dropped() -> None:
    assert _reasons(MEETING, "Bruno | Bruno | next | Alice, can you share the deck?") == [
        "said_outside_quote"
    ]


def test_a_pointer_at_unattributed_words_is_dropped() -> None:
    runs: tuple[Run, ...] = (
        (7, "Okay. Alice, can you share the deck?"),
        (None, "Sure, sharing it."),
        (3, "There it is."),
    )

    assert _reasons(runs, "Alice | Alice | next | Alice, can you share the deck?") == [
        "unattributed"
    ]


def test_a_pointer_from_unattributed_words_is_dropped() -> None:
    # The asker's own words lost their speaker: counted, the answers would name
    # the label beside them, and no check could see whose words they were.
    runs: tuple[Run, ...] = (
        (1, "Okay then, next item."),
        (None, "Carol, what do you think?"),
        (2, "Looks fine to me."),
        (1, "Right, and pricing."),
        (None, "Carol, what do you think about pricing?"),
        (2, "Also fine."),
    )
    lines = (
        "Carol | Carol | next | next item. Carol, what do you think?",
        "Carol | Carol | previous | Carol, what do you think about pricing?",
    )

    assert _reasons(runs, *lines) == ["unattributed", "unattributed"]

    assert _reasons(
        MEETING,
        "Dmitri | Dmitri | next | Dmitri sent the notes yesterday.",
        "Alice | Alice | previous | Okay, let's start. Alice, can",
    ) == ["no_turn", "no_turn"]


def test_a_name_counts_once_per_turn() -> None:
    runs: tuple[Run, ...] = (
        (7, "Alice, can you share the deck? Alice, are you there?"),
        (3, "Sorry, sharing it now."),
    )

    # The same place listed twice, and a second place in the same turn.
    assert _reasons(
        runs,
        "Alice | Alice | next | Alice, can you share the deck?",
        "Alice | Alice | next | Alice, can you share the",
        "Alice | Alice | next | deck? Alice, are you there?",
    ) == [None, "repeat", "repeat"]


def test_an_unknown_kind_and_a_short_line_are_dropped() -> None:
    assert _reasons(
        MEETING,
        "Alice | Alice | addressed | Alice, can you share the deck?",
        "Alice | Alice | next",
        "Bruno | <spk:1> | next | Bruno, what do you think of it?",
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
        st.sampled_from(["Alice", "Bruno", "Maria"]),
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
            mention
            for mention in named.evidence
            if mention.name == name
            and mention.kind != "self"
            and speaker in (mention.by, *mention.by_one_of)
        ]
    unnamed = turns_from_speakers(words, spoken)
    renamed = turns_from_speakers(words, spoken, named.names)
    assert [turn.model_copy(update={"speaker": ""}) for turn in renamed] == [
        turn.model_copy(update={"speaker": ""}) for turn in unnamed
    ]
