from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scribe.cleanup import CleanupRequest
from scribe.fidelity import ContentSpan, MovedSpan, check_fidelity
from scribe.schema import Turn

Spec = tuple[str, str]


def _request(
    specs: list[Spec],
    *,
    speaker_key: dict[str, str] | None = None,
    glossary: dict[str, str] | None = None,
) -> CleanupRequest:
    return CleanupRequest(
        turns=[
            Turn(speaker=speaker, start=float(index), end=index + 0.9, text=text)
            for index, (speaker, text) in enumerate(specs)
        ],
        speaker_key=speaker_key or {},
        glossary=glossary or {},
    )


SHIP: list[Spec] = [
    ("Speaker 1", "we ship on monday"),
    ("Speaker 2", "the deploy is on thursday"),
]


def test_a_faithful_copy_edit_finds_nothing() -> None:
    result = check_fidelity(
        _request(SHIP), "Speaker 1: We ship on Monday.\n\nSpeaker 2: The deploy is on Thursday.\n"
    )

    assert result.words_checked == 9
    assert result.moved_words == 0
    assert result.content_edit_words == 0
    assert result.content_edits_per_1000 == 0.0


def test_a_sentence_given_to_the_other_speaker_is_a_move() -> None:
    result = check_fidelity(
        _request(SHIP), "Speaker 1: We ship on Monday. The deploy is on Thursday.\n"
    )

    assert result.moved_words == 5
    assert result.moved_spans == [
        MovedSpan(
            turn=2,
            from_speaker="Speaker 2",
            to_speaker="Speaker 1",
            words="the deploy is on thursday",
        )
    ]
    assert result.content_edit_words == 0


def test_a_backchannel_moved_into_the_next_speakers_paragraph_is_a_move() -> None:
    request = _request([("Speaker 1", "we ship on monday yep"), ("Speaker 2", "okay cool")])

    result = check_fidelity(request, "Speaker 1: We ship on Monday.\n\nSpeaker 2: Yep. Okay, cool.")

    assert result.moved_words == 1
    assert result.moved_spans[0].words == "yep"


def test_fillers_stutters_and_case_are_not_content_edits() -> None:
    request = _request(
        [
            ("Speaker 1", "okay um so we we need the bu- budget you know by friday"),
            ("Speaker 2", "uh right I mean sort of the the whole plan plan"),
        ]
    )

    result = check_fidelity(
        request,
        "Speaker 1: OKAY, so we need the budget by Friday.\n\nSpeaker 2: Right, the whole plan.",
    )

    assert result.content_edit_words == 0
    assert result.moved_words == 0


def test_a_paraphrase_with_no_number_in_it_is_a_content_edit() -> None:
    request = _request([("Speaker 1", "we should probably wait for the review")])

    result = check_fidelity(request, "Speaker 1: We must wait for the review.\n")

    assert result.moved_words == 0
    assert result.content_edit_words == 2
    assert result.content_edits_per_1000 == 285.7
    assert result.content_spans == [ContentSpan(turn=1, before="should probably", after="must")]


def test_a_glossary_substitution_is_not_counted_made_or_not() -> None:
    request = _request(
        [("Speaker 1", "we asked cloud code about ackme")],
        glossary={"cloud code": "Claude Code", "ackme": "Acme"},
    )

    made = check_fidelity(request, "Speaker 1: We asked Claude Code about Acme.\n")
    missed = check_fidelity(request, "Speaker 1: We asked cloud code about ackme.\n")

    assert made.content_edit_words == 0
    assert missed.content_edit_words == 0


def test_the_same_substitution_without_a_glossary_entry_is_counted() -> None:
    result = check_fidelity(
        _request([("Speaker 1", "we asked cloud about it")]), "Speaker 1: We asked Claude about it."
    )

    assert result.content_edit_words == 1


def test_a_value_rewritten_as_the_number_check_accepts_is_not_counted() -> None:
    request = _request([("Speaker 1", "about ten percent of two million dollars")])

    result = check_fidelity(request, "Speaker 1: About 10% of $2M.\n")

    assert result.content_edit_words == 0


def test_a_compound_joined_with_a_hyphen_is_not_counted() -> None:
    result = check_fidelity(
        _request([("Speaker 1", "book a follow up call")]), "Speaker 1: Book a follow-up call."
    )

    assert result.content_edit_words == 0


def test_a_word_said_on_both_sides_of_a_turn_boundary_pairs_with_its_own_speaker() -> None:
    # Matched on words alone, the kept "right" pairs with the first copy, which
    # is Speaker 1's, and reads as moved to Speaker 2.
    request = _request([("Speaker 1", "right"), ("Speaker 2", "right")])

    result = check_fidelity(request, "Speaker 2: Right.\n")

    assert result.moved_words == 0
    assert result.content_edit_words == 1
    assert result.content_spans == [ContentSpan(turn=1, before="right", after="")]


def test_relabeled_and_upper_cased_labels_are_read_as_the_key_names_them() -> None:
    request = _request(SHIP, speaker_key={"Speaker 1": "Ann Lee", "Speaker 2": "Bo"})

    result = check_fidelity(
        request, "**ANN LEE:** We ship on Monday.\n\nBO: The deploy is on Thursday."
    )

    assert result.moved_words == 0
    assert result.content_edit_words == 0


def test_a_paragraph_without_a_label_continues_the_speaker_above_it() -> None:
    kept = check_fidelity(
        _request(SHIP), "Speaker 1: We ship\n\non Monday.\n\nSpeaker 2: The deploy is on Thursday."
    )
    moved = check_fidelity(
        _request(SHIP), "Speaker 1: We ship on Monday.\n\nThe deploy is on Thursday.\n"
    )

    assert kept.moved_words == 0
    # Read under Speaker 1, whose paragraph it follows, Speaker 2's words moved.
    assert moved.moved_words == 5


def test_speech_before_any_label_is_never_a_move() -> None:
    result = check_fidelity(
        _request(SHIP), "We ship on Monday.\n\nSpeaker 2: The deploy is on Thursday."
    )

    assert result.moved_words == 0


def test_spans_are_capped_and_short() -> None:
    specs: list[Spec] = [
        (
            f"Speaker {1 + index % 2}",
            f"kept{index} alpha beta gamma delta epsilon zeta eta theta iota",
        )
        for index in range(30)
    ]
    text = "\n\n".join(
        f"{speaker}: kept{index} rewritten." for index, (speaker, _) in enumerate(specs)
    )

    result = check_fidelity(_request(specs), text)

    assert result.content_edit_words == 270
    assert len(result.content_spans) == 20
    assert result.content_spans[3] == ContentSpan(
        turn=4, before="alpha beta gamma delta epsilon zeta eta theta", after="rewritten"
    )


def test_an_empty_input_checks_nothing() -> None:
    result = check_fidelity(_request([]), "")

    assert result.words_checked == 0
    assert result.content_edits_per_1000 == 0.0


# No word here begins another, so a stutter fragment is the only prefix.
_VOCABULARY = ["budget", "vendor", "plan", "launch", "review", "hiring", "agenda", "ackme"]
_SPOKEN_FILLERS = ["um", "uh", "er", "you know", "like", "I mean", "sort of", "kind of"]
_HEDGES = {"sort of", "kind of"}
_KEY = {"Speaker 1": "Ann Lee", "Speaker 2": "Bo"}
_GLOSSARY = {"ackme": "Acme"}


@st.composite
def _allowed_edits(draw: st.DrawFn) -> tuple[CleanupRequest, str]:
    """Turns, and a cleaned text made of them by allowed edits only."""
    keyed = draw(st.booleans())
    specs: list[Spec] = []
    paragraphs: list[str] = []
    last = None
    for _ in range(draw(st.integers(min_value=1, max_value=5))):
        speaker = draw(st.sampled_from(["Speaker 1", "Speaker 2", "Speaker 3"]))
        words = draw(st.lists(st.sampled_from(_VOCABULARY), min_size=1, max_size=5, unique=True))
        if last is not None and draw(st.booleans()):
            words = [last, *(word for word in words if word != last)]
        spoken: list[str] = []
        written: list[str] = []
        for word in words:
            if filler := draw(st.sampled_from([None, *_SPOKEN_FILLERS])):
                spoken.append(filler)
                # Before a bare word a hedge qualifies it; before an article it is filler.
                if filler in _HEDGES:
                    spoken.append("the")
                    written.append("the")
            stutter = draw(st.sampled_from(["", word, f"{word[:2]}-"]))
            spoken.extend([stutter, word] if stutter else [word])
            shown = _GLOSSARY.get(word, word)
            shown = draw(st.sampled_from([shown, shown.upper(), shown.capitalize()]))
            written.append(shown + draw(st.sampled_from(["", ",", ".", "?"])))
        if draw(st.booleans()):
            spoken.append("ten percent")
            written.append("10%.")
        specs.append((speaker, " ".join(spoken)))
        last = words[-1]
        label = _KEY.get(speaker, speaker) if keyed else speaker
        prefix = draw(
            st.sampled_from([f"{label}:", f"{label.upper()}:", f"**{label}:**", f"**{label}**:"])
        )
        cut = draw(st.integers(min_value=1, max_value=len(written)))
        paragraph = f"{prefix} {' '.join(written[:cut])}"
        if written[cut:]:
            paragraph += f"\n\n{' '.join(written[cut:])}"
        paragraphs.append(paragraph)
    request = _request(specs, speaker_key=_KEY if keyed else None, glossary=_GLOSSARY)
    return request, "\n\n".join(paragraphs) + "\n"


@settings(deadline=None)
@given(_allowed_edits())
def test_allowed_edits_alone_move_and_change_nothing(case: tuple[CleanupRequest, str]) -> None:
    request, text = case

    result = check_fidelity(request, text)

    assert result.moved_words == 0
    assert result.content_edit_words == 0


def test_an_insertion_is_reported_at_the_turn_it_follows() -> None:
    request = _request([("Speaker 1", "we ship"), ("Speaker 2", "on friday")])

    result = check_fidelity(request, "Speaker 1: We ship today.\n\nSpeaker 2: On Friday.")

    assert result.content_spans == [ContentSpan(turn=1, before="", after="today")]


def test_an_insertion_with_no_input_word_to_follow_is_reported_at_the_first_id() -> None:
    result = check_fidelity(_request([("Speaker 1", "um uh")]), "Speaker 1: Hello.")

    assert result.content_spans == [ContentSpan(turn=1, before="", after="hello")]


def test_moved_spans_are_capped_but_every_move_is_counted() -> None:
    specs: list[Spec] = [(f"Speaker {1 + index % 2}", f"said{index} here") for index in range(50)]
    text = "Speaker 1: " + " ".join(f"said{index} here" for index in range(50))

    result = check_fidelity(_request(specs), text)

    assert result.moved_words == 50
    assert len(result.moved_spans) == 20


def test_a_glossary_entry_with_no_letters_to_match_is_skipped() -> None:
    request = _request([("Speaker 1", "we ship on monday")], glossary={"...": "etc"})

    result = check_fidelity(request, "Speaker 1: We ship on Monday...")

    assert result.content_edit_words == 0


def test_a_sentence_moved_and_reordered_past_kept_speech_is_a_move() -> None:
    request = _request(
        [
            ("Speaker 1", "we ship on monday"),
            ("Speaker 2", "the budget is approved"),
            ("Speaker 1", "and we test on tuesday"),
        ]
    )

    result = check_fidelity(
        request, "Speaker 1: We ship on Monday and we test on Tuesday. The budget is approved.\n"
    )

    assert result.moved_words == 4
    assert result.moved_spans == [
        MovedSpan(
            turn=2, from_speaker="Speaker 2", to_speaker="Speaker 1", words="the budget is approved"
        )
    ]
    assert result.content_edit_words == 0


_MONOLOGUE = [f"part{n} alpha beta gamma delta." for n in range(5)]


@pytest.mark.parametrize(
    ("pieces", "max_words"),
    [pytest.param(1, 6000, id="whole"), *[pytest.param(n, 5, id=f"{n}-pieces") for n in (3, 4, 5)]],
)
def test_opening_words_moved_to_the_next_speaker_are_a_move_however_the_turn_was_split(
    pieces: int, max_words: int
) -> None:
    # Reach counts diarized turns: a monologue sent in pieces is still one turn
    # away from the speaker after it.
    monologue = " ".join(_MONOLOGUE[:pieces] if pieces > 1 else _MONOLOGUE)
    request = _request([("Speaker 1", monologue), ("Speaker 2", "that sounds good to me")])
    kept = monologue.split(" ", 3)[3]

    result = check_fidelity(
        request,
        f"Speaker 1: {kept}\n\nSpeaker 2: Part0 alpha beta. That sounds good to me.\n",
        max_words=max_words,
    )

    assert result.moved_words == 3
    # Named by the id of the piece the words were said in.
    assert result.moved_spans == [
        MovedSpan(
            turn=1, from_speaker="Speaker 1", to_speaker="Speaker 2", words="part0 alpha beta"
        )
    ]


def test_a_short_run_reordered_to_another_speaker_stays_a_content_edit() -> None:
    # Two common words recur by chance across a meeting; pairing them far from
    # where they were said would report moves nobody made.
    request = _request(
        [("Speaker 1", "we ship on monday"), ("Speaker 2", "yeah sure"), ("Speaker 1", "then test")]
    )

    result = check_fidelity(request, "Speaker 1: We ship on Monday, then test. Yeah, sure.\n")

    assert result.moved_words == 0
    assert result.content_edit_words == 4


def test_a_run_reordered_to_a_turn_far_away_stays_a_content_edit() -> None:
    specs: list[Spec] = [
        ("Speaker 1", "we ship on monday"),
        ("Speaker 2", "the budget is approved"),
        *[(f"Speaker {1 + index % 2}", f"filler{index} words here") for index in range(6)],
    ]
    text = "\n\n".join(f"{speaker}: {spoken}" for speaker, spoken in [specs[0], *specs[2:]])

    result = check_fidelity(_request(specs), f"{text} The budget is approved.\n")

    assert result.moved_words == 0
    assert result.content_edit_words == 8


def test_a_whole_word_the_next_word_begins_with_is_a_deleted_word() -> None:
    request = _request(
        [("Speaker 1", "i did not notice it"), ("Speaker 2", "there are no notes yet")]
    )

    result = check_fidelity(
        request, "Speaker 1: I did notice it.\n\nSpeaker 2: There are notes yet."
    )

    assert result.content_edit_words == 2
    assert result.content_spans == [
        ContentSpan(turn=1, before="not", after=""),
        ContentSpan(turn=2, before="no", after=""),
    ]


def test_a_run_reordered_under_its_own_speaker_is_not_a_move() -> None:
    request = _request([("Speaker 1", "alpha beta gamma delta"), ("Speaker 2", "that sounds good")])

    result = check_fidelity(
        request, "Speaker 2: That sounds good.\n\nSpeaker 1: Alpha beta gamma delta."
    )

    assert result.moved_words == 0
    assert result.content_edit_words == 6


def test_a_replaced_run_sharing_no_words_is_a_content_edit() -> None:
    request = _request(
        [("Speaker 1", "we ship on monday"), ("Speaker 2", "red green blue"), ("Speaker 1", "then")]
    )

    result = check_fidelity(
        request,
        "Speaker 1: We ship on Monday.\n\nSpeaker 2: Cyan magenta yellow.\n\nSpeaker 1: Then.",
    )

    assert result.moved_words == 0
    assert result.content_spans == [
        ContentSpan(turn=2, before="red green blue", after="cyan magenta yellow")
    ]


def test_an_er_removed_is_not_a_content_edit() -> None:
    result = check_fidelity(
        _request([("Speaker 1", "we er ship on monday")]), "Speaker 1: We ship on Monday."
    )

    assert result.content_edit_words == 0


@pytest.mark.parametrize(
    ("spoken", "cleaned", "dropped"),
    [
        ("i kind of agree with the plan", "I agree with the plan.", "kind of"),
        ("the vendor is sort of expensive", "The vendor is expensive.", "sort of"),
        ("that kind of budget needs review", "That budget needs review.", "kind of"),
        ("we like the plan", "We the plan.", "like"),
        ("i'd like a review", "I'd a review.", "like"),
    ],
)
def test_a_hedge_or_a_verb_removed_is_a_content_edit(
    spoken: str, cleaned: str, dropped: str
) -> None:
    result = check_fidelity(_request([("Speaker 1", spoken)]), f"Speaker 1: {cleaned}")

    assert result.content_spans == [ContentSpan(turn=1, before=dropped, after="")]


@pytest.mark.parametrize(
    ("spoken", "cleaned"),
    [
        ("it is kind of a mess", "It is a mess."),
        ("sort of the whole plan", "The whole plan."),
        ("it was sort of like a review", "It was a review."),
        ("we shipped it kind of", "We shipped it."),
        ("the budget like is late", "The budget is late."),
        ("we could do like a review", "We could do a review."),
        ("i kind of agree", "I kind of agree."),
        ("we like the plan", "We like the plan."),
    ],
)
def test_a_filler_removed_or_a_hedge_kept_is_not_a_content_edit(spoken: str, cleaned: str) -> None:
    result = check_fidelity(_request([("Speaker 1", spoken)]), f"Speaker 1: {cleaned}")

    assert result.content_edit_words == 0
