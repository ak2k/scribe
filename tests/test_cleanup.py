from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scribe.cleanup import (
    CLEANUP_PROMPT_VERSION,
    PRIOR_TAIL_HEADER,
    CleanResult,
    CleanupBackend,
    CleanupRequest,
    NumberDiff,
    chunk_turns,
    clean,
    final_speakers,
    parse_pairs,
    provenance_header,
    read_pairs_file,
    render_system_prompt,
    render_user_prompt,
    strip_speaker_labels,
    turn_label,
    verify_numbers,
)
from scribe.errors import InputValidationError
from scribe.schema import Engine, Source, Turn
from tests.cleanup_fakes import FakeBackend, sent_turns

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

Spec = tuple[str, str]

TWO_SPEAKERS: list[Spec] = [
    ("Speaker 1", "Okay, um, ready?"),
    ("Speaker 2", "Yes, let us start."),
]


def _turns(specs: list[Spec]) -> list[Turn]:
    return [
        Turn(speaker=speaker, start=float(index), end=index + 0.9, text=text)
        for index, (speaker, text) in enumerate(specs)
    ]


def _request(
    specs: list[Spec] = TWO_SPEAKERS,
    *,
    speaker_key: dict[str, str] | None = None,
    glossary: dict[str, str] | None = None,
    context: str | None = None,
) -> CleanupRequest:
    return CleanupRequest(
        turns=_turns(specs),
        speaker_key=speaker_key or {},
        glossary=glossary or {},
        context=context,
    )


def test_chunk_turns_splits_only_on_a_turn_boundary() -> None:
    turns = _turns([("A", "one two three"), ("B", "four five"), ("A", "six")])

    chunks = chunk_turns(turns, max_words=4)

    assert [[turn.text for turn in chunk] for chunk in chunks] == [
        ["one two three"],
        ["four five", "six"],
    ]


def test_a_turn_without_sentences_splits_on_words_at_the_ceiling() -> None:
    turns = _turns([("A", "one two"), ("B", " ".join(f"w{n}" for n in range(9))), ("A", "last")])

    chunks = chunk_turns(turns, max_words=3)

    assert [[(turn.speaker, turn.text) for turn in chunk] for chunk in chunks] == [
        [("A", "one two")],
        [("B", "w0 w1 w2")],
        [("B", "w3 w4 w5")],
        [("B", "w6 w7 w8")],
        [("A", "last")],
    ]


def test_a_turn_longer_than_the_ceiling_splits_between_sentences() -> None:
    turns = _turns([("A", "One two three. Four five. Six seven eight nine ten. Eleven.")])

    chunks = chunk_turns(turns, max_words=5)

    assert [[turn.text for turn in chunk] for chunk in chunks] == [
        ["One two three. Four five."],
        ["Six seven eight nine ten."],
        ["Eleven."],
    ]


def _sentences(words: int, length: int) -> str:
    return " ".join(f"w{n}." if n % length == length - 1 else f"w{n}" for n in range(words))


def test_an_oversize_turn_is_cleaned_in_chunks_under_the_ceiling() -> None:
    monologue = _sentences(10_000, 7)
    backend = FakeBackend()

    result = clean(_request([("Speaker 1", monologue)]), backend, max_words=1000)

    assert len(backend.calls) >= 10
    sections = [
        " ".join(text for _, text in sent_turns(user)).split() for _system, user in backend.calls
    ]
    assert all(len(section) <= 1000 for section in sections)
    assert result.truncated_chunks == []
    # Every piece keeps the turn's label, and the pieces rejoin in order.
    assert strip_speaker_labels(result.text, ["Speaker 1"]).split() == monologue.split()
    assert all(line.startswith("Speaker 1: ") for line in result.text.strip().split("\n\n"))


def test_a_chunk_filled_exactly_to_the_ceiling_is_not_split() -> None:
    exact = _turns([("A", "one two three"), ("B", "four")])
    over = _turns([("A", "one two three"), ("B", "four five")])

    assert len(chunk_turns(exact, max_words=4)) == 1
    assert len(chunk_turns(over, max_words=4)) == 2


def test_turns_that_fit_together_stay_one_chunk() -> None:
    turns = _turns(TWO_SPEAKERS)

    assert chunk_turns(turns) == [turns]


def test_no_turns_is_no_chunks() -> None:
    assert chunk_turns([]) == []


def test_a_thousands_separator_is_not_a_difference() -> None:
    assert verify_numbers("it came to $1,234.56", "it came to $1234.56") == NumberDiff(checked=1)


def test_a_percentage_and_a_time_survive_unchanged() -> None:
    before = "45.5% of them joined at 10:30"
    after = "45.5% of them joined at 10:30."

    assert verify_numbers(before, after) == NumberDiff(checked=2)


def test_a_space_before_the_percent_sign_is_not_a_difference() -> None:
    assert verify_numbers("45.5% of them joined", "45.5 % of them joined") == NumberDiff(checked=1)


def test_a_space_after_the_currency_symbol_is_not_a_difference() -> None:
    assert verify_numbers("it came to $1,234.56", "it came to $ 1234.56") == NumberDiff(checked=1)


def test_an_hour_written_with_a_leading_zero_is_the_same_time() -> None:
    assert verify_numbers("we met at 09:05", "we met at 9:05") == NumberDiff(checked=1)


def test_a_time_rewritten_with_a_dot_is_still_an_alarm() -> None:
    # The separator carries the meaning here, so this one is a real difference.
    assert verify_numbers("we met at 10:30", "we met at 10.30") == NumberDiff(
        missing=["10:30"], added=["10.30"], checked=1
    )


def test_a_dropped_year_is_reported_missing() -> None:
    diff = verify_numbers("paid $1,234.56 at 10:30 in 2026", "paid $1234.56 at 10:30")

    assert diff == NumberDiff(missing=["2026"], added=[], checked=3)


def test_a_value_kept_fewer_times_is_reduced_not_missing() -> None:
    assert verify_numbers("10 then 10 again", "10 again") == NumberDiff(reduced=["10"], checked=2)


def test_a_merged_restatement_is_reduced_not_missing() -> None:
    diff = verify_numbers(
        "There are two things. There are two things here.", "There are 2 things here."
    )

    assert diff == NumberDiff(missing=[], reduced=["2"], checked=2)


def test_a_value_gone_entirely_is_missing_with_its_multiplicity() -> None:
    assert verify_numbers("10 then 10 and 20", "20") == NumberDiff(missing=["10", "10"], checked=3)


def test_a_dropped_value_beside_a_kept_one_is_missing() -> None:
    assert verify_numbers("10 and 20", "10") == NumberDiff(missing=["20"], checked=2)


def test_one_copy_of_a_repeated_value_changed_is_drift() -> None:
    diff = verify_numbers("5 engineers and 5 designers", "5 engineers and 7 designers")

    assert diff == NumberDiff(reduced=["5"], added=["7"], checked=2)
    assert diff.drifted


@pytest.mark.parametrize(
    ("diff", "drifted"),
    [
        (NumberDiff(checked=2), False),
        (NumberDiff(missing=["5"], checked=2), True),
        (NumberDiff(reduced=["5"], checked=2), False),
        (NumberDiff(added=["7"], checked=2), False),
        (NumberDiff(reduced=["5"], added=["7"], checked=2), True),
    ],
)
def test_drift_is_a_lost_value_or_a_count_drop_beside_a_new_one(
    diff: NumberDiff, *, drifted: bool
) -> None:
    assert diff.drifted is drifted


# The list is unbounded, so one example can outlast any fixed deadline on a loaded machine.
@settings(deadline=None)
@given(values=st.lists(st.integers(min_value=0, max_value=99_999), min_size=1), data=st.data())
def test_only_a_value_gone_entirely_is_missing(values: list[int], data: st.DataObject) -> None:
    # " and " is no stutter gap, and no scale or unit follows a value to fold it.
    target = data.draw(st.sampled_from(values))
    copies = values.count(target)
    kept = data.draw(st.integers(min_value=0, max_value=copies - 1))
    after: list[int] = []
    for value in values:
        if value != target or kept > after.count(target):
            after.append(value)

    diff = verify_numbers(" and ".join(map(str, values)), " and ".join(map(str, after)))

    if kept:
        assert str(target) not in diff.missing
        assert str(target) in diff.reduced
    else:
        assert diff.missing.count(str(target)) == copies
        assert str(target) not in diff.reduced


def test_an_invented_number_is_reported_added() -> None:
    assert verify_numbers("about half", "about 50%") == NumberDiff(missing=[], added=["50%"])


def test_a_trailing_period_is_not_part_of_the_number() -> None:
    assert verify_numbers("we start in 2026.", "we start in 2026") == NumberDiff(checked=1)


def test_the_user_prompt_applies_the_relabel_key() -> None:
    prompt = render_user_prompt(_request(speaker_key={"Speaker 1": "Ann Lee"}))

    assert prompt == (
        '<t id=1 speaker="Ann Lee">Okay, um, ready?</t>\n\n'
        '<t id=2 speaker="Speaker 2">Yes, let us start.</t>\n'
    )
    assert PRIOR_TAIL_HEADER not in prompt


def test_the_prior_tail_precedes_the_turns_under_its_own_line_with_no_id() -> None:
    prompt = render_user_prompt(_request(), prior_tail="  Earlier, already cleaned.  ", first_id=7)

    assert prompt == (
        f"{PRIOR_TAIL_HEADER}\n\nEarlier, already cleaned.\n\n"
        '<t id=7 speaker="Speaker 1">Okay, um, ready?</t>\n\n'
        '<t id=8 speaker="Speaker 2">Yes, let us start.</t>\n'
    )


def test_a_quote_in_a_label_cannot_end_its_attribute() -> None:
    prompt = render_user_prompt(_request(speaker_key={"Speaker 1": 'Ann "the PM" Lee'}))

    assert prompt.startswith('<t id=1 speaker="Ann &quot;the PM&quot; Lee">Okay, um, ready?</t>')


def test_turn_text_that_looks_like_a_tag_is_sent_escaped_and_comes_back_as_said() -> None:
    # STT never writes "<", but a typed transcript can, and unescaped it would
    # open or close a turn of its own.
    said = 'she typed <t id=9 speaker="X">no</t> & "</t>" into it'
    backend = FakeBackend()

    result = clean(_request([("Speaker 1", said), ("Speaker 2", "right")]), backend)

    assert "&lt;t id=9" in backend.calls[0][1]
    assert "no</t>" not in backend.calls[0][1]
    assert result.text == f"Speaker 1: {said}\n\nSpeaker 2: right\n"
    assert result.malformed_chunks == []


def test_the_prior_tail_is_sent_escaped() -> None:
    # The tail is the previous chunk's text as said, so it can hold a tag too.
    backend = FakeBackend()

    clean(_request([("A", "x <t id=9> & y"), ("B", "four five six")]), backend, max_words=5)

    assert _carried(backend.calls[1][1]) == "A: x &lt;t id=9&gt; &amp; y"


def test_the_system_prompt_lists_final_names_and_never_the_key() -> None:
    request = _request(speaker_key={"Speaker 1": "Ann Lee"})

    prompt = render_system_prompt(request)

    assert final_speakers(request) == ["Ann Lee", "Speaker 2"]
    assert "- Ann Lee\n- Speaker 2" in prompt
    # The relabel is already applied in the user prompt, so handing the model
    # the mapping would ask it to perform a substitution a second time.
    assert "Speaker 1" not in prompt


def test_the_system_prompt_appends_the_glossary_and_the_context() -> None:
    prompt = render_system_prompt(_request(glossary={"ackme": "Acme"}, context="  a budget call  "))

    assert '- "ackme" is written as "Acme"' in prompt
    assert prompt.rstrip().endswith("a budget call")


def test_a_passage_left_as_recognized_still_takes_the_glossary() -> None:
    prompt = render_system_prompt(CleanupRequest(turns=[]))
    rule = prompt.split("\n4. ", 1)[1].split("\n5. ", 1)[0]

    assert "exactly" not in rule
    assert "glossary" in rule
    # The pass is nondeterministic, so the recorded version is the only way to
    # tell outputs of the reworded prompt from those of the one before it.
    assert CLEANUP_PROMPT_VERSION == "verbatim-3"


def test_the_system_prompt_asks_for_each_turn_back_under_its_id_and_no_label() -> None:
    prompt = render_system_prompt(CleanupRequest(turns=[]))

    assert '<t id=N speaker="Label">text</t>' in prompt
    assert "numbered from 1" in prompt
    assert "as <t id=N>cleaned text</t> with the same id" in prompt
    assert "never write a speaker label" in prompt
    assert "`Label:`" not in prompt


def test_a_request_with_no_extras_carries_no_extra_sections() -> None:
    prompt = render_system_prompt(CleanupRequest(turns=[]))

    assert prompt.rstrip().endswith("no preamble, no headings, no analysis.")


def _carried(user: str) -> str:
    """The tail a prompt carries ahead of its turns, or "" when it carries none."""
    if not user.startswith(PRIOR_TAIL_HEADER):
        return ""
    return user.removeprefix(f"{PRIOR_TAIL_HEADER}\n\n").split("\n\n<t id=", 1)[0]


def test_clean_makes_one_call_per_chunk_and_carries_the_tail_forward() -> None:
    backend = FakeBackend()
    request = _request([("A", "one two three"), ("B", "four five six")])

    result = clean(request, backend, max_words=3)

    assert result.chunks == 2
    assert len(backend.calls) == 2
    assert len(result.completions) == 2
    assert result.truncated_chunks == []
    first_user = backend.calls[0][1]
    second_user = backend.calls[1][1]
    assert first_user == '<t id=1 speaker="A">one two three</t>\n'
    # Ids number the whole recording, so the second chunk goes on from the first.
    assert second_user == (
        f'{PRIOR_TAIL_HEADER}\n\nA: one two three\n\n<t id=2 speaker="B">four five six</t>\n'
    )
    assert result.text == "A: one two three\n\nB: four five six\n"
    # One system prompt for the whole run: the speakers, glossary and context
    # do not change between chunks.
    assert backend.calls[0][0] == backend.calls[1][0]


def test_clean_keeps_only_the_last_two_cleaned_paragraphs_as_the_tail() -> None:
    backend = FakeBackend(
        mutate=lambda reply: reply.replace("one two three", "para one\n\npara two\n\npara three")
    )
    request = _request([("A", "one two three"), ("B", "four five six")])

    clean(request, backend, max_words=3)

    assert _carried(backend.calls[1][1]) == "A: para two\n\npara three"


def test_the_carried_tail_is_capped_in_words() -> None:
    # A cut monologue replies in one paragraph, so two paragraphs of tail
    # would be the whole previous chunk.
    backend = FakeBackend()

    clean(_request([("Speaker 1", _sentences(10_000, 7))]), backend, max_words=1000)

    tails = [_carried(user) for _system, user in backend.calls[1:]]
    assert len(tails) == len(backend.calls) - 1
    # Cut inside the one paragraph, the tail still says whose words it holds.
    assert all(tail.startswith("Speaker 1: ") for tail in tails)
    speech = [tail.removeprefix("Speaker 1: ").split() for tail in tails]
    assert all(0 < len(words) <= 200 for words in speech)
    # The last words of the previous reply, not its first.
    [(_, first_piece)] = sent_turns(backend.calls[0][1])
    assert speech[0] == first_piece.split()[-len(speech[0]) :]


def _causes(result: CleanResult) -> list[tuple[int, str]]:
    return [(item.chunk, item.cause) for item in result.malformed_chunks]


def _echoes_the_carried_tail(user: str) -> str:
    """Reply against the prompt: copy the tail back ahead of the turns, upper-cased."""
    tags = "".join(f"<t id={turn_id}>{text.upper()}</t>" for turn_id, text in sent_turns(user))
    return f"{_carried(user)}\n\n{tags}"


def test_an_echoed_tail_sets_its_chunk_aside() -> None:
    backend = FakeBackend(respond=_echoes_the_carried_tail)
    request = _request([("A", "one two three"), ("B", "four five six")])

    result = clean(request, backend, max_words=3)

    # The first chunk carries no tail, so it alone came back well formed.
    assert result.text == "A: ONE TWO THREE\n\nB: four five six\n"
    assert _causes(result) == [(1, "outside_text")]


def test_a_reply_returning_no_turn_keeps_every_input_text() -> None:
    backend = FakeBackend(mutate=lambda _reply: "   ")
    request = _request([("A", "one two three"), ("B", "four five six")])

    result = clean(request, backend, max_words=3)

    assert result.text == "A: one two three\n\nB: four five six\n"
    assert result.truncated_chunks == [0, 1]
    assert _causes(result) == [(0, "missing_id"), (1, "missing_id")]
    # The input text stands in for the reply, so it is what the next chunk
    # is sent as context.
    assert _carried(backend.calls[1][1]) == "A: one two three"


THREE: list[Spec] = [
    ("Speaker 1", "we ship on monday"),
    ("Speaker 2", "the deploy is on thursday"),
    ("Speaker 1", "then we rest"),
]


def _replying(*replies: str) -> Callable[[str], str]:
    """Answer each call with the next canned reply, whatever it was sent."""
    queue = iter(replies)
    return lambda _reply: next(queue)


def test_labels_come_from_the_input_whatever_the_reply_writes() -> None:
    # A label written inside a turn, on its first line or a later one, would
    # read as a change of speaker at that line.
    reply = (
        "<t id=1>Speaker 2: We ship on Monday.</t>\n\n"
        "<t id=2>The deploy is on Thursday.\n\n**Speaker 1:** After lunch.</t>\n\n"
        "<t id=3>SPEAKER 1: Then we rest.</t>"
    )
    backend = FakeBackend(mutate=_replying(reply))

    result = clean(_request(THREE), backend)

    assert result.text == (
        "Speaker 1: We ship on Monday.\n\n"
        "Speaker 2: The deploy is on Thursday.\n\nAfter lunch.\n\n"
        "Speaker 1: Then we rest.\n"
    )
    assert result.stripped_labels == 3
    assert result.truncated_chunks == []


def test_every_label_opening_a_line_is_stripped_and_counted() -> None:
    # A second label left behind would put its "1" on the cleaned side of the
    # number check, and read as a change of speaker on a later line.
    reply = (
        "<t id=1>We ship on Monday.</t>\n\n"
        "<t id=2>Speaker 2: Speaker 1: The deploy is on Thursday.\n\n"
        "**Speaker 2:** SPEAKER 2: Speaker 1: After lunch.</t>\n\n"
        "<t id=3>Then we rest.</t>"
    )
    backend = FakeBackend(mutate=_replying(reply))

    result = clean(_request(THREE), backend)

    assert result.text == (
        "Speaker 1: We ship on Monday.\n\n"
        "Speaker 2: The deploy is on Thursday.\n\nAfter lunch.\n\n"
        "Speaker 1: Then we rest.\n"
    )
    assert result.stripped_labels == 5


def test_a_reply_copying_the_speaker_attribute_is_read() -> None:
    backend = FakeBackend(
        mutate=lambda reply: re.sub(r"<t id=(\d+)>", r'<t id=\1 speaker="Speaker 9">', reply)
    )

    result = clean(_request(THREE), backend)

    # Read for its id alone: the input still names the speaker.
    assert result.text == (
        "Speaker 1: we ship on monday\n\n"
        "Speaker 2: the deploy is on thursday\n\n"
        "Speaker 1: then we rest\n"
    )
    assert result.malformed_chunks == []


# As said, THREE differs from every reply below, so the text shows whether the
# chunk was rebuilt from its reply or kept its input.
THREE_AS_SAID = (
    "Speaker 1: we ship on monday\n\n"
    "Speaker 2: the deploy is on thursday\n\n"
    "Speaker 1: then we rest\n"
)
THREE_CLEANED = (
    "<t id=1>We ship on Monday.</t>\n\n<t id=2>The deploy is on Thursday.</t>\n\n"
    "<t id=3>Then we rest.</t>"
)


@pytest.mark.parametrize(
    ("reply", "cause"),
    [
        pytest.param(
            "<t id=1>We ship on</t> Monday.<t id=2>The deploy is on Thursday.</t>"
            "<t id=3>Then we rest.</t>",
            "outside_text",
            id="closed-early",
        ),
        pytest.param(
            "<t id=1>We ship on Monday.</t>The deploy <t id=2>is on Thursday.</t>"
            "<t id=3>Then we rest.</t>",
            "outside_text",
            id="next-turn-opening-before-its-tag",
        ),
        pytest.param(
            "<t id=1>We ship on Monday.</t>The deploy is on Thursday.<t id=2></t>"
            "<t id=3>Then we rest.</t>",
            "outside_text",
            id="next-turn-before-its-empty-tag",
        ),
        pytest.param(
            "We ship <t id=1>on Monday.</t><t id=2>The deploy is on Thursday.</t>"
            "<t id=3>Then we rest.</t>",
            "outside_text",
            id="first-turn-opening-before-its-tag",
        ),
        pytest.param(
            f"Here is the cleaned transcript:\n\nSpeaker 1: we ship\n\n{THREE_CLEANED}",
            "outside_text",
            id="preamble",
        ),
        pytest.param(
            "<t id=1>We ship.</t><t id=2>Thursday.</t><t id=3>Then we</t>\n\nrest.",
            "outside_text",
            id="after-the-last-tag",
        ),
        pytest.param(
            f"{THREE_CLEANED}<t id=9>Stray</t> words here.",
            "outside_text",
            id="outside-text-before-foreign-id",
        ),
        pytest.param(
            "<t id=1>We ship on Monday.\n\n<t id=2>The deploy is on Thursday.</t>"
            "<t id=3>Then we rest.</t>",
            "outside_text",
            id="unclosed-tag",
        ),
        pytest.param(
            "<t id=1>We</t><t id=99>ship on Monday.</t><t id=2>The deploy is on Thursday.</t>"
            "<t id=3>Then we rest.</t>",
            "foreign_id",
            id="stub-and-a-foreign-tag-holding-its-words",
        ),
        pytest.param(
            f"<t id=0>Hello.</t>{THREE_CLEANED}<t id=4>Bye.</t>", "foreign_id", id="ids-0-and-4"
        ),
        pytest.param(
            "<t id=1>We ship on Monday.</t><t id=2>The deploy is on Thursday.</t>"
            "<t id=5>Then we rest.</t>",
            "foreign_id",
            id="foreign-before-missing",
        ),
        pytest.param(
            "<t id=1>We ship on Monday.</t><t id=2>Noted.</t>"
            "<t id=2>The deploy is on Thursday.</t><t id=3>Then we rest.</t>",
            "repeated_id",
            id="stub-then-a-full-copy",
        ),
        pytest.param(
            f"{THREE_CLEANED}<t id=2>The deploy is on Thursday.</t>",
            "repeated_id",
            id="identical-copies",
        ),
        pytest.param(
            "<t id=1>We ship on Monday.</t><t id=1>We ship on Monday.</t>"
            "<t id=2>The deploy is on Thursday.</t>",
            "repeated_id",
            id="repeated-before-missing",
        ),
        pytest.param(
            # Refilling id 2 alone would say its words twice.
            "<t id=1>We ship on Monday. The deploy is on Thursday.</t><t id=3>Then we rest.</t>",
            "missing_id",
            id="merged-into-the-tag-before",
        ),
        pytest.param("", "missing_id", id="no-tags"),
        pytest.param(
            "<t id=2>The deploy is on Thursday.</t><t id=1>We ship on Monday.</t>",
            "missing_id",
            id="missing-before-out-of-order",
        ),
        pytest.param(
            "<t id=3>Then we rest.</t><t id=1>We ship on Monday.</t>"
            "<t id=2>The deploy is on Thursday.</t>",
            "out_of_order",
            id="out-of-order",
        ),
        pytest.param("<t id=1></t><t id=2></t><t id=3></t>", "wordless", id="empty-tags"),
        # The labels are not speech, so tags of labels alone hold no words.
        pytest.param(
            "<t id=1>Speaker 1:</t><t id=2>**Speaker 2**:</t><t id=3>SPEAKER 1: </t>",
            "wordless",
            id="labels-alone",
        ),
    ],
)
def test_a_malformed_reply_leaves_its_chunk_as_said_and_names_the_first_rule_broken(
    reply: str, cause: str
) -> None:
    backend = FakeBackend(mutate=_replying(reply))

    result = clean(_request(THREE), backend)

    assert result.text == THREE_AS_SAID
    assert _causes(result) == [(0, cause)]
    assert result.truncated_chunks == [0]
    assert (result.emptied_turns, result.stripped_labels) == (0, 0)


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(THREE_CLEANED, id="well-formed"),
        pytest.param(f"Sure: {THREE_CLEANED}", id="with-outside-text"),
    ],
)
def test_a_reply_stopped_at_the_output_limit_is_set_aside_whatever_it_holds(reply: str) -> None:
    backend = FakeBackend(mutate=_replying(reply), stop_reasons=["max_tokens"])

    result = clean(_request(THREE), backend)

    assert result.text == THREE_AS_SAID
    assert _causes(result) == [(0, "max_tokens")]
    assert result.completions[0].stop_reason == "max_tokens"


def test_a_malformed_chunk_leaves_the_next_one_cleaned() -> None:
    backend = FakeBackend(
        mutate=_replying(
            "<t id=1>We ship on</t> Monday.<t id=2>The deploy is on Thursday.</t>",
            "<t id=3>Then we rest.</t><t id=4>See you then.</t>",
        )
    )
    request = _request([*THREE, ("Speaker 2", "see you then")])

    result = clean(request, backend, max_words=9)

    assert result.chunks == 2
    assert result.text == (
        "Speaker 1: we ship on monday\n\nSpeaker 2: the deploy is on thursday\n\n"
        "Speaker 1: Then we rest.\n\nSpeaker 2: See you then.\n"
    )
    assert _causes(result) == [(0, "outside_text")]
    # The input stands in for the reply, so it is what the next chunk is sent as context.
    assert _carried(backend.calls[1][1]) == (
        "Speaker 1: we ship on monday\n\nSpeaker 2: the deploy is on thursday"
    )


def test_a_label_the_key_renamed_is_stripped_inside_a_turn_and_counted() -> None:
    # Left in, the raw label would read as a speaker the transcript no longer has.
    reply = "<t id=1>We ship on Monday.\n\nSpeaker 1: The big boat leaves at noon today.</t>"
    backend = FakeBackend(mutate=_replying(reply))
    request = _request(
        [("Speaker 1", "we ship on monday the big boat leaves at noon today")],
        speaker_key={"Speaker 1": "Ann"},
    )

    result = clean(request, backend)

    assert result.text == "Ann: We ship on Monday.\n\nThe big boat leaves at noon today.\n"
    assert result.stripped_labels == 1
    assert result.malformed_chunks == []


def test_an_emptied_turn_leaves_no_line_and_is_counted() -> None:
    backend = FakeBackend(mutate=lambda reply: reply.replace(">um uh<", "><"))

    result = clean(_request([("Speaker 1", "we ship"), ("Speaker 2", "um uh")]), backend)

    assert result.text == "Speaker 1: we ship\n"
    assert result.emptied_turns == 1
    assert result.truncated_chunks == []


@pytest.mark.parametrize(
    "emptied",
    [
        pytest.param("", id="empty-tags"),
        # The labels are not speech, so a reply of labels alone holds no words.
        pytest.param("Speaker 1:\n\n**Speaker 2**:", id="labels-alone"),
    ],
)
def test_a_reply_whose_turns_hold_no_words_keeps_its_input_and_is_truncated(emptied: str) -> None:
    backend = FakeBackend(
        mutate=lambda reply: re.sub(r"(<t id=\d+>)[^<]*", rf"\g<1>{emptied}", reply)
    )
    request = _request([("Speaker 1", "we ship"), ("Speaker 2", "um uh")])

    result = clean(request, backend)

    assert result.text == "Speaker 1: we ship\n\nSpeaker 2: um uh\n"
    assert _causes(result) == [(0, "wordless")]
    assert (result.emptied_turns, result.stripped_labels) == (0, 0)


def test_a_turn_with_no_words_in_its_input_leaves_no_line() -> None:
    result = clean(_request([("A", "  ")]), FakeBackend())

    # Nothing was sent to come back, so no line is invented, but a reply of no
    # words is still no evidence of a cleanup.
    assert result.text == "\n"
    assert (_causes(result), result.emptied_turns) == ([(0, "wordless")], 0)


def test_ids_number_every_turn_of_the_recording_from_one() -> None:
    backend = FakeBackend()
    # The 5-word turn is cut into two pieces sharing its label: each piece is
    # a turn of its own to the model, with its own id.
    specs: list[Spec] = [("A", "one two"), ("B", "three. four five six seven."), ("A", "eight")]

    result = clean(_request(specs), backend, max_words=4)

    assert [sent_turns(user) for _, user in backend.calls] == [
        [(1, "one two"), (2, "three.")],
        [(3, "four five six seven.")],
        [(4, "eight")],
    ]
    assert result.text == "A: one two\n\nB: three.\n\nB: four five six seven.\n\nA: eight\n"


def test_each_turn_that_left_a_line_is_kept_with_its_text() -> None:
    # Both pieces of the cut turn keep its times; the emptied turn leaves no
    # pair, as it leaves no line.
    backend = FakeBackend(
        mutate=lambda reply: reply.replace(">um uh<", "><").replace(
            ">three.<", ">Three.\n\nAnd more.<"
        )
    )
    specs: list[Spec] = [
        ("A", "one two"),
        ("B", "three. four five six seven."),
        ("A", "um uh"),
        ("B", "eight"),
    ]
    request = _request(specs, speaker_key={"A": "Ann Lee"})

    result = clean(request, backend, max_words=4)

    assert [(turn.speaker, turn.start, text) for turn, text in result.kept] == [
        ("A", 0.0, "one two"),
        ("B", 1.0, "Three.\n\nAnd more."),
        ("B", 1.0, "four five six seven."),
        ("B", 3.0, "eight"),
    ]
    assert result.emptied_turns == 1
    assert (
        result.text
        == "\n\n".join(f"{turn_label(request, turn)}: {text}" for turn, text in result.kept) + "\n"
    )
    assert result.text.startswith("Ann Lee: one two\n\nB: Three.\n\nAnd more.\n\nB: four")


_WORDS = st.lists(st.text(alphabet="abcdefg", min_size=1, max_size=4), min_size=1, max_size=3)
_TEXT = _WORDS.map(" ".join)


@settings(deadline=None)
@given(
    turns=st.lists(st.tuples(st.sampled_from(["A", "B", "C"]), _TEXT), min_size=1, max_size=5),
    data=st.data(),
)
def test_a_chunk_comes_back_whole_from_its_reply_or_whole_as_said(
    turns: list[Spec], data: st.DataObject
) -> None:
    ids = range(1, len(turns) + 1)
    returned = data.draw(
        st.lists(st.tuples(st.integers(0, len(turns) + 1), _TEXT | st.just("")), max_size=8)
    )
    outside = data.draw(st.sampled_from(["", "stray words ", "A: the tail again\n\n"]))
    reply = outside + "".join(f"<t id={turn_id}>{text}</t>" for turn_id, text in returned)
    backend = FakeBackend(mutate=_replying(reply))

    result = clean(_request(turns), backend)

    # The transcript is under the ratio's floor, so the rules left are text
    # outside the tags, the ids, and a word at all.
    well_formed = (
        not outside
        and [turn_id for turn_id, _ in returned] == list(ids)
        and any(text for _, text in returned)
    )
    texts = [text for _, text in returned] if well_formed else [text for _, text in turns]
    assert (
        result.text
        == "\n\n".join(
            f"{speaker}: {text}" for (speaker, _), text in zip(turns, texts, strict=True) if text
        )
        + "\n"
    )
    assert result.truncated_chunks == ([] if well_formed else [0])
    request = _request(turns)
    assert (
        result.text
        == "\n\n".join(f"{turn_label(request, turn)}: {text}" for turn, text in result.kept) + "\n"
    )


def _tail_after(first: list[Spec]) -> str:
    """The tail sent with the next chunk once the turns of `first` come back as its texts."""
    reply = "\n\n".join(
        f"<t id={turn_id}>{text}</t>" for turn_id, (_, text) in enumerate(first, start=1)
    )
    backend = FakeBackend(mutate=_replying(reply, ""))
    # One word a turn, so every turn of `first` fills the first chunk exactly.
    specs = [(speaker, "x") for speaker, _ in first] + [("Speaker 2", "four")]

    clean(_request(specs), backend, max_words=len(first))

    return _carried(backend.calls[1][1])


def _words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{n}" for n in range(count))


def test_a_tail_cut_inside_a_labeled_paragraph_keeps_its_label() -> None:
    tail = _tail_after([("Speaker 2", "Short one."), ("Speaker 1", _words(250))])

    assert tail == f"Speaker 1: {_words(250).split(' ', 50)[50]}"


def test_a_tail_cut_inside_a_continuation_takes_the_label_before_it() -> None:
    # A reply may break one turn into paragraphs, and only the first is labeled.
    turn = f"Opening.\n\n{_words(100, 'a')}\n\n{_words(150, 'b')}"

    tail = _tail_after([("Speaker 2", "Hi."), ("Speaker 1", turn)])

    kept = _words(100, "a").split(" ", 50)[50]
    assert tail == f"Speaker 1: {kept}\n\n{_words(150, 'b')}"


def test_a_tail_cut_inside_a_label_carries_it_once() -> None:
    assert _tail_after([("Speaker 1", _words(199))]) == f"Speaker 1: {_words(199)}"


def test_a_tail_cut_inside_a_turns_later_line_takes_the_turns_label() -> None:
    # A line break without a blank line keeps one paragraph, and the label on
    # its first line is still the one the tail is cut under.
    turn = f"{_words(10, 'b')}\n{_words(250, 'c')}"

    tail = _tail_after([("Speaker 1", _words(50, "a")), ("Speaker 2", turn)])

    assert tail == f"Speaker 2: {_words(250, 'c').split(' ', 50)[50]}"


def test_an_uncut_tail_opening_on_a_continuation_takes_the_label_before_it() -> None:
    tail = _tail_after([("Speaker 1", "Opening words.\n\nMore of speaker one.\n\nAnd the end.")])

    assert tail == "Speaker 1: More of speaker one.\n\nAnd the end."


def test_a_tail_under_a_label_holding_a_line_break_still_carries_the_last_words() -> None:
    # Nothing stops a label holding a line break, and then no line carries the
    # whole label: the tail cannot name its speaker, but it still goes.
    tail = _tail_after([("Ann\nLee", _words(250))])

    assert tail == _words(250).split(" ", 50)[50]


def _first_word_of_each_turn(reply: str) -> str:
    return re.sub(r"(<t id=\d+>)(\S+)[^<]*", r"\1\2", reply)


FORTY = " ".join(["alpha bravo charlie delta echo foxtrot golf hotel india juliet"] * 4)


def _empties_turn_2(reply: str) -> str:
    return reply.replace(f"<t id=2>{FORTY}", "<t id=2>")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_first_word_of_each_turn, id="every-turn-cut"),
        # A turn returned empty counts its input words against the chunk.
        pytest.param(_empties_turn_2, id="one-emptied"),
    ],
)
def test_a_reply_keeping_every_id_but_few_words_is_short(mutate: Callable[[str], str]) -> None:
    backend = FakeBackend(mutate=mutate)

    result = clean(_request([("A", FORTY), ("B", FORTY)]), backend)

    assert result.text == f"A: {FORTY}\n\nB: {FORTY}\n"
    assert _causes(result) == [(0, "short")]


def test_a_short_transcript_is_not_judged_by_the_ratio() -> None:
    # Under fifty words, filler removal alone can account for the shortfall.
    backend = FakeBackend(mutate=_first_word_of_each_turn)

    result = clean(_request([("A", _words(40))]), backend)

    assert result.truncated_chunks == []


def test_chunking_does_not_decide_whether_the_ratio_runs() -> None:
    # Sixty words is over the floor, so splitting it into under-floor chunks
    # must not switch the check off for every one of them.
    backend = FakeBackend(mutate=_first_word_of_each_turn)
    request = _request([("A", _words(30, "a")), ("B", _words(30, "b"))])

    result = clean(request, backend, max_words=30)

    assert result.chunks == 2
    assert _causes(result) == [(0, "short"), (1, "short")]


def test_a_wordless_chunk_is_reported_wordless_not_short() -> None:
    backend = FakeBackend(mutate=lambda reply: re.sub(r"(<t id=\d+>)[^<]*", r"\1", reply))

    result = clean(_request([("A", FORTY), ("B", FORTY)]), backend)

    assert _causes(result) == [(0, "wordless")]


def test_a_glossary_term_spelled_as_a_filler_returned_empty_is_wordless() -> None:
    # "ER" reads as a filler, but the glossary makes it a term: an empty tag drops it.
    backend = FakeBackend(mutate=lambda reply: reply.replace("<t id=2>ER</t>", "<t id=2></t>"))
    request = _request(
        [("Speaker 1", _words(60)), ("Speaker 2", "ER")], glossary={"ER": "emergency room"}
    )

    result = clean(request, backend, max_words=60)

    assert result.chunks == 2
    assert result.text == f"Speaker 1: {_words(60)}\n\nSpeaker 2: ER\n"
    assert _causes(result) == [(1, "wordless")]


def test_a_chunk_of_filler_answered_with_an_invented_sentence_is_short() -> None:
    reply = "<t id=1>The board approved a merger for 5000000 dollars yesterday in private.</t>"
    backend = FakeBackend(mutate=_replying(reply))
    fillers = " ".join(["um"] * 80)

    result = clean(_request([("Ann", fillers)]), backend)

    assert result.text == f"Ann: {fillers}\n"
    assert _causes(result) == [(0, "short")]


def test_a_fake_backend_satisfies_the_protocol() -> None:
    backend: CleanupBackend = FakeBackend()

    assert backend.complete("system", '<t id=4 speaker="A">hi</t>\n').text == "<t id=4>hi</t>"


def test_speaker_labels_are_stripped_before_the_number_check() -> None:
    text = "Speaker 1: we paid 10\n**Ann Lee**: yes\nplain line: not a label\n"

    assert strip_speaker_labels(text, ["Speaker 1", "Ann Lee"]) == (
        "we paid 10\nyes\nplain line: not a label\n"
    )


def test_a_label_bolded_through_the_colon_is_stripped() -> None:
    # As plausible a markdown rendering as **Label**:, and it used to leave the
    # closing asterisks at the head of the line.
    assert strip_speaker_labels("**Speaker 1:** we paid 10", ["Speaker 1"]) == "we paid 10"


@pytest.mark.parametrize(
    "line",
    [
        "Speaker 1: **really** important",
        "*Speaker 1*: **really** important",
        "*Speaker 1:* **really** important",
        "**Speaker 1**: **really** important",
        "**Speaker 1:** **really** important",
    ],
)
def test_emphasis_opening_the_speech_survives_the_label(line: str) -> None:
    assert strip_speaker_labels(line, ["Speaker 1"]) == "**really** important"


@pytest.mark.parametrize(
    "label",
    [
        "Speaker 1:",
        "*Speaker 1*:",
        "*Speaker 1:*",
        "**Speaker 1**:",
        "**Speaker 1:**",
    ],
)
def test_every_label_form_is_stripped(label: str) -> None:
    assert strip_speaker_labels(f"{label} *so* it goes", ["Speaker 1"]) == "*so* it goes"


def test_an_upper_cased_label_is_stripped() -> None:
    # An unstripped label puts its digits on the cleaned side of the number check.
    assert strip_speaker_labels("SPEAKER 1: we paid 10", ["Speaker 1"]) == "we paid 10"


def test_the_longest_matching_label_wins() -> None:
    assert strip_speaker_labels("Ann Lee: hello", ["Ann", "Ann Lee"]) == "hello"


def test_stripping_no_labels_leaves_the_text_alone() -> None:
    assert strip_speaker_labels("Ann: hello", []) == "Ann: hello"


def test_the_provenance_header_is_a_yaml_block() -> None:
    header = provenance_header(
        title="Budget call",
        source=Source(kind="audio", ref="numbers.mp3", sha256=None),
        engine=Engine(name="xai-stt", model="grok-voice-transcribe-2.0", params={}),
        backend_name="claude-cli",
        backend_model="opus",
        speaker_key={"Speaker 1": "Ann Lee"},
        number_diff=NumberDiff(missing=["2026"], reduced=["10"], added=[], checked=4),
        generated_at=datetime(2026, 9, 22, 10, 30, tzinfo=UTC),
    )

    assert header == (
        "---\n"
        'title: "Budget call"\n'
        'source_kind: "audio"\n'
        'source_ref: "numbers.mp3"\n'
        'stt_engine: "xai-stt"\n'
        'stt_model: "grok-voice-transcribe-2.0"\n'
        'cleanup_backend: "claude-cli"\n'
        'cleanup_model: "opus"\n'
        f'cleanup_prompt_version: "{CLEANUP_PROMPT_VERSION}"\n'
        'generated_at: "2026-09-22T10:30:00+00:00"\n'
        "speakers:\n"
        '  "Speaker 1": "Ann Lee"\n'
        "numbers_checked: 4\n"
        'numbers_missing: ["2026"]\n'
        'numbers_reduced: ["10"]\n'
        "numbers_added: []\n"
        "---\n"
    )


def test_an_absent_stt_model_and_an_empty_key_stay_valid_yaml() -> None:
    header = provenance_header(
        title='a "quoted" title',
        source=Source(kind="krisp", ref="meeting-1"),
        engine=Engine(name="krisp"),
        backend_name="claude-cli",
        backend_model="opus",
        speaker_key={},
        number_diff=NumberDiff(),
        generated_at=datetime(2026, 9, 22, 10, 30, tzinfo=UTC),
    )

    assert "stt_model: null\n" in header
    assert "speakers: {}\n" in header
    assert 'title: "a \\"quoted\\" title"\n' in header


def test_pairs_split_on_the_first_equals_only() -> None:
    assert parse_pairs(["Speaker 1 = Ann Lee", "x=a=b"], "--speaker") == {
        "Speaker 1": "Ann Lee",
        "x": "a=b",
    }


def test_a_pair_without_an_equals_is_a_caller_error() -> None:
    with pytest.raises(InputValidationError, match="--speaker expects KEY=VALUE"):
        parse_pairs(["Speaker 1"], "--speaker")


def test_a_pair_with_an_empty_key_is_a_caller_error() -> None:
    with pytest.raises(InputValidationError, match="KEY=VALUE"):
        parse_pairs([" =Ann Lee"], "--speaker")


def test_a_pair_with_an_empty_value_is_a_caller_error() -> None:
    # An empty final name reaches the label alternation as an empty branch,
    # which strips the head of any line that starts with a colon.
    with pytest.raises(InputValidationError, match="KEY=VALUE"):
        parse_pairs(["x="], "--speaker")


def test_a_pairs_file_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "speakers.txt"
    path.write_text("# who is who\n\nSpeaker 1=Ann Lee\nSpeaker 2=Bo Chen\n", encoding="utf-8")

    assert read_pairs_file(path, "--speakers-file") == {
        "Speaker 1": "Ann Lee",
        "Speaker 2": "Bo Chen",
    }


def test_a_byte_order_mark_is_not_part_of_the_first_key(tmp_path: Path) -> None:
    path = tmp_path / "glossary.txt"
    path.write_text("\N{BYTE ORDER MARK}ackme=Acme\n", encoding="utf-8")

    assert read_pairs_file(path, "--glossary-file") == {"ackme": "Acme"}


def test_an_unreadable_pairs_file_is_a_caller_error(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="cannot read --speakers-file"):
        read_pairs_file(tmp_path / "absent.txt", "--speakers-file")


def test_a_non_utf8_pairs_file_is_a_caller_error(tmp_path: Path) -> None:
    path = tmp_path / "speakers.txt"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(InputValidationError, match="cannot read --speakers-file"):
        read_pairs_file(path, "--speakers-file")
