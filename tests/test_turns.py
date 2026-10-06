"""Canonical test shape for a stage: a table of word specs in, turns out."""

from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scribe.schema import Source, Track, Word
from scribe.turns import build_turns, turns_from_speakers, word_speakers

Spec = tuple[int | None, float, float]

# The smallest table in which a micro-turn collapse can erase a speaker.
MICRO_BETWEEN_RUNS: list[Spec] = [
    (7, 0.0, 0.6),
    (7, 0.7, 1.4),
    (3, 1.5, 2.0),
    (7, 2.1, 2.5),
    (7, 2.6, 3.0),
    (7, 3.1, 3.5),
]


def _words(specs: list[Spec]) -> list[Word]:
    """Build words named w0, w1, ... from (speaker, start, end) triples."""
    return [
        Word(text=f"w{index}", start=start, end=end, speaker=speaker)
        for index, (speaker, start, end) in enumerate(specs)
    ]


def _run(specs: list[Spec]) -> list[tuple[str, str]]:
    return [(turn.speaker, turn.text) for turn in build_turns(_words(specs))]


def _said(*runs: tuple[int | None, str], seconds: float = 0.3) -> list[Word]:
    """Build words `seconds` apart from (speaker, "words of one raw run") pairs."""
    said = [(speaker, text) for speaker, line in runs for text in line.split()]
    return [
        Word(text=text, start=index * seconds, end=(index + 0.8) * seconds, speaker=speaker)
        for index, (speaker, text) in enumerate(said)
    ]


def _spoken(words: list[Word], *, snap_words: int = 5) -> list[tuple[str, str]]:
    return [(turn.speaker, turn.text) for turn in build_turns(words, snap_words=snap_words)]


def test_empty_input_yields_no_turns() -> None:
    assert build_turns([]) == []


def test_consecutive_same_speaker_words_form_one_turn() -> None:
    specs: list[Spec] = [
        (7, 0.0, 0.5),
        (7, 0.6, 1.1),
        (7, 1.2, 1.7),
        (4, 1.8, 2.3),
        (4, 2.4, 2.9),
        (4, 3.0, 3.5),
    ]
    turns = build_turns(_words(specs))

    assert [(t.speaker, t.text) for t in turns] == [
        ("Speaker 1", "w0 w1 w2"),
        ("Speaker 2", "w3 w4 w5"),
    ]
    assert (turns[0].start, turns[0].end) == (0.0, 1.7)
    assert (turns[1].start, turns[1].end) == (1.8, 3.5)


def test_undiarized_words_are_one_speaker() -> None:
    specs: list[Spec] = [(None, 0.0, 0.5), (None, 0.6, 1.1), (None, 1.2, 1.7)]
    assert _run(specs) == [("Speaker 1", "w0 w1 w2")]


def test_labels_rank_by_first_appearance_not_by_api_id() -> None:
    specs: list[Spec] = [
        (7, 0.0, 0.5),
        (7, 0.6, 1.1),
        (7, 1.2, 1.7),
        (4, 1.8, 2.3),
        (4, 2.4, 2.9),
        (4, 3.0, 3.5),
        (7, 3.6, 4.1),
        (7, 4.2, 4.7),
        (7, 4.8, 5.3),
    ]
    assert [speaker for speaker, _ in _run(specs)] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_micro_turn_in_the_middle_merges_into_the_previous_turn() -> None:
    # The brief's worked example: raw ids [7,7,3,7,7,7,5,5,5,5]; the opening
    # id-7 turn spans 1.4 s, so only the lone id-3 word is a micro-turn.
    specs: list[Spec] = [
        (7, 0.0, 0.6),
        (7, 0.7, 1.4),
        (3, 1.5, 2.0),
        (7, 2.1, 2.5),
        (7, 2.6, 3.0),
        (7, 3.1, 3.5),
        (5, 3.6, 4.0),
        (5, 4.1, 4.5),
        (5, 4.6, 5.0),
        (5, 5.1, 5.5),
    ]
    turns = build_turns(_words(specs))

    assert [(t.speaker, t.text) for t in turns] == [
        ("Speaker 1", "w0 w1 w2 w3 w4 w5"),
        ("Speaker 2", "w6 w7 w8 w9"),
    ]
    assert (turns[0].start, turns[0].end) == (0.0, 3.5)


def test_consecutive_micro_turns_between_different_speakers_keep_their_own_turns() -> None:
    # [7][3][5][7]: neither micro-turn sits between two turns of one speaker, so
    # neither is a diarization flicker and both speakers keep their words.
    specs: list[Spec] = [
        (7, 0.0, 0.3),
        (7, 0.4, 0.7),
        (7, 0.8, 1.1),
        (3, 1.2, 1.5),
        (5, 1.6, 1.9),
        (7, 2.0, 2.3),
        (7, 2.4, 2.7),
        (7, 2.8, 3.1),
    ]
    assert _run(specs) == [
        ("Speaker 1", "w0 w1 w2"),
        ("Speaker 2", "w3"),
        ("Speaker 3", "w4"),
        ("Speaker 1", "w5 w6 w7"),
    ]


def test_an_opening_micro_turn_keeps_its_own_speaker() -> None:
    # Speaker 7 opens with a two-word aside and speaks again at the end; with
    # no turn before it, the aside cannot be a flicker inside one speaker's run.
    specs: list[Spec] = [
        (7, 0.0, 0.3),
        (7, 0.4, 0.7),
        (4, 0.8, 1.1),
        (4, 1.2, 1.5),
        (4, 1.6, 1.9),
        (4, 2.0, 2.3),
        (7, 2.4, 2.9),
        (7, 3.0, 3.5),
        (7, 3.6, 4.1),
    ]
    turns = build_turns(_words(specs))

    assert [(t.speaker, t.text) for t in turns] == [
        ("Speaker 1", "w0 w1"),
        ("Speaker 2", "w2 w3 w4 w5"),
        ("Speaker 1", "w6 w7 w8"),
    ]
    assert (turns[0].start, turns[0].end) == (0.0, 0.7)


def test_a_closing_micro_turn_keeps_its_own_speaker() -> None:
    specs: list[Spec] = [(7, 0.0, 0.5), (7, 0.6, 1.1), (7, 1.2, 1.7), (4, 1.8, 2.1)]
    assert _run(specs) == [("Speaker 1", "w0 w1 w2"), ("Speaker 2", "w3")]


def test_flicker_merging_repeats_until_a_pass_changes_nothing() -> None:
    # Pass one folds the id-5 flicker into the id-3 run around it; that merged
    # run is itself a micro-turn at min_turn_words=4, so only a second pass
    # folds it into the id-7 run.
    specs: list[Spec] = [
        (7, 0.0, 0.5),
        (7, 0.6, 1.1),
        (7, 1.2, 1.7),
        (7, 1.8, 2.3),
        (3, 2.4, 2.6),
        (5, 2.7, 2.9),
        (3, 3.0, 3.2),
        (7, 3.3, 3.8),
        (7, 3.9, 4.4),
        (7, 4.5, 5.0),
        (7, 5.1, 5.6),
    ]
    turns = build_turns(_words(specs), min_turn_words=4)

    assert [(t.speaker, t.text) for t in turns] == [
        ("Speaker 1", "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10")
    ]
    assert (turns[0].start, turns[0].end) == (0.0, 5.6)


def test_a_lone_micro_turn_has_nowhere_to_go_and_survives() -> None:
    assert _run([(7, 0.0, 0.3)]) == [("Speaker 1", "w0")]


def test_a_speaker_collapsed_away_never_earns_a_label() -> None:
    assert {speaker for speaker, _ in _run(MICRO_BETWEEN_RUNS)} == {"Speaker 1"}


def test_thresholds_are_tunable() -> None:
    # min_turn_words=1 makes no turn short enough to collapse: the lone id-3
    # word keeps its own turn, and the two id-7 runs stay separate.
    kept = build_turns(_words(MICRO_BETWEEN_RUNS), min_turn_words=1)
    assert [t.speaker for t in kept] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_a_backchannel_between_two_different_speakers_stays_its_own_turn() -> None:
    words = _said((7, "the first item is the budget."), (4, "Right."), (5, "we review it later."))
    assert _spoken(words) == [
        ("Speaker 1", "the first item is the budget."),
        ("Speaker 2", "Right."),
        ("Speaker 3", "we review it later."),
    ]


def test_a_flicker_island_mid_sentence_merges_into_the_speaker_around_it() -> None:
    words = _said((7, "we can review the"), (4, "numbers"), (7, "after lunch today."))
    assert _spoken(words) == [("Speaker 1", "we can review the numbers after lunch today.")]


def test_an_island_after_a_sentence_end_stays_its_own_turn() -> None:
    words = _said((7, "the first item is the budget."), (4, "Right."), (7, "we review it later."))
    assert _spoken(words) == [
        ("Speaker 1", "the first item is the budget."),
        ("Speaker 2", "Right."),
        ("Speaker 1", "we review it later."),
    ]


# Words 1.5 s apart are never micro-turns, so only the snap moves anything.
SLOW = 1.5


def test_a_mid_sentence_switch_snaps_left_to_the_nearest_sentence_end() -> None:
    words = _said((7, "we are done. and so"), (4, "the next item is here"), seconds=SLOW)
    assert _spoken(words) == [
        ("Speaker 1", "we are done."),
        ("Speaker 2", "and so the next item is here"),
    ]


def test_a_mid_sentence_switch_snaps_right_to_the_nearest_sentence_end() -> None:
    words = _said((7, "so we start with the"), (4, "budget. then the next item"), seconds=SLOW)
    assert _spoken(words) == [
        ("Speaker 1", "so we start with the budget."),
        ("Speaker 2", "then the next item"),
    ]


def test_the_left_sentence_end_wins_a_tie() -> None:
    words = _said((7, "we stopped. and"), (4, "then. more words here"), seconds=SLOW)
    assert _spoken(words) == [
        ("Speaker 1", "we stopped."),
        ("Speaker 2", "and then. more words here"),
    ]


def test_a_switch_never_snaps_across_another_switch() -> None:
    # The id-4 switch's only sentence end lies past the id-5 switch, so it
    # stays put; the id-5 switch then snaps right onto that end.
    words = _said((7, "we then"), (4, "maybe"), (5, "go. on and"), seconds=SLOW)
    assert _spoken(words) == [
        ("Speaker 1", "we then"),
        ("Speaker 2", "maybe go."),
        ("Speaker 3", "on and"),
    ]


def test_a_switch_never_snaps_onto_the_last_word() -> None:
    words = _said((7, "and we reach the"), (4, "end."), seconds=SLOW)
    assert _spoken(words) == [("Speaker 1", "and we reach the"), ("Speaker 2", "end.")]


def test_zero_snap_words_leaves_switches_where_they_fell() -> None:
    words = _said((7, "we are done. and so"), (4, "the next item is here"), seconds=SLOW)
    assert _spoken(words, snap_words=0) == [
        ("Speaker 1", "we are done. and so"),
        ("Speaker 2", "the next item is here"),
    ]


@given(
    said=st.lists(
        st.tuples(
            st.sampled_from([None, 0, 1, 2]),
            st.sampled_from(["", ".", "?", "!", ",", ". "]),
            st.floats(min_value=0.05, max_value=2.0),
        ),
        max_size=40,
    ),
    min_turn_words=st.integers(min_value=0, max_value=5),
    snap_words=st.integers(min_value=0, max_value=8),
)
def test_turns_keep_every_word_once_in_order(
    said: list[tuple[int | None, str, float]], min_turn_words: int, snap_words: int
) -> None:
    words: list[Word] = []
    clock = 0.0
    for index, (speaker, mark, length) in enumerate(said):
        words.append(Word(text=f"w{index}{mark}", start=clock, end=clock + length, speaker=speaker))
        clock += length + 0.1

    turns = build_turns(words, min_turn_words=min_turn_words, snap_words=snap_words)

    assert " ".join(turn.text for turn in turns) == " ".join(word.text for word in words)
    assert all(first.speaker != second.speaker for first, second in pairwise(turns))


@given(
    said=st.lists(
        st.tuples(
            st.sampled_from([None, 0, 1, 2]),
            st.sampled_from(["", ".", "?"]),
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=0.05, max_value=3.0),
        ),
        min_size=1,
        max_size=30,
    ),
    snap_words=st.integers(min_value=0, max_value=8),
)
def test_every_word_lies_inside_its_turn(
    said: list[tuple[int | None, str, float, float]], snap_words: int
) -> None:
    # Timings overlap, so a word can end after a later word does.
    words: list[Word] = []
    clock = 0.0
    for index, (speaker, mark, gap, length) in enumerate(said):
        clock += gap
        words.append(Word(text=f"w{index}{mark}", start=clock, end=clock + length, speaker=speaker))

    turns = build_turns(words, snap_words=snap_words)

    remaining = iter(words)
    for turn in turns:
        held = [next(remaining) for _ in turn.text.split()]
        assert all(turn.start <= word.start and word.end <= turn.end for word in held)


def test_word_speakers_carry_the_merge_and_the_snap() -> None:
    words = _said((7, "so the plan"), (4, "is"), (7, "set. and"), (4, "then it goes"))

    assert word_speakers(words) == [7, 7, 7, 7, 7, 4, 4, 4, 4]
    assert word_speakers(words, snap_words=0) == [7, 7, 7, 7, 7, 7, 4, 4, 4]


def test_words_without_a_speaker_among_diarized_ones_are_speaker_question_and_unranked() -> None:
    words = _said((None, "so"), (7, "the plan is set."), (None, "yes indeed okay"), (4, "then go."))

    assert _spoken(words) == [
        ("Speaker ?", "so"),
        ("Speaker 1", "the plan is set."),
        ("Speaker ?", "yes indeed okay"),
        ("Speaker 2", "then go."),
    ]


def test_words_without_a_speaker_take_no_part_in_the_merge_or_the_snap() -> None:
    # Unattributed words neither join a flicker's speaker nor shield a flicker from its merge.
    assert word_speakers(_said((7, "so the plan"), (None, "uh"), (7, "is set."))) == [
        *(7, 7, 7, None, 7, 7)
    ]
    assert word_speakers(_said((7, "so the"), (None, "uh"), (4, "plan"), (7, "is set."))) == [
        *(7, 7, None, 7, 7, 7)
    ]


@given(said=st.lists(st.tuples(st.sampled_from([None, 0, 1]), st.sampled_from(["w", "w."]))))
def test_words_with_a_speaker_are_settled_as_if_the_others_were_absent(
    said: list[tuple[int | None, str]],
) -> None:
    words = [
        Word(text=text, start=index * 0.3, end=index * 0.3 + 0.2, speaker=speaker)
        for index, (speaker, text) in enumerate(said)
    ]
    attributed = [word for word in words if word.speaker is not None]

    speakers = word_speakers(words)

    if attributed:
        assert [s for s, w in zip(speakers, words, strict=True) if w.speaker is not None] == (
            word_speakers(attributed)
        )
        assert all(s is None for s, w in zip(speakers, words, strict=True) if w.speaker is None)
    else:
        assert speakers == [None] * len(words)


def test_turns_from_speakers_merges_and_snaps_nothing() -> None:
    words = _said((1, "so the plan is set. and then"))

    turns = turns_from_speakers(words, [5, 5, 9, 5, 5, 2, 2])

    assert [(turn.speaker, turn.text) for turn in turns] == [
        ("Speaker 1", "so the"),
        ("Speaker 2", "plan"),
        ("Speaker 1", "is set."),
        ("Speaker 3", "and then"),
    ]


def test_turns_from_speakers_needs_one_id_per_word() -> None:
    with pytest.raises(ValueError, match="zip"):
        turns_from_speakers(_said((1, "two words")), [1])


def test_a_named_id_carries_its_name_and_the_others_keep_their_rank() -> None:
    words = _said((5, "so the plan"), (None, "um"), (9, "is set."), (5, "and then"))
    speakers = [word.speaker for word in words]
    unnamed = turns_from_speakers(words, speakers)

    named = turns_from_speakers(words, speakers, names={5: "Alice"})

    assert [turn.speaker for turn in unnamed] == [
        "Speaker 1",
        "Speaker ?",
        "Speaker 2",
        "Speaker 1",
    ]
    # Rank 2 stays rank 2 although rank 1 now has a name.
    assert [turn.speaker for turn in named] == ["Alice", "Speaker ?", "Speaker 2", "Alice"]
    assert [(turn.start, turn.end, turn.text) for turn in named] == [
        (turn.start, turn.end, turn.text) for turn in unnamed
    ]


@given(
    said=st.lists(
        st.tuples(
            st.sampled_from([None, 0, 1, 2]),
            st.sampled_from(["", ".", "?", ","]),
            st.floats(min_value=0.05, max_value=2.0),
        ),
        max_size=40,
    ),
    min_turn_words=st.integers(min_value=0, max_value=5),
    snap_words=st.integers(min_value=0, max_value=8),
)
def test_build_turns_is_word_speakers_then_turns_from_speakers(
    said: list[tuple[int | None, str, float]], min_turn_words: int, snap_words: int
) -> None:
    words: list[Word] = []
    clock = 0.0
    for index, (speaker, mark, length) in enumerate(said):
        words.append(Word(text=f"w{index}{mark}", start=clock, end=clock + length, speaker=speaker))
        clock += length + 0.1

    speakers = word_speakers(words, min_turn_words=min_turn_words, snap_words=snap_words)

    assert len(speakers) == len(words)
    assert turns_from_speakers(words, speakers) == build_turns(
        words, min_turn_words=min_turn_words, snap_words=snap_words
    )


def _tracked(*runs: tuple[int | None, int, str], seconds: float = 0.3) -> list[Word]:
    """Words `seconds` apart from (speaker, track, "words of one raw run") triples."""
    said = [(speaker, track, text) for speaker, track, line in runs for text in line.split()]
    return [
        Word(text=text, start=at * seconds, end=(at + 0.8) * seconds, speaker=speaker, track=track)
        for at, (speaker, track, text) in enumerate(said)
    ]


TRACKS = [
    Track(
        role="mic", label="Alice", source=Source(kind="audio", ref="mic.wav"), transcript_sha256="a"
    ),
    Track(role="app", source=Source(kind="audio", ref="app.wav"), transcript_sha256="b"),
]


def test_an_operator_word_inside_a_remote_sentence_keeps_its_own_speaker() -> None:
    words = _tracked((3, 1, "so the plan"), (9, 0, "Yeah."), (3, 1, "is set."))

    assert word_speakers(words) == [3, 3, 3, 9, 3, 3]


@given(
    said=st.lists(
        st.tuples(
            st.sampled_from([None, 0, 1, 2]),
            st.sampled_from([0, 1]),
            st.sampled_from(["", ".", ","]),
            st.floats(min_value=0.05, max_value=2.0),
        ),
        max_size=40,
    ),
)
def test_each_track_settles_as_if_alone(said: list[tuple[int | None, int, str, float]]) -> None:
    words = [
        Word(
            text=f"w{at}{mark}", start=at * 0.5, end=at * 0.5 + length, speaker=speaker, track=track
        )
        for at, (speaker, track, mark, length) in enumerate(said)
    ]

    settled = word_speakers(words)

    for track in (0, 1):
        held = [at for at, word in enumerate(words) if word.track == track]
        assert [settled[at] for at in held] == word_speakers([words[at] for at in held])


def test_tracks_label_the_mic_by_name_and_rank_app_speakers_alone() -> None:
    words = _tracked((9, 0, "Hello there."), (5, 1, "Hi."), (None, 1, "uh"), (2, 1, "Morning."))

    turns = turns_from_speakers(words, [word.speaker for word in words], tracks=TRACKS)

    assert [(turn.speaker, turn.text) for turn in turns] == [
        ("Alice", "Hello there."),
        ("Speaker 1", "Hi."),
        ("Speaker ?", "uh"),
        ("Speaker 2", "Morning."),
    ]
