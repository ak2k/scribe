"""The vote stage: backbone words changed or added to only where two hypotheses agree."""

from __future__ import annotations

from itertools import groupby

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scribe.gaps import MIN_DROP_SECONDS, MIN_DROP_WORDS, find_holes
from scribe.schema import Engine, Source, Transcript, Word
from scribe.vote import (
    first_decrease,
    is_filler,
    norm_tokens,
    vote_transcripts,
    vote_words,
    word_key,
)

Spec = tuple[str, float, float]

VOCAB = ["the", "cat", "Cat,", "sat.", "um", "uh", "mm", "hm", "yeah", "okay", "twenty-five", "..."]
FILLERS = {"um", "uh", "mm", "hm"}


def _words(*specs: Spec, speaker: int | None = None) -> list[Word]:
    return [Word(text=text, start=start, end=end, speaker=speaker) for text, start, end in specs]


def _texts(words: list[Word]) -> list[str]:
    return [word.text for word in words]


def test_a_token_both_hypotheses_hold_replaces_the_backbones() -> None:
    backbone = _words(("the", 0.0, 0.4), ("cat", 0.5, 0.9), ("sat", 1.0, 1.4), speaker=3)
    hypothesis = _words(("the", 0.0, 0.4), ("hat", 0.6, 0.9), ("sat", 1.0, 1.4))

    voted = vote_words(backbone, hypothesis, hypothesis)

    assert _texts(voted) == ["the", "hat", "sat"]
    # A substituted word keeps the backbone's slot: its times and speaker.
    assert voted[1] == Word(text="hat", start=0.5, end=0.9, speaker=3)


def test_one_dissenting_hypothesis_keeps_the_backbone() -> None:
    backbone = _words(("the", 0.0, 0.4), ("cat", 0.5, 0.9))
    primary = _words(("the", 0.0, 0.4), ("hat", 0.5, 0.9))
    secondary = _words(("the", 0.0, 0.0), ("bat", 0.5, 0.5))

    assert vote_words(backbone, primary, secondary) == backbone


def test_a_filler_is_never_substituted_in() -> None:
    backbone = _words(("and", 0.0, 0.4))
    hypothesis = _words(("um", 0.0, 0.4))

    assert vote_words(backbone, hypothesis, hypothesis) == backbone


def test_a_changed_word_takes_primary_spelling_and_backbone_punctuation_and_capital() -> None:
    backbone = _words(("Cat,", 0.0, 0.4))
    primary = _words(("hat.", 0.0, 0.4))
    secondary = _words(("hat", 0.1, 0.1))

    assert _texts(vote_words(backbone, primary, secondary)) == ["Hat,"]


def test_a_multi_token_backbone_word_changes_to_its_joined_tokens() -> None:
    backbone = _words(("Twenty-five,", 0.0, 0.8))
    hypothesis = _words(("twenty", 0.0, 0.4), ("six", 0.4, 0.8))

    assert _texts(vote_words(backbone, hypothesis, hypothesis)) == ["twenty six"]


def test_agreed_gap_words_are_inserted_with_primary_times_and_nearest_speaker() -> None:
    # The gap is under 2 s, so no hole: the nearest word's speaker applies.
    backbone = [Word(text="the", start=0.0, end=1.2, speaker=1)]
    backbone.append(Word(text="sat", start=3.0, end=3.4, speaker=2))
    primary = _words(("the", 0.0, 0.4), ("Cat", 2.4, 2.8), ("sat", 3.0, 3.4))
    secondary = _words(("the", 0.0, 0.0), ("cat", 2.4, 2.4), ("sat", 3.0, 3.0))

    voted = vote_words(backbone, primary, secondary)

    assert voted == [backbone[0], Word(text="Cat", start=2.4, end=2.8, speaker=2), backbone[1]]


@pytest.mark.parametrize("gap", ["yeah", "mm hm", "okay yeah"])
def test_a_run_of_only_backchannels_is_not_inserted(gap: str) -> None:
    backbone = _words(("so", 0.0, 0.4), ("then", 3.0, 3.4))
    said = [(text, 1.0 + index * 0.4, 1.3 + index * 0.4) for index, text in enumerate(gap.split())]
    hypothesis = _words(("so", 0.0, 0.4), *said, ("then", 3.0, 3.4))

    assert vote_words(backbone, hypothesis, hypothesis) == backbone


def test_a_filler_is_never_inserted_but_its_run_still_is() -> None:
    backbone = _words(("so", 0.0, 0.4), ("then", 3.0, 3.4))
    hypothesis = _words(("so", 0.0, 0.4), ("um", 1.0, 1.3), ("cat", 1.4, 1.7), ("then", 3.0, 3.4))

    assert _texts(vote_words(backbone, hypothesis, hypothesis)) == ["so", "cat", "then"]


def test_a_mistimed_backbone_word_is_not_inserted_again() -> None:
    # The backbone's "cat" sits 0.6 s from the hypotheses', too far to pair,
    # and within 1 s of the agreed insertion: a timing miss, not a missing word.
    backbone = _words(("the", 0.0, 0.3), ("cat", 1.4, 1.7))
    hypothesis = _words(("the", 0.0, 0.3), ("cat", 0.5, 0.8))

    assert vote_words(backbone, hypothesis, hypothesis) == backbone


def test_an_inserted_word_starts_between_its_neighbors() -> None:
    backbone = _words(("a", 0.0, 0.4), ("b", 1.0, 1.4), ("c", 3.0, 3.4))
    # Both land between "b" and "c": "y" starts before the voted "b" and is
    # raised to it, "x" starts after "c" and is lowered to it.
    primary = _words(("a", 0.0, 0.4), ("b", 0.7, 0.9), ("y", 0.8, 0.95), ("x", 3.1, 3.2))
    primary += _words(("c", 3.15, 3.4))
    secondary = _words(("a", 0.0, 0.0), ("b", 0.7, 0.7), ("y", 0.8, 0.8), ("x", 3.1, 3.1))
    secondary += _words(("c", 3.15, 3.15))

    voted = vote_words(backbone, primary, secondary)

    assert [(word.text, word.start, word.end) for word in voted] == [
        ("a", 0.0, 0.4),
        ("b", 1.0, 1.4),
        ("y", 1.0, 1.0),
        ("x", 3.0, 3.2),
        ("c", 3.0, 3.4),
    ]


def test_an_agreed_insertion_in_a_backbone_hole_goes_before_the_word_after_the_hole() -> None:
    # The hole is longer than the band, so words heard late in it lie over 6 s
    # after the backbone word before it.
    backbone = _words(("alpha", 0.0, 0.3), ("zulu", 20.0, 20.4))
    heard = _words(
        ("alpha", 0.0, 0.3),
        ("one", 10.0, 10.4),
        ("two", 10.5, 10.9),
        ("three", 11.0, 11.4),
        ("four", 11.5, 11.9),
        ("zulu", 20.0, 20.4),
    )

    voted = vote_words(backbone, heard, heard)

    assert [(w.text, w.start, w.end) for w in voted] == [(w.text, w.start, w.end) for w in heard]


def test_a_tie_between_pairing_and_deleting_pairs() -> None:
    # "hat" overlaps both slots: pairing it with "cat" costs what deleting "cat" does.
    backbone = _words(("the", 0.0, 0.4), ("cat", 0.5, 0.9))
    hypothesis = _words(("hat", 0.3, 0.6))

    assert _texts(vote_words(backbone, hypothesis, hypothesis)) == ["the", "hat"]


def test_a_changed_word_keeps_the_primarys_spelling_inside_it() -> None:
    backbone = _words(("phone,", 0.0, 0.4))
    primary = _words(("iPhone", 0.0, 0.4))
    secondary = _words(("iphone", 0.1, 0.1))

    assert _texts(vote_words(backbone, primary, secondary)) == ["iPhone,"]


def test_a_primary_word_inserted_in_part_is_written_as_its_kept_tokens() -> None:
    backbone = _words(("so", 0.0, 0.4), ("then", 3.0, 3.4))
    primary = _words(("so", 0.0, 0.4), ("Um-Cat", 1.0, 1.5), ("then", 3.0, 3.4))
    secondary = _words(("so", 0.0, 0.0), ("um", 1.0, 1.0), ("cat", 1.2, 1.2), ("then", 3.0, 3.0))

    assert _texts(vote_words(backbone, primary, secondary)) == ["so", "cat", "then"]


def test_a_backbone_word_only_one_hypothesis_left_unpaired_still_guards_a_repeat() -> None:
    # The secondary pairs the backbone's "cat"; the primary's lies 0.6 s off it.
    backbone = _words(("the", 0.0, 0.3), ("cat", 1.4, 1.7))
    primary = _words(("the", 0.0, 0.3), ("cat", 0.5, 0.8))
    secondary = _words(("the", 0.0, 0.0), ("cat", 0.5, 0.5), ("cat", 1.5, 1.5))

    assert vote_words(backbone, primary, secondary) == backbone


def test_the_band_does_not_narrow_after_a_long_backbone_word() -> None:
    # Row "b" alone would reach only 6.9 s, cutting "x" off from the rows after it.
    backbone = _words(("a", 0.0, 10.0), ("b", 0.5, 0.9), ("c", 12.0, 12.4))
    hypothesis = _words(("a", 0.0, 0.4), ("b", 0.5, 0.9), ("x", 8.0, 8.3), ("c", 12.0, 12.4))

    assert _texts(vote_words(backbone, hypothesis, hypothesis)) == ["a", "b", "x", "c"]


def test_a_word_inserted_inside_a_multi_token_backbone_word_follows_it() -> None:
    backbone = _words(("twenty-five", 0.0, 1.0))
    hypothesis = _words(("twenty", 0.0, 0.3), ("odd", 0.35, 0.5), ("five", 0.6, 1.0))

    assert _texts(vote_words(backbone, hypothesis, hypothesis)) == ["twenty-five", "odd"]


@pytest.mark.parametrize(
    ("before_end", "inserted_at", "speaker"),
    [(1.5, 2.25, 1), (2.0, 2.4, 1)],
    ids=["a tie goes to the earlier word", "distance counts from either end"],
)
def test_an_inserted_word_takes_the_speaker_nearest_in_time(
    before_end: float, inserted_at: float, speaker: int
) -> None:
    backbone = [Word(text="so", start=0.0, end=before_end, speaker=1)]
    backbone.append(Word(text="then", start=3.0, end=3.4, speaker=2))
    primary = _words(("so", 0.0, 0.4), ("cat", inserted_at, inserted_at + 0.2), ("then", 3.0, 3.4))
    secondary = _words(("so", 0.0, 0.0), ("cat", inserted_at, inserted_at), ("then", 3.0, 3.0))

    voted = vote_words(backbone, primary, secondary)

    assert (voted[1].text, voted[1].speaker) == ("cat", speaker)


def test_words_inserted_where_the_backbone_has_no_slot_have_no_speaker() -> None:
    backbone = _words(("...", 0.0, 0.4), speaker=1)
    hypothesis = _words(("cat", 0.0, 0.4))

    assert vote_words(backbone, hypothesis, hypothesis) == [
        backbone[0],
        Word(text="cat", start=0.0, end=0.4),
    ]


def test_words_inserted_in_a_hole_in_the_backbone_have_no_speaker() -> None:
    # The backbone may have dropped another speaker's turn in a 2 s hole; a
    # word missed in a short pause is almost always the same speaker's.
    backbone = _words(("so", 0.0, 0.4), ("sat", 0.7, 1.0), ("then", 20.0, 20.4), speaker=1)
    lost = _words(("we", 10.0, 10.3), ("lost", 10.4, 10.7), ("it", 10.8, 11.0))
    heard = [*_words(("so", 0.0, 0.4), ("cat", 0.45, 0.65), ("sat", 0.7, 1.0)), *lost]
    heard += _words(("then", 20.0, 20.4))

    voted = vote_words(backbone, heard, heard)

    cat = Word(text="cat", start=0.45, end=0.65, speaker=1)
    assert voted == [backbone[0], cat, backbone[1], *lost, backbone[2]]


@pytest.mark.parametrize(("count", "speaker"), [(1, 1), (2, 1), (3, None)])
def test_only_three_words_or_more_in_a_row_in_a_backbone_hole_lose_their_speaker(
    count: int, speaker: int | None
) -> None:
    # A word or two is too little speech for its speaker to matter.
    backbone = _words(("so", 0.0, 0.4), ("then", 5.4, 5.8), speaker=1)
    texts = ["we", "lost", "it"][:count]
    run = _words(*[(text, 2.0 + 0.4 * at, 2.3 + 0.4 * at) for at, text in enumerate(texts)])
    heard = [*_words(("so", 0.0, 0.4)), *run, *_words(("then", 5.4, 5.8))]

    voted = vote_words(backbone, heard, heard)

    assert [(word.text, word.speaker) for word in voted[1:-1]] == [
        (word.text, speaker) for word in run
    ]


def test_the_holes_run_to_the_backbones_duration_when_it_has_one() -> None:
    source = Source(kind="audio", ref="meeting.mp3")
    said = _words(("so", 0.0, 0.4), speaker=1)
    heard = _words(("so", 0.0, 0.4), ("we", 10.0, 10.3), ("lost", 10.4, 10.7), ("it", 10.8, 11.0))
    backbone = Transcript(source=source, engine=Engine(name="xai-stt"), text="", words=said)
    lasting = backbone.model_copy(update={"duration": 20.0})
    hypothesis = Transcript(source=source, engine=Engine(name="parakeet"), text="", words=heard)

    voted = vote_transcripts(lasting, hypothesis, hypothesis)

    assert [word.speaker for word in voted.words] == [1, None, None, None]
    assert voted.engine.params["words_inserted_unattributed"] == 3
    # With no duration the last hole ends at the backbone's last word.
    unbounded = vote_transcripts(backbone, hypothesis, hypothesis)
    assert [word.speaker for word in unbounded.words] == [1, 1, 1, 1]
    assert unbounded.engine.params["words_inserted_unattributed"] == 0


@pytest.mark.parametrize(
    ("said", "lost"),
    [
        (
            [("so", 5.0, 5.4), ("then", 8.0, 8.4)],
            [("hello", 0.0, 0.3), ("this", 0.4, 0.7), ("bob", 0.8, 1.1)],
        ),
        (
            [("so", 0.0, 1.0), ("then", 6.0, 6.4)],
            [("hello", 1.0, 1.3), ("this", 1.4, 1.7), ("bob", 1.8, 2.1)],
        ),
    ],
    ids=["at 0 s", "at the end of the word before"],
)
def test_a_word_starting_exactly_where_a_backbone_hole_starts_is_inside_it(
    said: list[Spec], lost: list[Spec]
) -> None:
    source = Source(kind="audio", ref="meeting.mp3")
    words = _words(*said, speaker=1)
    backbone = Transcript(source=source, engine=Engine(name="xai-stt"), text="", words=words)
    heard = sorted([*_words(*said), *_words(*lost)], key=lambda word: word.start)
    hypothesis = Transcript(source=source, engine=Engine(name="parakeet"), text="", words=heard)

    voted = vote_transcripts(backbone, hypothesis, hypothesis)

    assert [word for word in voted.words if word.speaker is not None] == words
    assert [word.text for word in voted.words if word.speaker is None] == ["hello", "this", "bob"]
    assert voted.engine.params["words_inserted_unattributed"] == 3


def test_hypothesis_starts_that_decrease_are_refused() -> None:
    backbone = _words(("a", 0.0, 0.4))
    unsorted = _words(("a", 1.0, 1.4), ("b", 0.5, 0.9))

    assert first_decrease(unsorted) == 1
    assert first_decrease(backbone) is None
    with pytest.raises(ValueError, match="secondary"):
        vote_words(backbone, backbone, unsorted)


def test_norm_tokens_folds_case_quotes_dashes_and_punctuation() -> None:
    assert norm_tokens("Don\N{RIGHT SINGLE QUOTATION MARK}t-stop/now, 'OK'!") == [
        "don't",
        "stop",
        "now",
        "ok",
    ]
    assert norm_tokens("...") == []


def test_vote_transcripts_records_the_three_engines_and_keeps_the_backbones_metadata() -> None:
    source = Source(kind="audio", ref="meeting.mp3")
    backbone = Transcript(
        source=source,
        engine=Engine(name="xai-stt", model="m", params={"diarize": True}),
        language="en",
        duration=9.5,
        text="the cat",
        words=_words(("the", 0.0, 0.4), ("cat", 0.5, 0.9)),
    )
    hypothesis = _words(("the", 0.0, 0.4), ("hat", 0.5, 0.9))
    primary = Transcript(source=source, engine=Engine(name="parakeet"), text="", words=hypothesis)
    secondary = Transcript(source=source, engine=Engine(name="gemini"), text="", words=hypothesis)

    voted = vote_transcripts(backbone, primary, secondary)

    assert voted == Transcript(
        source=source,
        engine=Engine(
            name="rover",
            params={
                "variant": "B",
                "tolerance_s": 0.5,
                "band_s": 6.0,
                "backbone": "xai-stt",
                "primary": "parakeet",
                "secondary": "gemini",
                "words_inserted": 0,
                "words_inserted_unattributed": 0,
                "words_substituted": 1,
            },
        ),
        language="en",
        duration=9.5,
        text="the hat",
        words=_words(("the", 0.0, 0.4), ("hat", 0.5, 0.9)),
    )


# Backbone words end on the half second and no hypothesis word does, so a
# voted word ending there is a backbone slot and any other was inserted.
_SAID = st.lists(st.tuples(st.integers(0, 8), st.sampled_from(VOCAB)), max_size=10).map(sorted)


def _timed(said: list[tuple[int, str]], *, offset: float, length: float) -> list[Word]:
    return [
        Word(text=text, start=slot + offset, end=slot + offset + length, speaker=index)
        for index, (slot, text) in enumerate(said)
    ]


@st.composite
def _ballots(draw: st.DrawFn) -> tuple[list[Word], list[Word], list[Word]]:
    backbone = draw(_SAID.filter(bool))
    primary = draw(_SAID)
    kept = draw(st.lists(st.booleans(), min_size=len(primary), max_size=len(primary)))
    # Part copied from the primary, so the two hypotheses often agree.
    copied = [said for said, keep in zip(primary, kept, strict=True) if keep]
    secondary = sorted([*copied, *draw(_SAID)])
    return (
        _timed(backbone, offset=0.0, length=0.5),
        _timed(primary, offset=0.25, length=0.5),
        _timed(secondary, offset=0.25, length=0.0),
    )


def _split(voted: list[Word]) -> tuple[list[Word], list[Word]]:
    """Return the voted words holding backbone slots, then the inserted ones."""
    return [word for word in voted if word.end % 1 == 0.5], [
        word for word in voted if word.end % 1 != 0.5
    ]


def _introduced(before: Word, after: Word) -> list[str]:
    old, new = norm_tokens(before.text), norm_tokens(after.text)
    assert len(new) == len(old)
    return [token for token, was in zip(new, old, strict=True) if token != was]


def _holds(words: list[Word], token: str, slot: Word) -> bool:
    return any(
        token in norm_tokens(word.text) and max(word.start - slot.end, slot.start - word.end) <= 0.5
        for word in words
    )


@given(_ballots())
def test_identical_inputs_give_the_backbone_back(ballots: tuple[list[Word], ...]) -> None:
    backbone = ballots[0]
    assert vote_words(backbone, backbone, backbone) == backbone


@given(_ballots())
def test_every_backbone_slot_survives_in_order(ballots: tuple[list[Word], ...]) -> None:
    slots, _ = _split(vote_words(*ballots))
    assert [(word.start, word.end, word.speaker) for word in slots] == [
        (word.start, word.end, word.speaker) for word in ballots[0]
    ]


@given(_ballots())
def test_every_changed_or_inserted_token_is_one_both_hypotheses_hold(
    ballots: tuple[list[Word], ...],
) -> None:
    backbone, primary, secondary = ballots
    slots, inserted = _split(vote_words(*ballots))
    for before, after in zip(backbone, slots, strict=True):
        for token in _introduced(before, after):
            assert _holds(primary, token, before)
            assert _holds(secondary, token, before)
    agreed = {token for word in primary for token in norm_tokens(word.text)}
    agreed &= {token for word in secondary for token in norm_tokens(word.text)}
    assert {token for word in inserted for token in norm_tokens(word.text)} <= agreed


@given(_ballots())
def test_no_filler_is_ever_introduced(ballots: tuple[list[Word], ...]) -> None:
    slots, inserted = _split(vote_words(*ballots))
    for before, after in zip(ballots[0], slots, strict=True):
        assert not FILLERS.intersection(_introduced(before, after))
    assert not FILLERS.intersection(token for word in inserted for token in norm_tokens(word.text))


@given(_ballots())
def test_the_vote_is_deterministic(ballots: tuple[list[Word], ...]) -> None:
    copies = [[word.model_copy() for word in words] for words in ballots]
    assert vote_words(*ballots) == vote_words(*copies)


@given(_ballots())
def test_the_recorded_counts_are_the_words_inserted_and_changed(
    ballots: tuple[list[Word], ...],
) -> None:
    source = Source(kind="audio", ref="meeting.mp3")
    backbone, primary, secondary = (
        Transcript(source=source, engine=Engine(name=name), text="", words=words)
        for name, words in zip(("xai-stt", "parakeet", "gemini"), ballots, strict=True)
    )

    voted = vote_transcripts(backbone, primary, secondary)

    slots, inserted = _split(voted.words)
    changed = sum(
        before.text != after.text for before, after in zip(ballots[0], slots, strict=True)
    )
    assert voted.engine.params["words_inserted"] == len(inserted)
    assert voted.engine.params["words_inserted_unattributed"] == sum(
        word.speaker is None for word in inserted
    )
    assert voted.engine.params["words_substituted"] == changed


@given(_ballots())
def test_an_inserted_word_has_no_speaker_exactly_when_its_run_fills_a_backbone_hole(
    ballots: tuple[list[Word], ...],
) -> None:
    backbone = ballots[0]
    holes = find_holes(backbone, max(word.end for word in backbone), MIN_DROP_SECONDS)
    # With no token in the backbone there is no neighbor to take a speaker from.
    slotted = any(norm_tokens(word.text) for word in backbone)

    def hole_of(word: Word) -> int | None:
        if word.end % 1 == 0.5:  # a backbone slot, which ends a run
            return None
        inside = (i for i, hole in enumerate(holes) if hole.start <= word.start < hole.end)
        return next(inside, None)

    for hole, members in groupby(vote_words(*ballots), key=hole_of):
        run = list(members)
        spoken = sum(not is_filler(word_key(word.text)) for word in run)
        dropped = hole is not None and spoken >= MIN_DROP_WORDS
        for word in _split(run)[1]:
            assert (word.speaker is None) == (dropped or not slotted)
