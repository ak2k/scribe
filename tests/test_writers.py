from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from scribe.schema import Engine, Source, Transcript, Turn, Word
from scribe.turns import build_turns
from scribe.writers import build_cues, to_markdown, to_srt, to_vtt

SOURCE = Source(kind="audio", ref="fixture.mp3")
ENGINE = Engine(name="xai-stt", model="grok-voice-transcribe-2.0")

SIX_WORDS = [
    Word(text="Hello", start=0.0, end=0.5, speaker=7),
    Word(text="there", start=0.6, end=1.1, speaker=7),
    Word(text="friend.", start=1.2, end=1.7, speaker=7),
    Word(text="Good", start=1.8, end=2.3, speaker=4),
    Word(text="to", start=2.4, end=2.9, speaker=4),
    Word(text="see.", start=3.0, end=3.5, speaker=4),
]


def _transcript(words: list[Word]) -> Transcript:
    built = build_turns(words)
    return Transcript(
        source=SOURCE,
        engine=ENGINE,
        text=" ".join(word.text for word in words),
        words=words,
        turns=built,
    )


def test_markdown_golden() -> None:
    assert to_markdown(_transcript(SIX_WORDS)) == (
        "**Speaker 1** [00:00:00]\nHello there friend.\n\n"
        "**Speaker 2** [00:00:01]\nGood to see.\n\n"
    )


def test_srt_golden() -> None:
    assert to_srt(_transcript(SIX_WORDS)) == (
        "1\n00:00:00,000 --> 00:00:01,700\nSpeaker 1: Hello there friend.\n\n"
        "2\n00:00:01,800 --> 00:00:03,500\nSpeaker 2: Good to see.\n\n"
    )


def test_vtt_starts_with_the_header_and_a_blank_line() -> None:
    vtt = to_vtt(_transcript(SIX_WORDS))
    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.700\nSpeaker 1: Hello there friend." in vtt


def test_long_turn_splits_into_cues_within_the_duration_limit() -> None:
    words = [
        Word(text=f"word{index:02d}", start=float(index), end=index + 0.9, speaker=None)
        for index in range(20)
    ]
    transcript = _transcript(words)
    assert len(transcript.turns) == 1

    cues = build_cues(transcript)

    assert len(cues) > 1
    assert all(cue.end - cue.start <= 7.0 for cue in cues)
    assert all(cue.speaker == "Speaker 1" for cue in cues)
    assert " ".join(cue.text for cue in cues) == transcript.turns[0].text


def test_cues_split_on_the_character_limit_too() -> None:
    # Ten words inside one second: only the 84-character limit can split these.
    words = [
        Word(text="antidisestablishment", start=index / 10, end=(index + 1) / 10, speaker=None)
        for index in range(10)
    ]
    cues = build_cues(_transcript(words))

    assert len(cues) > 1
    assert all(len(cue.text) <= 84 for cue in cues)


def test_a_turns_only_transcript_keeps_each_turn_as_one_cue() -> None:
    turns = [
        Turn(speaker="Speaker 1", start=0.0, end=30.0, text="a turn with no word timings"),
        Turn(speaker="Speaker 2", start=30.0, end=61.0, text="and another"),
    ]
    transcript = Transcript(source=SOURCE, engine=ENGINE, text="", turns=turns)

    assert build_cues(transcript) == turns
    assert to_srt(transcript).startswith("1\n00:00:00,000 --> 00:00:30,000\nSpeaker 1: a turn")
    assert "00:00:30.000 --> 00:01:01.000" in to_vtt(transcript)


def test_a_word_at_a_shared_turn_boundary_lands_in_one_cue_only() -> None:
    # A zero-duration word on the boundary lies in both turns' inclusive spans;
    # it is in one cue, under the turn that holds it.
    words = [
        Word(text="one", start=0.0, end=2.0, speaker=7),
        Word(text="two", start=2.1, end=4.0, speaker=7),
        Word(text="boundary", start=4.0, end=4.0, speaker=4),
        Word(text="four", start=4.1, end=6.0, speaker=4),
        Word(text="five", start=6.1, end=8.0, speaker=4),
    ]
    turns = [
        Turn(speaker="Speaker 1", start=0.0, end=4.0, text="one two"),
        Turn(speaker="Speaker 2", start=4.0, end=8.0, text="boundary four five"),
    ]
    transcript = Transcript(source=SOURCE, engine=ENGINE, text="", words=words, turns=turns)

    cues = build_cues(transcript)

    cue_words = [word for cue in cues for word in cue.text.split()]
    assert cue_words.count("boundary") == 1
    assert len(cue_words) == len(words)
    assert [cue.speaker for cue in cues if "boundary" in cue.text] == ["Speaker 2"]


def test_a_word_that_outlasts_its_turn_stays_in_that_turns_cue() -> None:
    # The turn holds the word whatever its span says, and the cue is timed by it.
    words = [Word(text="straddles", start=1.5, end=3.0, speaker=None)]
    turns = [Turn(speaker="Speaker 1", start=0.0, end=2.0, text="straddles")]
    transcript = Transcript(source=SOURCE, engine=ENGINE, text="", words=words, turns=turns)

    assert build_cues(transcript) == [
        Turn(speaker="Speaker 1", start=1.5, end=3.0, text="straddles")
    ]


def test_no_turns_renders_empty() -> None:
    transcript = Transcript(source=SOURCE, engine=ENGINE, text="")
    assert to_markdown(transcript) == ""
    assert to_srt(transcript) == ""
    assert to_vtt(transcript) == "WEBVTT\n\n"


def test_an_interjection_inside_another_turn_is_one_cue_under_its_own_speaker() -> None:
    # Speaker 4 speaks entirely inside speaker 7's span: both turns' spans hold
    # "Wait", and it used to land in the first turn's cue and again as the
    # second turn's whole-turn cue.
    words = [
        Word(text="Proceed", start=0.0, end=3.0, speaker=7),
        Word(text="Wait", start=0.5, end=2.5, speaker=4),
    ]
    transcript = _transcript(words)

    cues = build_cues(transcript)

    assert [(cue.speaker, cue.text, cue.start, cue.end) for cue in cues] == [
        ("Speaker 1", "Proceed", 0.0, 3.0),
        ("Speaker 2", "Wait", 0.5, 2.5),
    ]


def test_a_cue_ends_with_its_last_ending_word() -> None:
    # Overlapping timings leave words in start order but not in end order.
    words = [
        Word(text="long", start=0.0, end=3.0, speaker=None),
        Word(text="short", start=0.5, end=2.5, speaker=None),
    ]
    turns = [Turn(speaker="Speaker 1", start=0.0, end=3.0, text="long short")]
    transcript = Transcript(source=SOURCE, engine=ENGINE, text="", words=words, turns=turns)

    assert build_cues(transcript) == turns


def test_a_word_snapped_onto_a_turn_it_outlasts_stays_in_the_subtitles() -> None:
    # The snap moves "budget." to the speaker before it; "Review" still ends
    # after it, and the turn has to reach that end for its cue to hold "Review".
    words = [
        Word(text="Review", start=0.0, end=3.0, speaker=0),
        Word(text="budget.", start=0.5, end=1.0, speaker=1),
        Word(text="Then", start=1.1, end=1.5, speaker=1),
        Word(text="the", start=1.6, end=1.9, speaker=1),
        Word(text="schedule", start=2.0, end=2.6, speaker=1),
        Word(text="next.", start=2.7, end=3.2, speaker=1),
    ]

    cues = build_cues(_transcript(words))

    assert [(cue.speaker, cue.text) for cue in cues] == [
        ("Speaker 1", "Review budget."),
        ("Speaker 2", "Then the schedule next."),
    ]


def test_turns_not_cut_from_the_words_are_each_one_cue() -> None:
    # Both turns claim the same two words, so no word can be given to one of
    # them; the subtitles show the turns as markdown does.
    words = [
        Word(text="a", start=1.2, end=1.8, speaker=7),
        Word(text="b", start=1.3, end=1.9, speaker=7),
    ]
    turns = [
        Turn(speaker="Speaker 1", start=0.0, end=5.0, text="a b"),
        Turn(speaker="Speaker 2", start=1.0, end=2.0, text="a b"),
    ]
    transcript = Transcript(source=SOURCE, engine=ENGINE, text="", words=words, turns=turns)

    cues = build_cues(transcript)

    assert cues == turns


def test_a_turn_spanning_another_speakers_word_leaves_it_to_that_speaker() -> None:
    # "Well" runs to 4.0 s, so Speaker 1's turn spans "Yes"; the word is still
    # Speaker 2's.
    words = [
        Word(text="Well", start=0.0, end=4.0, speaker=0),
        Word(text="so.", start=0.2, end=0.6, speaker=0),
        Word(text="Yes", start=1.0, end=2.0, speaker=1),
        Word(text="indeed.", start=2.1, end=5.0, speaker=1),
    ]

    cues = build_cues(_transcript(words))

    assert [(cue.speaker, cue.text) for cue in cues] == [
        ("Speaker 1", "Well so."),
        ("Speaker 2", "Yes indeed."),
    ]


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
)
def test_cues_hold_each_word_once_under_its_own_turns_speaker(
    said: list[tuple[int | None, str, float, float]],
) -> None:
    # Timings overlap across speakers, so spans nest and cross.
    words: list[Word] = []
    clock = 0.0
    for index, (speaker, mark, gap, length) in enumerate(said):
        clock += gap
        words.append(Word(text=f"w{index}{mark}", start=clock, end=clock + length, speaker=speaker))
    transcript = _transcript(words)

    cues = build_cues(transcript)

    assert [(cue.speaker, word) for cue in cues for word in cue.text.split()] == [
        (turn.speaker, word) for turn in transcript.turns for word in turn.text.split()
    ]
