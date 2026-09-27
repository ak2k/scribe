"""Naming unattributed words: only where the diarizer's cluster and the voice match agree."""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scribe.attribution import (
    CENTROID_GUARD_S,
    CENTROID_SEGMENTS,
    Plan,
    Segment,
    Speech,
    Timeline,
    Vector,
    admits,
    can_name,
    centroid_segments,
    centroids,
    cluster_speakers,
    diarized_speakers,
    name_unattributed,
    nearest,
    plan,
    shuffled,
    unattributed_runs,
    unit,
    windows,
)
from scribe.schema import Word

VOICE: dict[int, Vector] = {7: (1.0, 0.0, 0.0), 4: (0.0, 1.0, 0.0)}


def _said(*runs: tuple[int | None, int]) -> tuple[list[Word], list[int | None]]:
    """(speaker, word count) runs: a word a second, each 0.9 s long."""
    speakers = [speaker for speaker, count in runs for _ in range(count)]
    words = [Word(text=f"w{index}", start=index, end=index + 0.9) for index in range(len(speakers))]
    return words, speakers


def _speech(words: list[Word], labels: list[str | None]) -> list[Speech]:
    """One stretch of speech per word, under its label; no speech where it is None."""
    return [
        Speech(word.start, word.end, label)
        for word, label in zip(words, labels, strict=True)
        if label is not None
    ]


def _clusters(speakers: list[int | None], unattributed: str | None) -> list[str | None]:
    """Cluster c<id> for each attributed word, and `unattributed` for the others."""
    return [unattributed if speaker is None else f"c{speaker}" for speaker in speakers]


def _embeddings(
    planned: Plan, heard: list[Vector | None], voices: dict[int, Vector | None] | None = None
) -> list[Vector | None]:
    """`heard` for the windows, run by run; then each requested segment its speaker's voice."""
    assert len(heard) == sum(len(run.windows) for run in planned.runs)
    own: dict[int, Vector | None] = {**VOICE, **(voices or {})}
    return [*heard, *(own[planned.segments[index].speaker] for index in planned.requested)]


# Three-word turns (2.9 s, one centroid segment each) around a two-word run at 30-31 s.
TALK = [*[(7, 3), (4, 3)] * 5, (None, 2), *[(7, 3), (4, 3)] * 5]


def _name(
    unattributed: str | None,
    heard: Vector | None,
    voices: dict[int, Vector | None] | None = None,
) -> list[int | None]:
    words, speakers = _said(*TALK)
    planned = plan(words, speakers)
    embeddings = _embeddings(planned, [heard], voices)
    speech = _speech(words, _clusters(speakers, unattributed))
    named = name_unattributed(planned, words, speakers, speech, embeddings)
    return list(named.speakers[30:32])


def test_a_word_both_readings_give_one_speaker_is_named_that_speaker() -> None:
    assert _name("c7", VOICE[7]) == [7, 7]


@pytest.mark.parametrize("cluster", [None, "c9"], ids=["no-speech", "unmapped-cluster"])
def test_a_word_with_no_mapped_cluster_stays_unattributed(cluster: str | None) -> None:
    assert _name(cluster, VOICE[7]) == [None, None]


def test_a_word_heard_in_a_window_with_no_embedding_stays_unattributed() -> None:
    assert _name("c7", None) == [None, None]


def test_a_word_with_no_centroid_to_match_stays_unattributed() -> None:
    assert _name("c7", VOICE[7], voices={7: None, 4: None}) == [None, None]


def test_a_word_the_two_readings_disagree_on_stays_unattributed() -> None:
    assert _name("c7", VOICE[4]) == [None, None]


def test_a_voice_as_near_two_centroids_is_heard_as_the_lower_speaker() -> None:
    # Midway between 4's and 7's voices; the centroids are compared speakers ascending.
    assert _name("c4", unit((1.0, 1.0, 0.0))) == [4, 4]


def test_a_run_with_one_speakers_centroid_stays_unattributed() -> None:
    # 4 never says more than one 0.9 s word at a time, so no segment of 4's is long enough.
    words, speakers = _said(*[(7, 3), (4, 1)] * 8, (None, 2), *[(7, 3), (4, 1)] * 8)
    planned = plan(words, speakers)
    assert {segment.speaker for segment in planned.segments} == {7}
    speech = _speech(words, _clusters(speakers, "c7"))

    # The run sounds like 4, nothing like 7's centroid, the only one there is.
    named = name_unattributed(planned, words, speakers, speech, _embeddings(planned, [VOICE[4]]))

    assert named.speakers[32:34] == (None, None)
    assert nearest(VOICE[7], {7: VOICE[7]}) is None


def test_the_counts_follow_the_named_words() -> None:
    words, speakers = _said(*TALK)
    planned = plan(words, speakers)
    speech = _speech(words, _clusters(speakers, "c4"))

    named = name_unattributed(planned, words, speakers, speech, _embeddings(planned, [VOICE[4]]))

    assert (named.unattributed, named.named) == (2, 2)
    assert [(run.start, run.end, run.words, run.named) for run in named.runs] == [
        (30.0, 31.9, 2, ((4, 2),))
    ]


def test_a_word_with_a_speaker_keeps_it_whatever_the_audio_says() -> None:
    words, speakers = _said(*TALK)
    planned = plan(words, speakers)
    # Every word, attributed or not, sits in cluster c7 and is heard as 7.
    speech = _speech(words, ["c7"] * len(words))

    named = name_unattributed(planned, words, speakers, speech, _embeddings(planned, [VOICE[7]]))

    assert named.speakers == (*speakers[:30], 7, 7, *speakers[32:])


def test_a_cluster_maps_to_the_speaker_most_of_its_attributed_words_have() -> None:
    labels: list[str | None] = ["a", "a", "a", "b", "b", "a", None, "c"]
    speakers: list[int | None] = [7, 4, 7, 4, 7, None, 4, None]

    # b is a tie, won by the speaker it met first; c has no attributed word.
    assert cluster_speakers(labels, speakers) == {"a": 7, "b": 4}


def test_every_word_gets_its_clusters_speaker_attributed_or_not() -> None:
    words, speakers = _said((7, 2), (4, 1), (None, 1), (4, 1), (7, 1))
    labels: list[str | None] = ["x", "x", "y", "x", "y", None]

    diarized = diarized_speakers(words, speakers, _speech(words, labels))

    assert diarized == [7, 7, 4, 7, 4, None]


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0.5, 1.5, "a"),  # 0.5 s in a, 0.25 s in b
        (1.125, 1.5, "b"),
        (0.75, 1.5, "a"),  # a tie goes to the speech that starts first
        (1.0, 1.0, "a"),  # an instant takes the speech holding it, ends included
        (3.0, 3.0, "c"),  # a shared edge goes to the speech that starts there
        (1.0625, 1.125, None),  # between a and b
        (1.125, 1.125, None),
        (9.0, 9.5, None),
        (2.5, 2.0, None),  # a word ending before it starts overlaps nothing
    ],
)
def test_a_word_takes_the_speech_it_overlaps_most(
    start: float, end: float, expected: str | None
) -> None:
    timeline = Timeline([Speech(3.0, 4.0, "c"), Speech(0.0, 1.0, "a"), Speech(1.25, 3.0, "b")])

    assert timeline.label(start, end) == expected


def test_a_long_speech_is_found_from_a_word_far_past_its_start() -> None:
    timeline = Timeline([Speech(0.0, 100.0, "long"), Speech(10.0, 11.0, "short")])

    assert timeline.label(50.0, 51.0) == "long"
    assert timeline.label(10.0, 10.5) == "long"


def test_windows_of_a_2_s_run_and_a_7_s_run() -> None:
    assert windows(10.0, 12.0) == ((10.0, 12.0),)
    assert windows(10.0, 17.0) == ((10.0, 13.0), (11.5, 14.5), (13.0, 16.0), (14.0, 17.0))


def test_a_run_ending_near_its_last_window_gets_no_window_of_its_own() -> None:
    assert windows(10.0, 16.2) == ((10.0, 13.0), (11.5, 14.5), (13.0, 16.0))


def test_a_word_is_heard_in_the_window_whose_center_is_nearest() -> None:
    # Seven words from 30 s to 36.9 s: windows centered at 31.5, 33, 34.5 and 35.4.
    words, speakers = _said(*[(7, 3), (4, 3)] * 5, (None, 7), *[(7, 3), (4, 3)] * 5)
    planned = plan(words, speakers)
    assert planned.runs[0].windows == ((30.0, 33.0), (31.5, 34.5), (33.0, 36.0), (33.9, 36.9))
    speech = _speech(words, _clusters(speakers, "c7"))

    heard: list[Vector | None] = [VOICE[7], VOICE[4], VOICE[7], VOICE[4]]
    named = name_unattributed(planned, words, speakers, speech, _embeddings(planned, heard))

    # Midpoints 30.45 ... 36.45 are nearest windows 0, 0, 1, 1, 2, 3, 3: 35.45 lies
    # inside window 2 too, but nearer window 3's center.
    assert list(named.speakers[30:37]) == [7, 7, None, None, 7, None, None]


def test_a_word_midway_between_two_window_centers_is_heard_in_the_first() -> None:
    words, speakers = _said(*[(7, 3), (4, 3)] * 5, (None, 6), *[(7, 3), (4, 3)] * 5)
    words[32] = Word(text="w32", start=32.0, end=32.5)
    planned = plan(words, speakers)
    assert planned.runs[0].windows == ((30.0, 33.0), (31.5, 34.5), (32.9, 35.9))
    speech = _speech(words, _clusters(speakers, "c7"))

    heard: list[Vector | None] = [VOICE[7], VOICE[4], VOICE[4]]
    named = name_unattributed(planned, words, speakers, speech, _embeddings(planned, heard))

    # w32's midpoint, 32.25, is 0.75 s from the centers of windows 0 and 1.
    assert list(named.speakers[30:36]) == [7, 7, 7, None, None, None]


# Two runs, one at 30-31.9 s and one at 62-63.9 s, amid three-word turns.
TWO_RUNS = [*TALK, (None, 2), *[(7, 3), (4, 3)] * 5]


def test_each_run_is_heard_in_its_own_windows() -> None:
    words, speakers = _said(*TWO_RUNS)
    planned = plan(words, speakers)
    labels = _clusters(speakers, "c7")
    labels[62:64] = ["c4", "c4"]

    embeddings = _embeddings(planned, [VOICE[7], VOICE[4]])

    named = name_unattributed(planned, words, speakers, _speech(words, labels), embeddings)

    assert named.speakers[30:32] == (7, 7)
    assert named.speakers[62:64] == (4, 4)
    assert [(run.start, run.words, run.named) for run in named.runs] == [
        (30.0, 2, ((7, 2),)),
        (62.0, 2, ((4, 2),)),
    ]


def test_each_runs_centroids_leave_out_only_the_speech_near_that_run() -> None:
    # Runs at 18-19.9 s and 38-39.9 s. Near the second, 4's segments sound nearer
    # its voice than 7's centroid does; in its centroid they would turn 7's win into 4's.
    turns = [(7, 3), (4, 3)] * 3
    words, speakers = _said(*turns, (None, 2), *turns, (None, 2), *turns)
    planned = plan(words, speakers)
    second = planned.runs[1]
    heard = unit((1.0, 0.9, 0.0))

    def voice(segment: Segment) -> Vector | None:
        near = not admits(segment, second.start, second.end)
        return heard if segment.speaker == 4 and near else VOICE[segment.speaker]

    segments = [voice(planned.segments[index]) for index in planned.requested]
    speech = _speech(words, _clusters(speakers, "c7"))

    named = name_unattributed(planned, words, speakers, speech, [VOICE[7], heard, *segments])

    assert named.speakers[18:20] == (7, 7)
    assert named.speakers[38:40] == (7, 7)


def test_a_segment_near_the_run_is_left_out_of_its_centroids() -> None:
    heard = unit((1.0, 0.9, 0.0))
    far = Segment(4, 0.0, 3.0)
    for_seven = Segment(7, 100.0, 103.0)
    vectors = [VOICE[4], VOICE[7], heard]

    def top(flipping: Segment) -> int | None:
        segments = [far, for_seven, flipping]
        return nearest(heard, centroids(segments, {4: (0, 2), 7: (1,)}, vectors, 50.0, 53.0))

    # At the guard's edge the segment counts, and turns 7's win into 4's.
    assert top(Segment(4, 37.0, 50.0 - CENTROID_GUARD_S)) == 4
    assert top(Segment(4, 53.0 + CENTROID_GUARD_S, 66.0)) == 4
    assert top(Segment(4, 37.0, 40.5)) == 7
    assert top(Segment(4, 62.9, 66.0)) == 7


def test_a_centroid_takes_the_first_30_segments_with_embeddings_and_no_more() -> None:
    count = CENTROID_SEGMENTS + 2
    segments = [Segment(7, 100.0 + 5 * index, 103.0 + 5 * index) for index in range(count)]
    # Segment k's vector points along axis k; the third in order has none.
    vectors: list[Vector | None] = [
        tuple(float(axis == index) for axis in range(count)) for index in range(count)
    ]
    vectors[2] = None
    order = tuple(reversed(range(count)))

    mean = centroids(segments, {7: order}, vectors, 0.0, 1.0)[7]

    used = [axis for axis in range(count) if mean[axis]]
    # Taken 31, 30, ..., 3, then 2 (none), then 1 as the 30th: 0 would be the 31st.
    assert used == [1, *range(3, count)]
    assert all(math.isclose(mean[axis], 1 / math.sqrt(CENTROID_SEGMENTS)) for axis in used)


def test_the_plan_asks_for_a_segment_past_a_speakers_30th_in_case_one_has_no_embedding() -> None:
    words, speakers = _said(*[(7, 3), (4, 3)] * 18, (None, 2), *[(7, 3), (4, 3)] * 18)
    planned = plan(words, speakers)
    run = planned.runs[0]
    fours = [
        index for index in planned.order[4] if admits(planned.segments[index], run.start, run.end)
    ]
    assert len(fours) > CENTROID_SEGMENTS
    # Of 4's segments only the 31st in order embeds, so it alone makes 4's centroid.
    only = fours[CENTROID_SEGMENTS]

    def voice(index: int) -> Vector | None:
        speaker = planned.segments[index].speaker
        return VOICE[7] if speaker == 7 else VOICE[4] if index == only else None

    segments = [voice(index) for index in planned.requested]
    speech = _speech(words, _clusters(speakers, "c4"))

    named = name_unattributed(planned, words, speakers, speech, [VOICE[4], *segments])

    assert (run.first, run.stop) == (108, 110)
    assert named.speakers[108:110] == (4, 4)


def test_a_segment_counts_toward_its_centroid_by_direction_not_length() -> None:
    words, speakers = _said(*TALK)
    planned = plan(words, speakers)
    run = planned.runs[0]
    heard = unit((1.0, 0.9, 0.0))

    # 4's segments before the run embed long along 4's axis, those after it short and
    # between the axes. Weighted by length, 4's centroid would lose the run to 7's.
    def voice(segment: Segment) -> Vector:
        if segment.speaker == 7:
            return VOICE[7]
        return (0.0, 20.0, 0.0) if segment.start < run.start else (0.8, 0.6, 0.0)

    segments = [voice(planned.segments[index]) for index in planned.requested]
    speech = _speech(words, _clusters(speakers, "c4"))

    named = name_unattributed(planned, words, speakers, speech, [heard, *segments])

    assert named.speakers[30:32] == (4, 4)


def test_segments_break_at_pauses_speaker_changes_and_unattributed_words() -> None:
    spoken = [
        (7, 0.0, 1.0),
        (7, 1.4, 2.5),  # a 0.4 s pause: same segment, 0-2.5
        (7, 3.0, 4.0),  # a 0.5 s pause ends it; 3-4 is too short
        (4, 4.1, 6.2),  # a speaker change: 4.1-6.2
        (None, 6.3, 6.4),
        (4, 6.5, 8.6),  # an unattributed word between: 6.5-8.6
        (4, 8.7, 20.0),  # 6.5-20 is cut to its first 10 s
    ]
    words = [Word(text="w", start=start, end=end) for _, start, end in spoken]
    speakers = [speaker for speaker, _, _ in spoken]

    assert centroid_segments(words, speakers) == [
        Segment(7, 0.0, 2.5),
        Segment(4, 4.1, 6.2),
        Segment(4, 6.5, 16.5),
    ]


def test_a_segment_ends_at_its_latest_word_end() -> None:
    words = [Word(text="w", start=0.0, end=2.5), Word(text="w", start=1.0, end=1.5)]

    assert centroid_segments(words, [7, 7]) == [Segment(7, 0.0, 2.5)]


def test_the_segments_are_drawn_once_per_call_speakers_ascending() -> None:
    segments = [Segment(speaker, 0.0, 3.0) for speaker in (9, 2, 9, 2, 9, 2, 9)]
    draws = random.Random("45")  # noqa: S311  # the draw the rule makes, not a secret
    twos, nines = [1, 3, 5], [0, 2, 4, 6]
    draws.shuffle(twos)
    draws.shuffle(nines)

    assert shuffled(segments) == {2: tuple(twos), 9: tuple(nines)}
    assert list(shuffled(segments)) == [2, 9]


def test_the_plan_lists_every_window_then_the_segments_some_run_may_use() -> None:
    words, speakers = _said((7, 3), (4, 3), (None, 1), *[(7, 3), (4, 3)] * 5, (None, 1))

    planned = plan(words, speakers)

    assert [(run.first, run.stop) for run in planned.runs] == [(6, 7), (37, 38)]
    assert planned.intervals[:2] == ((6.0, 6.9), (37.0, 37.9))
    # Each segment is 10 s or more from one run or the other, so every one is there.
    assert planned.requested == tuple(range(len(planned.segments)))
    assert planned.intervals[2:] == tuple((seg.start, seg.end) for seg in planned.segments)


def test_a_segment_no_runs_guard_admits_is_not_requested() -> None:
    words, speakers = _said((7, 3), (4, 3), (None, 1), (7, 3), (4, 3), (7, 30), (4, 30))

    planned = plan(words, speakers)

    kept = [planned.segments[index] for index in planned.requested]
    assert [segment.start for segment in planned.segments] == [0.0, 3.0, 7.0, 10.0, 13.0, 43.0]
    assert [segment.start for segment in kept] == [43.0]


def test_the_same_inputs_name_the_same_words() -> None:
    words, speakers = _said(*TALK)
    speech = _speech(words, _clusters(speakers, "c4"))

    first, second = plan(words, speakers), plan(words, speakers)
    assert first == second
    embeddings = _embeddings(first, [VOICE[4]])
    assert name_unattributed(first, words, speakers, speech, embeddings) == name_unattributed(
        second, words, speakers, speech, embeddings
    )


def test_embeddings_that_do_not_match_the_plan_are_refused() -> None:
    words, speakers = _said(*TALK)
    planned = plan(words, speakers)

    with pytest.raises(ValueError, match="planned intervals"):
        name_unattributed(planned, words, speakers, [], [])


@pytest.mark.parametrize(
    ("speakers", "expected"),
    [
        ([7, None, 4], True),
        ([7, 4], False),
        ([7, None, 7], False),
        ([None, None], False),
        ([], False),
    ],
)
def test_naming_needs_an_unattributed_word_and_two_speakers(
    speakers: list[int | None], expected: bool
) -> None:
    assert can_name(speakers) is expected


def test_runs_are_maximal_in_list_order() -> None:
    assert unattributed_runs([None, 7, None, None, 4, None]) == [(0, 1), (2, 4), (5, 6)]


def test_a_vector_without_direction_has_no_unit() -> None:
    assert unit((0.0, 0.0)) is None
    assert unit(None) is None
    assert unit((3.0, 4.0)) == (0.6, 0.8)


_speaker = st.one_of(st.none(), st.sampled_from([1, 2, 3]))
_cluster = st.one_of(st.none(), st.sampled_from(["a", "b", "c"]))
_voice = st.one_of(st.none(), st.sampled_from([(1.0, 0.0), (0.0, 1.0), (0.6, 0.8)]))


@given(st.lists(st.tuples(_speaker, _cluster, st.floats(0.1, 4.0)), max_size=40), st.data())
def test_only_unattributed_words_change_and_only_to_their_clusters_speaker(
    said: list[tuple[int | None, str | None, float]], data: st.DataObject
) -> None:
    words: list[Word] = []
    at = 0.0
    for _, _, length in said:
        words.append(Word(text="w", start=at, end=at + length))
        at += length + 0.2
    speakers = [speaker for speaker, _, _ in said]
    speech = _speech(words, [cluster for _, cluster, _ in said])
    planned = plan(words, speakers)
    count = len(planned.intervals)
    embeddings = data.draw(st.lists(_voice, min_size=count, max_size=count))

    named = name_unattributed(planned, words, speakers, speech, embeddings)

    diarized = diarized_speakers(words, speakers, speech)
    for before, after, cluster_speaker in zip(speakers, named.speakers, diarized, strict=True):
        assert (after == before) if before is not None else (after in (None, cluster_speaker))
    assert named.unattributed == speakers.count(None)
    assert named.named == sum(1 for b, a in zip(speakers, named.speakers, strict=True) if b != a)
