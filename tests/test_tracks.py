from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from scribe.schema import Engine, Source, Transcript, Word
from scribe.tracks import BLEED_WINDOW_S, Merged, TrackFile, merge_tracks
from scribe.vote import word_key

Said = tuple[str, float]


def _side(
    *said: Said, speaker: int | None = 0, params: dict[str, float | int | bool | str] | None = None
) -> Transcript:
    """A transcript of words 0.2 s long, all by `speaker`."""
    return Transcript(
        source=Source(kind="audio", ref="stem.wav"),
        engine=Engine(name="xai-stt", model="grok", params=params or {}),
        text=" ".join(text for text, _ in said),
        words=[
            Word(text=text, start=start, end=start + 0.2, speaker=speaker) for text, start in said
        ],
    )


def _merge(mic: Transcript, app: Transcript, *, offset: float = 0.0) -> Merged:
    return merge_tracks(
        TrackFile(mic, Path("mic.json"), "mic-sha"),
        TrackFile(app, Path("app.json"), "app-sha"),
        me="Alice",
        offset=offset,
    )


def _dropped(merged: Merged) -> list[tuple[str, float, int]]:
    return [(drop.word.text, drop.word.start, drop.run) for drop in merged.dropped]


# The far side's words; each test plants its own mic words against them.
APP = _side(
    ("Okay", 1.0), ("so", 1.3), ("forty?", 2.0), ("we", 4.0), ("can", 4.3), ("ship", 4.6),
    ("it", 4.9), ("Friday.", 5.2), ("yeah.", 8.0), ("Great.", 9.0),
    speaker=3,
)  # fmt: skip


def test_a_one_word_copy_is_dropped() -> None:
    merged = _merge(_side(("Hello", 0.2), ("Yeah.", 8.05), ("thanks", 8.6)), APP)

    assert _dropped(merged) == [("Yeah.", 8.05, 1)]
    assert merged.offset == 0.0


def test_a_repeat_back_is_kept() -> None:
    merged = _merge(_side(("Forty.", 2.6)), APP)

    assert merged.dropped == ()
    assert [word.text for word in merged.transcript.words if word.track == 0] == ["Forty."]


def test_two_mic_copies_of_one_app_word_lose_only_one() -> None:
    merged = _merge(_side(("Yeah.", 8.05), ("yeah", 8.2)), APP)

    assert _dropped(merged) == [("Yeah.", 8.05, 1)]


def test_the_window_includes_its_edge_and_nothing_past_it() -> None:
    # Dyadic times, so the 0.25 s edge is exact in binary.
    assert _dropped(_merge(_side(("yeah", 8.25)), APP)) == [("yeah", 8.25, 1)]
    assert _dropped(_merge(_side(("yeah", 7.75)), APP)) == [("yeah", 7.75, 1)]
    assert _dropped(_merge(_side(("yeah", 8.75)), APP, offset=0.5)) == [("yeah", 8.75, 1)]
    assert _merge(_side(("yeah", 8.3)), APP).dropped == ()


def test_each_mic_word_takes_the_earliest_free_app_word_in_its_window() -> None:
    app = _side(("yeah", 8.0), ("yeah", 8.25))
    # The closest app word for the first would leave the second none in reach.
    mic = _side(("yeah", 8.25), ("yeah", 8.5))

    assert _dropped(_merge(mic, app)) == [("yeah", 8.25, 2), ("yeah", 8.5, 2)]


def test_a_repeat_back_of_three_words_is_kept() -> None:
    app = _side(("Friday", 5.0), ("at", 5.25), ("three.", 5.5), speaker=3)
    mic = _side(("Friday", 5.9), ("at", 6.15), ("three?", 6.4))

    merged = _merge(mic, app)

    assert (merged.dropped, merged.offset) == ((), 0.0)


def test_a_shared_goodbye_sets_no_offset_for_the_rest_of_the_call() -> None:
    app = _side(("so", 30.0), ("yeah", 30.3), ("Thank", 90.0), ("you,", 90.25), ("bye.", 90.5))
    mic = _side(("yeah", 30.8), ("Thank", 90.4), ("you,", 90.65), ("bye.", 90.9))

    merged = _merge(mic, app)

    assert (merged.dropped, merged.offset) == ((), 0.0)
    assert merged.transcript.engine.params["merge_offset_s"] == 0.0


def test_a_given_offset_drops_only_the_planted_copies() -> None:
    copies = [(word.text, word.start + 0.3) for word in APP.words if word.start >= 4.0]
    mic = _side(
        ("Hello", 0.2), ("Forty.", 2.0 + 0.3 + 0.6), *copies, ("right", 7.0), ("thanks", 8.6)
    )

    merged = _merge(mic, APP, offset=0.3)

    assert merged.offset == 0.3
    assert merged.transcript.engine.params["merge_offset_s"] == 0.3
    assert sorted((text, start) for text, start, _ in _dropped(merged)) == sorted(copies)
    assert [run for *_, run in _dropped(merged)] == [5, 5, 5, 5, 5, 1, 1]


def test_disjoint_texts_drop_nothing() -> None:
    merged = _merge(_side(("Hello", 1.0), ("there", 4.0), ("friend", 8.0)), APP)

    assert (merged.dropped, merged.offset) == ((), 0.0)


def test_a_step_back_is_sorted_and_counted() -> None:
    mic = _side(("Hello", 3.0), ("there", 0.5), ("friend", 3.5))

    merged = _merge(mic, APP).transcript

    assert [w.text for w in merged.words if w.track == 0] == ["there", "Hello", "friend"]
    assert (merged.engine.params["merge_mic_moved"], merged.engine.params["merge_app_moved"]) == (
        2,
        0,
    )


def test_an_app_word_comes_first_on_an_equal_start() -> None:
    merged = _merge(_side(("Hello", 1.0)), APP).transcript

    assert [(w.text, w.track) for w in merged.words[:2]] == [("Okay", 1), ("Hello", 0)]


def test_the_merged_record_carries_only_its_own_params_and_the_fill_ranges() -> None:
    mic = _side(
        ("Yeah.", 8.05),
        speaker=None,
        params={"fill_ranges": "[[6.0, 7.0]]", "pick_record": "[]", "fill_counts": "[3]"},
    )
    copy = mic.words[0]
    app = APP.model_copy(
        update={
            "engine": Engine(name="xai-stt", model="grok-2", params={"fill_ranges": "[[1.0, 2.0]]"})
        }
    )

    merged = _merge(mic, app).transcript

    assert (merged.engine.name, merged.engine.model) == ("xai-stt", "grok-2")
    assert merged.engine.params == {
        "merge_rule": "bleed-1",
        "merge_window_s": 0.25,
        "merge_offset_s": 0.0,
        "merge_mic_words": 1,
        "merge_app_words": 10,
        "merge_mic_moved": 0,
        "merge_app_moved": 0,
        "merge_dropped_run_1": 1,
        "merge_dropped_run_2": 0,
        "merge_dropped_run_3plus": 0,
        "merge_dropped": json.dumps([[copy.start, copy.end, copy.text, 1]]),
        "merge_mic_sha256": "mic-sha",
        "merge_app_sha256": "app-sha",
        "fill_ranges": "[[1.0, 2.0], [6.0, 7.0]]",
    }
    assert merged.tracks is not None
    assert [(t.role, t.label, t.transcript_sha256) for t in merged.tracks] == [
        ("mic", "Alice", "mic-sha"),
        ("app", None, "app-sha"),
    ]
    assert merged.text == " ".join(word.text for word in merged.words)


def test_mic_words_take_one_speaker_no_app_word_holds() -> None:
    mic = _side(("Hello", 0.5), ("there", 0.8), speaker=None)
    app = _side(("Okay", 1.0), speaker=3).model_copy(
        update={
            "words": [*_side(("Okay", 1.0), speaker=3).words, Word(text="so", start=1.3, end=1.5)]
        }
    )

    merged = _merge(mic, app).transcript

    mic_speakers = {w.speaker for w in merged.words if w.track == 0}
    assert len(mic_speakers) == 1
    assert None not in mic_speakers


_WORDS = st.lists(
    st.tuples(
        st.sampled_from(["yeah", "Yeah.", "so", "the", "--", "plan"]),
        st.floats(min_value=0.0, max_value=6.0),
        st.sampled_from([None, 0, 1]),
    ),
    min_size=1,
    max_size=25,
)


def _twinned(dropped: list[Word], app: list[Word], offset: float) -> bool:
    """Whether each dropped word can hold its own same-key app word in its window."""
    held: dict[int, int] = {}

    def twins(word: Word) -> list[int]:
        key = word_key(word.text)
        return [
            index
            for index, other in enumerate(app)
            if key
            and word_key(other.text) == key
            and abs(other.start - (word.start - offset)) <= BLEED_WINDOW_S
        ]

    def place(at: int, seen: set[int]) -> bool:
        for index in twins(dropped[at]):
            if index not in seen:
                seen.add(index)
                if index not in held or place(held[index], seen):
                    held[index] = at
                    return True
        return False

    return all(place(at, set()) for at in range(len(dropped)))


@given(
    mic_said=_WORDS,
    app_said=_WORDS,
    offset=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
)
def test_app_words_stay_and_mic_words_are_kept_or_dropped(
    mic_said: list[tuple[str, float, int | None]],
    app_said: list[tuple[str, float, int | None]],
    offset: float,
) -> None:
    def side(said: list[tuple[str, float, int | None]]) -> Transcript:
        words = [Word(text=t, start=s, end=s + 0.1, speaker=k) for t, s, k in said]
        return _side().model_copy(update={"words": words})

    mic, app = side(mic_said), side(app_said)

    merged = _merge(mic, app, offset=offset)
    words = merged.transcript.words

    def fields(word: Word) -> tuple[str, float, float]:
        return word.text, word.start, word.end

    assert Counter(w.model_copy(update={"track": None}) for w in words if w.track == 1) == Counter(
        app.words
    )
    kept = [fields(w) for w in words if w.track == 0]
    assert Counter(kept + [fields(d.word) for d in merged.dropped]) == Counter(
        map(fields, mic.words)
    )
    assert [w.start for w in words] == sorted(w.start for w in words)
    assert merged.offset == offset
    assert _twinned([d.word for d in merged.dropped], app.words, offset)
