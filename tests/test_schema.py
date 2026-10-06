from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scribe.errors import ExternalServiceError, InputValidationError
from scribe.schema import Engine, Source, Track, Transcript, Word, from_xai_response

FIXTURES = Path(__file__).parent / "fixtures"

SOURCE = Source(kind="audio", ref="fixture.mp3")
ENGINE = Engine(name="xai-stt", model="grok-voice-transcribe-2.0")


def _payload(name: str) -> dict[str, object]:
    loaded: dict[str, object] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return loaded


def test_dump_load_round_trips(tmp_path: Path) -> None:
    original = Transcript.load(FIXTURES / "transcript_two_speakers.json")
    path = tmp_path / "out.json"
    original.dump(path)
    assert path.read_text(encoding="utf-8").endswith("}\n")
    assert Transcript.load(path) == original


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Word.model_validate({"text": "hi", "start": 0.0, "end": 0.1, "confidence": 0.9})


def test_load_reports_a_missing_file_as_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError):
        Transcript.load(tmp_path / "absent.json")


def test_load_reports_a_malformed_file_as_input_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"text": "hi"}', encoding="utf-8")
    with pytest.raises(InputValidationError) as caught:
        Transcript.load(path)
    assert "\n" not in str(caught.value), "the message must stay printable on one CLI line"


def test_load_reports_a_non_utf8_file_as_input_error(tmp_path: Path) -> None:
    path = tmp_path / "not-utf8.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(InputValidationError) as caught:
        Transcript.load(path)
    assert "\n" not in str(caught.value), "the message must stay printable on one CLI line"


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ('"start": 0.2,', '"start": Infinity,'),
        ('"turns": []', '"turns": [{"speaker": "S", "start": 0.0, "end": NaN, "text": "hi"}]'),
        ('"duration": 29.8,', '"duration": -Infinity,'),
        ('"params": {}', '"params": {"temperature": Infinity}'),
    ],
)
def test_load_rejects_a_non_finite_float(tmp_path: Path, before: str, after: str) -> None:
    # json accepts the non-RFC literals Infinity and NaN; dumped back they become
    # null, a transcript this schema cannot load.
    raw = (FIXTURES / "transcript_two_speakers.json").read_text(encoding="utf-8")
    assert before in raw
    path = tmp_path / "non-finite.json"
    path.write_text(raw.replace(before, after, 1), encoding="utf-8")

    with pytest.raises(InputValidationError):
        Transcript.load(path)


def test_a_non_finite_param_is_reported_as_a_number(tmp_path: Path) -> None:
    raw = (FIXTURES / "transcript_two_speakers.json").read_text(encoding="utf-8")
    path = tmp_path / "nan-param.json"
    path.write_text(raw.replace('"params": {}', '"params": {"t": NaN}', 1), encoding="utf-8")

    with pytest.raises(InputValidationError) as caught:
        Transcript.load(path)

    assert "finite number" in str(caught.value)


def test_params_keep_their_json_types_through_a_round_trip(tmp_path: Path) -> None:
    params = '{"s": "3", "t": "true", "i": 3, "z": 0, "f": 1.5, "b": true}'
    raw = (FIXTURES / "transcript_two_speakers.json").read_text(encoding="utf-8")
    path = tmp_path / "params.json"
    path.write_text(raw.replace('"params": {}', f'"params": {params}', 1), encoding="utf-8")

    loaded = Transcript.load(path)
    loaded.dump(tmp_path / "again.json")

    assert loaded.engine.params == {"s": "3", "t": "true", "i": 3, "z": 0, "f": 1.5, "b": True}
    assert [type(value) for value in loaded.engine.params.values()] == [
        str,
        str,
        int,
        int,
        float,
        bool,
    ]
    assert Transcript.load(tmp_path / "again.json") == loaded


def test_from_xai_response_keeps_every_word_undiarized() -> None:
    payload = _payload("xai_say_clip.json")
    transcript = from_xai_response(payload, source=SOURCE, engine=ENGINE)

    assert len(transcript.words) == 29
    assert transcript.language == "en"
    assert transcript.turns == []
    assert all(word.speaker is None for word in transcript.words)
    assert transcript.text.startswith("In the beginning")


def test_from_xai_response_keeps_speaker_ids_when_diarized() -> None:
    payload = _payload("xai_diarized_two_speakers.json")
    transcript = from_xai_response(payload, source=SOURCE, engine=ENGINE)

    assert len(transcript.words) == 40
    assert transcript.words[0].speaker == 7
    assert {word.speaker for word in transcript.words} == {7, 4}


def test_from_xai_response_wraps_shape_drift() -> None:
    with pytest.raises(ExternalServiceError):
        from_xai_response({"text": "hi", "surprise": 1}, source=SOURCE, engine=ENGINE)


def test_from_xai_response_rejects_an_infinite_word_boundary() -> None:
    # json.loads accepts the non-RFC literals Infinity and NaN, so a 2xx body
    # can carry a float that this schema cannot serialize back.
    payload: dict[str, object] = {
        "text": "hi",
        "words": [{"text": "hi", "start": float("inf"), "end": 2.0}],
    }
    with pytest.raises(ExternalServiceError):
        from_xai_response(payload, source=SOURCE, engine=ENGINE)


@pytest.mark.parametrize("confidence", [0.93, 1.0000001, -1e-9, None])
def test_from_xai_response_accepts_and_drops_word_confidence(confidence: float | None) -> None:
    word: dict[str, object] = {"text": "hi", "start": 0.0, "end": 0.4, "speaker": 3}
    bare: dict[str, object] = {"text": "hi", "words": [word]}
    scored: dict[str, object] = {"text": "hi", "words": [{**word, "confidence": confidence}]}

    expected = from_xai_response(bare, source=SOURCE, engine=ENGINE)
    actual = from_xai_response(scored, source=SOURCE, engine=ENGINE)

    assert expected.words == [Word(text="hi", start=0.0, end=0.4, speaker=3)]
    assert actual.model_dump() == expected.model_dump()


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf")])
def test_from_xai_response_rejects_a_non_finite_word_confidence(confidence: float) -> None:
    payload: dict[str, object] = {
        "text": "hi",
        "words": [{"text": "hi", "start": 0.0, "end": 0.4, "confidence": confidence}],
    }
    with pytest.raises(ExternalServiceError):
        from_xai_response(payload, source=SOURCE, engine=ENGINE)


def test_from_xai_response_wraps_word_shape_drift() -> None:
    payload: dict[str, object] = {
        "text": "hi",
        "words": [{"text": "hi", "start": 0.0, "end": 0.4, "surprise": 1}],
    }
    with pytest.raises(ExternalServiceError):
        from_xai_response(payload, source=SOURCE, engine=ENGINE)


def test_from_xai_response_rejects_a_not_a_number_duration() -> None:
    payload: dict[str, object] = {"text": "hi", "duration": float("nan")}
    with pytest.raises(ExternalServiceError):
        from_xai_response(payload, source=SOURCE, engine=ENGINE)


def test_stages_copy_rather_than_mutate() -> None:
    transcript = Transcript.load(FIXTURES / "transcript_two_speakers.json")
    updated = transcript.model_copy(update={"turns": []})
    assert updated is not transcript
    with pytest.raises(ValidationError):
        transcript.text = "edited"


def test_a_one_track_dump_is_unchanged(tmp_path: Path) -> None:
    fixture = FIXTURES / "transcript_two_speakers.json"
    path = tmp_path / "out.json"

    Transcript.load(fixture).dump(path)

    assert path.read_bytes() == fixture.read_bytes()
    assert '"track' not in path.read_text(encoding="utf-8")


def _merged(
    *words: tuple[int | None, int | None], role: str = "mic", label: str | None = "Alice"
) -> str:
    """A merged transcript's JSON with one word per (speaker, track) pair."""
    tracks = [
        {
            "role": role,
            "label": label,
            "source": {"kind": "audio", "ref": "mic.wav"},
            "transcript_sha256": "a",
        },
        {"role": "app", "source": {"kind": "audio", "ref": "app.wav"}, "transcript_sha256": "b"},
    ]
    return json.dumps(
        {
            "source": {"kind": "other", "ref": "mic.wav + app.wav"},
            "engine": {"name": "xai-stt"},
            "text": " ".join("w" for _ in words),
            "words": [
                {
                    "text": "w",
                    "start": float(index),
                    "end": index + 0.5,
                    "speaker": speaker,
                    "track": track,
                }
                for index, (speaker, track) in enumerate(words)
            ],
            "tracks": tracks,
        }
    )


def test_a_merged_transcript_round_trips_with_its_tracks(tmp_path: Path) -> None:
    path = tmp_path / "merged.json"
    path.write_text(_merged((2, 0), (0, 1), (None, 1)), encoding="utf-8")

    loaded = Transcript.load(path)
    loaded.dump(tmp_path / "again.json")

    assert [word.track for word in loaded.words] == [0, 1, 1]
    assert loaded.tracks is not None
    assert [(track.role, track.label) for track in loaded.tracks] == [
        ("mic", "Alice"),
        ("app", None),
    ]
    assert Transcript.load(tmp_path / "again.json") == loaded


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_merged((2, 0), (0, None)), "every word of a merged transcript needs a track"),
        (_merged((2, 0), (0, 2)), "every word of a merged transcript needs a track"),
        (_merged((2, 0), (0, -1)), "every word of a merged transcript needs a track"),
        (_merged((2, 0), (0, 0)), "mic track 0's words must share one speaker"),
        (_merged((2, 0), (2, 1)), "mic track 0's words must share one speaker"),
        (_merged((None, 0), (None, 1)), "mic track 0's words must share one speaker"),
        (_merged((2, 0), (0, 1), label=None), "a mic track needs a label"),
    ],
    ids=[
        "no-track",
        "past-the-end",
        "negative",
        "two-mic-speakers",
        "mic-speaker-on-app",
        "unattributed-on-both",
        "unlabeled-mic",
    ],
)
def test_load_refuses_words_that_do_not_fit_the_tracks(
    tmp_path: Path, raw: str, message: str
) -> None:
    path = tmp_path / "merged.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(InputValidationError) as caught:
        Transcript.load(path)

    assert message in str(caught.value)


def test_load_refuses_a_track_on_a_word_without_tracks(tmp_path: Path) -> None:
    raw = (FIXTURES / "transcript_two_speakers.json").read_text(encoding="utf-8")
    path = tmp_path / "stray.json"
    path.write_text(raw.replace('"speaker": 7', '"speaker": 7, "track": 0', 1), encoding="utf-8")

    with pytest.raises(InputValidationError) as caught:
        Transcript.load(path)

    assert "names a track, but the transcript lists none" in str(caught.value)


def test_the_schema_gains_only_the_optional_track_fields() -> None:
    assert not Word.model_fields["track"].is_required()
    assert not Transcript.model_fields["tracks"].is_required()
    assert set(Track.model_fields) == {"role", "label", "source", "transcript_sha256"}
    assert '"Track"' in json.dumps(Transcript.model_json_schema())
