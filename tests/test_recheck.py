"""Re-reading each fill and unresolved span of a transcript as a clip of its own."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import stamina

from scribe import recheck
from scribe.ear import RECOGNIZERS, Heard
from scribe.errors import EarError, InputValidationError, ToolMissingError
from scribe.schema import Engine, Source, Transcript
from scribe.xai_stt import DEFAULT_MODEL, XaiStt

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from scribe.recheck import Recheck

REFERENCE = "parakeet-cpp ggml-parakeet-tdt-0.6b-v3-f16.bin"
_FIELD_NAME = re.compile(rb'(?:;|\s)name="([^"]*)"')


@pytest.fixture(autouse=True)
def instant_retries() -> Iterator[None]:
    with stamina.set_testing(True, attempts=5, cap=True):
        yield


def _transcript(
    *,
    fills: Sequence[tuple[float, float]] = (),
    unresolved: Sequence[tuple[float, float]] = (),
    model: str | None = DEFAULT_MODEL,
    params: Mapping[str, float | str] | None = None,
    duration: float | None = 120.0,
) -> Transcript:
    recorded: dict[str, float | str] = {
        "fill_reference": REFERENCE,
        "fill_ranges": json.dumps([list(span) for span in fills]),
        "fill_unresolved_ranges": json.dumps([list(span) for span in unresolved]),
        "keyterms": json.dumps(["Alpha Term", "Beta Term"]),
    }
    recorded |= params or {}
    return Transcript(
        source=Source(kind="audio", ref="recording.mp3"),
        engine=Engine(name="xai-stt", model=model, params=recorded),
        duration=duration,
        text="",
        words=[],
    )


class Ffmpeg:
    """A stand-in for ffmpeg that records each argv and writes each clip."""

    def __init__(self, failing: Sequence[int] = ()) -> None:
        self.calls: list[list[str]] = []
        self.failing = set(failing)

    def run(self, argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
        code = 1 if len(self.calls) in self.failing else 0
        self.calls.append(argv)
        if code == 0:
            Path(argv[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(argv, code, stdout="", stderr="Invalid data")


def _parts(request: httpx.Request) -> list[tuple[str, bytes]]:
    boundary = request.headers["content-type"].partition("boundary=")[2].strip('"').encode()
    fields: list[tuple[str, bytes]] = []
    for section in request.read().split(b"--" + boundary):
        head, separator, body = section.partition(b"\r\n\r\n")
        match = _FIELD_NAME.search(head)
        if not separator or match is None:
            continue
        fields.append((match.group(1).decode(), body.removesuffix(b"\r\n")))
    return fields


def _reply(*words: tuple[str, float, float]) -> httpx.Response:
    body = {
        "text": " ".join(text for text, _, _ in words),
        "language": "en",
        "duration": 9.0,
        "words": [{"text": text, "start": start, "end": end} for text, start, end in words],
    }
    return httpx.Response(200, json=body)


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[XaiStt, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        request.read()
        seen.append(request)
        return handler(request)

    return XaiStt("test-key", transport=httpx.MockTransport(record)), seen


def _ok(_request: httpx.Request) -> httpx.Response:
    return _reply(("one", 1.0, 1.5), ("two", 2.0, 2.5))


def _refused(_audio: Path, _clips: Sequence[tuple[float, float]]) -> tuple[Heard, ...]:
    raise EarError("the recognizers need Apple silicon (macOS on arm64); this is Linux x86_64")


def _run(
    transcript: Transcript,
    tmp_path: Path,
    *,
    handler: Callable[[httpx.Request], httpx.Response] = _ok,
    ffmpeg: Ffmpeg | None = None,
    hear: Callable[[Path, Sequence[tuple[float, float]]], tuple[Heard, ...]] = _refused,
) -> tuple[Recheck, list[httpx.Request], Ffmpeg]:
    client, seen = _client(handler)
    fake = ffmpeg or Ffmpeg()
    audio = tmp_path / "recording.mp3"
    audio.write_bytes(b"pretend this is audio")
    result = recheck.recheck(
        transcript,
        tmp_path / "meeting.json",
        audio,
        client,
        hear,
        transcript_sha256="t" * 64,
        source_sha256="s" * 64,
        run=fake.run,
        which=lambda _name: "/usr/bin/ffmpeg",
    )
    return result, seen, fake


def test_spans_from_params() -> None:
    transcript = _transcript(fills=[(30.0, 32.0), (5.0, 6.0)], unresolved=[(100.0, 130.0)])

    found = recheck.find_spans(transcript, Path("meeting.json"))

    assert [(span.kind, span.start, span.end) for span in found] == [
        ("fill", 5.0, 6.0),
        ("fill", 30.0, 32.0),
        ("unresolved", 100.0, 130.0),
    ]


def test_a_range_before_the_recording_is_refused() -> None:
    transcript = _transcript(fills=[(-1.0, 2.0)])

    with pytest.raises(InputValidationError, match="fill_ranges"):
        recheck.find_spans(transcript, Path("meeting.json"))


def test_clip_pad_and_shift(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(2.0, 4.0)], unresolved=[(112.0, 118.0)], duration=120.0)

    result, seen, fake = _run(transcript, tmp_path)

    assert [span.clip for span in result.spans] == [(0.0, 9.0), (107.0, 120.0)]
    timings = [call[call.index("-ss") : call.index("-ss") + 4] for call in fake.calls]
    assert timings == [["-ss", "0.000", "-t", "9.000"], ["-ss", "107.000", "-t", "13.000"]]
    assert len(seen) == 2
    words = result.spans[1].votes[0].words
    assert words is not None
    # The reply's times are the clip's; the record's are the recording's.
    assert [(word.text, word.start, word.end) for word in words] == [
        ("one", 108.0, 108.5),
        ("two", 109.0, 109.5),
    ]


def test_keyterms_and_model_reused(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)])

    result, seen, _ = _run(transcript, tmp_path)

    fields = _parts(seen[0])
    assert dict(fields)["model"] == DEFAULT_MODEL.encode()
    # Diarization off is the field left out, the API's default.
    assert "diarize" not in dict(fields)
    assert [body for name, body in fields if name == "keyterm"] == [b"Alpha Term", b"Beta Term"]
    assert result.model == DEFAULT_MODEL
    assert result.spans[0].votes[0].model == DEFAULT_MODEL


def test_a_transcript_of_another_model_is_refused(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], model="some-other-model")

    with pytest.raises(InputValidationError, match="some-other-model"):
        _run(transcript, tmp_path)


def test_a_transcript_naming_no_model_is_refused(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], model=None)

    with pytest.raises(InputValidationError, match=DEFAULT_MODEL):
        _run(transcript, tmp_path)


def test_span_failure_recorded(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0), (50.0, 52.0)])
    calls: list[int] = []

    def first_fails(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        # The first span's five attempts all fail; the second span is answered.
        return httpx.Response(500, text="upstream down") if len(calls) <= 5 else _ok(_request)

    result, _, _ = _run(transcript, tmp_path, handler=first_fails)

    failed, heard = (span.votes[0] for span in result.spans)
    assert failed.engine == "xai-stt"
    assert failed.status == "failed"
    assert failed.reason is not None
    assert "500" in failed.reason
    assert failed.words is None
    assert heard.status == "heard"
    assert heard.text == "one two"


def test_a_failed_cut_is_recorded_and_sends_nothing(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0), (50.0, 52.0)])

    result, seen, _ = _run(transcript, tmp_path, ffmpeg=Ffmpeg(failing=[0]))

    assert len(seen) == 1
    first, second = (span.votes[0] for span in result.spans)
    assert first.status == "failed"
    assert first.reason is not None
    assert "ffmpeg exited 1" in first.reason
    assert second.status == "heard"


def test_no_spans_writes_empty(tmp_path: Path) -> None:
    heard: list[int] = []

    def counted(_audio: Path, _clips: Sequence[tuple[float, float]]) -> tuple[Heard, ...]:
        heard.append(1)
        return ()

    result, seen, fake = _run(_transcript(), tmp_path, hear=counted)

    assert result.spans == []
    assert seen == []
    assert fake.calls == []
    assert heard == []


def test_no_ffmpeg_is_refused_before_any_request(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    with pytest.raises(ToolMissingError):
        recheck.recheck(
            _transcript(fills=[(10.0, 12.0)]),
            tmp_path / "meeting.json",
            tmp_path / "recording.mp3",
            client,
            _refused,
            transcript_sha256="t" * 64,
            source_sha256="s" * 64,
            run=Ffmpeg().run,
            which=lambda _name: None,
        )
    assert seen == []


def test_a_refused_local_ear_is_recorded_as_not_run(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], unresolved=[(60.0, 90.0)])

    result, _, _ = _run(transcript, tmp_path)

    for span in result.spans:
        engines = [vote.engine for vote in span.votes]
        assert engines == ["xai-stt", *(spec.name for spec in RECOGNIZERS)]
        for vote in span.votes[1:]:
            assert vote.status == "not run"
            assert vote.reason is not None
            assert "Apple silicon" in vote.reason
            assert vote.text is None


def test_local_ears_vote_on_every_span(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], unresolved=[(60.0, 90.0)])
    asked: list[list[tuple[float, float]]] = []

    def hear(_audio: Path, clips: Sequence[tuple[float, float]]) -> tuple[Heard, ...]:
        asked.append(list(clips))
        return tuple(
            Heard(spec, {}, "mps", "float16", 1.0, tuple(f"{spec.name} {i}" for i in range(2)))
            for spec in RECOGNIZERS
        )

    result, _, _ = _run(transcript, tmp_path, hear=hear)

    assert asked == [[(5.0, 17.0), (55.0, 95.0)]]
    for index, span in enumerate(result.spans):
        local = span.votes[1:]
        assert [(vote.engine, vote.model, vote.status) for vote in local] == [
            (spec.name, spec.model, "heard") for spec in RECOGNIZERS
        ]
        assert [vote.text for vote in local] == [f"{spec.name} {index}" for spec in RECOGNIZERS]


def test_each_span_names_the_engine_that_proposed_it(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], unresolved=[(60.0, 90.0)])

    result, _, _ = _run(transcript, tmp_path)

    assert [span.proposed_by for span in result.spans] == [REFERENCE, REFERENCE]
    assert result.transcript_sha256 == "t" * 64
    assert result.source_sha256 == "s" * 64
    assert result.pad_s == recheck.PAD_S == 5.0


def test_malformed_keyterms_are_refused(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], params={"keyterms": "not a list"})

    with pytest.raises(InputValidationError, match="keyterms"):
        _run(transcript, tmp_path)


def test_spans_with_no_fill_reference_are_refused(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], params={"fill_reference": " "})

    with pytest.raises(InputValidationError, match="fill_reference"):
        _run(transcript, tmp_path)


def test_a_clip_runs_its_full_pad_when_the_duration_is_unknown() -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], duration=None)

    assert [span.clip for span in recheck.find_spans(transcript, Path("m.json"))] == [(5.0, 17.0)]


def test_a_span_past_the_recordings_end_is_recorded_as_failed(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(130.0, 131.0)], duration=120.0)

    result, seen, fake = _run(transcript, tmp_path)

    vote = result.spans[0].votes[0]
    assert vote.status == "failed"
    assert vote.reason is not None
    assert "past the recording's end" in vote.reason
    assert (seen, fake.calls) == ([], [])


def test_a_malformed_reply_is_recorded_as_failed(tmp_path: Path) -> None:
    def odd(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"words": []})

    result, _, _ = _run(_transcript(fills=[(10.0, 12.0)]), tmp_path, handler=odd)

    vote = result.spans[0].votes[0]
    assert vote.status == "failed"
    assert vote.reason is not None
    assert "malformed xAI transcription response" in vote.reason
