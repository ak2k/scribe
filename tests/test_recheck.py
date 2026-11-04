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
from scribe.ear import RECOGNIZERS, Heard, LocalEars
from scribe.errors import EarError, InputValidationError, ToolMissingError
from scribe.schema import Engine, Source, Transcript
from scribe.xai_stt import DEFAULT_MODEL, XaiStt
from tests.diarizer_fakes import FakeWorker, found

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from scribe.ear import Recognizer
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
    params: Mapping[str, float | bool | str] | None = None,
    duration: float | None = 120.0,
    without: Sequence[str] = (),
) -> Transcript:
    recorded: dict[str, float | bool | str] = {
        "fill_reference": REFERENCE,
        "fill_ranges": json.dumps([list(span) for span in fills]),
        "fill_unresolved_ranges": json.dumps([list(span) for span in unresolved]),
        "keyterms": json.dumps(["Alpha Term", "Beta Term"]),
        "language": "en",
        "format": True,
        "vad_threshold": 0.0,
    }
    recorded |= params or {}
    for key in without:
        del recorded[key]
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


APPLE_ONLY = "the recognizers need Apple silicon (macOS on arm64); this is Linux x86_64"


class Ears:
    """Stand-in local recognizers: refused by the host, or each hearing or failing as told."""

    def __init__(self, *, refused: str | None = None, failing: Mapping[str, str] | None = None):
        self.refused = refused
        self.failing = failing or {}
        self.asked: list[tuple[str, list[tuple[float, float]], int]] = []

    def resolve(self, _recognizers: Sequence[Recognizer] = RECOGNIZERS, /) -> None:
        if self.refused is not None:
            raise EarError(self.refused)

    def hear(
        self,
        _audio: Path,
        clips: Sequence[tuple[float, float]],
        *,
        max_new_tokens: int = 256,
        recognizers: Sequence[Recognizer] = RECOGNIZERS,
    ) -> tuple[Heard, ...]:
        self.resolve()
        for spec in recognizers:
            self.asked.append((spec.name, list(clips), max_new_tokens))
            if spec.name in self.failing:
                raise EarError(self.failing[spec.name])
        return tuple(
            Heard(spec, {}, "mps", "float16", 1.0, tuple(f"{spec.name} {clip}" for clip in clips))
            for spec in recognizers
        )


def _refused() -> Ears:
    return Ears(refused=APPLE_ONLY)


def _run(
    transcript: Transcript,
    tmp_path: Path,
    *,
    handler: Callable[[httpx.Request], httpx.Response] = _ok,
    ffmpeg: Ffmpeg | None = None,
    ears: recheck.Ears | None = None,
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
        ears or _refused(),
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
    timings = [call[call.index("-ss") : call.index("-ss") + 4] for call in fake.calls[1:]]
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


@pytest.mark.parametrize(
    ("stored", "sent"),
    [
        ({"language": "", "format": False, "vad_threshold": 0.4}, {"vad_threshold": b"0.4"}),
        (
            {"language": "fr", "format": False, "vad_threshold": 0.0},
            {"language": b"fr", "vad_threshold": b"0"},
        ),
        (
            {"language": "en", "format": True, "vad_threshold": 0.25},
            {"language": b"en", "format": b"true", "vad_threshold": b"0.25"},
        ),
    ],
    ids=["no-language", "no-format", "formatted"],
)
def test_the_stored_request_settings_are_replayed(
    tmp_path: Path, stored: dict[str, float | bool | str], sent: dict[str, bytes]
) -> None:
    _, seen, _ = _run(_transcript(fills=[(10.0, 12.0)], params=stored), tmp_path)

    fields = dict(_parts(seen[0]))
    settings = {"language", "format", "vad_threshold"}
    assert {name: body for name, body in fields.items() if name in settings} == sent
    assert "diarize" not in fields


@pytest.mark.parametrize(
    ("change", "without", "named"),
    [
        ({}, ("language",), "language"),
        ({}, ("format",), "format"),
        ({}, ("vad_threshold",), "vad_threshold"),
        ({"language": 1.0}, (), "language"),
        ({"format": "yes"}, (), "format"),
        ({"vad_threshold": True}, (), "vad_threshold"),
        ({"vad_threshold": 1.5}, (), "VAD threshold"),
    ],
    ids=[
        "no-language",
        "no-format",
        "no-vad",
        "number-language",
        "text-format",
        "bool-vad",
        "vad-past-1",
    ],
)
def test_missing_or_malformed_request_settings_are_refused_before_any_cut(
    tmp_path: Path, change: dict[str, float | bool | str], without: tuple[str, ...], named: str
) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], params=change, without=without)
    fake = Ffmpeg()

    with pytest.raises(InputValidationError, match=named):
        _run(transcript, tmp_path, ffmpeg=fake)
    assert fake.calls == []


def test_each_clip_is_cut_from_one_decode_of_the_recording(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], unresolved=[(60.0, 90.0)])

    _, seen, fake = _run(transcript, tmp_path)

    decode, *cuts = fake.calls
    recording = str((tmp_path / "recording.mp3").absolute())
    # A container's timestamps can jump; the decoded samples are the timeline the spans are on.
    assert decode[decode.index("-i") + 1] == recording
    assert "-ss" not in decode
    assert decode[-1].endswith(".wav")
    assert [cut[cut.index("-i") + 1] for cut in cuts] == [decode[-1]] * 2
    assert len(seen) == 2


def test_a_recording_that_does_not_decode_fails_every_xai_vote(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], unresolved=[(60.0, 90.0)])

    result, seen, fake = _run(transcript, tmp_path, ffmpeg=Ffmpeg(failing=[0]))

    assert seen == []
    assert len(fake.calls) == 1
    for span in result.spans:
        vote = span.votes[0]
        assert vote.status == "failed"
        assert vote.reason is not None
        assert "ffmpeg exited 1" in vote.reason


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

    result, seen, _ = _run(transcript, tmp_path, ffmpeg=Ffmpeg(failing=[1]))

    assert len(seen) == 1
    first, second = (span.votes[0] for span in result.spans)
    assert first.status == "failed"
    assert first.reason is not None
    assert "ffmpeg exited 1" in first.reason
    assert second.status == "heard"


def test_no_spans_writes_empty(tmp_path: Path) -> None:
    ears = Ears()

    result, seen, fake = _run(_transcript(), tmp_path, ears=ears)

    assert result.spans == []
    assert seen == []
    assert fake.calls == []
    assert ears.asked == []


def test_no_ffmpeg_is_refused_before_any_request(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    with pytest.raises(ToolMissingError):
        recheck.recheck(
            _transcript(fills=[(10.0, 12.0)]),
            tmp_path / "meeting.json",
            tmp_path / "recording.mp3",
            client,
            _refused(),
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
    ears = Ears()

    result, _, _ = _run(transcript, tmp_path, ears=ears)

    clips = [(5.0, 17.0), (55.0, 95.0)]
    assert [(name, asked) for name, asked, _ in ears.asked] == [
        (spec.name, clips) for spec in RECOGNIZERS
    ]
    for clip, span in zip(clips, result.spans, strict=True):
        local = span.votes[1:]
        assert [(vote.engine, vote.model, vote.status) for vote in local] == [
            (spec.name, spec.model, "heard") for spec in RECOGNIZERS
        ]
        assert [vote.text for vote in local] == [f"{spec.name} {clip}" for spec in RECOGNIZERS]


def test_a_recognizer_that_fails_is_failed_and_the_other_still_votes(tmp_path: Path) -> None:
    cohere, qwen = RECOGNIZERS
    ears = Ears(failing={cohere.name: f"{cohere.name} exited 1: no GPU"})

    result, _, _ = _run(_transcript(fills=[(10.0, 12.0)]), tmp_path, ears=ears)

    _, first, second = result.spans[0].votes
    assert (first.engine, first.status, first.reason) == (
        cohere.name,
        "failed",
        f"{cohere.name} exited 1: no GPU",
    )
    assert (second.engine, second.status, second.reason) == (qwen.name, "heard", None)


def test_an_uncached_recognizer_is_not_run_and_the_other_still_votes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohere, qwen = RECOGNIZERS
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    snapshot = hub / f"models--{qwen.model.replace('/', '--')}" / "snapshots" / qwen.revision
    snapshot.mkdir(parents=True)
    for name in ("config.json", "model.safetensors"):
        (snapshot / name).write_text("{}", encoding="utf-8")

    def answer(intervals: list[list[float]]) -> dict[str, object]:
        texts = [f"heard {start:g}" for start, _ in intervals]
        return {"versions": {}, "device": "mps", "dtype": "bf16", "runtime_s": 1.0, "texts": texts}

    worker = FakeWorker(answer)
    ears = LocalEars(run=worker.run, which=found, host=lambda: ("Darwin", "arm64"))

    result, _, _ = _run(_transcript(fills=[(10.0, 12.0)]), tmp_path, ears=ears)

    _, first, second = result.spans[0].votes
    assert (first.engine, first.status) == (cohere.name, "not run")
    assert first.reason is not None
    assert f"{cohere.model} at {cohere.revision} is not in the Hugging Face cache" in first.reason
    assert (second.engine, second.status, second.text) == (qwen.name, "heard", "heard 5")


def test_local_token_cap_fits_the_longest_clip(tmp_path: Path) -> None:
    # A 60 s span: a 70 s clip once padded.
    ears = Ears()

    result, _, _ = _run(
        _transcript(fills=[(10.0, 12.0)], unresolved=[(20.0, 80.0)]), tmp_path, ears=ears
    )

    caps = {cap for _, _, cap in ears.asked}
    assert len(caps) == 1
    (cap,) = caps
    assert cap >= recheck.TOKENS_PER_S * 70.0
    assert result.local_max_new_tokens == cap


def test_a_span_past_the_end_leaves_the_other_spans_votes(tmp_path: Path) -> None:
    ears = Ears()
    transcript = _transcript(fills=[(10.0, 12.0), (130.0, 131.0)], duration=120.0)

    result, seen, fake = _run(transcript, tmp_path, ears=ears)

    assert {tuple(asked) for _, asked, _ in ears.asked} == {((5.0, 17.0),)}
    assert [vote.status for vote in result.spans[0].votes] == ["heard"] * 3
    past = result.spans[1]
    assert past.clip is None
    assert [vote.status for vote in past.votes] == ["out of range"] * 3
    assert len(seen) == len(fake.calls) - 1 == 1


def test_a_span_starting_within_the_pad_after_the_end_hears_nothing(tmp_path: Path) -> None:
    ears = Ears()
    transcript = _transcript(fills=[(122.0, 123.0)], duration=120.0)

    result, seen, fake = _run(transcript, tmp_path, ears=ears)

    votes = result.spans[0].votes
    assert [vote.status for vote in votes] == ["out of range"] * 3
    assert all(vote.reason is not None and "120" in vote.reason for vote in votes)
    assert (seen, fake.calls, ears.asked) == ([], [], [])


def test_spans_without_a_known_duration_are_refused(tmp_path: Path) -> None:
    transcript = _transcript(fills=[(10.0, 12.0)], duration=None)

    with pytest.raises(InputValidationError, match="duration"):
        _run(transcript, tmp_path)


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


def test_a_malformed_reply_is_recorded_as_failed(tmp_path: Path) -> None:
    def odd(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"words": []})

    result, _, _ = _run(_transcript(fills=[(10.0, 12.0)]), tmp_path, handler=odd)

    vote = result.spans[0].votes[0]
    assert vote.status == "failed"
    assert vote.reason is not None
    assert "malformed xAI transcription response" in vote.reason
