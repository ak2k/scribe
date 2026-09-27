from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import stamina
from typer.testing import CliRunner

from scribe import gemini_stt
from scribe.cli import app
from scribe.errors import ExternalServiceError, InputValidationError, SpendCapError
from scribe.gemini_stt import Chunk, retime
from scribe.schema import Engine, Source, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

KEY = "gemini-test-key-never-logged"
SOURCE = Source(kind="audio", ref="meeting.mp3")
# `_reply`'s usage (15,362 in, 5,000 out, 1,000 thinking) and one attempt's worst case,
# at $2 / $12 per 1M.
REPLY_USD = 15362 * 2e-6 + (5000 + 1000) * 12e-6
WORST_USD = 25_000 * 2e-6 + 32768 * 12e-6
cli = CliRunner()


@pytest.fixture(autouse=True)
def instant_retries() -> Iterator[None]:
    with stamina.set_testing(True, attempts=gemini_stt.RETRY_ATTEMPTS, cap=True):
        yield


def _anchor(*timed: tuple[str, float]) -> Transcript:
    return Transcript(
        source=SOURCE,
        engine=Engine(name="parakeet"),
        text=" ".join(text for text, _ in timed),
        words=[Word(text=text, start=start, end=start + 0.2) for text, start in timed],
    )


def _runner(
    duration: str = "100.0",
    *,
    calls: list[list[str]] | None = None,
    missing: str | None = None,
    failing: str | None = None,
    write: bool = True,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    seen = [] if calls is None else calls

    def run(argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
        if argv[0] == missing:
            raise FileNotFoundError(2, "No such file or directory")
        seen.append(argv)
        if argv[0] == "ffmpeg" and write:
            Path(argv[-1]).write_bytes(b"mp3 bytes")
        return subprocess.CompletedProcess(
            argv, int(argv[0] == failing), stdout=f"{duration}\n", stderr="Invalid data"
        )

    return run


def _reply(
    text: str = "Hello, there.",
    *,
    finish: str = "STOP",
    body: str | None = None,
    usage: bool = True,
    output_tokens: int = 5000,
) -> httpx.Response:
    segments = {"segments": [{"start_seconds": 1.5, "speaker": "S1", "text": text}]}
    parts = [
        {"text": "thinking it over", "thought": True},
        {"text": json.dumps(segments) if body is None else body},
    ]
    payload: dict[str, object] = {
        "candidates": [{"content": {"parts": parts}, "finishReason": finish}],
        "modelVersion": gemini_stt.MODEL,
    }
    if usage:
        payload["usageMetadata"] = {
            "promptTokenCount": 15362,
            "candidatesTokenCount": output_tokens,
            "thoughtsTokenCount": 1000,
        }
    return httpx.Response(200, json=payload)


def _serve(*answers: httpx.Response | Exception) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Answer requests in order, repeating the last answer."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        answer = answers[min(len(seen), len(answers) - 1)]
        seen.append(request)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return httpx.MockTransport(handle), seen


def _transcribe(
    transport: httpx.MockTransport,
    *,
    max_usd: float = 3.0,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    anchor: Transcript | None = None,
) -> Transcript:
    return gemini_stt.transcribe(
        Path("meeting.mp3"),
        _anchor(("Hello", 1.0), ("there", 1.5)) if anchor is None else anchor,
        source=SOURCE,
        api_key=KEY,
        max_usd=max_usd,
        run=_runner() if run is None else run,
        transport=transport,
    )


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
    *args: str,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[int, str]:
    run = _runner() if run is None else run
    patched = partial(gemini_stt.transcribe, run=run, transport=transport)
    monkeypatch.setattr(gemini_stt, "transcribe", patched)
    (tmp_path / "meeting.mp3").write_bytes(b"audio")
    _anchor(("Hello,", 1.0), ("there.", 1.5)).dump(tmp_path / "anchor.json")
    argv = ["gemini", str(tmp_path / "meeting.mp3"), "--anchor", str(tmp_path / "anchor.json")]
    result = cli.invoke(app, [*argv, *args])
    return result.exit_code, result.output


def test_retime_takes_anchor_times_and_keeps_each_chunk_to_its_half_of_the_overlap() -> None:
    anchor = _anchor(
        ("One,", 1.0),
        ("two", 2.0),
        ("three.", 3.0),
        ("four", 596.0),
        ("five", 598.0),
        ("six", 599.0),
        ("seven", 600.0),
    ).words
    chunks = [Chunk(0.0, 600.0), Chunk(595.0, 20.0)]
    replies = [["Uh, one two", "and three", "four five six"], ["Four five\u2014six seven eight"]]

    words = retime(chunks, replies, anchor)

    assert [(word.text, round(word.start, 9)) for word in words] == [
        ("uh", 0.7),
        ("one", 1.0),
        ("two", 2.0),
        ("and", 2.5),
        ("three", 3.0),
        ("four", 596.0),
        ("five", 598.0),
        ("six", 599.0),
        ("seven", 600.0),
        ("eight", 600.3),
    ]
    assert all(word.end == word.start and word.speaker is None for word in words)


def test_retime_puts_a_chunk_with_nothing_aligned_at_its_offset() -> None:
    words = retime([Chunk(0.0, 9.0)], [["Don\u2019t stop/go"]], [])

    assert [(word.text, word.start) for word in words] == [
        ("don't", 0.0),
        ("stop", 0.0),
        ("go", 0.0),
    ]


def test_retime_aligns_to_anchor_words_up_to_a_second_outside_the_chunk() -> None:
    anchor = _anchor(("alpha", 19.5), ("mid", 25.0), ("omega", 30.5)).words

    words = retime([Chunk(20.0, 10.0)], [["alpha mid omega"]], anchor)

    assert [(word.text, word.start) for word in words] == [
        ("alpha", 19.5),
        ("mid", 25.0),
        ("omega", 30.5),
    ]


def test_retime_lets_the_last_chunk_keep_a_tail_too_short_for_its_own_chunk() -> None:
    anchor = _anchor(("one", 1.0), ("late", 1192.6), ("word", 1192.8)).words
    chunks = [Chunk(0.0, 600.0), Chunk(595.0, 598.0)]

    words = retime(chunks, [["One"], ["late word"]], anchor)

    assert [(word.text, word.start) for word in words] == [
        ("one", 1.0),
        ("late", 1192.6),
        ("word", 1192.8),
    ]


@pytest.mark.parametrize(
    ("duration", "planned"),
    [
        (0.0, []),
        (1190.0, [(0.0, 600.0), (595.0, 595.0)]),
        (1195.0, [(0.0, 600.0), (595.0, 600.0)]),
        (1195.5, [(0.0, 600.0), (595.0, 600.0), (1190.0, 5.5)]),
    ],
)
def test_a_tail_inside_the_previous_chunk_gets_no_chunk_of_its_own(
    duration: float, planned: list[tuple[float, float]]
) -> None:
    chunks = gemini_stt.plan_chunks(duration)

    assert [(chunk.offset, chunk.duration) for chunk in chunks] == planned


def test_a_short_tail_is_not_sent() -> None:
    transport, seen = _serve(_reply())

    _transcribe(transport, run=_runner("1193"))

    assert len(seen) == 2


def test_the_command_writes_retimed_words_and_the_key_stays_in_its_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, seen = _serve(_reply())
    calls: list[list[str]] = []
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    code, output = _invoke(tmp_path, monkeypatch, transport, run=_runner(calls=calls))

    assert code == 0, output
    written = tmp_path / "meeting.gemini.json"
    transcript = Transcript.load(written)
    assert [(word.text, word.start, word.end) for word in transcript.words] == [
        ("hello", 1.0, 1.0),
        ("there", 1.5, 1.5),
    ]
    assert transcript.text == "hello there"
    assert transcript.source.sha256 == hashlib.sha256(b"audio").hexdigest()
    assert transcript.engine.model == "gemini-3.1-pro-preview"
    assert transcript.engine.params == {
        "chunk_seconds": 600.0,
        "step_seconds": 595.0,
        "cost_usd": round(REPLY_USD, 4),
        "anchor_engine": "parakeet",
    }
    assert f"for ${round(REPLY_USD, 4)}" in output
    assert "gemini.preflight" in output
    assert str(seen[0].url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.1-pro-preview:generateContent"
    )
    assert seen[0].headers["x-goog-api-key"] == KEY
    assert KEY not in str(seen[0].url)
    assert KEY not in output
    assert KEY not in written.read_text(encoding="utf-8")
    sent = seen[0].content.decode()
    assert json.dumps(gemini_stt.PROMPT.format(dur=100.0)) in sent
    assert '"maxOutputTokens":32768' in sent
    audio = str((tmp_path / "meeting.mp3").absolute())
    assert calls[1][:-1] == [
        *["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "0.000", "-t", "600.000"],
        *["-i", audio, "-ac", "1", "-ar", "16000", "-b:a", "64k"],
    ]
    assert not Path(calls[1][-1]).parent.exists()


@pytest.mark.parametrize(
    ("args", "run", "message"),
    [
        (["--max-usd", "0.01"], _runner(), "the expected cost of $0.02 plus room"),
        ([], _runner(missing="ffmpeg"), "cannot run ffmpeg: No such file or directory"),
        ([], _runner(missing="ffprobe"), "cannot run ffprobe: No such file or directory"),
        *(
            ([f"--max-usd={cap}"], _runner(), "--max-usd must be a finite amount above 0")
            for cap in ("nan", "inf", "0", "-1")
        ),
    ],
)
def test_a_run_refused_before_any_call_is_one_line_and_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    run: Callable[..., subprocess.CompletedProcess[str]],
    message: str,
) -> None:
    transport, seen = _serve(_reply())
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    code, output = _invoke(tmp_path, monkeypatch, transport, *args, run=run)

    assert code == 2
    assert output.startswith(f"scribe: {message}")
    assert len(output.splitlines()) == 1
    assert seen == []
    assert not (tmp_path / "meeting.gemini.json").exists()


def test_the_command_logs_its_preflight_but_not_each_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, seen = _serve(_reply())
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    code, output = _invoke(tmp_path, monkeypatch, transport, run=_runner("1000"))

    assert code == 0, output
    assert len(seen) == 2
    assert output.count("gemini.preflight") == 1
    assert "gemini.chunk" not in output


def test_preflight_prices_the_chunks_after_running_only_ffprobe() -> None:
    calls: list[list[str]] = []

    planned = gemini_stt.preflight(Path("meeting.mp3"), 3.0, _runner("1195.5", calls=calls))

    assert planned.duration == 1195.5
    assert [(chunk.offset, chunk.duration) for chunk in planned.chunks] == [
        (0.0, 600.0),
        (595.0, 600.0),
        (1190.0, 5.5),
    ]
    assert planned.expected_usd == gemini_stt.expected_usd(planned.chunks)
    assert [argv[0] for argv in calls] == ["ffprobe"]


def test_preflight_refuses_a_cap_that_is_no_amount_before_running_anything() -> None:
    calls: list[list[str]] = []

    with pytest.raises(InputValidationError, match="finite amount above 0"):
        gemini_stt.preflight(Path("meeting.mp3"), float("nan"), _runner(calls=calls))

    assert calls == []


def test_a_missing_key_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport, seen = _serve(_reply())
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    code, output = _invoke(tmp_path, monkeypatch, transport)

    assert (code, output, seen) == (2, "scribe: GEMINI_API_KEY is not set\n", [])


def test_a_reply_off_its_schema_twice_exits_two_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, seen = _serve(_reply(body='{"segments": [{"text": "no time or speaker"}]}'))
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    code, output = _invoke(tmp_path, monkeypatch, transport)

    assert code == 2
    assert "reply to chunk 1 of 1 was unusable twice" in output
    assert "Field required (at segments.0.start_seconds)" in output
    assert len(seen) == 2
    assert not (tmp_path / "meeting.gemini.json").exists()


def test_a_transcript_that_cannot_be_written_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = Transcript.dump

    def refuse(self: Transcript, path: Path) -> None:
        if path.name.endswith(".gemini.json"):
            raise OSError("disk full")
        dump(self, path)

    monkeypatch.setattr(Transcript, "dump", refuse)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    code, output = _invoke(tmp_path, monkeypatch, _serve(_reply())[0])

    assert code == 2
    assert "cannot write transcript to" in output


def test_a_reply_cut_short_is_asked_once_more_and_charged_in_full() -> None:
    transport, seen = _serve(_reply(finish="MAX_TOKENS", usage=False), _reply())

    transcript = _transcribe(transport)

    assert len(seen) == 2
    assert transcript.text == "hello there"
    assert transcript.engine.params["cost_usd"] == round(WORST_USD + REPLY_USD, 4)


def test_a_cap_of_exactly_one_worst_case_affords_one_call() -> None:
    transport, seen = _serve(_reply())

    transcript = _transcribe(transport, max_usd=gemini_stt.WORST_CASE_USD)

    assert len(seen) == 1
    assert transcript.text == "hello there"


def test_a_429_is_retried() -> None:
    transport, seen = _serve(httpx.Response(429, text="slow down"), _reply())

    transcript = _transcribe(transport)

    assert len(seen) == 2
    assert transcript.text == "hello there"
    # An HTTP error is not billed, so only the reply is charged.
    assert transcript.engine.params["cost_usd"] == round(REPLY_USD, 4)


@pytest.mark.parametrize(
    ("answer", "sent", "spent", "message"),
    [
        (httpx.Response(400, text=f"bad key {KEY}"), 1, 0.0, "HTTP 400: bad key [redacted]"),
        (httpx.Response(200, text="not json"), 1, WORST_USD, "not a reply envelope"),
        (
            httpx.ConnectError("unreachable"),
            gemini_stt.RETRY_ATTEMPTS,
            gemini_stt.RETRY_ATTEMPTS * WORST_USD,
            "request failed",
        ),
    ],
)
def test_a_failed_request_is_an_error_naming_the_spend(
    answer: httpx.Response | Exception, sent: int, spent: float, message: str
) -> None:
    transport, seen = _serve(answer)

    with pytest.raises(ExternalServiceError, match=re.escape(message)) as raised:
        _transcribe(transport)

    assert len(seen) == sent
    assert f"${spent:.4f} spent" in str(raised.value)
    assert KEY not in str(raised.value)


@pytest.mark.parametrize(
    ("duration", "answers", "max_usd", "sent"),
    [
        # A first chunk dearer than expected leaves no room for the second's worst case.
        ("1000", [_reply(output_tokens=20000)], 0.6, 1),
        # A reply cut short without usage is charged in full before the re-ask.
        ("100", [_reply(finish="MAX_TOKENS", usage=False)], 0.6, 1),
        # A timeout may still be billed, so it is charged its worst case before the retry.
        ("100", [httpx.ReadTimeout("slow"), _reply()], 0.88, 1),
    ],
)
def test_an_attempt_whose_worst_case_passes_the_cap_is_not_sent(
    duration: str, answers: list[httpx.Response | Exception], max_usd: float, sent: int
) -> None:
    transport, seen = _serve(*answers)

    with pytest.raises(SpendCapError, match=re.escape("worst case of $0.44 would pass")):
        _transcribe(transport, max_usd=max_usd, run=_runner(duration))

    assert len(seen) == sent


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (_runner("N/A"), "ffprobe reported no usable duration"),
        (_runner("inf"), "ffprobe reported no usable duration"),
        (_runner(failing="ffmpeg"), "ffmpeg exited 1: Invalid data"),
        (_runner(write=False), "cannot read a cut chunk"),
    ],
)
def test_a_tool_failure_is_an_error_and_leaves_no_chunks_behind(
    run: Callable[..., subprocess.CompletedProcess[str]], message: str
) -> None:
    calls: list[list[str]] = []
    transport, seen = _serve(_reply())

    def recorded(argv: list[str], **settings: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return run(argv, **settings)

    with pytest.raises(ExternalServiceError, match=message):
        _transcribe(transport, run=recorded)

    assert seen == []
    assert all(not Path(argv[-1]).parent.exists() for argv in calls if argv[0] == "ffmpeg")


@pytest.mark.parametrize(("duration", "max_usd"), [("18000", 3.0), ("12600", 2.0)])
def test_a_run_the_last_call_could_not_afford_is_refused_before_any_call(
    duration: str, max_usd: float
) -> None:
    # Expected cost under the cap, but not with room for the last call's worst case.
    calls: list[list[str]] = []
    transport, seen = _serve(_reply())

    with pytest.raises(SpendCapError, match="plus room for the last call's worst case"):
        _transcribe(transport, max_usd=max_usd, run=_runner(duration, calls=calls))

    assert seen == []
    assert [argv[0] for argv in calls] == ["ffprobe"]


def test_an_anchor_without_words_is_refused() -> None:
    transport, seen = _serve(_reply())

    with pytest.raises(InputValidationError, match="no words"):
        _transcribe(transport, anchor=_anchor())

    assert seen == []
