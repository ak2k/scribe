from __future__ import annotations

import asyncio
import json
import re
import stat
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast, override

import anyio
import anyio.to_thread
import httpx
import pytest
import stamina
import uvicorn
from structlog.testing import capture_logs
from uvicorn.lifespan.on import LifespanOn
from uvicorn.protocols.http.h11_impl import H11Protocol

from scribe.serve import ENDPOINT, create_app, dictation_client, run
from scribe.session_sources import SessionTerms, local_source, remote_source
from scribe.vocab import TermsFile
from tests.xai_fixtures import xai_payload

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from pathlib import Path

    from starlette.types import ASGIApp, Message

    from scribe.xai_stt import XaiStt

KEY = "xai-test-key-never-logged"
AUDIO = b"RIFF-pretend-wav-bytes"
VOICEINK_FIELDS = {"model": "scribe", "response_format": "json", "temperature": "0"}
KEPT_NAME = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{6}")
pytestmark = pytest.mark.anyio
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
NO_SESSIONS = SessionTerms([])


@pytest.fixture(autouse=True)
def instant_retries() -> Iterator[None]:
    with stamina.set_testing(True, attempts=5, cap=True):
        yield


def _fields(request: httpx.Request) -> list[tuple[str, bytes]]:
    boundary = request.headers["content-type"].partition("boundary=")[2].encode()
    fields: list[tuple[str, bytes]] = []
    for section in request.read().split(b"--" + boundary):
        head, separator, body = section.partition(b"\r\n\r\n")
        match = re.search(rb'(?:;|\s)name="([^"]*)"', head)
        if separator and match is not None:
            fields.append((match.group(1).decode(), body.removesuffix(b"\r\n")))
    return fields


class Xai:
    """A fake xAI endpoint recording each request it answers."""

    def __init__(self, *replies: httpx.Response) -> None:
        self.replies = list(replies) or [httpx.Response(200, json=xai_payload())]
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.seen.append(request)
        return self.replies[min(len(self.seen), len(self.replies)) - 1]


def _terms(tmp_path: Path, text: str = "") -> TermsFile:
    path = tmp_path / "terms.txt"
    path.write_text(text, encoding="utf-8")
    return TermsFile(path, required=True)


def _client(
    xai: Xai,
    tmp_path: Path,
    *,
    terms: str = "",
    keep: bool = True,
    max_bytes: int = 25 * 1024 * 1024,
    host: str = "127.0.0.1:8765",
    sessions: SessionTerms = NO_SESSIONS,
) -> httpx.AsyncClient:
    app = create_app(
        dictation_client(KEY, transport=httpx.MockTransport(xai)),
        _terms(tmp_path, terms),
        session_terms=sessions,
        keep=tmp_path / "keep" if keep else None,
        flags={"host": "127.0.0.1", "port": 8765},
        max_bytes=max_bytes,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{host}")


async def _post(
    client: httpx.AsyncClient,
    *,
    data: dict[str, str] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return await client.post(
        "/v1/audio/transcriptions",
        data=VOICEINK_FIELDS if data is None else data,
        files={"file": ("rec-1.wav", AUDIO, "audio/wav")} if files is None else files,
        headers={"Authorization": "Bearer anything", **(headers or {})},
    )


def _kept(tmp_path: Path) -> list[Path]:
    root = tmp_path / "keep"
    return sorted(root.iterdir()) if root.exists() else []


def _json(path: Path) -> dict[str, object]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))  # pyright: ignore[reportAny]  # json.loads is Any
    assert isinstance(decoded, dict)
    return cast("dict[str, object]", decoded)


async def test_health_answers_200(tmp_path: Path) -> None:
    async with _client(Xai(), tmp_path) as client:
        response = await client.get("/health")

    assert response.status_code == 200


def _session_file(*blocks: tuple[str, datetime, Sequence[str]]) -> str:
    return "".join(
        f"# session {session} ranked {ranked.isoformat()}"
        f" expires {(ranked + timedelta(minutes=30)).isoformat()}"
        ' {"cwd": "/w", "titles": ["A tab"]}\n' + "".join(f"{term}\n" for term in terms)
        for session, ranked, terms in blocks
    )


class Ssh:
    """A fake ssh runner answering one fixed file, or raising."""

    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.calls: list[list[str]] = []
        self.polled = threading.Event()

    def __call__(self, argv: Sequence[str], _timeout: float) -> str:
        self.calls.append(list(argv))
        self.polled.set()
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _sessions(tmp_path: Path, **hosts: Ssh) -> SessionTerms:
    local = tmp_path / "current.txt"
    local.write_text(
        _session_file(("local-s", NOW - timedelta(minutes=10), ["local_a", "local_b"])),
        encoding="utf-8",
    )
    sources = [local_source(local)]
    sources += [remote_source(host, runner=runner) for host, runner in hosts.items()]
    for source in sources:
        source.refresh(lambda: NOW)
    return SessionTerms(sources, clock=lambda: NOW)


def _two_hosts() -> dict[str, Ssh]:
    return {
        "box-a": Ssh(_session_file(("a-s", NOW - timedelta(minutes=1), ["kbx25", "herdr"]))),
        "box-b": Ssh(_session_file(("b-s", NOW - timedelta(minutes=5), ["b_term"]))),
    }


async def test_session_terms_follow_the_static_terms_newest_session_first(
    tmp_path: Path,
) -> None:
    xai = Xai()
    hosts = _two_hosts()
    sessions = _sessions(tmp_path, **hosts)
    async with _client(xai, tmp_path, terms="herdr\nVoiceInk\n", sessions=sessions) as client:
        await _post(client)

    keyterms = [body for name, body in _fields(xai.seen[0]) if name == "keyterm"]
    assert keyterms == [b"herdr", b"VoiceInk", b"kbx25", b"b_term", b"local_a", b"local_b"]
    (kept,) = _kept(tmp_path)
    result = _json(kept / "result.json")
    assert result["terms"] == [term.decode() for term in keyterms]
    assert result["session_terms"] == {"local file": 2, "box-a": 1, "box-b": 1}
    assert "A tab" not in (kept / "result.json").read_text(encoding="utf-8")


async def test_session_terms_are_snapped_like_static_terms(tmp_path: Path) -> None:
    heard = ["check", "kbx", "25"]
    payload = {
        "text": " ".join(heard),
        "duration": 1.0,
        "words": [{"text": text, "start": n, "end": n + 0.5} for n, text in enumerate(heard)],
    }
    sessions = _sessions(tmp_path, **_two_hosts())
    async with _client(Xai(httpx.Response(200, json=payload)), tmp_path, sessions=sessions) as c:
        response = await _post(c)

    assert response.json() == {"text": "check kbx25"}


async def test_a_dictation_never_runs_ssh(tmp_path: Path) -> None:
    ssh = Ssh(AssertionError("ssh ran during a request"))
    sessions = SessionTerms([remote_source("box-a", runner=ssh)], clock=lambda: NOW)
    async with _client(Xai(), tmp_path, sessions=sessions) as client:
        response = await _post(client)
        health = await client.get("/health")

    assert response.status_code == 200
    assert health.status_code == 200
    assert ssh.calls == []


async def test_health_counts_each_source_without_terms_or_session_ids(tmp_path: Path) -> None:
    hosts = {**_two_hosts(), "box-c": Ssh(RuntimeError("down"))}
    async with _client(
        Xai(), tmp_path, terms="herdr\nVoiceInk\n", sessions=_sessions(tmp_path, **hosts)
    ) as client:
        response = await client.get("/health")

    assert response.json() == {
        "status": "ok",
        "static_terms": 2,
        "session_terms": [
            {"source": "local file", "blocks": 1, "terms": 2, "age_seconds": 0},
            {"source": "box-a", "blocks": 1, "terms": 2, "age_seconds": 0},
            {"source": "box-b", "blocks": 1, "terms": 1, "age_seconds": 0},
            {"source": "box-c", "blocks": 0, "terms": 0, "age_seconds": None},
        ],
    }
    for secret in ("local_a", "kbx25", "b_term", "a-s", "b-s", "local-s", "A tab", "/w"):
        assert secret not in response.text


async def test_the_request_log_line_counts_session_terms_and_names_none(tmp_path: Path) -> None:
    with capture_logs() as logs:
        async with _client(Xai(), tmp_path, sessions=_sessions(tmp_path, **_two_hosts())) as c:
            await _post(c)

    (line,) = [entry for entry in logs if entry["event"] == "serve.request"]
    assert line["session_terms"] == 5
    assert "local_a" not in repr(logs)
    assert "A tab" not in repr(logs)


async def test_the_poller_runs_from_startup_until_shutdown(tmp_path: Path) -> None:
    ssh = Ssh(_session_file(("s", NOW, ["polled_term"])))
    sessions = SessionTerms([remote_source("box-a", runner=ssh, interval=0.01)])
    app = create_app(
        dictation_client(KEY, transport=httpx.MockTransport(Xai())),
        _terms(tmp_path),
        session_terms=sessions,
        keep=None,
        flags={},
    )
    sent: list[Message] = []
    shut = False

    async def receive() -> Message:
        nonlocal shut
        if not sent:
            return {"type": "lifespan.startup"}
        assert await anyio.to_thread.run_sync(ssh.polled.wait, 2)
        shut = True
        return {"type": "lifespan.shutdown"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app({"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}, receive, send)

    assert shut
    assert [message["type"] for message in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]
    calls = len(ssh.calls)
    await anyio.sleep(0.1)
    assert len(ssh.calls) == calls


async def test_a_voiceink_upload_gets_xais_words_joined_by_single_spaces(tmp_path: Path) -> None:
    xai = Xai()
    async with _client(xai, tmp_path, terms="herdr\nVoiceInk\n") as client:
        response = await _post(client)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    words = cast("list[dict[str, object]]", xai_payload()["words"])
    assert response.json() == {"text": " ".join(str(word["text"]) for word in words)}
    assert len(xai.seen) == 1
    sent = _fields(xai.seen[0])
    assert [name for name, _ in sent] == [
        "model",
        "language",
        "format",
        "keyterm",
        "keyterm",
        "vad_threshold",
        "file",
    ]
    assert dict(sent)["language"] == b"en"
    assert [body for name, body in sent if name == "keyterm"] == [b"herdr", b"VoiceInk"]
    assert dict(sent)["vad_threshold"] == b"0"
    assert dict(sent)["file"] == AUDIO


async def test_the_requests_language_reaches_xai(tmp_path: Path) -> None:
    xai = Xai()
    async with _client(xai, tmp_path) as client:
        await _post(client, data={**VOICEINK_FIELDS, "language": "de"})

    assert dict(_fields(xai.seen[0]))["language"] == b"de"


async def test_aliases_and_snapping_shape_the_delivered_text(tmp_path: Path) -> None:
    heard = ["ask", "herder", "about", "voice", "ink,", "kbx", "25."]
    payload = {
        "text": " ".join(heard),
        "duration": 2.5,
        "words": [
            {"text": text, "start": index, "end": index + 0.5} for index, text in enumerate(heard)
        ],
    }
    xai = Xai(httpx.Response(200, json=payload))
    terms = "herdr\nVoiceInk\nkbx25\nherder => herdr\n"
    async with _client(xai, tmp_path, terms=terms) as client:
        response = await _post(client)

    assert response.json() == {"text": "ask herdr about VoiceInk, kbx25."}
    (kept,) = _kept(tmp_path)
    result = _json(kept / "result.json")
    assert result["edits"] == [
        {"rule": "alias", "from": "herder", "to": "herdr"},
        {"rule": "snap", "from": "voice ink,", "to": "VoiceInk,"},
        {"rule": "snap", "from": "kbx 25.", "to": "kbx25."},
    ]
    assert result["aliases"] == [{"heard": "herder", "written": "herdr"}]


async def test_text_without_words_is_delivered_as_is(tmp_path: Path) -> None:
    xai = Xai(httpx.Response(200, json={"text": "just text", "words": []}))
    async with _client(xai, tmp_path) as client:
        response = await _post(client)

    assert response.json() == {"text": "just text"}


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://example.com"},
        {"Origin": "null"},
        {"Host": "evil.example:8765"},
        {"Host": "127.0.0.1.example"},
        {"Host": "127.0.0.1:8765@evil.example"},
        {"Host": "localhost:80:80"},
        {"Host": "[::1]x"},
    ],
)
async def test_a_cross_origin_or_foreign_host_request_is_refused_before_anything_runs(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    xai = Xai()
    async with _client(xai, tmp_path) as client:
        response = await _post(client, headers=headers)
        health = await client.get("/health", headers=headers)

    assert response.status_code == 403
    assert health.status_code == 403
    assert set(cast("dict[str, dict[str, str]]", response.json())["error"]) == {"message", "type"}
    assert xai.seen == []
    assert _kept(tmp_path) == []


@pytest.mark.parametrize("host", ["localhost:9000", "[::1]:8765", "127.0.0.1", "LOCALHOST"])
async def test_every_loopback_host_name_is_served(tmp_path: Path, host: str) -> None:
    async with _client(Xai(), tmp_path, host=host) as client:
        response = await _post(client)

    assert response.status_code == 200


async def test_an_upload_past_the_cap_is_413(tmp_path: Path) -> None:
    xai = Xai()
    async with _client(xai, tmp_path, max_bytes=1000) as client:
        declared = await _post(client, files={"file": ("a.wav", b"x" * 2000, "audio/wav")})

        async def chunks() -> AsyncIterator[bytes]:
            for _ in range(10):
                yield b"x" * 200

        streamed = await client.post(
            "/v1/audio/transcriptions",
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=b"},
        )

    assert declared.status_code == 413
    assert streamed.status_code == 413
    assert xai.seen == []


@pytest.mark.parametrize(
    ("data", "files", "word"),
    [
        (VOICEINK_FIELDS, {}, "file"),
        ({**VOICEINK_FIELDS, "response_format": "text"}, None, "response_format"),
        ({**VOICEINK_FIELDS, "response_format": "verbose_json"}, None, "response_format"),
    ],
)
async def test_a_bad_request_is_400_with_an_openai_error_body(
    tmp_path: Path,
    data: dict[str, str],
    files: dict[str, tuple[str, bytes, str]] | None,
    word: str,
) -> None:
    xai = Xai()
    async with _client(xai, tmp_path) as client:
        response = await _post(client, data=data, files=files)

    assert response.status_code == 400
    error = cast("dict[str, dict[str, str]]", response.json())["error"]
    assert word in error["message"]
    assert error["type"] == "invalid_request_error"
    assert xai.seen == []


async def test_fields_other_apps_send_are_accepted_or_ignored(tmp_path: Path) -> None:
    data = {
        "prompt": "their dictionary",
        "keywords[]": "x",
        "timestamp_granularities[]": "word",
        "stream": "false",
    }
    async with _client(Xai(), tmp_path) as client:
        response = await _post(client, data=data)

    assert response.status_code == 200


async def test_an_xai_outage_is_tried_twice_then_a_502_naming_the_cause(tmp_path: Path) -> None:
    xai = Xai(httpx.Response(503, text="upstream unavailable"))
    async with _client(xai, tmp_path) as client:
        response = await _post(client)

    assert response.status_code == 502
    assert len(xai.seen) == 2
    assert "503" in response.text
    assert KEY not in response.text
    assert "Traceback" not in response.text
    (kept,) = _kept(tmp_path)
    result = _json(kept / "result.json")
    assert result["outcome"] == "xai_failed"
    assert "503" in str(result["cause"])


async def test_a_request_xai_refuses_is_not_retried(tmp_path: Path) -> None:
    xai = Xai(httpx.Response(400, text="bad audio"))
    async with _client(xai, tmp_path) as client:
        response = await _post(client)

    assert response.status_code == 502
    assert len(xai.seen) == 1


async def test_a_malformed_xai_reply_is_a_502(tmp_path: Path) -> None:
    xai = Xai(httpx.Response(200, json={"words": "nope"}))
    async with _client(xai, tmp_path) as client:
        response = await _post(client)

    assert response.status_code == 502
    assert "malformed" in response.text


async def test_each_dictation_is_kept_in_a_private_directory(tmp_path: Path) -> None:
    async with _client(Xai(), tmp_path, terms="herdr\nherder => herdr\n") as client:
        response = await _post(client)

    (kept,) = _kept(tmp_path)
    assert KEPT_NAME.fullmatch(kept.name)
    assert stat.S_IMODE(kept.stat().st_mode) == 0o700
    assert stat.S_IMODE(kept.parent.stat().st_mode) == 0o700
    assert sorted(path.name for path in kept.iterdir()) == ["audio.wav", "result.json", "xai.json"]
    assert (kept / "audio.wav").read_bytes() == AUDIO
    assert _json(kept / "xai.json") == xai_payload()
    result = _json(kept / "result.json")
    assert result["text"] == cast("dict[str, str]", response.json())["text"]
    assert result["xai_text"] == xai_payload()["text"]
    assert result["outcome"] == "ok"
    assert result["terms"] == ["herdr"]
    assert result["request"] == {
        "model": "scribe",
        "language": None,
        "response_format": "json",
        "filename": "rec-1.wav",
        "content_type": "audio/wav",
        "bytes": len(AUDIO),
    }
    assert set(result) >= {
        "schema_version",
        "received_at",
        "latency_ms",
        "scribe_version",
        "flags",
        "cause",
    }
    assert KEY not in (kept / "result.json").read_text(encoding="utf-8")


async def test_only_the_newest_1000_dictations_are_kept(tmp_path: Path) -> None:
    root = tmp_path / "keep"
    for index in range(1000):
        (root / f"20200101T{index // 60:02d}{index % 60:02d}00Z-000000").mkdir(parents=True)
    oldest = min(root.iterdir())

    async with _client(Xai(), tmp_path) as client:
        await _post(client)

    assert len(_kept(tmp_path)) == 1000
    assert not oldest.exists()


async def test_no_keep_writes_nothing(tmp_path: Path) -> None:
    async with _client(Xai(), tmp_path, keep=False) as client:
        response = await _post(client)

    assert response.status_code == 200
    assert _kept(tmp_path) == []


async def test_a_keep_failure_logs_one_line_and_never_fails_the_response(tmp_path: Path) -> None:
    (tmp_path / "keep").write_text("a file where the directory should be", encoding="utf-8")
    with capture_logs() as logs:
        async with _client(Xai(), tmp_path) as client:
            response = await _post(client)

    assert response.status_code == 200
    assert [entry["event"] for entry in logs if entry["log_level"] == "error"] == [
        "serve.keep_failed"
    ]


async def test_one_log_line_per_request_holds_no_text_and_no_key(tmp_path: Path) -> None:
    with capture_logs() as logs:
        async with _client(Xai(), tmp_path, terms="herdr\n") as client:
            await _post(client)

    (line,) = [entry for entry in logs if entry["event"] == "serve.request"]
    assert line["status"] == 200
    assert line["outcome"] == "ok"
    assert line["audio_seconds"] == xai_payload()["duration"]
    assert line["terms"] == 1
    assert line["edits"] == 0
    assert {"total_ms", "xai_ms"} <= set(line)
    rendered = repr(logs)
    assert KEY not in rendered
    assert "beginning" not in rendered


async def test_an_unknown_path_gets_an_openai_error_body(tmp_path: Path) -> None:
    async with _client(Xai(), tmp_path) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 404
    assert set(cast("dict[str, dict[str, str]]", response.json())["error"]) == {"message", "type"}


async def test_one_client_serves_every_dictation_with_an_8_s_budget(tmp_path: Path) -> None:
    xai = Xai()
    closed: list[bool] = []

    class Transport(httpx.MockTransport):
        @override
        def close(self) -> None:
            closed.append(True)

    app = create_app(
        dictation_client(KEY, transport=Transport(xai)),
        _terms(tmp_path),
        session_terms=NO_SESSIONS,
        keep=None,
        flags={},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as client:
        await _post(client)
        await _post(client)

    assert len(xai.seen) == 2
    assert closed == []
    timeouts: object = xai.seen[0].extensions["timeout"]  # pyright: ignore[reportAny]  # httpx types extensions as Any
    assert isinstance(timeouts, dict)
    assert 0 < cast("dict[str, float]", timeouts)["read"] <= 8


async def test_a_garbled_multipart_body_is_400_with_an_openai_error_body(tmp_path: Path) -> None:
    xai = Xai()
    async with _client(xai, tmp_path) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            content=b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n',
            headers={"content-type": "multipart/form-data; boundary=b"},
        )

    assert response.status_code == 400
    assert set(cast("dict[str, dict[str, str]]", response.json())["error"]) == {"message", "type"}
    assert xai.seen == []


@pytest.fixture
def workdirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the server's temporary directories at one the test can inspect."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(work))
    return work


async def test_an_internal_error_is_a_500_that_is_kept_and_leaves_no_temporary_audio(
    tmp_path: Path, workdirs: Path
) -> None:
    lone_surrogate = (
        b'{"text": "x", "duration": 1.0, "words": [{"text": "\\ud800", "start": 0, "end": 1}]}'
    )
    reply = httpx.Response(
        200, content=lone_surrogate, headers={"content-type": "application/json"}
    )
    async with _client(Xai(reply), tmp_path) as client:
        response = await _post(client)

    assert response.status_code == 500
    assert set(cast("dict[str, dict[str, str]]", response.json())["error"]) == {"message", "type"}
    assert [entry async for entry in anyio.Path(workdirs).iterdir()] == []
    (kept,) = _kept(tmp_path)
    assert (kept / "audio.wav").read_bytes() == AUDIO
    assert _json(kept / "result.json")["outcome"] == "error"


async def test_audio_that_cannot_be_sent_is_a_502_naming_no_path_and_is_still_kept(
    tmp_path: Path, workdirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def lost(_path: anyio.Path, _data: bytes) -> int:
        return 0

    monkeypatch.setattr(anyio.Path, "write_bytes", lost)
    async with _client(Xai(), tmp_path) as client:
        response = await _post(client)

    assert response.status_code == 502
    assert str(workdirs) not in response.text
    (kept,) = _kept(tmp_path)
    assert (kept / "audio.wav").read_bytes() == AUDIO
    assert "cannot read audio file" in str(_json(kept / "result.json")["cause"])


async def test_pruning_skips_a_link_or_file_named_like_a_kept_dictation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scribe.serve.KEEP_NEWEST", 3)
    root = tmp_path / "keep"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "20000101T000001Z-aaaaaa"
    link.symlink_to(outside)
    plain = root / "20000101T000002Z-bbbbbb"
    plain.write_text("not a directory", encoding="utf-8")

    with capture_logs() as logs:
        async with _client(Xai(), tmp_path) as client:
            for _ in range(5):
                await _post(client)

    assert len([path for path in root.iterdir() if path.is_dir() and not path.is_symlink()]) == 3
    assert link.is_symlink()
    assert outside.is_dir()
    assert plain.is_file()
    assert [entry for entry in logs if entry["log_level"] == "error"] == []


async def test_a_keep_root_that_already_exists_is_made_private(tmp_path: Path) -> None:
    (tmp_path / "keep").mkdir(mode=0o755)

    async with _client(Xai(), tmp_path) as client:
        await _post(client)

    assert stat.S_IMODE((tmp_path / "keep").stat().st_mode) == 0o700


async def test_the_xai_client_is_closed_when_the_server_shuts_down(tmp_path: Path) -> None:
    closed: list[bool] = []

    class Transport(httpx.MockTransport):
        @override
        def close(self) -> None:
            closed.append(True)

    app = create_app(
        dictation_client(KEY, transport=Transport(Xai())),
        _terms(tmp_path),
        session_terms=NO_SESSIONS,
        keep=None,
        flags={},
    )
    events: list[Message] = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent: list[Message] = []

    async def receive() -> Message:
        return events.pop(0)

    async def send(message: Message) -> None:
        sent.append(message)

    await app({"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}, receive, send)

    assert [message["type"] for message in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]
    assert closed == [True]


async def test_a_client_that_hangs_up_mid_upload_is_one_log_line(tmp_path: Path) -> None:
    xai = Xai()
    app = create_app(
        dictation_client(KEY, transport=httpx.MockTransport(xai)),
        _terms(tmp_path),
        session_terms=NO_SESSIONS,
        keep=tmp_path / "keep",
        flags={},
    )
    events: list[Message] = [
        {"type": "http.request", "body": b"--b\r\n", "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive() -> Message:
        return events.pop(0) if events else {"type": "http.disconnect"}

    async def send(_message: Message) -> None:
        return None

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": ENDPOINT,
        "raw_path": ENDPOINT.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"content-type", b"multipart/form-data; boundary=b"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
    }
    with capture_logs() as logs:
        await app(scope, receive, send)

    assert [entry["outcome"] for entry in logs] == ["disconnected"]
    assert xai.seen == []
    assert _kept(tmp_path) == []


async def test_a_slowly_trickled_xai_reply_ends_in_a_502_within_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scribe.serve.XAI_DEADLINE_SECONDS", 0.3)
    release = threading.Event()

    class Trickle(httpx.SyncByteStream):
        @override
        def __iter__(self) -> Iterator[bytes]:
            yield b'{"text": "late", '
            release.wait(3)
            yield b'"words": []}'

    reply = httpx.Response(200, headers={"content-type": "application/json"}, stream=Trickle())
    started = time.monotonic()
    try:
        async with _client(Xai(reply), tmp_path) as client:
            response = await _post(client)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert response.status_code == 502
    assert elapsed < 2
    (kept,) = _kept(tmp_path)
    assert _json(kept / "result.json")["outcome"] == "xai_failed"


async def test_missed_deadlines_leave_no_worker_reading_from_xai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scribe.serve.XAI_DEADLINE_SECONDS", 0.3)
    release = threading.Event()
    opened: list[bool] = []
    closed: list[bool] = []

    class Trickle(httpx.SyncByteStream):
        """A reply that sends a byte often enough to beat every read timeout."""

        @override
        def __iter__(self) -> Iterator[bytes]:
            while not release.wait(0.02):
                yield b" "

        @override
        def close(self) -> None:
            closed.append(True)

    def trickling(request: httpx.Request) -> httpx.Response:
        request.read()
        opened.append(True)
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=Trickle())

    app = create_app(
        dictation_client(KEY, transport=httpx.MockTransport(trickling)),
        _terms(tmp_path),
        session_terms=NO_SESSIONS,
        keep=None,
        flags={},
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client:
            statuses = [(await _post(client)).status_code for _ in range(4)]
        # A worker may outlive its deadline by one read timeout, which is at most the deadline.
        await anyio.sleep(0.3 + 0.3 + 0.2)
        live = len(opened) - len(closed)
    finally:
        release.set()

    assert statuses == [502] * 4
    assert len(opened) == 4
    assert live == 0


class _Wire(asyncio.Transport):
    """An in-memory connection to uvicorn's HTTP protocol: no socket is opened."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    @override
    def get_extra_info(self, name: str, default: object = None) -> object:
        return {"sockname": ("127.0.0.1", 8765), "peername": ("127.0.0.1", 50000)}.get(
            name, default
        )

    @override
    def write(self, data: bytes | bytearray | memoryview) -> None:
        return None

    @override
    def is_closing(self) -> bool:
        return self.closed

    @override
    def close(self) -> None:
        self.closed = True

    @override
    def pause_reading(self) -> None:
        return None

    @override
    def resume_reading(self) -> None:
        return None


async def test_a_stalled_upload_does_not_hold_off_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scribe.serve.SHUTDOWN_GRACE_SECONDS", 0.2)
    closed: list[bool] = []

    class Transport(httpx.MockTransport):
        @override
        def close(self) -> None:
            closed.append(True)

    held = dictation_client(KEY, transport=Transport(Xai()))

    def holding(_key: str) -> XaiStt:
        return held

    monkeypatch.setattr("scribe.serve.dictation_client", holding)
    started: list[tuple[ASGIApp, dict[str, object]]] = []

    def record(app: ASGIApp, **settings: object) -> None:
        started.append((app, settings))

    monkeypatch.setattr("uvicorn.run", record)
    run(
        api_key=KEY,
        terms=_terms(tmp_path),
        session_terms=NO_SESSIONS,
        keep=None,
        host="127.0.0.1",
        port=8765,
    )
    ((app, settings),) = started

    config = uvicorn.Config(app, **settings, log_config=None)  # pyright: ignore[reportArgumentType]  # the settings run passed
    config.load()
    server = uvicorn.Server(config)
    server.servers = []
    lifespan = LifespanOn(config)
    server.lifespan = lifespan
    await lifespan.startup()
    protocol = H11Protocol(config, server.server_state, lifespan.state)
    wire = _Wire()
    protocol.connection_made(wire)
    protocol.data_received(
        b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1:8765\r\n"
        b"Content-Type: multipart/form-data; boundary=b\r\nContent-Length: 1000\r\n\r\n--b\r\n"
    )
    await anyio.sleep(0.05)
    assert len(server.server_state.tasks) == 1

    with anyio.fail_after(3):
        await server.shutdown()

    assert closed == [True]
