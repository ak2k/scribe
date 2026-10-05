from __future__ import annotations

import json
import re
import stat
from typing import TYPE_CHECKING, cast, override

import httpx
import pytest
import stamina
from structlog.testing import capture_logs

from scribe.serve import create_app, dictation_client
from scribe.vocab import TermsFile
from tests.xai_fixtures import xai_payload

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path

KEY = "xai-test-key-never-logged"
AUDIO = b"RIFF-pretend-wav-bytes"
VOICEINK_FIELDS = {"model": "scribe", "response_format": "json", "temperature": "0"}
KEPT_NAME = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{6}")
pytestmark = pytest.mark.anyio


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
) -> httpx.AsyncClient:
    app = create_app(
        dictation_client(KEY, transport=httpx.MockTransport(xai)),
        _terms(tmp_path, terms),
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
    heard = ["ask", "herder", "about", "voice", "ink,", "akms", "25."]
    payload = {
        "text": " ".join(heard),
        "duration": 2.5,
        "words": [
            {"text": text, "start": index, "end": index + 0.5} for index, text in enumerate(heard)
        ],
    }
    xai = Xai(httpx.Response(200, json=payload))
    terms = "herdr\nVoiceInk\nakms25\nherder => herdr\n"
    async with _client(xai, tmp_path, terms=terms) as client:
        response = await _post(client)

    assert response.json() == {"text": "ask herdr about VoiceInk, akms25."}
    (kept,) = _kept(tmp_path)
    result = _json(kept / "result.json")
    assert result["edits"] == [
        {"rule": "alias", "from": "herder", "to": "herdr"},
        {"rule": "snap", "from": "voice ink,", "to": "VoiceInk,"},
        {"rule": "snap", "from": "akms 25.", "to": "akms25."},
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


async def test_a_408_from_xai_is_not_retried_by_a_dictation(tmp_path: Path) -> None:
    xai = Xai(httpx.Response(408, text="request timeout"))
    async with _client(xai, tmp_path) as client:
        response = await _post(client)

    assert response.status_code == 502
    assert len(xai.seen) == 1


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
