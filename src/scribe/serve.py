"""`scribe serve`: a local OpenAI-compatible transcription endpoint for dictation apps."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

import anyio
import anyio.to_thread
import structlog
import uvicorn
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.datastructures import Headers, UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from scribe import __version__
from scribe.errors import AppError
from scribe.schema import XaiResponse
from scribe.vocab import deliver
from scribe.xai_stt import XaiStt

if TYPE_CHECKING:
    import httpx
    from starlette.datastructures import FormData
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
    from structlog.stdlib import BoundLogger

    from scribe.vocab import TermsFile, Vocab

ENDPOINT = "/v1/audio/transcriptions"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
KEEP_NEWEST = 1000
# A dictation is waited on by someone at a keyboard: two quick tries, never minutes.
XAI_ATTEMPTS = 2
XAI_DEADLINE_SECONDS = 8.0
XAI_RETRY_STATUSES = frozenset({429, *range(500, 600)})
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_KEPT_NAME = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{6}")
_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,10}")

Flags = dict[str, str | int | None]


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


def is_loopback(host: str) -> bool:
    """True for a loopback host name, without brackets or port."""
    return host.lower() in LOOPBACK_NAMES


def dictation_client(api_key: str, *, transport: httpx.BaseTransport | None = None) -> XaiStt:
    """Return the xAI client a server holds for its lifetime, tuned for dictation."""
    return XaiStt(
        api_key,
        transport=transport,
        timeout_seconds=XAI_DEADLINE_SECONDS,
        attempts=XAI_ATTEMPTS,
        retry_statuses=XAI_RETRY_STATUSES,
        deadline_seconds=XAI_DEADLINE_SECONDS,
        keep_alive=True,
    )


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequestFields(_Record):
    """What a dictation request carried, apart from its audio."""

    model: str | None
    language: str | None
    response_format: str | None
    filename: str | None
    content_type: str | None
    bytes: int


class KeepRecord(_Record):
    """`result.json` in a kept dictation's directory."""

    schema_version: Literal[1] = 1
    received_at: str
    request: RequestFields
    text: str | None
    xai_text: str | None
    terms: list[str]
    aliases: list[dict[str, str]]
    edits: list[dict[str, str]]
    latency_ms: dict[str, int]
    outcome: Literal["ok", "xai_failed"]
    cause: str | None
    scribe_version: str
    flags: Flags


@dataclass(frozen=True)
class _Dictation:
    payload: dict[str, object] | None
    text: str | None
    xai_text: str | None
    duration: float | None
    edits: list[dict[str, str]]
    cause: str | None
    xai_ms: int


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def _host_name(host: str) -> str:
    if host.startswith("["):
        return host[1:].partition("]")[0]
    return host.partition(":")[0]


class _LocalOnly:
    """Refuse a request a web page could have sent, before anything else sees it."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            # A page can POST multipart cross-origin without a preflight, and DNS
            # rebinding reaches loopback under a foreign Host.
            if "origin" in headers or not is_loopback(_host_name(headers.get("host", ""))):
                _logger().info("serve.request", status=403, outcome="refused")
                response = _error(403, "only local, non-browser clients are served", "forbidden")
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _TooLargeError(AppError):
    """The request body passed the upload cap."""


async def _read_capped(request: Request, max_bytes: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise _TooLargeError
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > max_bytes:
            raise _TooLargeError
    return bytes(body)


async def _form(scope: Scope, body: bytes) -> FormData:
    async def replay() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    return await Request(scope, replay).form()


def _field(form: FormData, name: str) -> str | None:
    value = form.get(name)
    return value if isinstance(value, str) else None


def _dictate(stt: XaiStt, audio: Path, language: str, vocab: Vocab) -> _Dictation:
    started = time.monotonic()
    payload: dict[str, object] | None = None
    try:
        payload = stt.transcribe(
            audio,
            language=language,
            format_text=True,
            diarize=False,
            keyterms=vocab.terms,
            vad_threshold=0.0,
        )
        parsed = XaiResponse.model_validate(payload)
    except (AppError, ValidationError) as exc:
        cause = str(exc) if isinstance(exc, AppError) else "malformed xAI transcription response"
        xai_ms = round((time.monotonic() - started) * 1000)
        return _Dictation(payload, None, None, None, [], " ".join(cause.split()), xai_ms)
    xai_ms = round((time.monotonic() - started) * 1000)
    words, edits = deliver([word.text for word in parsed.words], vocab)
    text = " ".join(words) if parsed.words else parsed.text
    edited = [{"rule": edit.rule, "from": edit.before, "to": edit.after} for edit in edits]
    return _Dictation(payload, text, parsed.text, parsed.duration, edited, None, xai_ms)


def _keep(root: Path, record: KeepRecord, audio: Path, payload: dict[str, object] | None) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(record.received_at).strftime("%Y%m%dT%H%M%SZ")
    target = root / f"{stamp}-{secrets.token_hex(3)}"
    target.mkdir(mode=0o700)
    shutil.move(audio, target / audio.name)
    if payload is not None:
        (target / "xai.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (target / "result.json").write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    kept = sorted(path for path in root.iterdir() if _KEPT_NAME.fullmatch(path.name))
    for old in kept[:-KEEP_NEWEST]:
        shutil.rmtree(old)


class _Server:
    def __init__(
        self, stt: XaiStt, terms: TermsFile, keep: Path | None, flags: Flags, max_bytes: int
    ) -> None:
        self.stt = stt
        self.terms = terms
        self.keep = keep
        self.flags = flags
        self.max_bytes = max_bytes

    async def health(self, _request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def transcriptions(self, request: Request) -> Response:
        started = time.monotonic()
        received_at = datetime.now(UTC)
        try:
            form = await _form(request.scope, await _read_capped(request, self.max_bytes))
        except _TooLargeError:
            return self._refuse(413, f"the upload is over {self.max_bytes} bytes")
        try:
            parsed = await self._parse(form)
        finally:
            await form.close()
        if isinstance(parsed, Response):
            return parsed
        fields, audio = parsed
        vocab = self.terms.current()
        workdir = Path(await anyio.to_thread.run_sync(tempfile.mkdtemp))
        suffix = PurePosixPath(fields.filename or "").suffix
        path = workdir / f"audio{suffix if _SUFFIX.fullmatch(suffix) else ''}"
        await anyio.Path(path).write_bytes(audio)
        result = await anyio.to_thread.run_sync(
            _dictate, self.stt, path, fields.language or "en", vocab
        )
        total_ms = round((time.monotonic() - started) * 1000)
        if result.text is None:
            response = _error(502, result.cause or "xAI failed", "upstream_error")
        else:
            response = JSONResponse({"text": result.text})
        _logger().info(
            "serve.request",
            status=response.status_code,
            outcome="ok" if result.cause is None else "xai_failed",
            audio_seconds=result.duration,
            total_ms=total_ms,
            xai_ms=result.xai_ms,
            terms=len(vocab.terms),
            edits=len(result.edits),
        )
        record = KeepRecord(
            received_at=received_at.isoformat(),
            request=fields,
            text=result.text,
            xai_text=result.xai_text,
            terms=list(vocab.terms),
            aliases=[{"heard": alias.heard, "written": alias.written} for alias in vocab.aliases],
            edits=result.edits,
            latency_ms={"total": total_ms, "xai": result.xai_ms},
            outcome="ok" if result.cause is None else "xai_failed",
            cause=result.cause,
            scribe_version=__version__,
            flags=self.flags,
        )
        response.background = BackgroundTask(self._after, record, workdir, path, result.payload)
        return response

    async def _parse(self, form: FormData) -> Response | tuple[RequestFields, bytes]:
        # DIVERGE: unknown fields are ignored, not refused: client apps add their own.
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            return self._refuse(400, "the `file` field is required")
        response_format = _field(form, "response_format")
        if response_format not in {None, "json"}:
            return self._refuse(400, "only response_format=json is supported")
        audio = await upload.read()
        fields = RequestFields(
            model=_field(form, "model"),
            language=_field(form, "language") or None,
            response_format=response_format,
            filename=upload.filename,
            content_type=upload.content_type,
            bytes=len(audio),
        )
        return fields, audio

    def _refuse(self, status: int, message: str) -> Response:
        _logger().info("serve.request", status=status, outcome="refused")
        return _error(status, message, "invalid_request_error")

    def _after(
        self, record: KeepRecord, workdir: Path, audio: Path, payload: dict[str, object] | None
    ) -> None:
        """Keep the dictation once its response is sent; a failure here is only logged."""
        try:
            if self.keep is not None:
                _keep(self.keep, record, audio, payload)
        except OSError as exc:
            _logger().error("serve.keep_failed", error=str(exc))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


async def _http_error(_request: Request, exc: Exception) -> Response:
    status = exc.status_code if isinstance(exc, HTTPException) else 500
    message = exc.detail if isinstance(exc, HTTPException) else "internal error"
    return _error(status, message, "invalid_request_error" if status < 500 else "server_error")  # noqa: PLR2004  # 5xx is the server's fault


def create_app(
    stt: XaiStt,
    terms: TermsFile,
    *,
    keep: Path | None,
    flags: Flags,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Starlette:
    """Build the ASGI app.

    Args:
        stt: The xAI client, held for the app's lifetime.
        terms: The terms file, re-read when it changes.
        keep: Where dictations are kept, or None to keep nothing.
        flags: The server's settings, recorded with each kept dictation.
        max_bytes: Upload cap, counted as the body streams in.

    """
    server = _Server(stt, terms, keep, flags, max_bytes)
    return Starlette(
        routes=[
            Route(ENDPOINT, server.transcriptions, methods=["POST"]),
            Route("/health", server.health, methods=["GET"]),
        ],
        middleware=[Middleware(_LocalOnly)],
        exception_handlers={HTTPException: _http_error, Exception: _http_error},
    )


def run(*, api_key: str, terms: TermsFile, keep: Path | None, host: str, port: int) -> None:
    """Serve until interrupted."""
    stt = dictation_client(api_key)
    flags: Flags = {
        "host": host,
        "port": port,
        "terms": str(terms.path),
        "keep": None if keep is None else str(keep),
    }
    shown = f"[{host}]" if ":" in host else host
    _logger().info("serve.listening", voiceink_endpoint=f"http://{shown}:{port}{ENDPOINT}")
    try:
        uvicorn.run(
            create_app(stt, terms, keep=keep, flags=flags),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
    finally:
        stt.close()
