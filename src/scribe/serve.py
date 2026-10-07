"""`scribe serve`: a local OpenAI-compatible transcription endpoint for dictation apps."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
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
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from scribe import __version__
from scribe.errors import AppError, InputValidationError
from scribe.schema import XaiResponse
from scribe.vocab import deliver
from scribe.xai_stt import XaiStt

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import httpx
    from starlette.datastructures import FormData
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
    from structlog.stdlib import BoundLogger

    from scribe.session_sources import SessionTerms
    from scribe.vocab import TermsFile, Vocab

ENDPOINT = "/v1/audio/transcriptions"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
KEEP_NEWEST = 1000
# A dictation is waited on by someone at a keyboard: two quick tries, never minutes.
XAI_ATTEMPTS = 2
XAI_DEADLINE_SECONDS = 8.0
# Past the xAI budget plus cleanup; a client that stalls mid-upload is cut off then
# rather than holding off a stop forever.
SHUTDOWN_GRACE_SECONDS = 10
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
# A Host header: a name, or a bracketed IPv6 address, then an optional port.
_HOST = re.compile(r"(?:\[(?P<bracketed>[^\]]*)\]|(?P<name>[^:]*))(?::\d*)?")
_KEPT_NAME = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{6}")
_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,10}")

Flags = dict[str, str | int | list[str] | None]


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


class FocusRecord(_Record):
    """What the focused tab did for a kept dictation; nothing from the tab itself."""

    verdict: str
    terms: int


class KeepRecord(_Record):
    """`result.json` in a kept dictation's directory."""

    schema_version: Literal[1] = 1
    received_at: str
    request: RequestFields
    text: str | None
    xai_text: str | None
    terms: list[str]
    session_terms: dict[str, int]
    focus: FocusRecord
    aliases: list[dict[str, str]]
    edits: list[dict[str, str]]
    latency_ms: dict[str, int]
    outcome: Literal["ok", "xai_failed", "error"]
    cause: str | None
    scribe_version: str
    flags: Flags


@dataclass(frozen=True)
class _Dictation:
    """What one dictation came to; `cause` is recorded, `message` is what the client sees."""

    outcome: Literal["ok", "xai_failed", "error"]
    xai_ms: int
    payload: dict[str, object] | None = None
    text: str | None = None
    xai_text: str | None = None
    duration: float | None = None
    edits: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    cause: str | None = None
    message: str | None = None


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def _host_name(host: str) -> str:
    """The name in a Host header, or "" if the header is more than a name and a port."""
    match = _HOST.fullmatch(host)
    if match is None:
        return ""
    return match["bracketed"] or match["name"] or ""


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


def _ms_since(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


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
        # Its text names the temporary copy of the audio, which is no use to the client.
        message = (
            "the audio could not be sent to xAI" if isinstance(exc, InputValidationError) else cause
        )
        return _Dictation(
            "xai_failed",
            _ms_since(started),
            payload,
            cause=" ".join(cause.split()),
            message=message,
        )
    words, edits = deliver([word.text for word in parsed.words], vocab)
    return _Dictation(
        "ok",
        _ms_since(started),
        payload,
        text=" ".join(words) if parsed.words else parsed.text,
        xai_text=parsed.text,
        duration=parsed.duration,
        edits=[{"rule": edit.rule, "from": edit.before, "to": edit.after} for edit in edits],
    )


def _audio_name(fields: RequestFields) -> str:
    suffix = PurePosixPath(fields.filename or "").suffix
    return f"audio{suffix if _SUFFIX.fullmatch(suffix) else ''}"


def _keep(
    root: Path, record: KeepRecord, name: str, audio: bytes, payload: dict[str, object] | None
) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    # mkdir leaves the mode of a root that already exists as it was.
    root.chmod(0o700)
    stamp = datetime.fromisoformat(record.received_at).strftime("%Y%m%dT%H%M%SZ")
    target = root / f"{stamp}-{secrets.token_hex(3)}"
    target.mkdir(mode=0o700)
    (target / name).write_bytes(audio)
    if payload is not None:
        (target / "xai.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (target / "result.json").write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _prune(root)


def _prune(root: Path) -> None:
    """Remove all but the newest kept dictations; only real directories count as kept."""
    kept = sorted(
        path
        for path in root.iterdir()
        if _KEPT_NAME.fullmatch(path.name) and path.is_dir() and not path.is_symlink()
    )
    for old in kept[:-KEEP_NEWEST]:
        # Another request's prune may have removed it first.
        with suppress(FileNotFoundError):
            shutil.rmtree(old)


class _Server:
    def __init__(
        self,
        stt: XaiStt,
        terms: TermsFile,
        sessions: SessionTerms,
        keep: Path | None,
        flags: Flags,
        max_bytes: int,
    ) -> None:
        self.stt = stt
        self.terms = terms
        self.sessions = sessions
        self.keep = keep
        self.flags = flags
        self.max_bytes = max_bytes

    async def health(self, _request: Request) -> Response:
        return JSONResponse(
            {
                "status": "ok",
                "static_terms": len(self.terms.current().terms),
                "session_terms": self.sessions.health(),
                "focus": self.sessions.focus_health(),
            }
        )

    async def transcriptions(self, request: Request) -> Response:
        started = time.monotonic()
        received_at = datetime.now(UTC)
        try:
            form = await _form(request.scope, await _read_capped(request, self.max_bytes))
        except _TooLargeError:
            return self._refuse(413, f"the upload is over {self.max_bytes} bytes")
        except ClientDisconnect:
            _logger().info("serve.request", status=499, outcome="disconnected")
            return Response(status_code=499)
        try:
            parsed = await self._parse(form)
        finally:
            await form.close()
        if isinstance(parsed, Response):
            return parsed
        fields, audio = parsed
        static = self.terms.current()
        vocab, session_counts, focus = await self.sessions.vocab(static)
        workdir = Path(await anyio.to_thread.run_sync(tempfile.mkdtemp))
        dictated: _Dictation | None = None
        try:
            dictated = await self._xai(workdir / _audio_name(fields), fields, audio, vocab)
            if dictated.text is None:
                response = _error(502, dictated.message or "xAI failed", "upstream_error")
            else:
                response = JSONResponse({"text": dictated.text})
            result = dictated
        # Whatever fails, the client gets an OpenAI-shaped answer and the dictation is kept.
        except Exception as exc:  # noqa: BLE001  # logged by type; the record says what was lost
            _logger().error("serve.internal_error", error=type(exc).__name__)
            response = _error(500, "internal error", "server_error")
            result = _Dictation(
                "error",
                0 if dictated is None else dictated.xai_ms,
                None if dictated is None else dictated.payload,
                cause=f"internal error: {type(exc).__name__}",
            )
        total_ms = _ms_since(started)
        _logger().info(
            "serve.request",
            status=response.status_code,
            outcome=result.outcome,
            audio_seconds=result.duration,
            total_ms=total_ms,
            xai_ms=result.xai_ms,
            terms=len(vocab.terms),
            session_terms=sum(session_counts.values()),
            focus=focus.verdict,
            focus_terms=focus.terms,
            edits=len(result.edits),
        )
        record = KeepRecord(
            received_at=received_at.isoformat(),
            request=fields,
            text=result.text,
            xai_text=result.xai_text,
            # On disk a session term would outlive its window, so only the counts are kept.
            terms=list(static.terms),
            session_terms=session_counts,
            focus=FocusRecord(verdict=focus.verdict, terms=focus.terms),
            aliases=[{"heard": alias.heard, "written": alias.written} for alias in vocab.aliases],
            edits=result.edits,
            latency_ms={"total": total_ms, "xai": result.xai_ms},
            outcome=result.outcome,
            cause=result.cause,
            scribe_version=__version__,
            flags=self.flags,
        )
        response.background = BackgroundTask(
            self._after, record, workdir, _audio_name(fields), audio, result.payload
        )
        return response

    async def _xai(
        self, path: Path, fields: RequestFields, audio: bytes, vocab: Vocab
    ) -> _Dictation:
        """Send the audio to xAI, giving up once `XAI_DEADLINE_SECONDS` of wall time pass."""
        await anyio.Path(path).write_bytes(audio)
        started = time.monotonic()
        with anyio.move_on_after(XAI_DEADLINE_SECONDS):
            # Abandoned at the deadline: the thread's own deadline, which starts just
            # after this one, ends its request, and what it returns is dropped.
            return await anyio.to_thread.run_sync(
                _dictate, self.stt, path, fields.language or "en", vocab, abandon_on_cancel=True
            )
        cause = f"xAI did not answer within {XAI_DEADLINE_SECONDS:g} s"
        return _Dictation("xai_failed", _ms_since(started), cause=cause, message=cause)

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
        self,
        record: KeepRecord,
        workdir: Path,
        name: str,
        audio: bytes,
        payload: dict[str, object] | None,
    ) -> None:
        """Keep the dictation once its response is sent; a failure here is only logged."""
        try:
            if self.keep is not None:
                _keep(self.keep, record, name, audio, payload)
        except Exception as exc:  # noqa: BLE001  # a keep never fails the request it records
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
    session_terms: SessionTerms,
    keep: Path | None,
    flags: Flags,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Starlette:
    """Build the ASGI app.

    Args:
        stt: The xAI client, held for the app's lifetime and closed at its shutdown.
        terms: The terms file, re-read when it changes.
        session_terms: The session term sources, polled while the app runs.
        keep: Where dictations are kept, or None to keep nothing.
        flags: The server's settings, recorded with each kept dictation.
        max_bytes: Upload cap, counted as the body streams in.

    """
    server = _Server(stt, terms, session_terms, keep, flags, max_bytes)

    # Closed on shutdown, where SIGTERM lands too: uvicorn re-raises the signal
    # after shutting down, so code after its `run` never sees a launchd stop.
    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(session_terms.poll)
                yield
                group.cancel_scope.cancel()
        finally:
            stt.close()

    return Starlette(
        routes=[
            Route(ENDPOINT, server.transcriptions, methods=["POST"]),
            Route("/health", server.health, methods=["GET"]),
        ],
        middleware=[Middleware(_LocalOnly)],
        exception_handlers={HTTPException: _http_error, Exception: _http_error},
        lifespan=lifespan,
    )


def run(
    *,
    api_key: str,
    terms: TermsFile,
    session_terms: SessionTerms,
    keep: Path | None,
    host: str,
    port: int,
) -> None:
    """Serve until interrupted."""
    stt = dictation_client(api_key)
    flags: Flags = {
        "host": host,
        "port": port,
        "terms": str(terms.path),
        "session_terms": [source.name for source in session_terms.sources],
        "focus": None if session_terms.focus is None else session_terms.focus.name,
        "keep": None if keep is None else str(keep),
    }
    shown = f"[{host}]" if ":" in host else host
    _logger().info("serve.listening", voiceink_endpoint=f"http://{shown}:{port}{ENDPOINT}")
    uvicorn.run(
        create_app(stt, terms, session_terms=session_terms, keep=keep, flags=flags),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS,
    )
