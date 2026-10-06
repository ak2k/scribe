"""Client for the xAI batch speech-to-text endpoint (`POST /v1/stt`)."""

from __future__ import annotations

import os
import re
import stat
import time
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, TYPE_CHECKING, cast, override

import httpcore
import httpx
import stamina

from scribe.errors import ExternalServiceError, InputValidationError

if TYPE_CHECKING:
    import ssl
    from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

DEFAULT_MODEL = "grok-voice-transcribe-2.0"
DEFAULT_BASE_URL = "https://api.x.ai/v1"
MAX_UPLOAD_BYTES = 500_000_000
RETRY_ATTEMPTS = 5
# The documented ceilings for the `keyterm` field.
MAX_KEYTERMS = 100
MAX_KEYTERM_CHARS = 50
# DIVERGE: the API's own default gate of 0.5 skipped whole stretches of clear
# speech, 82 s of it in one meeting, which a threshold of 0 transcribed in full.
DEFAULT_VAD_THRESHOLD = 0.0

_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# httpx's own default, for a client that lives for one call.
_PER_CALL_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
# A kept client's idle connection outlives the pause between two dictations.
KEPT_ALIVE_SECONDS = 300.0
_BODY_EXCERPT_CHARS = 300
_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_UPLOAD_NAME_CHARS = 96

# stamina's default instrumentation reports each scheduled retry through
# structlog, which prints to stdout until something configures it. The CLI owns
# stdout for its one machine-readable line, so retries stay silent here until a
# verbose flag gives them a home on stderr.
stamina.instrumentation.set_on_retry_hooks([])

# One multipart field as httpx encodes it: (name, (filename, content, content_type)).
# A scalar rides as (None, value, None) so httpx encodes the whole body as
# multipart/form-data rather than urlencoding the scalars.
_Part = tuple[str, tuple[str | None, str | IO[bytes], str | None]]


def resolve_api_key(env: Mapping[str, str] = os.environ) -> str:
    """Read the xAI API key from the environment.

    Args:
        env: Environment to read from; defaults to the process environment.

    Returns:
        The key, stripped of surrounding whitespace.

    Raises:
        InputValidationError: the variable is unset or empty.

    """
    # DIVERGE: a bare environment read, not a field on config.Settings — that
    # class pins env_prefix="SCRIBE_", which would rename the variable the
    # vendor documents.
    key = env.get("XAI_API_KEY", "").strip()
    if not key:
        raise InputValidationError("XAI_API_KEY is not set")
    return key


def _is_retryable(exc: Exception) -> bool:
    """True for a transient failure: rate limit, request timeout, 5xx, transport error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUSES
    # A 4xx other than 408/429 needs the call fixed, so retrying it only burns quota.
    return isinstance(exc, httpx.TransportError)


def _upload_name(path: Path) -> str:
    """Return an ASCII-only multipart filename for `path`."""
    # httpx's multipart encoder and the API's parser disagree on how brackets,
    # spaces and apostrophes are quoted, and the API answers that disagreement
    # with a 400; ASCII-only names leave nothing to disagree about.
    name = _SAFE_FILENAME_CHARS.sub("_", path.name)
    if len(name) > _MAX_UPLOAD_NAME_CHARS:
        # httpx guesses the part's media type from this name, so a cut keeps the suffix.
        suffix = _SAFE_FILENAME_CHARS.sub("_", path.suffix)[:_MAX_UPLOAD_NAME_CHARS]
        name = name[: _MAX_UPLOAD_NAME_CHARS - len(suffix)] + suffix
    return name or "audio"


def check_keyterms(keyterms: Sequence[str]) -> None:
    """Raise `InputValidationError` unless `keyterms` is within the documented limits.

    Public so a caller can refuse a bad list before it needs an API key.

    Args:
        keyterms: Terms to bias recognition toward.

    Raises:
        InputValidationError: there are more than `MAX_KEYTERMS` terms, or a term
            is blank, holds a control character such as a newline, or is longer
            than `MAX_KEYTERM_CHARS` characters.

    """
    if len(keyterms) > MAX_KEYTERMS:
        raise InputValidationError(
            f"{len(keyterms)} keyterms given; xAI accepts at most {MAX_KEYTERMS}"
        )
    for term in keyterms:
        if not term.strip():
            raise InputValidationError("a keyterm is empty")
        if any(unicodedata.category(char) == "Cc" for char in term):
            raise InputValidationError(f"keyterm {term!r} holds a control character")
        if len(term) > MAX_KEYTERM_CHARS:
            raise InputValidationError(
                f"keyterm {term!r} is {len(term)} characters; "
                f"xAI accepts at most {MAX_KEYTERM_CHARS} characters per term"
            )


def check_vad_threshold(threshold: float) -> None:
    """Raise `InputValidationError` unless `threshold` is a probability from 0 to 1.

    Public so a caller can refuse a bad value before it needs an API key.
    """
    # Negated, so NaN, which fails every comparison, is refused too.
    if not 0 <= threshold <= 1:
        raise InputValidationError(
            f"the VAD threshold must be a number from 0 to 1, not {threshold}"
        )


def _scalar_parts(
    *,
    model: str,
    language: str | None,
    format_text: bool,
    diarize: bool,
    keyterms: Sequence[str],
    vad_threshold: float,
) -> list[_Part]:
    """Build the non-file multipart fields, in the order they are sent."""
    # Only enabled flags are sent: `format` and `diarize` both default to false
    # upstream, and the API rejects `format` unless `language` accompanies it.
    parts: list[_Part] = [("model", (None, model, None))]
    if language:
        parts.append(("language", (None, language, None)))
        if format_text:
            parts.append(("format", (None, "true", None)))
    if diarize:
        parts.append(("diarize", (None, "true", None)))
    parts.extend(("keyterm", (None, term, None)) for term in keyterms)
    # Always sent: leaving it out means the API's default, not this one.
    parts.append(("vad_threshold", (None, f"{vad_threshold:g}", None)))
    return parts


def check_input(path: Path, max_bytes: int) -> None:
    """Raise `InputValidationError` unless `path` is a regular file within `max_bytes`.

    Public because a caller that touches the file itself must run this first: a
    FIFO or a character device blocks whoever opens it.

    Args:
        path: Audio file the caller intends to upload.
        max_bytes: Upload ceiling; the client passes its own `max_bytes`.

    Raises:
        InputValidationError: the path is unreadable, not a regular file, or
            larger than `max_bytes`.

    """
    try:
        info = path.stat()
    except OSError as exc:
        raise InputValidationError(f"cannot read audio file {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise InputValidationError(f"not an audio file: {path}")
    if info.st_size > max_bytes:
        raise InputValidationError(
            f"{path} is {info.st_size} bytes, over the {max_bytes}-byte upload limit; "
            "segmenting large files is not implemented yet"
        )


def _post(
    client: httpx.Client,
    path: Path,
    scalars: list[_Part],
    timeout_seconds: float,
    ends_at: float | None,
) -> httpx.Response:
    """POST one attempt at `/stt`, with `file` as the last multipart field."""
    # A handle per attempt: httpx encodes multipart lazily at send time and
    # rewinds only handles it can seek, so sharing one across retries would make
    # the retried upload depend on that rewind.
    try:
        with path.open("rb") as handle:
            parts: list[_Part] = [
                *scalars,
                ("file", (_upload_name(path), handle, None)),
            ]
            timeout = httpx.Timeout(timeout_seconds)
            if ends_at is None:
                response = client.post("/stt", files=parts, timeout=timeout)
            else:
                response = _post_until(client, parts, timeout, ends_at)
    except OSError as exc:
        raise InputValidationError(f"cannot read audio file {path}: {exc}") from exc
    response.raise_for_status()
    return response


class _Until(httpx.SyncByteStream):
    """A reply body that stops at the first chunk arriving past `ends_at`."""

    def __init__(self, inner: httpx.SyncByteStream, ends_at: float) -> None:
        self._inner = inner
        self._ends_at = ends_at

    @override
    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            if time.monotonic() > self._ends_at:
                raise ExternalServiceError("xAI did not finish its reply before the deadline")
            yield chunk

    @override
    def close(self) -> None:
        self._inner.close()


def _post_until(
    client: httpx.Client, parts: list[_Part], timeout: httpx.Timeout, ends_at: float
) -> httpx.Response:
    """POST, reading the reply chunk by chunk and closing it once `ends_at` passes."""
    # A read timeout bounds only each wait on the socket, so a reply trickled in
    # slowly would otherwise hold its thread and connection long past the deadline.
    with client.stream("POST", "/stt", files=parts, timeout=timeout) as response:
        # A sync client's reply body is always a sync stream.
        response.stream = _Until(cast("httpx.SyncByteStream", response.stream), ends_at)
        _ = response.read()
    return response


# Per thread, not per client: concurrent dictations share one kept-alive pool.
_ENDS_AT: ContextVar[float | None] = ContextVar("xai_ends_at", default=None)


def _time_left(timeout: float | None, expired: type[httpcore.TimeoutException]) -> float | None:
    """Cut one socket wait to what is left of the current call's deadline."""
    ends_at = _ENDS_AT.get()
    if ends_at is None:  # pragma: no cover  # every attempt sets it; a guard, not a path
        return timeout
    left = ends_at - time.monotonic()
    if left <= 0:
        raise expired("the dictation deadline passed")
    return left if timeout is None else min(timeout, left)


class _DeadlineStream(httpcore.NetworkStream):
    """A connection whose every wait ends by the deadline of the call using it."""

    def __init__(self, inner: httpcore.NetworkStream) -> None:
        self._inner = inner

    @override
    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._inner.read(max_bytes, _time_left(timeout, httpcore.ReadTimeout))

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._inner.write(buffer, _time_left(timeout, httpcore.WriteTimeout))

    @override
    def close(self) -> None:
        self._inner.close()

    @override
    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        timeout = _time_left(timeout, httpcore.ConnectTimeout)
        return _DeadlineStream(self._inner.start_tls(ssl_context, server_hostname, timeout))

    @override
    def get_extra_info(self, info: str) -> object:
        return self._inner.get_extra_info(info)  # pyright: ignore[reportAny]  # httpcore returns Any


class _DeadlineBackend(httpcore.NetworkBackend):
    def __init__(self, inner: httpcore.NetworkBackend) -> None:
        self._inner = inner

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        timeout = _time_left(timeout, httpcore.ConnectTimeout)
        return _DeadlineStream(
            self._inner.connect_tcp(host, port, timeout, local_address, socket_options)
        )


class _DeadlineTransport(httpx.HTTPTransport):
    """httpx's transport over connections that honor the calling dictation's deadline."""

    # httpx's timeouts bound each socket wait, and response headers trickled in
    # one byte at a time reset that wait before the reply exists to be closed.
    def __init__(self, limits: httpx.Limits, network_backend: httpcore.NetworkBackend) -> None:
        super().__init__(limits=limits)
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            network_backend=_DeadlineBackend(network_backend),
        )


def _status_error(response: httpx.Response) -> ExternalServiceError:
    """Wrap a non-retryable HTTP status as a domain error carrying a body excerpt."""
    excerpt = " ".join(response.text[:_BODY_EXCERPT_CHARS].split())
    return ExternalServiceError(
        f"xAI transcription failed with HTTP {response.status_code}: {excerpt}"
    )


def _decode(response: httpx.Response) -> dict[str, object]:
    """Decode a 2xx body as a JSON object."""
    try:
        payload: object = response.json()  # pyright: ignore[reportAny]  # httpx returns Any
    except ValueError as exc:
        raise ExternalServiceError(f"xAI transcription response was not JSON: {exc}") from exc
    if isinstance(payload, dict):
        return cast("dict[str, object]", payload)
    raise ExternalServiceError(
        f"xAI transcription response was a JSON {type(payload).__name__}, not an object"
    )


class XaiStt:
    """Batch transcription against `POST /v1/stt`.

    Sync on purpose: the only caller is a CLI, so there is no event loop to
    share (AGENTS.md principle 8 applies where async already exists).
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 600,
        max_bytes: int = MAX_UPLOAD_BYTES,
        attempts: int = RETRY_ATTEMPTS,
        deadline_seconds: float | None = None,
        keep_alive: bool = False,
        network_backend: httpcore.NetworkBackend | None = None,
    ) -> None:
        """Hold request settings; no connection is opened until `transcribe`.

        Args:
            api_key: xAI API key. Never logged, printed, or written to an artifact.
            base_url: API root, so a test or a proxy can point elsewhere.
            transport: httpx transport override; tests inject `httpx.MockTransport`.
            timeout_seconds: Per-request timeout, generous because a large
                upload is slow.
            max_bytes: Upload ceiling enforced before any request.
            attempts: Most attempts one `transcribe` makes.
            deadline_seconds: Seconds after which one `transcribe` starts no
                new attempt, even one a retry backoff scheduled. Every
                connect, write and read waits at most until it, so no request
                is in flight once it passes. None for no bound.
            keep_alive: Hold one client, and its connections, across calls
                until `close`, rather than one client per call.
            network_backend: Socket layer under a deadline, when `transport`
                is not given; tests inject an in-memory one.

        """
        self._api_key = api_key
        self._base_url = base_url
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes
        self._attempts = attempts
        self._deadline_seconds = deadline_seconds
        self._network_backend = network_backend or httpcore.SyncBackend()
        self._shared = (
            self._new_client(httpx.Limits(keepalive_expiry=KEPT_ALIVE_SECONDS))
            if keep_alive
            else None
        )

    def _new_client(self, limits: httpx.Limits = _PER_CALL_LIMITS) -> httpx.Client:
        transport = self._transport
        if transport is None and self._deadline_seconds is not None:
            transport = _DeadlineTransport(limits, self._network_backend)
        return httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(self._timeout_seconds),
            transport=transport,
            limits=limits,
        )

    @contextmanager
    def _client(self) -> Generator[httpx.Client]:
        if self._shared is not None:
            yield self._shared
            return
        with self._new_client() as client:
            yield client

    def close(self) -> None:
        """Close the kept-alive client, if there is one."""
        if self._shared is not None:
            self._shared.close()

    def _attempt_timeout(self, started: float) -> float:
        """Per-attempt timeout: the client's, cut to what is left of the deadline."""
        if self._deadline_seconds is None:
            return self._timeout_seconds
        left = self._deadline_seconds - (time.monotonic() - started)
        # Never zero, which httpx reads as an immediate timeout on connect.
        return max(min(self._timeout_seconds, left), 0.1)

    def transcribe(
        self,
        path: Path,
        *,
        model: str = DEFAULT_MODEL,
        language: str | None = "en",
        format_text: bool = True,
        diarize: bool = True,
        keyterms: Sequence[str] = (),
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    ) -> dict[str, object]:
        """Transcribe one audio file.

        Args:
            path: Audio file to upload.
            model: Transcription model; always sent explicitly, never defaulted
                server-side.
            language: Language code, or None to send none. Only gates the
                formatting of numbers and currency, not what is recognized.
            format_text: Inverse text normalization, writing "one hundred
                dollars" as "$100". Sent only when `language` is set, as the
                API requires.
            diarize: Ask for an integer speaker id per word.
            keyterms: Names and terms to bias recognition toward, each sent
                as its own `keyterm` field; none are sent by default.
            vad_threshold: Speech probability below which the API's
                voice-activity gate skips audio as silence; 0 turns the gate off.

        Returns:
            The decoded response body, for `schema.from_xai_response`.

        Raises:
            InputValidationError: the path is missing, unreadable, or too large,
                `keyterms` is past the documented limits, or `vad_threshold`
                is not from 0 to 1.
            ExternalServiceError: the request failed, or the body was not a
                JSON object.

        """
        check_input(path, self._max_bytes)
        check_keyterms(keyterms)
        check_vad_threshold(vad_threshold)
        scalars = _scalar_parts(
            model=model,
            language=language,
            format_text=format_text,
            diarize=diarize,
            keyterms=keyterms,
            vad_threshold=vad_threshold,
        )
        started = time.monotonic()
        try:
            with self._client() as client:
                response = self._post_with_retries(client, path, scalars, started)
        except httpx.HTTPStatusError as exc:
            raise _status_error(exc.response) from exc
        # RequestError, not TransportError: a mis-framed or mis-encoded body
        # raises DecodingError, which is a RequestError outside that subtree and
        # would otherwise leave httpx's own exception facing the caller.
        except httpx.RequestError as exc:
            raise ExternalServiceError(f"xAI transcription request failed: {exc}") from exc
        return _decode(response)

    def _post_with_retries(
        self, client: httpx.Client, path: Path, scalars: list[_Part], started: float
    ) -> httpx.Response:
        ends_at = None if self._deadline_seconds is None else started + self._deadline_seconds
        for attempt in stamina.retry_context(
            on=_is_retryable, attempts=self._attempts, timeout=self._deadline_seconds
        ):
            with attempt:
                # stamina checks its timeout before the backoff sleep, not after,
                # so a retry can otherwise fall due past the deadline.
                if ends_at is not None and time.monotonic() >= ends_at:
                    raise ExternalServiceError("xAI did not answer before the deadline")
                token = _ENDS_AT.set(ends_at)
                try:
                    return _post(client, path, scalars, self._attempt_timeout(started), ends_at)
                finally:
                    _ENDS_AT.reset(token)
        raise AssertionError("unreachable: stamina re-raises the last failure")  # pragma: no cover
