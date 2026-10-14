from __future__ import annotations

import gzip
import json
import mimetypes
import os
import re
import threading
import time
from typing import TYPE_CHECKING, cast, override

import httpcore
import httpx
import pytest
import stamina

from scribe import serve, xai_stt
from scribe.errors import AppError, ExternalServiceError, InputValidationError
from scribe.xai_stt import KEPT_ALIVE_SECONDS, MAX_UPLOAD_BYTES, XaiStt, resolve_api_key
from tests.xai_fixtures import xai_payload

if TYPE_CHECKING:
    import ssl
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

KEY = "xai-test-key-never-logged"
# `filename="..."` also ends in `name="`, so only a delimited match finds the field name.
_FIELD_NAME = re.compile(rb'(?:;|\s)name="([^"]*)"')
_SHOUTED = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")


@pytest.fixture(autouse=True)
def instant_retries() -> Iterator[None]:
    # Testing mode drops stamina's backoff waits; capping rather than setting
    # attempts leaves the client's own limit in force, so a retry still retries.
    with stamina.set_testing(True, attempts=5, cap=True):
        yield


def _clip(tmp_path: Path, name: str = "my clip (take 2).wav", body: bytes = b"fake-audio1") -> Path:
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _parts(request: httpx.Request) -> list[tuple[str, bytes]]:
    """Split a recorded multipart body into its fields, in wire order."""
    boundary = request.headers["content-type"].partition("boundary=")[2].strip('"').encode()
    fields: list[tuple[str, bytes]] = []
    for section in request.read().split(b"--" + boundary):
        head, separator, body = section.partition(b"\r\n\r\n")
        match = _FIELD_NAME.search(head)
        if not separator or match is None:
            continue
        fields.append((match.group(1).decode(), body.removesuffix(b"\r\n")))
    return fields


def _client(
    handler: Callable[[httpx.Request], httpx.Response], *, max_bytes: int = MAX_UPLOAD_BYTES
) -> tuple[XaiStt, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        # Read here, not in the assertions: httpx encodes multipart lazily, so
        # the upload handle is only open while the request is in flight.
        request.read()
        seen.append(request)
        return handler(request)

    return XaiStt(KEY, transport=httpx.MockTransport(record), max_bytes=max_bytes), seen


def _ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=xai_payload())


def test_the_file_field_is_last_and_the_model_is_pinned(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    client.transcribe(_clip(tmp_path))

    names = [name for name, _ in _parts(seen[0])]
    assert names == ["model", "language", "format", "diarize", "vad_threshold", "file"]
    assert dict(_parts(seen[0]))["model"] == b"grok-voice-transcribe-2.0"
    assert str(seen[0].url) == "https://api.x.ai/v1/stt"
    # Spaces and parens in the on-disk name never reach the multipart header.
    assert b'filename="my_clip_take_2_.wav"' in seen[0].read()


def test_a_long_name_keeps_the_suffix_its_media_type_is_guessed_from(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    client.transcribe(_clip(tmp_path, name=f"{'a' * 200}.wav"))

    body = seen[0].read()
    filename = re.search(rb'filename="([^"]*)"', body)
    assert filename is not None
    assert len(filename.group(1)) <= 96
    assert filename.group(1).endswith(b".wav")
    expected = mimetypes.guess_type("clip.wav")[0]
    assert expected is not None
    assert f"Content-Type: {expected}".encode() in body


def test_disabled_options_are_omitted_from_the_body(tmp_path: Path) -> None:
    client, seen = _client(_ok)
    clip = _clip(tmp_path)

    client.transcribe(clip, language=None, diarize=False)
    client.transcribe(clip, format_text=False)

    assert [name for name, _ in _parts(seen[0])] == ["model", "vad_threshold", "file"]
    assert [name for name, _ in _parts(seen[1])] == [
        "model",
        "language",
        "diarize",
        "vad_threshold",
        "file",
    ]


def test_without_keyterms_no_keyterm_field_is_sent(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    client.transcribe(_clip(tmp_path))

    assert _parts(seen[0]) == [
        ("model", b"grok-voice-transcribe-2.0"),
        ("language", b"en"),
        ("format", b"true"),
        ("diarize", b"true"),
        ("vad_threshold", b"0"),
        ("file", b"fake-audio1"),
    ]


def test_each_keyterm_is_its_own_field_before_the_file(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    client.transcribe(_clip(tmp_path), keyterms=["Claude", "Understand The Universe"])

    assert _parts(seen[0]) == [
        ("model", b"grok-voice-transcribe-2.0"),
        ("language", b"en"),
        ("format", b"true"),
        ("diarize", b"true"),
        ("keyterm", b"Claude"),
        ("keyterm", b"Understand The Universe"),
        ("vad_threshold", b"0"),
        ("file", b"fake-audio1"),
    ]


@pytest.mark.parametrize(("threshold", "sent"), [(0.0, b"0"), (0.25, b"0.25"), (1.0, b"1")])
def test_the_vad_threshold_is_sent_on_every_request_just_before_the_file(
    tmp_path: Path, threshold: float, sent: bytes
) -> None:
    client, seen = _client(_ok)

    client.transcribe(_clip(tmp_path), keyterms=["Claude"], vad_threshold=threshold)

    assert _parts(seen[0])[-2:] == [("vad_threshold", sent), ("file", b"fake-audio1")]


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), float("inf")])
def test_a_vad_threshold_outside_zero_to_one_raises_before_any_request(
    tmp_path: Path, threshold: float
) -> None:
    client, seen = _client(_ok)

    with pytest.raises(InputValidationError, match="from 0 to 1"):
        client.transcribe(_clip(tmp_path), vad_threshold=threshold)

    assert seen == []


@pytest.mark.parametrize(
    ("keyterms", "reason"),
    [
        (["term"] * 101, "at most 100"),
        (["x" * 51], "at most 50 characters"),
        (["  "], "empty"),
        (["Ann\nLee"], "control character"),
        (["Acme\x07"], "control character"),
    ],
)
def test_keyterms_past_the_documented_limits_raise_before_any_request(
    tmp_path: Path, keyterms: list[str], reason: str
) -> None:
    client, seen = _client(_ok)

    with pytest.raises(InputValidationError, match=reason):
        client.transcribe(_clip(tmp_path), keyterms=keyterms)

    assert seen == []


@pytest.mark.parametrize(
    "keyterms",
    [
        [f"term{index}" for index in range(100)],
        ["x" * 50],
        ["\N{LATIN SMALL LETTER E WITH ACUTE}" * 50],
    ],
)
def test_keyterms_at_the_documented_limits_are_all_sent(
    tmp_path: Path, keyterms: list[str]
) -> None:
    client, seen = _client(_ok)

    client.transcribe(_clip(tmp_path), keyterms=keyterms)

    sent = [body.decode() for name, body in _parts(seen[0]) if name == "keyterm"]
    assert sent == keyterms


def test_the_authorization_header_carries_the_key(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    client.transcribe(_clip(tmp_path))

    assert seen[0].headers["authorization"] == f"Bearer {KEY}"


def test_the_response_body_is_returned_decoded(tmp_path: Path) -> None:
    client, _ = _client(_ok)

    assert client.transcribe(_clip(tmp_path)) == xai_payload()


def test_a_rate_limit_is_retried_with_the_same_file_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    replies = [httpx.Response(429, text="slow down"), httpx.Response(200, json=xai_payload())]
    client, seen = _client(lambda _request: replies.pop(0))

    client.transcribe(_clip(tmp_path, body=b"first-and-second-attempt-bytes"))

    assert len(seen) == 2
    uploads = [dict(_parts(request))["file"] for request in seen]
    assert uploads == [b"first-and-second-attempt-bytes"] * 2
    # The CLI puts one thing on stdout, the destination path; a retry must not
    # add a second writer to it.
    assert capsys.readouterr().out == ""


def test_a_transport_error_is_retried_then_wrapped(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, seen = _client(refuse)

    with pytest.raises(ExternalServiceError) as caught:
        client.transcribe(_clip(tmp_path))

    assert len(seen) == 5
    assert KEY not in str(caught.value)


def test_repeated_server_errors_are_retried_then_named_by_status(tmp_path: Path) -> None:
    client, seen = _client(lambda _request: httpx.Response(503, text="upstream unavailable"))

    with pytest.raises(ExternalServiceError) as caught:
        client.transcribe(_clip(tmp_path))

    assert len(seen) == 5
    assert "503" in str(caught.value)
    assert KEY not in str(caught.value)


def test_a_mislabeled_content_encoding_is_wrapped_as_an_external_error(tmp_path: Path) -> None:
    def garbled(_request: httpx.Request) -> httpx.Response:
        # httpx raises DecodingError, a RequestError that is not a TransportError.
        return httpx.Response(200, content=b"not-gzip", headers={"content-encoding": "gzip"})

    client, _ = _client(garbled)

    with pytest.raises(ExternalServiceError) as caught:
        client.transcribe(_clip(tmp_path))

    assert KEY not in str(caught.value)


def test_a_bad_request_is_not_retried_and_hides_the_key(tmp_path: Path) -> None:
    client, seen = _client(lambda _request: httpx.Response(400, text="format requires language"))

    with pytest.raises(AppError) as caught:
        client.transcribe(_clip(tmp_path))

    assert len(seen) == 1
    assert "format requires language" in str(caught.value)
    assert "400" in str(caught.value)
    assert KEY not in str(caught.value)


def test_a_long_error_body_is_excerpted_at_300_characters(tmp_path: Path) -> None:
    # A body without whitespace measures the slice itself: `_status_error` cuts
    # before it collapses runs of whitespace, so anything else only shrinks.
    client, _ = _client(lambda _request: httpx.Response(400, text="x" * 400))

    with pytest.raises(ExternalServiceError) as caught:
        client.transcribe(_clip(tmp_path))

    assert str(caught.value).endswith("x" * 300)
    assert "x" * 301 not in str(caught.value)


def test_a_body_that_is_not_an_object_is_an_external_error(tmp_path: Path) -> None:
    client, _ = _client(lambda _request: httpx.Response(200, json=[1, 2]))

    with pytest.raises(ExternalServiceError):
        client.transcribe(_clip(tmp_path))


def test_a_body_that_is_not_json_is_an_external_error(tmp_path: Path) -> None:
    client, _ = _client(lambda _request: httpx.Response(200, text="<html>nope</html>"))

    with pytest.raises(ExternalServiceError):
        client.transcribe(_clip(tmp_path))


def test_an_oversized_file_raises_before_any_request(tmp_path: Path) -> None:
    client, seen = _client(_ok, max_bytes=10)

    with pytest.raises(InputValidationError) as caught:
        client.transcribe(_clip(tmp_path, body=b"11-bytes-ok"))

    assert seen == []
    assert "segmenting" in str(caught.value)


def test_a_missing_path_raises_before_any_request(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    with pytest.raises(InputValidationError):
        client.transcribe(tmp_path / "absent.wav")

    assert seen == []


def test_a_directory_raises_before_any_request(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    with pytest.raises(InputValidationError):
        client.transcribe(tmp_path)

    assert seen == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 0o000 file, so nothing fails")
def test_an_unreadable_file_raises_before_any_request(tmp_path: Path) -> None:
    client, seen = _client(_ok)
    clip = _clip(tmp_path)
    clip.chmod(0o000)

    try:
        with pytest.raises(InputValidationError) as caught:
            client.transcribe(clip)
    finally:
        clip.chmod(0o600)

    assert seen == []
    assert "cannot read" in str(caught.value)


def test_the_key_comes_from_the_environment() -> None:
    assert resolve_api_key({"XAI_API_KEY": " k "}) == "k"


@pytest.mark.parametrize("env", [{}, {"XAI_API_KEY": "  "}])
def test_a_missing_key_names_only_the_variable(env: dict[str, str]) -> None:
    with pytest.raises(InputValidationError) as caught:
        resolve_api_key(env)

    assert _SHOUTED.findall(str(caught.value)) == ["XAI_API_KEY"]


class _ClosingTransport(httpx.MockTransport):
    """A mock transport that counts how often a client closes it."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        super().__init__(handler)
        self.closed = 0

    @override
    def close(self) -> None:
        self.closed += 1


def _read_timeout(request: httpx.Request) -> float:
    timeouts: object = request.extensions["timeout"]  # pyright: ignore[reportAny]  # httpx types extensions as Any
    assert isinstance(timeouts, dict)
    return cast("dict[str, float]", timeouts)["read"]


def test_by_default_each_call_opens_and_closes_its_own_client(
    tmp_path: Path,
) -> None:
    transport = _ClosingTransport(_ok)
    client = XaiStt(KEY, transport=transport)
    clip = _clip(tmp_path)

    client.transcribe(clip)
    client.transcribe(clip)
    client.close()

    assert transport.closed == 2


def test_the_default_attempt_timeout_is_600_s(tmp_path: Path) -> None:
    client, seen = _client(_ok)

    client.transcribe(_clip(tmp_path))

    assert _read_timeout(seen[0]) == 600


def test_a_kept_alive_client_is_reused_until_closed(tmp_path: Path) -> None:
    transport = _ClosingTransport(_ok)
    client = XaiStt(KEY, transport=transport, keep_alive=True)
    clip = _clip(tmp_path)

    client.transcribe(clip)
    client.transcribe(clip)
    assert transport.closed == 0

    client.close()
    assert transport.closed == 1


def test_fewer_attempts_stop_retrying_sooner(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def unavailable(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503, text="upstream unavailable")

    client = XaiStt(KEY, transport=httpx.MockTransport(unavailable), attempts=2)

    with pytest.raises(ExternalServiceError, match="503"):
        client.transcribe(_clip(tmp_path))

    assert len(seen) == 2


def test_a_kept_alive_client_holds_idle_connections_for_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, object]] = []
    real = httpx.Client

    def recording(**settings: object) -> httpx.Client:
        made.append(settings)
        return real(**settings)  # pyright: ignore[reportArgumentType]  # forwards what the code passed

    monkeypatch.setattr(httpx, "Client", recording)

    XaiStt(KEY, keep_alive=True).close()

    limits = made[0].get("limits")
    assert isinstance(limits, httpx.Limits)
    assert limits.keepalive_expiry is not None
    assert limits.keepalive_expiry >= 60


def test_a_deadline_bounds_each_attempt_by_the_time_left(tmp_path: Path) -> None:
    timeouts: list[float] = []

    def stall(request: httpx.Request) -> httpx.Response:
        timeouts.append(_read_timeout(request))
        raise httpx.ReadTimeout("stalled", request=request)

    client = XaiStt(KEY, transport=httpx.MockTransport(stall), attempts=2, deadline_seconds=8)

    with pytest.raises(ExternalServiceError, match="stalled"):
        client.transcribe(_clip(tmp_path))

    assert len(timeouts) == 2
    assert 7 < timeouts[0] <= 8
    assert 0 < timeouts[1] <= timeouts[0]


def test_a_deadline_closes_a_reply_trickled_past_it(tmp_path: Path) -> None:
    closed: list[bool] = []

    class Trickle(httpx.SyncByteStream):
        @override
        def __iter__(self) -> Iterator[bytes]:
            for _ in range(100):
                time.sleep(0.02)
                yield b" "

        @override
        def close(self) -> None:
            closed.append(True)

    def trickling(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Trickle())

    client = XaiStt(KEY, transport=httpx.MockTransport(trickling), attempts=2, deadline_seconds=0.2)
    started = time.monotonic()

    with pytest.raises(ExternalServiceError, match="before the deadline"):
        client.transcribe(_clip(tmp_path))

    assert time.monotonic() - started < 0.6
    assert closed == [True]


def test_a_reply_read_under_a_deadline_is_decoded_as_sent(tmp_path: Path) -> None:
    def gzipped(_request: httpx.Request) -> httpx.Response:
        body = gzip.compress(json.dumps(xai_payload()).encode())
        stream = httpx.ByteStream(body)
        return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=stream)

    client = XaiStt(KEY, transport=httpx.MockTransport(gzipped), deadline_seconds=8)

    assert client.transcribe(_clip(tmp_path)) == xai_payload()


def test_no_retry_starts_after_the_deadline(tmp_path: Path) -> None:
    starts: list[float] = []

    def unavailable_at_the_end(_request: httpx.Request) -> httpx.Response:
        starts.append(time.monotonic())
        time.sleep(0.25)
        return httpx.Response(503, text="upstream unavailable")

    client = XaiStt(
        KEY, transport=httpx.MockTransport(unavailable_at_the_end), attempts=2, deadline_seconds=0.3
    )
    started = time.monotonic()

    # Real backoff, at least 0.1 s, so the retry falls due after the deadline.
    with stamina.set_testing(False), pytest.raises(ExternalServiceError):
        client.transcribe(_clip(tmp_path))

    assert [at for at in starts if at >= started + 0.3] == []


class _Wire:
    """Connections opened and closed over an in-memory network."""

    def __init__(self, *, honors_timeouts: bool = False) -> None:
        self.honors_timeouts = honors_timeouts
        self.opened = 0
        self.closed = 0
        # The connection, by its opening order from 1, that carried each request.
        self.requests: list[int] = []


class _TrickledHeaders(httpcore.NetworkStream):
    """A server that sends its response headers one byte every 20 ms."""

    def __init__(self, wire: _Wire) -> None:
        self._wire = wire
        self._chunks = iter(
            [b"HTTP/1.1 200 OK\r\nX-Pad: ", *[b"a"] * 100, b"\r\nContent-Length: 2\r\n\r\n{}"]
        )
        wire.opened += 1

    @override
    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._wire.honors_timeouts and timeout is not None and timeout < 0.02:
            time.sleep(timeout)
            raise httpcore.ReadTimeout
        time.sleep(0.02)
        return next(self._chunks, b"")

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        pass

    @override
    def close(self) -> None:
        self._wire.closed += 1

    @override
    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        return self


class _TrickleBackend(httpcore.NetworkBackend):
    def __init__(self, wire: _Wire) -> None:
        self._wire = wire

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return _TrickledHeaders(self._wire)


# A socket that ignores its timeout still blocks no read begun past the deadline.
@pytest.mark.parametrize("honors_timeouts", [True, False])
def test_a_deadline_closes_a_connection_whose_headers_trickle_past_it(
    tmp_path: Path, *, honors_timeouts: bool
) -> None:
    wire = _Wire(honors_timeouts=honors_timeouts)
    client = XaiStt(
        KEY,
        network_backend=_TrickleBackend(wire),
        attempts=2,
        deadline_seconds=0.2,
        keep_alive=True,
    )
    started = time.monotonic()

    with pytest.raises(ExternalServiceError, match="deadline"):
        client.transcribe(_clip(tmp_path))

    assert time.monotonic() - started < 0.6
    assert (wire.opened, wire.closed) == (1, 1)
    client.close()


_REPLY_BODY = json.dumps(xai_payload()).encode()
_REPLY = (
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
    + f"Content-Length: {len(_REPLY_BODY)}\r\n\r\n".encode()
    + _REPLY_BODY
)


class _Replying(httpcore.NetworkStream):
    """A server that answers each request at once or, when it stalls, never."""

    def __init__(self, wire: _Wire, *, stalls: bool) -> None:
        wire.opened += 1
        self._wire = wire
        self._number = wire.opened
        self._stalls = stalls
        self._reply = b""

    @override
    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._stalls:
            assert timeout is not None
            time.sleep(timeout)
            raise httpcore.ReadTimeout
        reply, self._reply = self._reply, b""
        return reply

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if buffer.startswith(b"POST "):
            self._wire.requests.append(self._number)
            self._reply = _REPLY

    @override
    def close(self) -> None:
        self._wire.closed += 1

    @override
    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        return self


class _Uploading(_Replying):
    """A healthy server that takes 50 ms to accept each large chunk of an upload."""

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if len(buffer) > 1024:
            time.sleep(0.05)
        super().write(buffer, timeout)


class _ReplyingBackend(httpcore.NetworkBackend):
    """Connections that stall for the first `stalled` opened, then answer."""

    def __init__(
        self, wire: _Wire, *, stalled: int = 0, stream: type[_Replying] = _Replying
    ) -> None:
        self._wire = wire
        self._stalled = stalled
        self._stream = stream

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return self._stream(self._wire, stalls=self._wire.opened < self._stalled)


def _dictation_client(
    monkeypatch: pytest.MonkeyPatch, backend: httpcore.NetworkBackend, *, deadline: float
) -> XaiStt:
    """The server's xAI client over `backend`, a stalled attempt yielding after 0.2 s."""
    monkeypatch.setattr(serve, "XAI_DEADLINE_SECONDS", deadline)
    monkeypatch.setattr(serve, "XAI_ATTEMPT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(httpcore, "SyncBackend", lambda: backend)
    return serve.dictation_client(KEY)


def test_a_stalled_attempt_yields_to_a_retry_on_a_new_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = _Wire()
    client = _dictation_client(monkeypatch, _ReplyingBackend(wire, stalled=1), deadline=1.0)
    started = time.monotonic()

    try:
        reply: object = client.transcribe(_clip(tmp_path))
    except ExternalServiceError as exc:
        reply = exc
    elapsed = time.monotonic() - started
    client.close()

    assert reply == xai_payload()
    assert wire.requests == [1, 2]
    assert elapsed < 1.0


def test_an_upload_slower_than_the_cap_is_one_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = _Wire()
    client = _dictation_client(monkeypatch, _ReplyingBackend(wire, stream=_Uploading), deadline=1.0)
    clip = _clip(tmp_path, body=bytes(9 * 64 * 1024))

    try:
        reply: object = client.transcribe(clip)
    except ExternalServiceError as exc:
        reply = exc
    client.close()

    assert reply == xai_payload()
    assert wire.requests == [1]


def test_a_reply_trickled_past_the_cap_is_not_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = json.dumps(xai_payload()).encode()
    step = len(body) // 5 + 1

    class Trickle(httpx.SyncByteStream):
        @override
        def __iter__(self) -> Iterator[bytes]:
            for start in range(0, len(body), step):
                time.sleep(0.1)
                yield body[start : start + step]

    seen: list[httpx.Request] = []

    def trickling(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, stream=Trickle())

    monkeypatch.setattr(serve, "XAI_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(serve, "XAI_ATTEMPT_SECONDS", 0.2, raising=False)
    client = serve.dictation_client(KEY, transport=httpx.MockTransport(trickling))

    try:
        reply: object = client.transcribe(_clip(tmp_path))
    except ExternalServiceError as exc:
        reply = exc

    assert reply == xai_payload()
    assert len(seen) == 1


def test_the_last_attempt_keeps_the_time_left_to_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx.Request] = []

    def outage_then_slow(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(503, text="upstream unavailable")
        # A server that answers after 0.4 s, behind a socket that honors its timeout.
        if _read_timeout(request) < 0.4:
            time.sleep(_read_timeout(request))
            raise httpx.ReadTimeout("stalled", request=request)
        time.sleep(0.4)
        return _ok(request)

    monkeypatch.setattr(serve, "XAI_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(serve, "XAI_ATTEMPT_SECONDS", 0.2)
    client = serve.dictation_client(KEY, transport=httpx.MockTransport(outage_then_slow))

    try:
        reply: object = client.transcribe(_clip(tmp_path))
    except ExternalServiceError as exc:
        reply = exc

    assert reply == xai_payload()
    assert len(seen) == 2


def test_a_normal_dictation_is_one_request_on_the_kept_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = _Wire()
    client = _dictation_client(monkeypatch, _ReplyingBackend(wire), deadline=1.0)
    clip = _clip(tmp_path)

    assert client.transcribe(clip) == xai_payload()
    assert client.transcribe(clip) == xai_payload()
    client.close()

    assert wire.requests == [1, 1]


def test_a_connection_idle_through_a_sleep_carries_no_dictation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The wall clock counts a sleep; the monotonic clock under httpcore's own
    # keep-alive expiry does not, so it moves only the milliseconds this test takes.
    now = [1_000_000.0]
    monkeypatch.setattr(xai_stt, "_wall_clock", lambda: now[0], raising=False)
    wire = _Wire()
    client = XaiStt(
        KEY, network_backend=_ReplyingBackend(wire), deadline_seconds=8, keep_alive=True
    )
    clip = _clip(tmp_path)

    client.transcribe(clip)
    now[0] += KEPT_ALIVE_SECONDS + 1
    client.transcribe(clip)

    assert wire.requests == [1, 2]
    assert wire.closed == 1
    client.close()


def test_a_replacement_never_closes_a_connection_a_call_is_using(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1_000_000.0]
    monkeypatch.setattr(xai_stt, "_wall_clock", lambda: now[0])
    wire = _Wire()
    in_read = threading.Event()
    release = threading.Event()

    class Held(_Replying):
        """Holds the reply to the second request on the first connection until released."""

        @override
        def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            if wire.requests == [1, 1]:
                in_read.set()
                assert release.wait(5)
            return super().read(max_bytes, timeout)

    client = XaiStt(
        KEY,
        network_backend=_ReplyingBackend(wire, stream=Held),
        deadline_seconds=8,
        keep_alive=True,
    )
    clip = _clip(tmp_path)
    client.transcribe(clip)
    replies: list[dict[str, object]] = []
    held = threading.Thread(target=lambda: replies.append(client.transcribe(clip)))
    held.start()
    try:
        assert in_read.wait(5)
        # A call that arrives after a sleep while the one before it is still running.
        now[0] += KEPT_ALIVE_SECONDS + 1
        client.transcribe(clip)
        closed_while_held = wire.closed
    finally:
        release.set()
        held.join(5)
    client.close()

    assert closed_while_held == 0
    assert replies == [xai_payload()]


def test_connections_idle_through_a_sleep_after_overlapping_calls_carry_no_dictation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1_000_000.0]
    monkeypatch.setattr(xai_stt, "_wall_clock", lambda: now[0])
    wire = _Wire()
    entered = {number: threading.Event() for number in (1, 2, 3)}
    release = {number: threading.Event() for number in (1, 2, 3)}
    # Connections whose peer dropped them during the sleep: they never answer again.
    dead: set[int] = set()

    class Overlapping(_Replying):
        """Holds each of the first three connections' first reply until released."""

        @override
        def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            if self._number in release:
                entered[self._number].set()
                assert release[self._number].wait(5)
            if self._number in dead:
                assert timeout is not None
                time.sleep(timeout)
                raise httpcore.ReadTimeout
            return super().read(max_bytes, timeout)

    client = _dictation_client(
        monkeypatch, _ReplyingBackend(wire, stream=Overlapping), deadline=1.0
    )
    clip = _clip(tmp_path)
    calls: dict[int, threading.Thread] = {}
    try:
        for number in (1, 2, 3):
            calls[number] = threading.Thread(target=client.transcribe, args=(clip,))
            calls[number].start()
            assert entered[number].wait(5)
        for number in (1, 2):
            release[number].set()
            calls[number].join(5)
        # The machine sleeps past the keep-alive while the third call is in flight.
        now[0] += KEPT_ALIVE_SECONDS + 1
        dead.update({1, 2})
        release[3].set()
        calls[3].join(5)

        try:
            reply: object = client.transcribe(clip)
        except ExternalServiceError as exc:
            reply = exc
    finally:
        for event in release.values():
            event.set()
    client.close()

    assert wire.requests == [1, 2, 3, 4]
    assert reply == xai_payload()


def _clocks(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Drive the wall and monotonic clocks the kept client reads; both start far from zero."""
    clocks = {"wall": 1_000_000.0, "monotonic": 5_000.0}
    monkeypatch.setattr(xai_stt, "_wall_clock", lambda: clocks["wall"])
    monkeypatch.setattr(xai_stt, "_monotonic", lambda: clocks["monotonic"], raising=False)
    return clocks


def test_a_sleep_shorter_than_the_keep_alive_also_replaces_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clocks = _clocks(monkeypatch)
    wire = _Wire()
    client = XaiStt(
        KEY, network_backend=_ReplyingBackend(wire), deadline_seconds=8, keep_alive=True
    )
    clip = _clip(tmp_path)

    client.transcribe(clip)
    clocks["wall"] += 60
    client.transcribe(clip)

    assert wire.requests == [1, 2]
    assert wire.closed == 1
    client.close()


def test_calls_with_no_sleep_between_them_keep_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clocks = _clocks(monkeypatch)
    wire = _Wire()
    client = XaiStt(
        KEY, network_backend=_ReplyingBackend(wire), deadline_seconds=8, keep_alive=True
    )
    clip = _clip(tmp_path)

    for awake in (0, 1, 200, 0.5):
        clocks["wall"] += awake
        clocks["monotonic"] += awake
        client.transcribe(clip)

    assert wire.requests == [1, 1, 1, 1]
    assert wire.closed == 0
    client.close()


class _SleepWire(_Wire):
    """A wire that holds chosen replies until released and loses connections in a sleep."""

    def __init__(self, *, held: Iterable[int]) -> None:
        super().__init__()
        # Keyed by request, numbered from 1 in the order the requests were sent.
        self.entered = {number: threading.Event() for number in held}
        self.release = {number: threading.Event() for number in held}
        # Connections whose peer dropped them during the sleep: they never answer again.
        self.dead: set[int] = set()


class _Sleeping(_Replying):
    def __init__(self, wire: _Wire, *, stalls: bool) -> None:
        super().__init__(wire, stalls=stalls)
        assert isinstance(wire, _SleepWire)
        self._sleep_wire = wire
        self._request = 0

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if buffer.startswith(b"POST "):
            self._request = len(self._sleep_wire.requests) + 1
        super().write(buffer, timeout)

    @override
    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        wire = self._sleep_wire
        if self._request in wire.release:
            wire.entered[self._request].set()
            assert wire.release[self._request].wait(5)
        if self._number in wire.dead:
            assert timeout is not None
            time.sleep(timeout)
            raise httpcore.ReadTimeout
        return super().read(max_bytes, timeout)


def _start_call(client: XaiStt, clip: Path, outcomes: list[object]) -> threading.Thread:
    def call() -> None:
        try:
            outcomes.append(client.transcribe(clip))
        except ExternalServiceError as exc:
            outcomes.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    return thread


def test_a_call_after_a_sleep_while_another_runs_retries_on_a_new_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clocks = _clocks(monkeypatch)
    wire = _SleepWire(held=(1, 2, 3))
    client = _dictation_client(monkeypatch, _ReplyingBackend(wire, stream=_Sleeping), deadline=1.0)
    clip = _clip(tmp_path)
    outcomes: list[object] = []
    calls: list[threading.Thread] = []
    try:
        for number in (1, 2, 3):
            calls.append(_start_call(client, clip, outcomes))
            assert wire.entered[number].wait(5)
        for number in (1, 2):
            wire.release[number].set()
            calls[number - 1].join(5)
        # The machine sleeps while the third call is in flight; the idle two die.
        clocks["wall"] += 60
        wire.dead.update({1, 2})

        try:
            reply: object = client.transcribe(clip)
        except ExternalServiceError as exc:
            reply = exc
    finally:
        for event in wire.release.values():
            event.set()
        for thread in calls:
            thread.join(5)
    client.close()

    assert wire.requests == [1, 2, 3, 1, 4]
    assert reply == xai_payload()


def test_a_call_whose_wait_spans_a_sleep_retries_on_a_new_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clocks = _clocks(monkeypatch)
    wire = _SleepWire(held=(1, 2, 3))
    client = _dictation_client(monkeypatch, _ReplyingBackend(wire, stream=_Sleeping), deadline=1.0)
    clip = _clip(tmp_path)
    outcomes: list[object] = []
    try:
        # Two overlapping calls leave two idle connections in the pool.
        overlapping: list[threading.Thread] = []
        for number in (1, 2):
            overlapping.append(_start_call(client, clip, outcomes))
            assert wire.entered[number].wait(5)
        for number in (1, 2):
            wire.release[number].set()
            overlapping[number - 1].join(5)
        held = _start_call(client, clip, outcomes)
        assert wire.entered[3].wait(5)
        # The machine sleeps while the call waits on its reply; both connections die.
        clocks["wall"] += 60
        wire.dead.update({1, 2})
        wire.release[3].set()
        held.join(5)
    finally:
        for event in wire.release.values():
            event.set()
    client.close()

    assert wire.requests == [1, 2, 1, 3]
    assert outcomes[2] == xai_payload()
