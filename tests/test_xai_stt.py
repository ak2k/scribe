from __future__ import annotations

import mimetypes
import os
import re
from typing import TYPE_CHECKING

import httpx
import pytest
import stamina

from scribe.errors import AppError, ExternalServiceError, InputValidationError
from scribe.xai_stt import MAX_UPLOAD_BYTES, XaiStt, resolve_api_key
from tests.xai_fixtures import xai_payload

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
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
