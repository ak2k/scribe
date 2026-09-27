"""The pyannote backend: the pinned uvx run, its request and its answer, all faked here."""

from __future__ import annotations

import contextlib
import json
import subprocess
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scribe.attribution import Speech
from scribe.diarizer import (
    DEFAULT_TIMEOUT_S,
    DIMENSION,
    FAILED,
    Diarization,
    PyannoteDiarizer,
    parse_output,
)
from scribe.errors import ExternalServiceError, ToolMissingError
from tests.diarizer_fakes import FFMPEG, TOKEN, UVX, FakeWorker, answer, found

if TYPE_CHECKING:
    from collections.abc import Callable

REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"


def _nothing(_name: str) -> str | None:
    return None


def _no_ffmpeg(name: str) -> str | None:
    return UVX if name == "uvx" else None


def _backend(
    fake: FakeWorker,
    *,
    host: tuple[str, str] = ("Darwin", "arm64"),
    which: Callable[[str], str | None] = found,
) -> PyannoteDiarizer:
    return PyannoteDiarizer(run=fake.run, which=which, host=lambda: host)


def _audio(tmp_path: Path) -> Path:
    clip = tmp_path / "-clip.mp3"
    clip.write_bytes(b"pretend this is audio")
    return clip


def test_the_run_decodes_the_audio_then_runs_the_pinned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    fake = FakeWorker()
    clip = _audio(tmp_path)

    result = _backend(fake).diarize(clip, [(1.0, 4.0), (2.5, 5.0)])

    decode, spawn = fake.calls
    wav = decode[-1]
    assert decode == [FFMPEG, "-nostdin", "-v", "error", "-i", str(clip.absolute()), "-vn",
        "-ac", "1", "-ar", "16000", wav]  # fmt: skip
    pinned = [UVX, "--python", "3.13", "--exclude-newer", "2026-09-24T00:00:00Z"]
    pinned += ["--from", "pyannote.audio==4.0.7", "--with", "torch==2.14.0"]
    pinned += ["--with", "numpy==2.5.3", "--with", "soundfile==0.14.0", "python", "-P"]
    assert spawn[: len(pinned)] == pinned
    assert len(spawn) == len(pinned) + 3
    worker = resources.files("scribe") / "diarize_worker.py"
    assert fake.workers == [worker.read_text(encoding="utf-8")]
    assert json.loads(fake.requests[0]) == {
        "audio": wav,
        "model": "pyannote/speaker-diarization-community-1",
        "revision": REVISION,
        "intervals": [[1.0, 4.0], [2.5, 5.0]],
    }
    # The token reaches the worker only through its environment.
    decode_env, worker_env = fake.envs
    assert worker_env["HF_TOKEN"] == TOKEN
    assert "HF_TOKEN" not in decode_env
    assert not any(TOKEN in part for argv in fake.calls for part in argv)
    assert TOKEN.encode() not in fake.requests[0]
    assert fake.timeouts == [DEFAULT_TIMEOUT_S, DEFAULT_TIMEOUT_S]
    assert not Path(wav).parent.exists()
    assert result == Diarization(
        device="mps",
        runtime_s=1.5,
        speech=(Speech(0.0, 2.0, "SPEAKER_00"),),
        embeddings=((0.5,) * DIMENSION, (0.5,) * DIMENSION),
    )


# Keys scribe's own backends read, a path to a key file, and a token under a
# prefix whose other names pass.
_SECRETS = (
    "XAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HONCHO_API_KEY",
    "SOPS_AGE_KEY_FILE",
    "UV_PUBLISH_TOKEN",
)
# What the children need to run and to find, or fetch, their packages and weights.
_NEEDED = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "UV_CACHE_DIR",
    "PARAKEET_CACHE_DIR",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "https_proxy",
)


def test_only_the_worker_gets_the_hugging_face_token_and_no_child_gets_other_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _SECRETS:
        monkeypatch.setenv(name, "secret")
    # uv would look for an interpreter there before resolving its own.
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
    for name in _NEEDED:
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "token"))
    fake = FakeWorker()

    _backend(fake).diarize(_audio(tmp_path), [(1.0, 4.0)])

    decode_env, worker_env = fake.envs
    assert worker_env["HF_TOKEN"] == TOKEN
    assert worker_env["HF_TOKEN_PATH"] == str(tmp_path / "token")
    assert not {"HF_TOKEN", "HF_TOKEN_PATH"} & decode_env.keys()
    for env in fake.envs:
        assert not {*_SECRETS, "VIRTUAL_ENV"} & env.keys()
        assert {name: env.get(name) for name in _NEEDED} == dict.fromkeys(_NEEDED, str(tmp_path))


def test_the_worker_marks_its_failures_as_this_module_reads_them() -> None:
    source = (resources.files("scribe") / "diarize_worker.py").read_text(encoding="utf-8")

    assert f'FAILED = "{FAILED}"' in source


_NOISE = "UserWarning: torchcodec is not installed\n" * 20


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (
            FakeWorker(
                returncode=1,
                stdout="Could not download config.yaml ...\n* visit https://hf.co/...\n",
                stderr=f"{_NOISE}{FAILED}cannot load it: set HF_TOKEN\n",
            ),
            "the diarizer exited 1: cannot load it: set HF_TOKEN",
        ),
        (
            FakeWorker(returncode=2, stderr="error: No solution found when resolving"),
            "the diarizer exited 2: error: No solution found",
        ),
        (FakeWorker(returncode=3, stdout="only stdout"), "exited 3: only stdout"),
        (FakeWorker(lambda _: None, stderr="warned"), "exited 0 without writing an answer: warned"),
        (FakeWorker(lambda _: "[not json"), "unexpected answer"),
        (FakeWorker(lambda i: answer(i, extra=1)), "at extra"),
        (FakeWorker(lambda i: answer(i, embeddings=[])), "embedded 0 of 1 intervals"),
        (FakeWorker(lambda i: answer(i, embeddings=[[0.5] * 3])), "3 values, not 256"),
        (
            FakeWorker(error=subprocess.TimeoutExpired(["uvx"], 1.0)),
            f"the diarizer did not finish within {DEFAULT_TIMEOUT_S:g} s",
        ),
        (FakeWorker(error=OSError("exec format error")), f"cannot run {UVX}"),
        (FakeWorker(decode_returncode=1), "ffmpeg exited 1: bad mp3"),
    ],
    ids=[
        "worker-line",
        "uvx",
        "stdout",
        "no-answer",
        "not-json",
        "extra-field",
        "count",
        "dimension",
        "timeout",
        "unrunnable",
        "decode",
    ],
)
def test_a_failed_run_is_one_service_error_naming_its_cause(
    tmp_path: Path, fake: FakeWorker, expected: str
) -> None:
    with pytest.raises(ExternalServiceError) as caught:
        _backend(fake).diarize(_audio(tmp_path), [(0.0, 3.0)])

    assert expected in str(caught.value)
    assert "\n" not in str(caught.value)


def test_a_long_report_keeps_both_ends() -> None:
    fake = FakeWorker(returncode=1, stderr="Resolving pyannote " + "x " * 400 + "the cause")

    with pytest.raises(ExternalServiceError, match=r"Resolving pyannote .* \.\.\. .*the cause$"):
        PyannoteDiarizer(run=fake.run, which=found, host=lambda: ("Darwin", "arm64")).diarize(
            Path("a.mp3"), []
        )


@pytest.mark.parametrize(
    ("host", "which", "error", "expected"),
    [
        (("Linux", "x86_64"), found, ExternalServiceError, "needs Apple silicon"),
        (("Darwin", "x86_64"), found, ExternalServiceError, "this is Darwin x86_64"),
        (("Darwin", "arm64"), _nothing, ExternalServiceError, "uvx is not on PATH"),
        (("Darwin", "arm64"), _no_ffmpeg, ToolMissingError, "ffmpeg is not on PATH"),
    ],
)
def test_an_unusable_machine_is_refused_before_anything_runs(
    tmp_path: Path,
    host: tuple[str, str],
    which: Callable[[str], str | None],
    error: type[ExternalServiceError],
    expected: str,
) -> None:
    fake = FakeWorker()

    with pytest.raises(error, match=expected):
        _backend(fake, host=host, which=which).diarize(_audio(tmp_path), [(0.0, 3.0)])
    assert fake.calls == []


def test_a_null_embedding_is_kept_as_none() -> None:
    raw = json.dumps(answer([[0.0, 1.0], [1.0, 2.0]], embeddings=[None, [0.0] * DIMENSION]))

    assert parse_output(raw.encode(), 2).embeddings == (None, (0.0,) * DIMENSION)


_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=4),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=4), inner),
    max_leaves=8,
)
_keys = st.sampled_from(["versions", "device", "runtime_s", "exclusive", "embeddings", "other"])


@given(st.one_of(st.binary(max_size=40), _json.map(json.dumps).map(str.encode)))
def test_any_answer_either_parses_or_is_a_service_error(raw: bytes) -> None:
    with contextlib.suppress(ExternalServiceError):
        parse_output(raw, 1)


@given(_keys, _json)
def test_a_well_formed_answer_with_one_field_changed_parses_or_is_a_service_error(
    key: str, value: object
) -> None:
    raw = json.dumps(answer([[0.0, 1.0]], **{key: value})).encode()

    with contextlib.suppress(ExternalServiceError):
        parse_output(raw, 1)
