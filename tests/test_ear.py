"""The local recognizers: the resolver, the pinned uvx runs and their answers, all faked."""

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

from scribe.ear import DEFAULT_TIMEOUT_S, FAILED, RECOGNIZERS, Heard, LocalEars, parse_output
from scribe.errors import EarError
from tests.child_env_cases import ALL_BUT_THE_WORKER, TOKENS
from tests.diarizer_fakes import FFMPEG, TOKEN, UVX, FakeWorker, found

if TYPE_CHECKING:
    from collections.abc import Callable

    from scribe.ear import Recognizer

COHERE, QWEN = RECOGNIZERS
_WEIGHTS = ("config.json", "model.safetensors")


def answer(intervals: list[list[float]], **changes: object) -> dict[str, object]:
    """A well-formed OUT for `intervals`, with any top-level key replaced."""
    body: dict[str, object] = {
        "versions": {"python": "3.12.13", "transformers": "5.18.0"},
        "device": "mps",
        "dtype": "bfloat16",
        "runtime_s": 2.5,
        "texts": [f"heard {start:g}" for start, _ in intervals],
    }
    return {**body, **changes}


def _snapshot(hub: Path, model: str, revision: str) -> Path:
    return hub / f"models--{model.replace('/', '--')}" / "snapshots" / revision


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Hugging Face cache holding both pinned snapshots."""
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    for spec in RECOGNIZERS:
        snapshot = _snapshot(hub, spec.model, spec.revision)
        snapshot.mkdir(parents=True)
        for name in _WEIGHTS:
            (snapshot / name).write_text("{}", encoding="utf-8")
    return hub


def _ears(
    fake: FakeWorker,
    *,
    host: tuple[str, str] = ("Darwin", "arm64"),
    which: Callable[[str], str | None] = found,
) -> LocalEars:
    return LocalEars(run=fake.run, which=which, host=lambda: host)


def _audio(tmp_path: Path) -> Path:
    clip = tmp_path / "-clip.mp3"
    clip.write_bytes(b"pretend this is audio")
    return clip


def test_each_recognizer_runs_in_turn_on_the_decoded_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    fake = FakeWorker(answer)
    clip = _audio(tmp_path)

    heard = _ears(fake).hear(clip, [(1.0, 4.0), (2.5, 5.0)], max_new_tokens=1024)

    decode, *spawns = fake.calls
    samples = decode[-1]
    assert decode == [FFMPEG, "-nostdin", "-v", "error", "-i", str(clip.absolute()), "-vn",
        "-ac", "1", "-ar", "16000", "-f", "f32le", samples]  # fmt: skip
    pinned = [UVX, "--python", "3.12", "--exclude-newer", "2026-09-30T00:00:00Z"]
    pinned += ["--from", "transformers==5.18.0", "--with", "torch==2.14.1"]
    pinned += ["--with", "numpy==2.5.3", "--with", "librosa==1.0.0", "python", "-P"]
    assert [spawn[: len(pinned)] for spawn in spawns] == [pinned, pinned]
    assert [len(spawn) for spawn in spawns] == [len(pinned) + 3] * 2
    worker = (resources.files("scribe") / "ear_worker.py").read_text(encoding="utf-8")
    assert fake.workers == [worker, worker]
    request = {"audio": samples, "intervals": [[1.0, 4.0], [2.5, 5.0]], "max_new_tokens": 1024}
    assert [json.loads(raw) for raw in fake.requests] == [
        {**request, "name": "cohere-transcribe", "model": "CohereLabs/cohere-transcribe-03-2026",
            "revision": "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"},
        {**request, "name": "qwen3-asr", "model": "Qwen/Qwen3-ASR-1.7B-hf",
            "revision": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"},
    ]  # fmt: skip
    assert not any(TOKEN in part for argv in fake.calls for part in argv)
    assert fake.timeouts == [DEFAULT_TIMEOUT_S] * 3
    assert not Path(samples).parent.exists()
    versions = {"python": "3.12.13", "transformers": "5.18.0"}
    assert heard == tuple(
        Heard(spec, versions, "mps", "bfloat16", 2.5, ("heard 1", "heard 2.5"))
        for spec in RECOGNIZERS
    )


# Given to the decode, Cohere's worker, Qwen's worker: the token only where the gated model is
# loaded, and the opt-out of sending one everywhere else.
@pytest.mark.parametrize(
    ("name", "given_to"),
    [(name, [False, True, False]) for name in TOKENS]
    + [(name, [True, False, True]) for name in ALL_BUT_THE_WORKER],
)
def test_each_child_is_given_only_the_hugging_face_names_it_needs(
    name: str, given_to: list[bool], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, "1")
    fake = FakeWorker(answer)

    _ears(fake).hear(_audio(tmp_path), [(1.0, 4.0)])

    assert [name in env for env in fake.envs] == given_to


@pytest.mark.parametrize("offline", [True, False])
def test_offline_keeps_both_workers_to_what_is_cached(
    offline: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ("UV_OFFLINE", "HF_HUB_OFFLINE")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    fake = FakeWorker(answer)

    _ears(fake).hear(_audio(tmp_path), [(1.0, 4.0)], offline=offline)

    given_names = [{name: env[name] for name in names if name in env} for env in fake.envs]
    expected = dict.fromkeys(names, "1") if offline else {}
    assert given_names == [{}, expected, expected]


def test_the_worker_marks_its_failures_as_this_module_reads_them() -> None:
    source = (resources.files("scribe") / "ear_worker.py").read_text(encoding="utf-8")

    assert f'FAILED = "{FAILED}"' in source


_NOISE = "UserWarning: something is deprecated\n" * 20


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (
            FakeWorker(answer, returncode=1, stderr=f"{_NOISE}{FAILED}OSError: not cached\n"),
            "cohere-transcribe exited 1: OSError: not cached",
        ),
        (
            FakeWorker(answer, returncode=2, stderr="error: Network connectivity is disabled"),
            "cohere-transcribe exited 2: error: Network connectivity is disabled",
        ),
        (FakeWorker(answer, returncode=3, stdout="only stdout"), "exited 3: only stdout"),
        (
            FakeWorker(lambda _: None, stderr="warned"),
            "cohere-transcribe exited 0 without writing an answer: warned",
        ),
        (FakeWorker(lambda _: "[not json"), "cohere-transcribe wrote an unexpected answer"),
        (FakeWorker(lambda i: answer(i, extra=1)), "at extra"),
        (FakeWorker(lambda i: answer(i, texts=[])), "cohere-transcribe heard 0 of 1 clips"),
        (
            FakeWorker(answer, error=subprocess.TimeoutExpired(["uvx"], 1.0)),
            f"cohere-transcribe did not finish within {DEFAULT_TIMEOUT_S:g} s",
        ),
        (FakeWorker(answer, error=OSError("exec format error")), f"cannot run {UVX}"),
        (FakeWorker(answer, decode_returncode=1), "ffmpeg exited 1: bad mp3"),
    ],
    ids=[
        "worker-line",
        "uvx",
        "stdout",
        "no-answer",
        "not-json",
        "extra-field",
        "count",
        "timeout",
        "unrunnable",
        "decode",
    ],
)
def test_a_failed_run_is_one_ear_error_naming_its_cause(
    tmp_path: Path, fake: FakeWorker, expected: str
) -> None:
    with pytest.raises(EarError) as caught:
        _ears(fake).hear(_audio(tmp_path), [(0.0, 3.0)])

    assert expected in str(caught.value)
    assert "\n" not in str(caught.value)


def _nothing(_name: str) -> str | None:
    return None


def _no_ffmpeg(name: str) -> str | None:
    return UVX if name == "uvx" else None


@pytest.mark.parametrize(
    ("host", "which", "expected"),
    [
        (("Linux", "x86_64"), found, "need Apple silicon"),
        (("Darwin", "x86_64"), found, "this is Darwin x86_64"),
        (("Darwin", "arm64"), _nothing, "uvx is not on PATH"),
        (("Darwin", "arm64"), _no_ffmpeg, "ffmpeg is not on PATH"),
    ],
)
def test_an_unusable_machine_is_refused_before_anything_runs(
    tmp_path: Path,
    host: tuple[str, str],
    which: Callable[[str], str | None],
    expected: str,
) -> None:
    fake = FakeWorker(answer)

    with pytest.raises(EarError, match=expected):
        _ears(fake, host=host, which=which).hear(_audio(tmp_path), [(0.0, 3.0)])
    assert fake.calls == []


@pytest.mark.parametrize("spec", RECOGNIZERS, ids=[spec.name for spec in RECOGNIZERS])
@pytest.mark.parametrize("name", _WEIGHTS)
def test_an_uncached_snapshot_is_refused_before_anything_runs(
    tmp_path: Path, cache: Path, spec: Recognizer, name: str
) -> None:
    (_snapshot(cache, spec.model, spec.revision) / name).unlink()
    fake = FakeWorker(answer)

    with pytest.raises(EarError, match=f"{spec.model} at {spec.revision} is not in the Hugging"):
        _ears(fake).hear(_audio(tmp_path), [(0.0, 3.0)])
    assert fake.calls == []


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ({"HF_HUB_CACHE": "a", "HUGGINGFACE_HUB_CACHE": "b", "HF_HOME": "c"}, "a"),
        ({"HUGGINGFACE_HUB_CACHE": "b", "HF_HOME": "c", "XDG_CACHE_HOME": "d"}, "b"),
        ({"HF_HOME": "c", "XDG_CACHE_HOME": "d"}, "c/hub"),
        ({"XDG_CACHE_HOME": "d"}, "d/huggingface/hub"),
        ({"HOME": "e"}, "e/.cache/huggingface/hub"),
    ],
)
def test_the_cache_is_found_where_hugging_face_looks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: dict[str, str], expected: str
) -> None:
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in names.items():
        monkeypatch.setenv(name, str(tmp_path / value))
    fake = FakeWorker(answer)

    with pytest.raises(EarError) as caught:
        _ears(fake).hear(_audio(tmp_path), [(0.0, 3.0)])

    assert f"under {_snapshot(tmp_path / expected, COHERE.model, COHERE.revision)}" in str(
        caught.value
    )


_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=4),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=4), inner),
    max_leaves=8,
)
_keys = st.sampled_from(["versions", "device", "dtype", "runtime_s", "texts", "other"])


@given(st.one_of(st.binary(max_size=40), _json.map(json.dumps).map(str.encode)))
def test_any_answer_either_parses_or_is_an_ear_error(raw: bytes) -> None:
    with contextlib.suppress(EarError):
        parse_output(COHERE, raw, 1)


@given(_keys, _json)
def test_a_well_formed_answer_with_one_field_changed_parses_or_is_an_ear_error(
    key: str, value: object
) -> None:
    raw = json.dumps(answer([[0.0, 1.0]], **{key: value})).encode()

    with contextlib.suppress(EarError):
        parse_output(COHERE, raw, 1)
