"""The worker's one line for a failure, with torch and transformers faked."""

from __future__ import annotations

import json
import runpy
import sys
from contextlib import nullcontext
from importlib import resources
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, NoReturn

import pytest

from scribe.diarizer import TOKEN_NAMES
from scribe.ear import FAILED, RECOGNIZERS
from tests.diarizer_fakes import TOKEN

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from scribe.ear import Recognizer

# transformers' refusal of a gated model, as it words it over the Hub's own error.
_GATED = (
    "You are trying to access a gated repo.\nMake sure to have access to it at https://hf.co/x."
)
# Hugging Face's client, finding no token saved or named.
_NO_SAVED_TOKEN: dict[str, object] = {"get_token": lambda: None}
# One second of decoded audio at the worker's rate, each sample its own index.
_AUDIO = tuple(range(16000))


class _Inputs(dict[str, object]):
    def to(self, *_args: object, **_kwargs: object) -> _Inputs:
        return self


def _decoded(*_args: object, **_kwargs: object) -> list[int]:
    return list(_AUDIO)


class _Ear:
    """Cohere's processor and model at once: a clip is heard as its first and last samples."""

    device = SimpleNamespace(type="mps")
    dtype = "bfloat16"

    def __init__(self) -> None:
        self.loads: list[str] = []

    def load(self, model: str, **_kwargs: object) -> _Ear:
        self.loads.append(model)
        return self

    def to(self, _device: str) -> _Ear:
        return self

    def eval(self) -> _Ear:
        return self

    def __call__(self, clip: list[int], **_kwargs: object) -> _Inputs:
        return _Inputs(clip=clip)

    def generate(self, *, clip: list[int], **_kwargs: object) -> list[int]:
        return clip

    def decode(self, out: list[int], **_kwargs: object) -> list[str]:
        return [f"samples {out[0]} to {out[-1]}" if out else "no samples"]


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec: Recognizer,
    *,
    mps: bool,
    refusal: str = _GATED,
    hub: Mapping[str, object] = _NO_SAVED_TOKEN,
    ear: _Ear | None = None,
    intervals: Sequence[Sequence[float]] = ((0.0, 1.0),),
) -> object:
    """Run the worker as uvx does, over stand-ins for what it imports; no real model loads.

    Each model refuses with `refusal` unless `ear` stands in for it.
    """

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError(refusal)

    gpu = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    classes = (
        "AutoProcessor",
        "CohereAsrForConditionalGeneration",
        "Qwen3ASRForConditionalGeneration",
    )
    load = refuse if ear is None else ear.load
    stand_ins: dict[str, dict[str, object]] = {
        "numpy": {"fromfile": _decoded},
        "torch": {"bfloat16": "bfloat16", "backends": gpu, "inference_mode": nullcontext},
        "transformers": {name: SimpleNamespace(from_pretrained=load) for name in classes},
        "huggingface_hub": dict(hub),
    }
    for name, attributes in stand_ins.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    request = tmp_path / "in.json"
    named = {"name": spec.name, "model": spec.model, "revision": spec.revision}
    asked = {"intervals": [list(clip) for clip in intervals], "max_new_tokens": 256}
    request.write_text(json.dumps({"audio": "a.f32", **named, **asked}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["ear_worker.py", str(request), str(tmp_path / "out.json")])
    with (
        resources.as_file(resources.files("scribe") / "ear_worker.py") as path,
        pytest.raises(SystemExit) as exited,
    ):
        runpy.run_path(str(path), run_name="__main__")
    return exited.value.code


@pytest.mark.parametrize("spec", RECOGNIZERS, ids=[spec.name for spec in RECOGNIZERS])
def test_a_model_that_cannot_load_is_one_line_naming_its_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    spec: Recognizer,
) -> None:
    code = _run(tmp_path, monkeypatch, spec, mps=True)

    _, err = capsys.readouterr()
    assert code == 1
    assert [line for line in err.splitlines() if line.startswith(FAILED)] == [
        f"{FAILED}OSError: {' '.join(_GATED.split())}"
    ]
    assert not (tmp_path / "out.json").exists()


def test_a_gpu_torch_cannot_use_is_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(tmp_path, monkeypatch, RECOGNIZERS[0], mps=False)

    _, err = capsys.readouterr()
    assert code == 1
    assert err.splitlines() == [f"{FAILED}torch cannot use the Metal GPU (mps) here"]
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("name", TOKEN_NAMES)
def test_a_failure_that_quotes_the_token_does_not_show_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    # The others set but empty, as `NAME=` leaves them; the token with the newline a file ends in.
    for other in TOKEN_NAMES:
        monkeypatch.setenv(other, "")
    monkeypatch.setenv(name, f"{TOKEN}\n")

    code = _run(tmp_path, monkeypatch, RECOGNIZERS[0], mps=True, refusal=f"401 {TOKEN}")

    out, err = capsys.readouterr()
    assert code == 1
    assert TOKEN not in out + err
    assert [line for line in err.splitlines() if line.startswith(FAILED)] == [
        f"{FAILED}OSError: 401 <hidden>"
    ]


def test_a_failure_that_quotes_a_saved_token_does_not_show_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)

    code = _run(
        tmp_path,
        monkeypatch,
        RECOGNIZERS[0],
        mps=True,
        refusal=f"401 {TOKEN}",
        hub={"get_token": lambda: TOKEN},
    )

    out, err = capsys.readouterr()
    assert code == 1
    assert TOKEN not in out + err
    assert [line for line in err.splitlines() if line.startswith(FAILED)] == [
        f"{FAILED}OSError: 401 <hidden>"
    ]


def _unreadable() -> NoReturn:
    raise PermissionError(13, "Permission denied", "/nowhere/huggingface/token")


@pytest.mark.parametrize(
    "hub", [{}, {"get_token": _unreadable}], ids=["no-get-token", "unreadable-token-file"]
)
def test_a_saved_token_that_cannot_be_read_leaves_the_line_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hub: dict[str, object],
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    code = _run(tmp_path, monkeypatch, RECOGNIZERS[0], mps=True, refusal=f"401 {TOKEN}", hub=hub)

    out, err = capsys.readouterr()
    assert code == 1
    assert TOKEN not in out + err
    assert [line for line in err.splitlines() if line.startswith(FAILED)] == [
        f"{FAILED}OSError: 401 <hidden>"
    ]


def test_each_clip_is_heard_as_the_audio_it_names_and_one_past_the_end_is_cut_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ear = _Ear()

    code = _run(
        tmp_path,
        monkeypatch,
        RECOGNIZERS[0],
        mps=True,
        ear=ear,
        intervals=[(0.0, 0.25), (0.5, 1.0), (0.75, 2.5)],
    )

    assert code == 0
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["texts"] == [
        "samples 0 to 3999",
        "samples 8000 to 15999",
        "samples 12000 to 15999",
    ]


# Each after a clip that is fine, in one second of decoded audio.
@pytest.mark.parametrize(
    ("clip", "named"),
    [
        ((-0.25, 0.5), "-0.25 s to 0.5 s"),
        ((-1.0, -0.25), "-1 s to -0.25 s"),
        ((0.5, 0.5), "0.5 s to 0.5 s"),
        ((0.5, 0.25), "0.5 s to 0.25 s"),
        ((0.5, 0.50001), "0.5 s to 0.50001 s"),
        ((1.0, 2.0), "1 s to 2 s"),
        ((1.5, 2.0), "1.5 s to 2 s"),
    ],
    ids=[
        "negative-start",
        "negative-end",
        "empty",
        "backward",
        "under-one-sample",
        "start-at-the-end",
        "start-past-the-end",
    ],
)
def test_a_clip_naming_no_stretch_of_the_audio_is_one_line_before_any_model_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    clip: tuple[float, float],
    named: str,
) -> None:
    ear = _Ear()

    code = _run(
        tmp_path, monkeypatch, RECOGNIZERS[0], mps=True, ear=ear, intervals=[(0.0, 0.5), clip]
    )

    _, err = capsys.readouterr()
    assert code == 1
    assert err.splitlines() == [
        f"{FAILED}clip 1 ({named}) is not a stretch of the 1 s of decoded audio"
    ]
    assert ear.loads == []
    assert not (tmp_path / "out.json").exists()
