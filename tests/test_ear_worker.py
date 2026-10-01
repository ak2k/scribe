"""The worker's one line for a failure, with torch and transformers faked."""

from __future__ import annotations

import json
import runpy
import sys
from importlib import resources
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, NoReturn

import pytest

from scribe.ear import FAILED, RECOGNIZERS

if TYPE_CHECKING:
    from pathlib import Path

    from scribe.ear import Recognizer

# transformers' refusal of a gated model, as it words it over the Hub's own error.
_GATED = (
    "You are trying to access a gated repo.\nMake sure to have access to it at https://hf.co/x."
)


def _refuse(*_args: object, **_kwargs: object) -> NoReturn:
    raise OSError(_GATED)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: Recognizer, *, mps: bool) -> object:
    """Run the worker as uvx does, over stand-ins for what it imports; no model loads."""
    gpu = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    classes = (
        "AutoProcessor",
        "CohereAsrForConditionalGeneration",
        "Qwen3ASRForConditionalGeneration",
    )
    stand_ins: dict[str, dict[str, object]] = {
        "numpy": {},
        "torch": {"bfloat16": "bfloat16", "backends": gpu},
        "transformers": {name: SimpleNamespace(from_pretrained=_refuse) for name in classes},
    }
    for name, attributes in stand_ins.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    request = tmp_path / "in.json"
    named = {"name": spec.name, "model": spec.model, "revision": spec.revision}
    request.write_text(
        json.dumps({"audio": "a.f32", **named, "intervals": [[0.0, 1.0]], "max_new_tokens": 256}),
        encoding="utf-8",
    )
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
