"""The worker's one line for a model it cannot load, with pyannote and the Hub faked."""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib import resources
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, cast

import pytest

from scribe.diarizer import FAILED, MODEL
from tests.diarizer_fakes import TOKEN

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


# The pinned huggingface_hub's error classes, as far as the worker tells them apart.
class HfHubHTTPError(OSError):
    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status)


class RepositoryNotFoundError(HfHubHTTPError): ...


class GatedRepoError(RepositoryNotFoundError): ...


class RevisionNotFoundError(HfHubHTTPError): ...


class EntryNotFoundError(Exception): ...


class LocalEntryNotFoundError(FileNotFoundError, EntryNotFoundError): ...


def _uncached(cause: BaseException | None) -> LocalEntryNotFoundError:
    """What hf_hub_download raises when the file is not cached and the Hub gave no file."""
    error = LocalEntryNotFoundError(f"cannot find the requested files {TOKEN}")
    error.__cause__ = cause
    return error


def _worker(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> Callable[[list[str]], int]:
    """Load the worker over stand-ins for what it imports; loading the model raises `error`."""

    def refuse(_model: str, *, revision: str) -> NoReturn:
        raise error

    errors = {
        "GatedRepoError": GatedRepoError,
        "HfHubHTTPError": HfHubHTTPError,
        "LocalEntryNotFoundError": LocalEntryNotFoundError,
    }
    stand_ins: dict[str, dict[str, object]] = {
        "numpy": {},
        "soundfile": {},
        "torch": {
            "device": str,
            "backends": SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        },
        "huggingface_hub": {},
        "huggingface_hub.errors": dict(errors),
        "pyannote": {},
        "pyannote.audio": {"Pipeline": SimpleNamespace(from_pretrained=refuse)},
    }
    for name, attributes in stand_ins.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    # Loading turns pyannote's telemetry off in this process's environment; this undoes it.
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
    with resources.as_file(resources.files("scribe") / "diarize_worker.py") as path:
        spec = importlib.util.spec_from_file_location("diarize_worker", path)
        assert spec is not None
        assert spec.loader is not None
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
    return cast("Callable[[list[str]], int]", vars(worker)["main"])


_TOKEN_ADVICE = (
    "; it is gated: HF_TOKEN, or a saved `hf auth login`, must hold a token whose account "
    f"accepted its terms at https://hf.co/{MODEL}"
)
_UNREACHABLE = ": it is not in the Hugging Face cache, and the Hub is unreachable"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            GatedRepoError(f"403 {TOKEN}", status=403),
            f" from Hugging Face (GatedRepoError){_TOKEN_ADVICE}",
        ),
        (
            RepositoryNotFoundError(f"401 {TOKEN}", status=401),
            f" from Hugging Face (RepositoryNotFoundError){_TOKEN_ADVICE}",
        ),
        (
            HfHubHTTPError(f"401 {TOKEN}", status=401),
            f" from Hugging Face (HfHubHTTPError){_TOKEN_ADVICE}",
        ),
        (
            _uncached(HfHubHTTPError(f"403 {TOKEN}", status=403)),
            f" from Hugging Face (LocalEntryNotFoundError){_TOKEN_ADVICE}",
        ),
        (_uncached(None), f"{_UNREACHABLE} (LocalEntryNotFoundError)"),
        (_uncached(ConnectionError(TOKEN)), f"{_UNREACHABLE} (LocalEntryNotFoundError)"),
        (
            _uncached(HfHubHTTPError(f"503 {TOKEN}", status=503)),
            f"{_UNREACHABLE} (LocalEntryNotFoundError)",
        ),
        (
            RevisionNotFoundError(f"404 {TOKEN}", status=404),
            " from Hugging Face: HTTP 404 (RevisionNotFoundError)",
        ),
        (
            HfHubHTTPError(f"429 {TOKEN}", status=429),
            " from Hugging Face: HTTP 429 (HfHubHTTPError)",
        ),
    ],
    ids=[
        "gated",
        "no-token",
        "bad-token",
        "forbidden-uncached",
        "offline",
        "no-connection",
        "hub-down",
        "no-such-revision",
        "rate-limited",
    ],
)
def test_a_model_that_cannot_load_is_one_line_naming_its_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    expected: str,
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    main = _worker(monkeypatch, error)
    request = tmp_path / "in.json"
    request.write_text(
        json.dumps({"audio": "a.wav", "model": MODEL, "revision": "abc", "intervals": []}),
        encoding="utf-8",
    )

    code = main([str(request), str(tmp_path / "out.json")])

    out, err = capsys.readouterr()
    assert code == 1
    assert err.splitlines() == [f"{FAILED}cannot load {MODEL} at abc{expected}"]
    assert TOKEN not in out + err
    assert not (tmp_path / "out.json").exists()
