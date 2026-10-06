from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner

from scribe import cli
from scribe.cli import app

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

runner = CliRunner()
KEY = "xai-test-key-never-logged"


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, object]]:
    """Record each server start instead of binding a port."""
    starts: list[dict[str, object]] = []

    def record(_app: object, **settings: object) -> None:
        starts.append(settings)

    monkeypatch.setattr("uvicorn.run", record)
    monkeypatch.setenv("XAI_API_KEY", KEY)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return starts


def test_it_serves_on_loopback_and_names_the_url_to_configure(
    served: list[dict[str, object]],
) -> None:
    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    assert served == [
        {"host": "127.0.0.1", "port": 8765, "log_level": "warning", "access_log": False}
    ]
    assert "http://127.0.0.1:8765/v1/audio/transcriptions" in result.output
    assert KEY not in result.output


def test_the_default_terms_file_is_read_from_the_config_home(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    terms = tmp_path / "config" / "scribe" / "terms.txt"
    terms.parent.mkdir(parents=True)
    terms.write_text("ok\nbad =>\n", encoding="utf-8")

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 2
    assert f"{terms}:2" in result.output
    assert served == []


@pytest.mark.parametrize(
    ("args", "why"),
    [
        (["--host", "0.0.0.0"], "loopback"),  # noqa: S104  # the refused value under test
        (["--host", "192.168.1.5"], "loopback"),
        (["--terms", "absent.txt"], "absent.txt"),
        (["--keep", "k", "--no-keep"], "--no-keep"),
    ],
)
def test_a_bad_start_exits_two_with_one_line_before_serving(
    served: list[dict[str, object]], args: list[str], why: str
) -> None:
    result = runner.invoke(app, ["serve", *args])

    assert result.exit_code == 2
    assert why in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert served == []


def test_without_a_key_it_exits_two(
    served: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XAI_API_KEY")

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 2
    assert "XAI_API_KEY" in result.output
    assert served == []


def test_ipv6_loopback_is_served_with_a_bracketed_url(served: list[dict[str, object]]) -> None:
    result = runner.invoke(app, ["serve", "--host", "::1", "--port", "9000"])

    assert result.exit_code == 0, result.output
    assert "http://[::1]:9000/v1/audio/transcriptions" in result.output


def test_the_cli_loads_without_the_server_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("scribe.serve", "uvicorn", "starlette"):
        monkeypatch.setitem(sys.modules, name, cast("ModuleType", None))
    spec = importlib.util.spec_from_file_location("scribe_cli_alone", cli.__file__)
    assert spec is not None
    assert spec.loader is not None

    spec.loader.exec_module(importlib.util.module_from_spec(spec))
