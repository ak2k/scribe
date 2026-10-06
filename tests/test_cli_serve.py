from __future__ import annotations

import importlib.util
import json
import sys
from typing import TYPE_CHECKING, cast

import anyio
import httpx
import pytest
from structlog.testing import capture_logs
from typer.testing import CliRunner

from scribe import cli, session_sources
from scribe.cli import app
from scribe.focus import Focus
from scribe.serve import dictation_client
from tests.xai_fixtures import xai_payload

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from starlette.types import ASGIApp

    from scribe.xai_stt import XaiStt

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
        {
            "host": "127.0.0.1",
            "port": 8765,
            "log_level": "warning",
            "access_log": False,
            "timeout_graceful_shutdown": 10,
        }
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
        (["--session-terms-host", "box-a", "--no-session-terms"], "--no-session-terms"),
        (["--session-terms-host", ""], "--session-terms-host"),
        (["--session-terms-host=-oProxyCommand=x"], "--session-terms-host"),
        (["--session-terms-host", "box a"], "--session-terms-host"),
        (["--session-terms-host", "box\x1b"], "--session-terms-host"),
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


@pytest.fixture
def served_apps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[ASGIApp]:
    """Record each app a server start would serve, with a fake xAI behind it."""
    apps: list[ASGIApp] = []

    def record(app_: ASGIApp, **_settings: object) -> None:
        apps.append(app_)

    def fake_xai(_key: str) -> XaiStt:
        reply = httpx.Response(200, json=xai_payload())
        return dictation_client(KEY, transport=httpx.MockTransport(lambda _request: reply))

    monkeypatch.setattr("uvicorn.run", record)
    monkeypatch.setattr("scribe.serve.dictation_client", fake_xai)
    monkeypatch.setenv("XAI_API_KEY", KEY)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return apps


def _dictate_and_check_health(app_: ASGIApp) -> dict[str, object]:
    async def go() -> dict[str, object]:
        transport = httpx.ASGITransport(app=app_)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            response = await client.post(
                "/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF", "audio/wav")}
            )
            assert response.status_code == 200
            return cast("dict[str, object]", (await client.get("/health")).json())

    # The CLI run bound its logging to a stream the runner has since closed.
    with capture_logs():
        return anyio.run(go)


def _kept_record(tmp_path: Path) -> dict[str, object]:
    (kept,) = (tmp_path / "state" / "scribe" / "serve").iterdir()
    return cast("dict[str, object]", json.loads((kept / "result.json").read_text()))


def test_session_terms_come_from_the_local_hook_file_and_each_host(
    served_apps: list[ASGIApp], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[Path] = []
    real = session_sources.local_source

    def recording(path: Path) -> session_sources.Source:
        read.append(path)
        return real(path)

    monkeypatch.setattr(session_sources, "local_source", recording)
    args = ["serve", "--session-terms-host", "box-a", "--session-terms-host", "box-b"]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert read == [tmp_path / "state" / "scribe" / "terms" / "current.txt"]
    (served,) = served_apps
    health = _dictate_and_check_health(served)
    assert [
        source["source"] for source in cast("list[dict[str, object]]", health["session_terms"])
    ] == [
        "local file",
        "box-a",
        "box-b",
    ]
    record = _kept_record(tmp_path)
    assert cast("dict[str, object]", record["flags"])["session_terms"] == [
        "local file",
        "box-a",
        "box-b",
    ]
    assert record["session_terms"] == {"local file": 0, "box-a": 0, "box-b": 0}


def test_no_session_terms_reads_no_source(served_apps: list[ASGIApp], tmp_path: Path) -> None:
    result = runner.invoke(app, ["serve", "--no-session-terms"])

    assert result.exit_code == 0, result.output
    (served,) = served_apps
    assert _dictate_and_check_health(served)["session_terms"] == []
    assert cast("dict[str, object]", _kept_record(tmp_path)["flags"])["session_terms"] == []


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> list[Focus]:
    """Record each focus poller the CLI builds; each answers that Ghostty is not in front."""
    made: list[Focus] = []

    def build() -> Focus:
        made.append(Focus(lambda _argv, _timeout: "back\n"))
        return made[-1]

    monkeypatch.setattr("scribe.focus.ghostty_focus", build)
    return made


def test_on_macos_focus_is_on_by_default(
    served_apps: list[ASGIApp],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory: list[Focus],
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    assert len(factory) == 1
    (served,) = served_apps
    assert _dictate_and_check_health(served)["focus"] == {"state": "never", "age_seconds": None}
    record = _kept_record(tmp_path)
    assert cast("dict[str, object]", record["flags"])["focus"] == "ghostty"
    assert record["focus"] == {"verdict": "miss", "terms": 0}


@pytest.mark.parametrize(
    ("platform", "args"),
    [
        ("darwin", ["--no-focus"]),
        ("darwin", ["--no-session-terms"]),
        ("linux", []),
    ],
)
def test_focus_is_off_when_disabled_without_session_terms_or_off_macos(
    served_apps: list[ASGIApp],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory: list[Focus],
    platform: str,
    args: list[str],
) -> None:
    monkeypatch.setattr(sys, "platform", platform)

    result = runner.invoke(app, ["serve", *args])

    assert result.exit_code == 0, result.output
    assert factory == []
    (served,) = served_apps
    assert _dictate_and_check_health(served)["focus"] is None
    record = _kept_record(tmp_path)
    assert cast("dict[str, object]", record["flags"])["focus"] is None
    assert record["focus"] == {"verdict": "off", "terms": 0}


def test_help_lists_no_focus() -> None:
    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--no-focus" in result.output
