"""Shared test fixtures."""

from collections.abc import Iterator
from typing import NoReturn

import pytest
import structlog
from hypothesis import settings

# A loaded machine can stretch any example past a fixed deadline, so no property
# test's verdict may depend on how long one example ran. The active profile stays
# the parent so a CI environment keeps the rest of what it selects.
settings.register_profile("scribe", settings.default, deadline=None)
settings.load_profile("scribe")


@pytest.fixture
def anyio_backend() -> str:
    """Default anyio backend for async tests. Override per-test if needed."""
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_structlog() -> Iterator[None]:
    """Undo a CLI run's logging setup, which binds the stderr that run captured."""
    yield
    structlog.reset_defaults()


@pytest.fixture(autouse=True)
def plain_typer_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render typer's help and error panels unstyled at 80 columns, whatever shell or CI runs."""
    # typer reads GITHUB_ACTIONS, FORCE_COLOR, PY_COLORS and TERMINAL_WIDTH once, at import,
    # so clearing them here would come too late. A fixed answer also keeps rich from
    # consulting TTY_COMPATIBLE and FORCE_COLOR on each render, and a fixed width wins over
    # COLUMNS. 80 is rich's own width when no stream is a terminal, as under pytest's capture.
    monkeypatch.setattr("typer.rich_utils.FORCE_TERMINAL", False)
    monkeypatch.setattr("typer.rich_utils.MAX_WIDTH", 80)


@pytest.fixture(autouse=True)
def no_real_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a CLI test that reaches the real backend, rather than let it spawn claude."""

    def refuse(**_settings: object) -> NoReturn:
        raise AssertionError(
            "reached the real claude backend; swap it in or pass --no-llm-speakers"
        )

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", refuse)


@pytest.fixture(autouse=True)
def no_real_parakeet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a CLI test that reaches the real parakeet backend, rather than let it spawn uvx."""

    def refuse() -> NoReturn:
        raise AssertionError("reached the real parakeet backend; swap it in")

    monkeypatch.setattr("scribe.cli.ParakeetMlx", refuse)


@pytest.fixture(autouse=True)
def no_real_diarizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a CLI test that reaches the real pyannote backend, rather than let it spawn uvx."""

    def refuse() -> NoReturn:
        raise AssertionError("reached the real pyannote backend; swap it in")

    monkeypatch.setattr("scribe.cli.PyannoteDiarizer", refuse)


@pytest.fixture(autouse=True)
def no_real_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that reaches the real server, rather than let it bind a port."""

    def refuse(*_args: object, **_settings: object) -> NoReturn:
        raise AssertionError("reached the real server; swap uvicorn.run out")

    monkeypatch.setattr("uvicorn.run", refuse)
