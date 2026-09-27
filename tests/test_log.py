"""Canonical log-configuration test pattern."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from scribe.log import configure

if TYPE_CHECKING:
    import pytest
    from structlog.stdlib import BoundLogger

    from scribe.config import LogLevel


def test_configure_runs_without_error() -> None:
    """Smoke test: the default configuration constructs the structlog chain."""
    configure(level="info", json=False)


def test_configure_idempotent_across_levels_and_renderers() -> None:
    """Reconfiguration mid-process should not raise."""
    levels: list[LogLevel] = ["debug", "info", "warning", "error"]
    for level in levels:
        for json_mode in (True, False):
            configure(level=level, json=json_mode)


def test_logs_go_to_stderr_and_leave_stdout_to_the_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Stdout carries the CLI's output paths and --stdout markdown, never a log line."""
    configure(level="info", json=True)
    try:
        logger: BoundLogger = structlog.get_logger()  # pyright: ignore[reportAny]  # structlog.get_logger returns Any
        logger.info("probe")
    finally:
        structlog.reset_defaults()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "probe" in captured.err
