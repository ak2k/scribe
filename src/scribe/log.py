"""Canonical logging configuration. Call `configure(...)` once at startup.

Demonstrates:
- structlog setup with an ordered processor chain.
- JSON output for production; ConsoleRenderer for dev.
- Type-safe level mapping (no `getattr` -> Any).
- Idempotent — safe to call from tests and app entry points.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from structlog.typing import Processor

    from scribe.config import LogLevel

LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure(level: LogLevel = "info", json: bool = False) -> None:
    """Configure structlog + stdlib logging.

    Args:
        level: Minimum log level to emit; below-level calls are dropped.
        json: True for JSON output (production); False for ConsoleRenderer (dev).

    """
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Color only on a terminal: a redirected log would carry the escapes.
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(LEVELS[level]),
        # structlog prints to stdout by default, and a CLI's stdout carries its
        # output paths and --stdout markdown.
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
