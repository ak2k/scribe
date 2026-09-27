"""Canonical configuration pattern. Copy this shape for new services.

Demonstrates:
- pydantic-settings for typed env-var loading at the process boundary.
- `extra="forbid"` + `frozen=True` so accidental kwargs fail loudly and
  settings can't drift mid-request.
- `.env` file support for local dev (production sets env vars directly).
- Field descriptions for self-documenting errors and tooling.
- Literal type for enumerated values, rather than bare `str`.
- `get_settings()` factory: the one place settings are constructed, so a
  required field added later needs its suppression here only — BaseSettings
  fills it from env, but type checkers flag the missing argument.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["debug", "info", "warning", "error"]


class Settings(BaseSettings):
    """Application settings sourced from environment + `.env` file."""

    model_config = SettingsConfigDict(
        env_prefix="SCRIBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    log_level: LogLevel = Field(
        default="info",
        description="Application log level.",
    )
    log_json: bool = Field(
        default=False,
        description="Emit logs as JSON (production) or pretty text (dev).",
    )


def get_settings() -> Settings:
    """Construct Settings from the environment and `.env`."""
    return Settings()
