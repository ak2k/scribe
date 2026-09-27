"""Canonical settings test pattern."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scribe.config import get_settings


def test_settings_default_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRIBE_LOG_LEVEL", raising=False)
    monkeypatch.delenv("SCRIBE_LOG_JSON", raising=False)
    settings = get_settings()
    assert settings.log_level == "info"
    assert settings.log_json is False


def test_env_overrides_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRIBE_LOG_JSON", "true")
    assert get_settings().log_json is True


def test_log_level_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad LogLevel literal value fails at construction, not at use."""
    monkeypatch.setenv("SCRIBE_LOG_LEVEL", "verbose")
    with pytest.raises(ValidationError):
        get_settings()
