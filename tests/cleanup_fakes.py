"""A cleanup backend that never leaves the process, shared by the cleanup tests."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from scribe.claude_cli import Completion

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

# Escaped as sent, so a turn's text holds no `<` and a label no `"`.
_SENT = re.compile(r'<t id=(\d+) speaker="[^"]*">([^<]*)</t>')


def keyed_reply(user: str) -> str:
    """Return every turn the prompt sent, unchanged, keyed as the prompt asks."""
    return "\n\n".join(f"<t id={turn_id}>{text}</t>" for turn_id, text in sent_turns(user))


def sent_turns(user: str) -> list[tuple[int, str]]:
    """The id and escaped text of each turn the prompt sent, in order."""
    return [(int(match.group(1)), match.group(2)) for match in _SENT.finditer(user)]


class FakeBackend:
    """Records what it was sent and returns a configurable mutation of a reply.

    The reply is `respond(user)`, by default `keyed_reply`, then `mutate` of it.
    """

    name = "fake-backend"

    def __init__(
        self,
        *,
        model: str = "fake-model",
        max_budget_usd: float = 0.0,
        mutate: Callable[[str], str] | None = None,
        respond: Callable[[str], str] | None = None,
        stop_reasons: list[str | None] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.calls: list[tuple[str, str]] = []
        self._mutate = mutate
        self._respond = respond or keyed_reply
        self._stop_reasons = stop_reasons or []
        self._fail_with = fail_with

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        if self._fail_with is not None:
            raise self._fail_with
        index = len(self.calls) - 1
        stop = self._stop_reasons[index] if index < len(self._stop_reasons) else "end_turn"
        reply = self._respond(user)
        text = reply if self._mutate is None else self._mutate(reply)
        return Completion(
            text=text,
            model=self.model,
            output_tokens=len(text.split()),
            stop_reason=stop,
            is_error=False,
        )


def patch_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: Callable[[str], str] | None = None,
    respond: Callable[[str], str] | None = None,
    stop_reasons: list[str | None] | None = None,
    fail_with: Exception | None = None,
) -> list[FakeBackend]:
    """Swap the CLI's backend for fakes; return the list they are recorded into."""
    made: list[FakeBackend] = []

    def factory(*, model: str, max_budget_usd: float) -> FakeBackend:
        backend = FakeBackend(
            model=model,
            max_budget_usd=max_budget_usd,
            mutate=mutate,
            respond=respond,
            stop_reasons=stop_reasons,
            fail_with=fail_with,
        )
        made.append(backend)
        return backend

    # Patched at the import site, which is where cli.py resolves the name.
    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)
    return made
