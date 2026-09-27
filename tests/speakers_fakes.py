"""A speaker backend that never leaves the process, shared by the speaker tests."""

from __future__ import annotations

import re
import threading
import time
from typing import TYPE_CHECKING

from scribe.claude_cli import Completion
from scribe.errors import ExternalServiceError

if TYPE_CHECKING:
    from collections.abc import Callable

_TARGET = re.compile(r"<target>\n(.*)\n</target>", re.DOTALL)


def target_of(user: str) -> str:
    """The tagged chunk a user prompt asks the model to return."""
    found = _TARGET.search(user)
    assert found is not None, user
    return found.group(1)


def echo(target: str) -> str:
    return target


def service_error() -> BaseException:
    return ExternalServiceError("claude -p exited 1 with subtype 'error'")


class FakeSpeakerBackend:
    """Answers each chunk with `reply(target)` inside <out> tags, from any thread.

    With `wrap=False` the reply is returned as is, <out> tags and all left to it.
    """

    name = "fake-backend"

    def __init__(
        self,
        *,
        model: str = "fake-model",
        reply: Callable[[str], str] = echo,
        fail_when: Callable[[str], bool] | None = None,
        fail_with: Callable[[], BaseException] = service_error,
        wrap: bool = True,
        delay_s: float = 0.0,
        stop_reason: str = "end_turn",
    ) -> None:
        self.model = model
        self._reply = reply
        self._fail_when = fail_when
        self._fail_with = fail_with
        self._wrap = wrap
        self._delay_s = delay_s
        self._stop_reason = stop_reason
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []
        self.in_flight = 0
        self.peak = 0

    def resolve(self) -> str:
        return "/opt/nowhere/bin/claude"

    def complete(self, system: str, user: str) -> Completion:
        with self._lock:
            self.calls.append((system, user))
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(self._delay_s)
            target = target_of(user)
            if self._fail_when is not None and self._fail_when(target):
                raise self._fail_with()
            text = self._reply(target)
            return Completion(
                text=f"<out>\n{text}\n</out>" if self._wrap else text,
                model=self.model,
                stop_reason=self._stop_reason,
            )
        finally:
            with self._lock:
                self.in_flight -= 1
