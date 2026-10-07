"""The focused Ghostty tab for `scribe serve`: asked for by each request, matched to a session.

A dictation goes to the frontmost app; when that is a Ghostty tab running Claude
Code, the tab's title and directory single out that session's block in the
local hook file, whose terms then lead the keyterms.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anyio
import anyio.to_thread
import structlog

from scribe.errors import AppError, InputValidationError
from scribe.session_sources import run_program

if TYPE_CHECKING:
    from collections.abc import Sequence

    from structlog.stdlib import BoundLogger

    from scribe.session_sources import Runner
    from scribe.session_terms import SessionBlock

# Fields are joined by linefeeds: a tab character can occur in a title.
SCRIPT = """\
if application id "com.mitchellh.ghostty" is not running then return "absent"
tell application id "com.mitchellh.ghostty"
  if not frontmost then return "back"
  if (count of windows) is 0 then return "back"
  set trm to focused terminal of selected tab of front window
  set ttl to name of trm
  set wd to working directory of trm
  if ttl is missing value then set ttl to ""
  if wd is missing value then set wd to ""
  return "front" & linefeed & ttl & linefeed & wd
end tell"""
# By absolute path, so the Automation grant belongs to the system osascript.
ARGV = ("/usr/bin/osascript", "-e", SCRIPT)
QUERY_TIMEOUT_SECONDS = 2.0
# How long a request waits for the answer: a warm query takes about 60 ms, and a
# dictation is better sent unfocused than late.
QUERY_CAP_SECONDS = 0.25
# Claude Code's terminal title before a session has one, as `title_key` reduces it.
UNTITLED_KEY = "claude code"
_CODE = re.compile(r"\((-\d{1,6})\)\s*$")

Verdict = Literal["off", "slow", "error", "away", "miss", "ambiguous", "elsewhere", "hit"]


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


def title_key(title: str) -> str:
    """The part of a title that a tab and its session's title share.

    Claude Code and Ghostty prepend status glyphs (spinner frames, activity
    stars, the bell) that change between releases, so any leading run of
    symbols, format characters, variation selectors and spaces is dropped.
    Punctuation is not a symbol and stays.
    """

    def marker(char: str) -> bool:
        category = unicodedata.category(char)
        return (
            category[0] == "S" or category == "Cf" or "\ufe00" <= char <= "\ufe0f" or char.isspace()
        )

    return "".join(itertools.dropwhile(marker, title)).strip().lower()


@dataclass(frozen=True)
class Tab:
    """The focused terminal of Ghostty's front window."""

    title: str
    cwd: str


def parse_answer(text: str) -> Tab | None:
    """Read the script's answer: a tab, or None when Ghostty is not running or not in front.

    Raises:
        InputValidationError: any other shape. The message quotes nothing from the text.

    """
    answer = text.removesuffix("\n")
    if answer in {"absent", "back"}:
        return None
    parts = answer.split("\n")
    if len(parts) != 3 or parts[0] != "front":  # noqa: PLR2004  # the marker, a title and a cwd
        raise InputValidationError("unparseable focus answer")
    return Tab(parts[1], parts[2])


def match(tab: Tab, blocks: Sequence[SessionBlock]) -> tuple[Verdict, SessionBlock | None]:
    """Pick the block of the session `tab` shows, among local blocks ranked within 24 h.

    A block is picked only when it alone holds the tab's title, now or among its
    earlier titles, and its session works in the tab's directory: a tab's title can
    lag its session's re-title, so whichever session holds that title now is no
    surer a pick than the one that held it before.
    """
    key = title_key(tab.title)
    # An untitled tab picks nothing: a session gets its block only at its first prompt,
    # so a new session's directory would find a sibling's block instead of its own.
    if key in {"", UNTITLED_KEY}:
        return "miss", None
    found = [b for b in blocks if any(title_key(title) == key for title in b.titles)]
    if not found:
        return "miss", None
    if len(found) > 1:
        return "ambiguous", None
    if not _same_directory(found[0].cwd, tab.cwd):
        return "elsewhere", None
    return "hit", found[0]


def _cause(exc: Exception) -> str:
    """A failure as a code that holds nothing from the tab: the error output can quote a title."""
    if isinstance(exc.__cause__, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(exc.__cause__, OSError):
        return "missing"
    code = _CODE.search(str(exc)) if isinstance(exc, AppError) else None
    return code[1] if code is not None else "unparseable"


def _resolved(cwd: str) -> str:
    try:
        return str(Path(cwd).resolve()) if cwd else ""
    # A loop raises RuntimeError, a NUL byte ValueError.
    except (OSError, RuntimeError, ValueError):
        return cwd


def _same_directory(session: str, tab: str) -> bool:
    # A tab reports the logical $PWD, a session its physical directory. Resolving keeps
    # the spelling it was given and a macOS volume ignores case, so the two can differ
    # in case alone.
    return bool(tab) and _resolved(session).casefold() == _resolved(tab).casefold()


class Focus:
    """The focused Ghostty tab, asked for by each request."""

    name = "ghostty"

    def __init__(self, runner: Runner, *, cap: float = QUERY_CAP_SECONDS) -> None:
        """Hold a query run with `runner`, which a request waits on for at most `cap` seconds."""
        self.cap = cap
        self._runner = runner
        self._state: Literal["never", "front", "away", "error", "slow"] = "never"
        self._failure: str | None = None

    def _ask(self) -> Tab | None:
        return parse_answer(self._runner(ARGV, QUERY_TIMEOUT_SECONDS))

    async def _query(self) -> Tab | str | None:
        """The tab, None when Ghostty is not in front, or the cause of a failure."""
        with anyio.move_on_after(self.cap):
            try:
                # Abandoned at the cap: the query's own timeout ends it soon after.
                return await anyio.to_thread.run_sync(self._ask, abandon_on_cancel=True)
            except Exception as exc:  # noqa: BLE001  # no query may fail a dictation; the code is the cause
                return _cause(exc)
        return "slow"

    async def pick(self, recent: Sequence[SessionBlock]) -> tuple[Verdict, SessionBlock | None]:
        """The focused block among `recent`, the local blocks ranked within 24 h.

        Ghostty is asked now, so a tab chosen just before the dictation counts and
        no earlier tab's session can be put first.
        """
        if not recent:
            return "miss", None
        answer = await self._query()
        if isinstance(answer, str):
            if answer != self._failure:
                _logger().warning("serve.focus_failed", error=answer)
            self._failure = answer
            self._state = "slow" if answer == "slow" else "error"
            return ("slow" if answer == "slow" else "error"), None
        if self._failure is not None:
            _logger().info("serve.focus_restored")
            self._failure = None
        if answer is None:
            self._state = "away"
            return "away", None
        self._state = "front"
        return match(answer, recent)

    def health(self) -> dict[str, object]:
        """The last query's outcome; nothing from the tab."""
        return {"state": self._state}


def ghostty_focus() -> Focus:
    """The focused-tab query, asking Ghostty with osascript."""
    return Focus(run_program)
