"""The focused session for `scribe serve`: asked for by each request, matched to a session block.

A dictation goes to the session being typed into. herdr's socket names the
focused pane's Claude Code session outright; failing that, when the frontmost app
is a Ghostty tab running Claude Code, the tab's title and directory single out
that session's block in the local hook file. The block's terms then lead the
keyterms.
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import threading
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anyio
import anyio.from_thread
import anyio.lowlevel
import structlog
from anyio.streams.buffered import BufferedByteReceiveStream
from pydantic import BaseModel, ConfigDict

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
# herdr answers in about 0.3 ms; a miss then costs a fifth of the Ghostty query's cap at most.
HERDR_CAP_SECONDS = 0.05
HERDR_REQUEST = b'{"id":"scribe","method":"pane.current","params":{}}\n'
# A pane's reply is about a kilobyte; past this it is not one.
MAX_HERDR_REPLY_BYTES = 1 << 16

Verdict = Literal["off", "slow", "error", "away", "miss", "ambiguous", "elsewhere", "hit"]
Source = Literal["herdr", "ghostty", "off"]
Check = Literal["agree", "disagree", "unmatched", "away", "unknown"]
type _Match = tuple[Verdict, SessionBlock | None]


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
    # An untitled tab picks nothing: Claude Code's default title is no session's own evidence.
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


class _HerdrModel(BaseModel):
    # DIVERGE: extra="ignore" because herdr adds pane fields across versions and its
    # reply carries many that are not read here.
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class _AgentSession(_HerdrModel):
    source: str
    kind: str
    value: str


class _Pane(_HerdrModel):
    focused: bool
    agent: str | None = None
    agent_session: _AgentSession | None = None


class _PaneCurrent(_HerdrModel):
    type: Literal["pane_current"]
    pane: _Pane


class _HerdrReply(_HerdrModel):
    result: _PaneCurrent


def herdr_match(reply: bytes, blocks: Sequence[SessionBlock]) -> SessionBlock | None:
    """The block of the Claude Code session in herdr's focused pane, or None.

    The session id must name exactly one block. Strings are only compared, so
    nothing here can stall on the filesystem.
    """
    pane = _HerdrReply.model_validate_json(reply).result.pane
    session = pane.agent_session
    if (
        not pane.focused
        or pane.agent != "claude"
        or session is None
        or session.source != "herdr:claude"
        or session.kind != "id"
        or not session.value
    ):
        return None
    found = [block for block in blocks if block.session_id == session.value]
    return found[0] if len(found) == 1 else None


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
    # the spelling it was given, and a macOS volume ignores case and Unicode form, so
    # the two can differ in those alone.
    return bool(tab) and _spelling(session) == _spelling(tab)


def _spelling(cwd: str) -> str:
    return unicodedata.normalize("NFC", _resolved(cwd)).casefold()


class Focus:
    """The focused session, asked of herdr and then Ghostty by each request."""

    def __init__(
        self,
        runner: Runner,
        *,
        cap: float = QUERY_CAP_SECONDS,
        herdr: Path | None = None,
        herdr_cap: float = HERDR_CAP_SECONDS,
    ) -> None:
        """Hold a query run with `runner`, which a request waits on for at most `cap` seconds.

        herdr's socket at `herdr`, when given, is asked first, for at most `herdr_cap` seconds.
        """
        self.cap = cap
        self.herdr_cap = herdr_cap
        self.name = "ghostty" if herdr is None else "herdr+ghostty"
        self._runner = runner
        self._herdr = herdr
        self._state: Literal["never", "front", "away", "error", "slow"] = "never"
        self._source: Source | None = None
        self._failure: str | None = None
        # A query past the cap is abandoned, not stopped, and a stalled mount can hold its
        # thread for good; one permit keeps that to one thread, however many requests come.
        self._permit = threading.Lock()

    def _ask(self, recent: Sequence[SessionBlock]) -> _Match | None:
        tab = parse_answer(self._runner(ARGV, QUERY_TIMEOUT_SECONDS))
        # Matched here, under the cap: resolving a directory can hang on a stalled mount.
        return None if tab is None else match(tab, recent)

    async def _query(self, recent: Sequence[SessionBlock]) -> _Match | str | None:
        """The match, None when Ghostty is not in front, or the cause of a failure.

        A request that finds the last query still running is "slow" at once, as that
        query's answer did not come within its own cap either.
        """
        if not self._permit.acquire(blocking=False):
            return "slow"
        token = anyio.lowlevel.current_token()
        done = anyio.Event()
        answer: list[_Match | str | None] = []

        def run() -> None:
            try:
                answer.append(self._ask(recent))
            except Exception as exc:  # noqa: BLE001  # no query may fail a dictation; the code is the cause
                answer.append(_cause(exc))
            finally:
                self._permit.release()
            # A request past the cap no longer waits, and its loop may have closed.
            with suppress(RuntimeError):
                anyio.from_thread.run_sync(done.set, token=token)

        # Not AnyIO's pool: a pooled job cancelled before a worker takes it strands that
        # worker, while a started thread always runs `run` and so frees the permit.
        try:
            threading.Thread(target=run, name="focus query", daemon=True).start()
        except Exception as exc:  # noqa: BLE001  # a thread that never starts never frees the permit
            self._permit.release()
            return _cause(exc)
        with anyio.move_on_after(self.cap):
            await done.wait()
            return answer[0]
        return "slow"

    async def _ask_herdr(self, herdr: Path, recent: Sequence[SessionBlock]) -> SessionBlock | None:
        with anyio.move_on_after(self.herdr_cap):
            try:
                async with await anyio.connect_unix(herdr) as stream:
                    await stream.send(HERDR_REQUEST)
                    reply = await BufferedByteReceiveStream(stream).receive_until(
                        b"\n", MAX_HERDR_REPLY_BYTES
                    )
                return herdr_match(reply, recent)
            # Absent, refused, closed early or any other shape: Ghostty is asked instead,
            # and the reason is not logged, as the reply can quote a title or a path.
            except Exception:  # noqa: BLE001  # no herdr answer may fail a dictation
                return None
        return None

    def _check(self, picked: SessionBlock, recent: Sequence[SessionBlock]) -> None:
        """Log whether Ghostty's own match agrees with a herdr pick, off the request path."""
        if not self._permit.acquire(blocking=False):
            _logger().info("serve.focus_check", agree="unknown")
            return

        def run() -> None:
            outcome: Check
            try:
                answer = self._ask(recent)
            except Exception:  # noqa: BLE001  # a check only measures; its failure is "unknown"
                outcome = "unknown"
            else:
                if answer is None:
                    outcome = "away"
                elif answer[1] is None:
                    outcome = "unmatched"
                else:
                    outcome = "agree" if answer[1] is picked else "disagree"
            finally:
                self._permit.release()
            _logger().info("serve.focus_check", agree=outcome)

        try:
            threading.Thread(target=run, name="focus check", daemon=True).start()
        except Exception:  # noqa: BLE001  # a thread that never starts never frees the permit
            self._permit.release()
            _logger().info("serve.focus_check", agree="unknown")

    async def pick(
        self, recent: Sequence[SessionBlock]
    ) -> tuple[Verdict, SessionBlock | None, Source]:
        """The focused block among `recent`, the local blocks ranked within 24 h, and who named it.

        herdr, then Ghostty, is asked now, so a pane or tab chosen just before the
        dictation counts and no earlier one's session can be put first.
        """
        if not recent:
            return "miss", None, "ghostty"
        # DIVERGE: herdr's pick stands even when another app is in front: focus only
        # orders which session's terms come first, and the pane typed into last is a
        # better guess then than the newest session.
        if self._herdr is not None:
            block = await self._ask_herdr(self._herdr, recent)
            if block is not None:
                self._source = "herdr"
                self._check(block, recent)
                return "hit", block, "herdr"
        self._source = "ghostty"
        answer = await self._query(recent)
        if isinstance(answer, str):
            if answer != self._failure:
                _logger().warning("serve.focus_failed", error=answer)
            self._failure = answer
            self._state = "slow" if answer == "slow" else "error"
            return ("slow" if answer == "slow" else "error"), None, "ghostty"
        if self._failure is not None:
            _logger().info("serve.focus_restored")
            self._failure = None
        if answer is None:
            self._state = "away"
            return "away", None, "ghostty"
        self._state = "front"
        return *answer, "ghostty"

    def health(self) -> dict[str, object]:
        """The last Ghostty query's outcome and the source of the last pick; nothing from either."""
        return {"state": self._state, "source": self._source}


def herdr_socket() -> Path:
    """herdr's socket: `$HERDR_SOCKET_PATH` when set, else herdr's default."""
    configured = os.environ.get("HERDR_SOCKET_PATH")
    return Path(configured) if configured else Path.home() / ".config" / "herdr" / "herdr.sock"


def ghostty_focus() -> Focus:
    """The focused-session query, asking herdr's socket and then Ghostty with osascript."""
    return Focus(run_program, herdr=herdr_socket())
