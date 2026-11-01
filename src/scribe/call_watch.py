"""Start and stop an Audio Hijack session when a call app holds the microphone.

The decision is a pure function of the state, the time and who holds input. The
reader, the command files and the notifications are the edges around it, and a
sync loop wires them together, since nothing here is concurrent.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import time
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol

import structlog

from scribe.errors import ExternalServiceError, UnsupportedPlatformError
from scribe.session_sources import run_program

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from structlog.stdlib import BoundLogger

    from scribe.session_sources import Runner

CONFIRM_SECONDS = 3.0
GRACE_SECONDS = 30.0
READBACK_SECONDS = 5.0
RESEND_SECONDS = 30.0
OUTAGE_SECONDS = 30.0
POLL_SECONDS = 1.0
HOOK_TIMEOUT_SECONDS = 10.0

DEFAULT_SESSION = "Call"
# Plain prefixes: Chrome's microphone sits on helpers that carry its ID, and Teams ships
# as com.microsoft.teams2, which a rule requiring a dot after the prefix would lose.
DEFAULT_APPS = (
    "com.google.Chrome",
    "us.zoom.",
    "com.microsoft.teams",
    "com.apple.avconferenced",
    "com.apple.TelephonyUtilities",
)
HIJACK_BUNDLE_ID = "com.rogueamoeba.audiohijack"

_NOTIFY_ARGV = (
    "/usr/bin/osascript",
    "-e",
    "on run argv",
    "-e",
    'display notification (item 1 of argv) with title "scribe call-watch"',
    "-e",
    "end run",
)
# Asks LaunchServices, so no Apple Event goes to Audio Hijack and nothing launches it.
# Prints a line naming the app when it runs, and only a line break when it does not.
_RUNNING_ARGV = ("/usr/bin/lsappinfo", "find", f"bundleid={HIJACK_BUNDLE_ID}")
_OPEN_ARGV = ("/usr/bin/open", "-g", "-b", HIJACK_BUNDLE_ID)

Marker = Literal["sent", "confirmed", "stopping"]
Verb = Literal["start", "stop"]
Action = Literal["start", "resend-start", "stop", "resend-stop"]
Level = Literal["info", "warning", "error"]


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


class Read(Enum):
    """The result of a read that has no set of bundle IDs to give."""

    FAILED = "failed"


@dataclass(frozen=True)
class Timings:
    """The watcher's waits, in seconds."""

    confirm_seconds: float = CONFIRM_SECONDS
    grace_seconds: float = GRACE_SECONDS
    readback_seconds: float = READBACK_SECONDS
    resend_seconds: float = RESEND_SECONDS
    outage_seconds: float = OUTAGE_SECONDS


TIMINGS = Timings()


@dataclass(frozen=True)
class Idle:
    """No session of this watcher's is believed to be running."""


@dataclass(frozen=True)
class Starting:
    """A start is sent and not read back.

    `window_from` is None after a restart with no call in progress: nothing is
    waiting on the read-back until a call holds input.
    """

    window_from: float | None = None
    last_sent_at: float = 0.0
    warned: bool = False


@dataclass(frozen=True)
class Recording:
    """A start is read back. `holding` is False once a release during the call was noted."""

    holding: bool = True


@dataclass(frozen=True)
class Stopping:
    """The call has ended.

    `marker` is "stopping" once a stop has gone out. `sent_at` is None until then, and
    after a restart, until the first good read starts the read-back again.
    """

    marker: Marker
    sent_at: float | None = None
    last_sent_at: float = 0.0
    warned: bool = False


type Session = Idle | Starting | Recording | Stopping


@dataclass(frozen=True)
class State:
    """What the watcher remembers between polls.

    Times other than `failing_since` are on a clock that stands still during an
    outage: nothing is known about who held input then, so it counts toward no wait.
    `stood_still` is how far that clock has fallen behind the monotonic one.
    """

    session: Session = Idle()
    held_since: float | None = None
    released_at: float | None = None
    failing_since: float | None = None
    outage_noted: bool = False
    stood_still: float = 0.0

    @property
    def marker(self) -> Marker | None:
        """What the marker file should hold."""
        match self.session:
            case Idle():
                return None
            case Starting():
                return "sent"
            case Recording():
                return "confirmed"
            case Stopping(marker=marker):
                return marker


@dataclass(frozen=True)
class Note:
    """A state change to log, and the notification text if it earns one."""

    event: str
    level: Level = "info"
    notify: str | None = None
    apps: tuple[str, ...] = ()


@dataclass(frozen=True)
class Step:
    """The result of one decision."""

    state: State
    action: Action | None = None
    notes: tuple[Note, ...] = ()


_START_UNCONFIRMED = Note(
    "call_watch.start_unconfirmed",
    "error",
    "A call holds the microphone, but Audio Hijack has not started recording. Retrying.",
)
_STOP_UNCONFIRMED = Note(
    "call_watch.stop_unconfirmed",
    "error",
    "The call ended, but Audio Hijack has not stopped recording. Retrying.",
)
_RELEASED = Note(
    "call_watch.recording_released",
    "warning",
    "This call is no longer recording: Audio Hijack released the microphone.",
)
_READS_FAILING = Note(
    "call_watch.reads_failing",
    "error",
    "Cannot read which apps hold the microphone, so calls are not being watched.",
)


def _is_hijack(bundle: str) -> bool:
    return bundle.lower() == HIJACK_BUNDLE_ID


def is_call_app(bundle: str, apps: Sequence[str]) -> bool:
    """Whether a bundle ID starts with one of `apps`, in any case.

    Audio Hijack never counts: its own hold would keep every session it starts alive.
    """
    lowered = bundle.lower()
    return not _is_hijack(bundle) and any(lowered.startswith(app.lower()) for app in apps)


def restored_state(marker: Marker | None) -> State:
    """The state to begin from, given the marker file left by an earlier run.

    The first good read completes it: a call resumes the session as it was left,
    and no call runs the end grace from zero. A stop on its way needs no call or
    grace, only its read-back.
    """
    match marker:
        case None:
            return State()
        case "sent":
            return State(session=Starting())
        case "confirmed":
            return State(session=Recording())
        case "stopping":
            return State(session=Stopping("stopping"))


def needs_running(state: State, holding: frozenset[str] | Read) -> bool:
    """Whether `decide` needs to know that Audio Hijack is running.

    Only a first stop does, and a process holding input is plainly running.
    """
    return (
        isinstance(state.session, Stopping)
        and state.session.marker != "stopping"
        and holding is not Read.FAILED
        and not any(_is_hijack(bundle) for bundle in holding)
    )


@dataclass(frozen=True)
class _Tick:
    state: State
    clock: float
    call_apps: tuple[str, ...]
    hijack: bool
    running: bool
    timings: Timings

    @property
    def call_held(self) -> bool:
        return bool(self.call_apps)

    @property
    def call_confirmed(self) -> bool:
        since = self.state.held_since
        return since is not None and self.clock - since >= self.timings.confirm_seconds

    @property
    def call_ended(self) -> bool:
        since = self.state.released_at
        return since is not None and self.clock - since >= self.timings.grace_seconds

    def step(
        self, session: Session, action: Action | None = None, notes: tuple[Note, ...] = ()
    ) -> Step:
        return Step(replace(self.state, session=session), action, notes)

    def start(self) -> Step:
        note = Note("call_watch.call_confirmed", apps=self.call_apps)
        return self.step(Starting(self.clock, self.clock), "start", (note,))


def _idle(tick: _Tick, session: Idle) -> Step:
    return tick.start() if tick.call_confirmed else tick.step(session)


def _starting(tick: _Tick, session: Starting) -> Step:
    if tick.hijack:
        return tick.step(Recording(), notes=(Note("call_watch.start_confirmed"),))
    if tick.call_ended:
        return tick.step(Stopping("sent"))
    window = session.window_from
    if window is None:
        if not tick.call_held:
            return tick.step(session)
        window = tick.clock
        session = Starting(window, window)
    notes: tuple[Note, ...] = ()
    if not session.warned and tick.clock - window >= tick.timings.readback_seconds:
        session = replace(session, warned=True)
        notes = (_START_UNCONFIRMED,)
    if tick.call_held and tick.clock - session.last_sent_at >= tick.timings.resend_seconds:
        return tick.step(replace(session, last_sent_at=tick.clock), "resend-start", notes)
    return tick.step(session, notes=notes)


def _recording(tick: _Tick, session: Recording) -> Step:
    if tick.call_ended:
        return tick.step(Stopping("confirmed"))
    # Only while the call holds input: a person's Stop as a call ends loses nothing.
    if tick.call_held and session.holding != tick.hijack:
        note = Note("call_watch.recording_resumed") if tick.hijack else _RELEASED
        return tick.step(Recording(tick.hijack), notes=(note,))
    return tick.step(session)


def _first_stop(tick: _Tick, marker: Marker) -> Step:
    # A stop now would end a session that is capturing the call that came back.
    if tick.call_held:
        if marker == "sent":
            return _starting(tick, Starting())
        return _recording(tick, Recording())
    if tick.hijack or tick.running:
        return tick.step(Stopping("stopping", tick.clock, tick.clock), "stop")
    return tick.step(Idle(), notes=(Note("call_watch.stop_not_needed"),))


def _stopping(tick: _Tick, session: Stopping) -> Step:
    if session.marker != "stopping":
        return _first_stop(tick, session.marker)
    sent_at = session.sent_at
    if sent_at is None:
        sent_at = tick.clock
        session = Stopping("stopping", sent_at, sent_at)
    if not tick.hijack:
        return tick.step(Idle(), notes=(Note("call_watch.stop_done"),))
    # A start before the stop lands would be dropped, and the landing read as a person's
    # Stop. Past the window the hold may be another session's, which never lets go.
    landing = tick.clock - session.last_sent_at < tick.timings.readback_seconds
    if tick.call_confirmed and not landing:
        return tick.start()
    notes: tuple[Note, ...] = ()
    if not session.warned and tick.clock - sent_at >= tick.timings.readback_seconds:
        session = replace(session, warned=True)
        notes = (_STOP_UNCONFIRMED,)
    if tick.clock - session.last_sent_at >= tick.timings.resend_seconds:
        return tick.step(replace(session, last_sent_at=tick.clock), "resend-stop", notes)
    return tick.step(session, notes=notes)


def _read_failed(state: State, now: float, timings: Timings) -> Step:
    since = now if state.failing_since is None else state.failing_since
    failing = replace(state, failing_since=since)
    if state.outage_noted or now - since < timings.outage_seconds:
        return Step(failing)
    return Step(replace(failing, outage_noted=True), notes=(_READS_FAILING,))


def decide(
    state: State,
    now: float,
    holding: frozenset[str] | Read,
    *,
    apps: Sequence[str] = DEFAULT_APPS,
    hijack_running: bool = True,
    timings: Timings = TIMINGS,
) -> Step:
    """Decide one poll.

    Args:
        state: What the last poll left.
        now: The monotonic time.
        holding: The bundle IDs now holding input, or `Read.FAILED`.
        apps: The bundle ID prefixes of call apps.
        hijack_running: Whether Audio Hijack is running. Read only when `needs_running`
            says so.
        timings: The waits.

    Returns:
        The new state, the action to take if any, and the notes to report.

    """
    if holding is Read.FAILED:
        return _read_failed(state, now, timings)
    notes: tuple[Note, ...] = ()
    stood_still = state.stood_still
    if state.failing_since is not None:
        stood_still += now - state.failing_since
        if state.outage_noted:
            notes = (Note("call_watch.reads_recovered"),)
    clock = now - stood_still
    call_apps = tuple(sorted(bundle for bundle in holding if is_call_app(bundle, apps)))
    held = state.held_since if state.held_since is not None else clock
    released = state.released_at if state.released_at is not None else clock
    tick = _Tick(
        replace(
            state,
            held_since=held if call_apps else None,
            released_at=None if call_apps else released,
            failing_since=None,
            outage_noted=False,
            stood_still=stood_still,
        ),
        clock,
        call_apps,
        any(_is_hijack(bundle) for bundle in holding),
        hijack_running,
        timings,
    )
    match state.session:
        case Idle() as session:
            step = _idle(tick, session)
        case Starting() as session:
            step = _starting(tick, session)
        case Recording() as session:
            step = _recording(tick, session)
        case Stopping() as session:
            step = _stopping(tick, session)
    return replace(step, notes=notes + step.notes)


class VanishedProcessError(Exception):
    """A process object went away during a walk. Not a failed read."""


class ProcessObjects(Protocol):
    """The process objects of the system's audio hardware."""

    def object_ids(self) -> Sequence[int]:
        """Return the objects now listed."""
        ...

    def holds_input(self, obj: int) -> bool:
        """Return whether a process object is running input."""
        ...

    def bundle_id(self, obj: int) -> str:
        """Return a process object's bundle ID, empty if it has none."""
        ...


def input_holders(objects: ProcessObjects) -> frozenset[str]:
    """Return the bundle IDs of the process objects holding input."""
    holders: set[str] = set()
    for obj in objects.object_ids():
        try:
            if objects.holds_input(obj) and (bundle := objects.bundle_id(obj)):
                holders.add(bundle)
        except VanishedProcessError:
            continue
    return frozenset(holders)


class _PropertyAddress(ctypes.Structure):
    _fields_ = (
        ("selector", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("element", ctypes.c_uint32),
    )


def _four_cc(code: bytes) -> int:
    value: int = struct.unpack(">i", code)[0]  # pyright: ignore[reportAny]  # the format gives one int
    return value


_SYSTEM_OBJECT = 1
_NO_ERROR = 0
_BAD_OBJECT = _four_cc(b"!obj")
_UTF8 = 0x08000100
_BUNDLE_BYTES = 1024


class CoreAudio:  # pragma: no cover
    """Property reads on the system's audio process objects.

    These only read the audio server's process list: no device opens and no prompt shows.
    """

    # DIVERGE: excluded from coverage because these lines need macOS's frameworks and CI
    # runs on Linux; `input_holders` carries the logic and is tested with a fake.

    def __init__(self) -> None:
        """Load the frameworks, which exist only on macOS.

        Raises:
            ExternalServiceError: a framework cannot be loaded.

        """
        try:
            audio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
            foundation = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
        except OSError as exc:
            raise ExternalServiceError(f"cannot load CoreAudio: {exc}") from exc
        address = ctypes.POINTER(_PropertyAddress)
        size = ctypes.POINTER(ctypes.c_uint32)
        get_size = audio.AudioObjectGetPropertyDataSize
        get_size.argtypes = (ctypes.c_uint32, address, ctypes.c_uint32, ctypes.c_void_p, size)
        get_size.restype = ctypes.c_int32
        get_data = audio.AudioObjectGetPropertyData
        get_data.argtypes = (
            ctypes.c_uint32,
            address,
            ctypes.c_uint32,
            ctypes.c_void_p,
            size,
            ctypes.c_void_p,
        )
        get_data.restype = ctypes.c_int32
        get_string = foundation.CFStringGetCString
        get_string.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32)
        get_string.restype = ctypes.c_ubyte
        release = foundation.CFRelease
        release.argtypes = (ctypes.c_void_p,)
        release.restype = None
        self._get_size: Callable[..., int] = get_size
        self._get_data: Callable[..., int] = get_data
        self._get_string: Callable[..., int] = get_string
        self._release: Callable[..., None] = release

    @staticmethod
    def _address(selector: bytes) -> _PropertyAddress:
        return _PropertyAddress(_four_cc(selector), _four_cc(b"glob"), 0)

    def _status(self, obj: int, selector: bytes, value: ctypes.c_uint32 | ctypes.c_void_p) -> int:
        size = ctypes.c_uint32(ctypes.sizeof(value))
        return self._get_data(
            obj,
            ctypes.byref(self._address(selector)),
            0,
            None,
            ctypes.byref(size),
            ctypes.byref(value),
        )

    def _read(self, obj: int, selector: bytes, value: ctypes.c_uint32 | ctypes.c_void_p) -> None:
        status = self._status(obj, selector, value)
        if status == _NO_ERROR:
            return
        # An exiting process can answer `who?` just before it reads `!obj`. Only the
        # re-read decides: `who?` for every object would otherwise read as nobody holding input.
        again = self._status(obj, selector, value)
        if again == _BAD_OBJECT:
            raise VanishedProcessError
        raise ExternalServiceError(
            f"CoreAudio status {status}, then {again}, reading {selector.decode()}"
        )

    def object_ids(self) -> Sequence[int]:
        """Return the process objects now listed.

        Raises:
            ExternalServiceError: the list cannot be read.

        """
        address = self._address(b"prs#")
        size = ctypes.c_uint32()
        status = self._get_size(_SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size))
        if status != _NO_ERROR:
            raise ExternalServiceError(f"CoreAudio status {status} sizing the process list")
        item = ctypes.sizeof(ctypes.c_uint32)
        buffer = (ctypes.c_uint32 * (size.value // item))()
        status = self._get_data(
            _SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size), buffer
        )
        if status != _NO_ERROR:
            raise ExternalServiceError(f"CoreAudio status {status} reading the process list")
        # The list can shrink between the two calls, and `size` says how much was written.
        return list(buffer)[: size.value // item]

    def holds_input(self, obj: int) -> bool:
        """Return whether a process object is running input."""
        value = ctypes.c_uint32()
        self._read(obj, b"piri", value)
        return value.value != 0

    def bundle_id(self, obj: int) -> str:
        """Return a process object's bundle ID, empty if it has none.

        Raises:
            ExternalServiceError: the ID cannot be copied out of its CFString.

        """
        value = ctypes.c_void_p()
        # The caller owns this CFString, and a watcher reads for months.
        try:
            self._read(obj, b"pbid", value)
            if not value.value:
                return ""
            text = ctypes.create_string_buffer(_BUNDLE_BYTES)
            if not self._get_string(value, text, _BUNDLE_BYTES, _UTF8):
                raise ExternalServiceError("a bundle ID does not fit its buffer")
            return ctypes.string_at(text).decode()
        finally:
            if value.value:
                self._release(value)


@cache
def core_audio() -> CoreAudio:  # pragma: no cover
    """Return the one loaded `CoreAudio`, loading it on first use."""
    return CoreAudio()


def read_input_holders() -> frozenset[str]:  # pragma: no cover
    """Return the bundle IDs now holding input.

    Raises:
        ExternalServiceError: the read failed, which is not the same as an empty set.

    """
    return input_holders(core_audio())


def command_text(session: str, verb: Verb) -> str:
    """Return the Audio Hijack script line that starts or stops a named session."""
    # JSON's escapes are JavaScript's, so any name stays one ASCII string on one line.
    return f"app.sessionWithName({json.dumps(session)}).{verb}()"


def _session_key(session: str) -> str:
    return hashlib.sha256(session.encode(errors="surrogatepass")).hexdigest()


def marker_path(directory: Path, session: str) -> Path:
    """Return the marker file of one session.

    A hash keeps any session name, however long or odd, safe as a filename.
    """
    return directory / f"{_session_key(session)}.marker"


def require_macos(platform: str) -> None:
    """Refuse a platform with no Audio Hijack.

    Raises:
        UnsupportedPlatformError: `platform` is not macOS.

    """
    if platform != "darwin":
        raise UnsupportedPlatformError("call-watch runs only on macOS")


@dataclass
class _Edges:
    """The watcher's side effects. None of them may end the loop."""

    directory: Path
    session: str
    runner: Runner
    saved: Marker | None = None
    marker_failing: bool = False

    @property
    def _marker_path(self) -> Path:
        return marker_path(self.directory, self.session)

    def load_marker(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            raw = self._marker_path.read_bytes()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ExternalServiceError(f"cannot read the call-watch state: {exc}") from exc
        if raw == b"confirmed":
            self.saved = "confirmed"
            return
        if raw == b"stopping":
            self.saved = "stopping"
            return
        # Only this watcher writes the file, so it stands for a session, whatever it holds.
        if raw != b"sent":
            _logger().error("call_watch.marker_unreadable", session=self.session)
        self.saved = "sent"

    def save_marker(self, marker: Marker | None) -> None:
        if marker == self.saved and (marker is None or self._marker_path.exists()):
            return
        try:
            if marker is None:
                self._marker_path.unlink(missing_ok=True)
            else:
                self.directory.mkdir(parents=True, exist_ok=True)
                partial = self._marker_path.with_suffix(".tmp")
                partial.write_text(marker)
                partial.replace(self._marker_path)
        except OSError as exc:
            if not self.marker_failing:
                _logger().error("call_watch.marker_failed", session=self.session, error=str(exc))
            self.marker_failing = True
            return
        self.saved = marker
        self.marker_failing = False

    def send(self, action: Action) -> None:
        verb: Verb = "start" if action in {"start", "resend-start"} else "stop"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Keyed as the marker is, so a watcher of another session never overwrites it.
            path = self.directory / f"{_session_key(self.session)}.{verb}.ahcommand"
            path.write_text(command_text(self.session, verb) + "\n")
            self.runner([*_OPEN_ARGV, str(path)], HOOK_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001  # whatever a hook raises, the loop outlives it
            _logger().error(
                "call_watch.hook_failed", session=self.session, action=action, error=str(exc)
            )
            return
        _logger().info("call_watch.sent", session=self.session, action=action)

    def report(self, note: Note, error: str | None) -> None:
        log = _logger().bind(session=self.session)
        if note.apps:
            log = log.bind(apps=list(note.apps))
        if error is not None:
            log = log.bind(error=error)
        match note.level:
            case "info":
                log.info(note.event)
            case "warning":
                log.warning(note.event)
            case "error":
                log.error(note.event)
        if note.notify is None:
            return
        try:
            self.runner([*_NOTIFY_ARGV, note.notify], HOOK_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001  # a failure to notify changes nothing else
            _logger().error("call_watch.notify_failed", session=self.session, error=str(exc))

    def hijack_running(self) -> bool:
        try:
            return bool(self.runner(_RUNNING_ARGV, HOOK_TIMEOUT_SECONDS).strip())
        except Exception as exc:  # noqa: BLE001  # whatever a hook raises, the loop outlives it
            _logger().error("call_watch.running_check_failed", session=self.session, error=str(exc))
            # A needless stop launches the app for nothing; a missed one lets it record on.
            return True


def _poll(
    state: State,
    now: float,
    reader: Callable[[], frozenset[str]],
    edges: _Edges,
    apps: Sequence[str],
    timings: Timings,
) -> State:
    holding: frozenset[str] | Read
    error: str | None = None
    try:
        holding = reader()
    except Exception as exc:  # noqa: BLE001  # any failure to read is a failed read
        holding = Read.FAILED
        error = str(exc)
    running = edges.hijack_running() if needs_running(state, holding) else True
    step = decide(state, now, holding, apps=apps, hijack_running=running, timings=timings)
    # Attempted before `open`, not required: recording the call outranks restart safety,
    # and a failed write is retried on every poll.
    edges.save_marker(step.state.marker)
    for note in step.notes:
        edges.report(note, error)
    if step.action is not None:
        edges.send(step.action)
    return step.state


def watch(
    directory: Path,
    *,
    session: str = DEFAULT_SESSION,
    apps: Sequence[str] = DEFAULT_APPS,
    timings: Timings = TIMINGS,
    reader: Callable[[], frozenset[str]] = read_input_holders,
    runner: Runner = run_program,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = POLL_SECONDS,
) -> NoReturn:
    """Watch who holds input, and start and stop `session` to match, until killed.

    Args:
        directory: Where the marker and the command files live.
        session: The Audio Hijack session to drive.
        apps: The bundle ID prefixes of call apps.
        timings: The waits.
        reader: Returns the bundle IDs holding input, or raises when it cannot.
        runner: Runs an argv with a timeout and returns its standard output.
        clock: The monotonic time.
        sleep: Waits between polls.
        poll_seconds: The time between polls.

    Raises:
        ExternalServiceError: the state directory cannot be used.

    """
    edges = _Edges(directory, session, runner)
    edges.load_marker()
    state = restored_state(edges.saved)
    while True:
        state = _poll(state, clock(), reader, edges, apps, timings)
        sleep(poll_seconds)
