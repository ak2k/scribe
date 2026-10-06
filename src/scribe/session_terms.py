"""Session vocabulary and the prompt log, kept by a Claude Code hook.

Each hook event rebuilds one session's term record from the tail of its
transcript; every update then rewrites one merged list, the terms of every
recently active session, which a dictation server can pass to xAI as keyterms.
Every submitted prompt is also logged, as truth for a later bake-off.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, ValidationError

from scribe.errors import InputValidationError
from scribe.xai_stt import MAX_KEYTERM_CHARS, MAX_KEYTERMS, check_keyterms

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Mapping
    from typing import BinaryIO

TAIL_BYTES = 256 * 1024
MERGED_CAP = 60
MIN_TERM_CHARS = 3
WINDOW = timedelta(minutes=30)
RETENTION = timedelta(days=7)
LOG_LIMIT = 1024 * 1024

# The events that are the operator's own activity, which rank a session.
_ACTIVE_EVENTS = frozenset({"UserPromptSubmit", "SessionStart"})
# Interactive sessions; any other entrypoint (such as "sdk-cli") is a headless worker.
_INTERACTIVE = "cli"

# Em and en dashes join prose words, never identifiers.
_SPLIT = re.compile(r"[\s\u2013\u2014]+")
_LEAD = "\"'`([{<*\u201c\u2018"
_TRAIL = "\"'`)]}>*\u201d\u2019.,;:!?"
# "xAI's" would otherwise sit beside "xAI" as a term of its own.
_POSSESSIVE = re.compile(r"['\u2019]s$")
# An identifier or path alphabet: code fragments, shell syntax, and the "=>" and
# leading "#" a terms file reads as an alias or a comment all fall outside it.
_TERM = re.compile(r"[A-Za-z0-9_.\-/~@+:]+")
_DIGIT_LED = re.compile(r"[~$]?\d")
# Case-uniform, as a hash or id prints; a digit is required so that words
# spelled only in a-f, such as "defaced", are not taken for one.
_HEX_RUN = re.compile(r"(?=[a-f]*\d)[\da-f]{7,}|(?=[A-F]*\d)[\dA-F]{7,}")
# A known key prefix followed by key material; "xai-stt" is not a key.
_KEY_PREFIX = re.compile(
    r"(?<![A-Za-z0-9])(?:gh[opusr]_|github_pat_|sk-|xai-|AKIA|xox[a-z]-)[A-Za-z0-9_-]{16,}"
)
_OPAQUE = re.compile(r"[A-Za-z0-9_-]{20,}")


class HookInput(BaseModel):
    """The fields of a Claude Code hook's stdin this hook reads."""

    # DIVERGE: extra="ignore", not "forbid": Claude Code adds hook input fields over
    # time, and a refused event would silently stop the vocabulary and the prompt log.
    model_config = ConfigDict(extra="ignore", frozen=True)

    # It names a file, so nothing that could leave the sessions directory.
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    hook_event_name: str
    cwd: str
    transcript_path: str | None = None
    prompt: str | None = None


class TermCount(BaseModel):
    """One term of a session, with what ranks it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    term: str
    count: int
    last_seen: AwareDatetime


class SessionRecord(BaseModel):
    """One session's terms, best first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    cwd: str
    updated: AwareDatetime
    ranked: AwareDatetime | None = None
    entrypoint: str | None = None
    terms: list[TermCount]


class _Part(BaseModel):
    # DIVERGE: extra="ignore": transcript records are Claude Code's internal format.
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str
    text: str | None = None
    tool_input: dict[str, JsonValue] | None = Field(default=None, alias="input")


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: str | list[_Part] | None = None


class _Origin(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: str | None = None


class _Record(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str = ""
    is_meta: bool = Field(default=False, alias="isMeta")
    git_branch: str | None = Field(default=None, alias="gitBranch")
    entrypoint: str | None = None
    origin: _Origin | None = None
    timestamp: datetime | None = None
    message: JsonValue = None


@dataclass
class _Transcript:
    texts: list[tuple[datetime | None, str]] = field(default_factory=list)
    branch: str | None = None
    entrypoint: str | None = None
    drifted: int = 0


def state_dir(environ: Mapping[str, str], *, home: Path) -> Path:
    """Return scribe's state directory under the XDG base directory rules."""
    configured = environ.get("XDG_STATE_HOME", "")
    base = Path(configured) if Path(configured).is_absolute() else home / ".local" / "state"
    return base / "scribe"


def _credential_shaped(term: str) -> bool:
    if _KEY_PREFIX.search(term):
        return True
    return bool(_OPAQUE.fullmatch(term)) and all(
        any(test(char) for char in term) for test in (str.isupper, str.islower, str.isdigit)
    )


def _admissible(term: str) -> bool:
    return (
        MIN_TERM_CHARS <= len(term) <= MAX_KEYTERM_CHARS
        and _TERM.fullmatch(term) is not None
        and any(char.isalpha() for char in term)
        # Flags, and all-caps words, which as keyterms pull ordinary speech toward them.
        and not term.startswith("-")
        and not (term.isalpha() and term.isupper())
        and _DIGIT_LED.match(term) is None
        and _HEX_RUN.search(term) is None
        and not _credential_shaped(term)
    )


def _identifier_shaped(term: str) -> bool:
    return not term.isalpha() or any(char.isupper() for char in term[1:])


def extract_terms(text: str) -> list[str]:
    """Return the identifier-shaped terms of `text`, in order, repeats included."""
    terms: list[str] = []
    for raw in _SPLIT.split(text):
        token = _POSSESSIVE.sub("", raw.lstrip(_LEAD).rstrip(_TRAIL)).rstrip(_TRAIL)
        # A call keeps its name: "run_hook()" arrives here as "run_hook(".
        token = token.removesuffix("(")
        if "://" in token or token.lower().startswith("www."):
            continue
        if "/" in token:
            base = token.rstrip("/").rsplit("/", 1)[-1]
            # A capitalized basename such as "Makefile" needs no other identifier
            # shape; a lowercase one such as "null" or "status" is an ordinary word.
            if _admissible(base) and (base[0].isupper() or _identifier_shaped(base)):
                terms.append(base)
            if _admissible(token):
                terms.append(token)
        elif _admissible(token) and _identifier_shaped(token):
            terms.append(token)
    return terms


def read_tail(path: Path) -> list[str]:
    """Return the complete lines of the last `TAIL_BYTES` of `path`."""
    with path.open("rb") as handle:
        start = max(0, handle.seek(0, os.SEEK_END) - TAIL_BYTES)
        # One byte early, so a line starting exactly at the cut is seen as complete.
        handle.seek(max(0, start - 1))
        data = handle.read(TAIL_BYTES + 1)
    if start:
        data = data[data.find(b"\n") + 1 :] if b"\n" in data else b""
    # Bytes split on newline only: JSON strings may hold U+2028, which splitlines breaks on.
    return [line.decode("utf-8", errors="replace") for line in data.split(b"\n") if line]


def repo_name(cwd: Path) -> str:
    """Return the basename of the git toplevel holding `cwd`, else of `cwd` itself."""
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory.name
    return cwd.name


def _strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _record_texts(record: _Record) -> list[str]:
    if record.type not in {"user", "assistant"} or record.is_meta:
        return []
    # Task notifications and peer messages arrive as user records too.
    if record.type == "user" and record.origin is not None and record.origin.kind != "human":
        return []
    content = _Message.model_validate(record.message).content
    if isinstance(content, str):
        return [content] if record.type == "user" else []
    texts: list[str] = []
    for part in content or []:
        if part.type == "text" and part.text is not None:
            texts.append(part.text)
        elif part.type == "tool_use" and part.tool_input is not None:
            texts.extend(_strings(part.tool_input))
    return texts


def _read_transcript(lines: list[str]) -> _Transcript:
    found = _Transcript()
    for line in lines:
        try:
            record = _Record.model_validate_json(line)
            texts = _record_texts(record)
        except ValidationError as exc:
            # The newest line may still be half written; that is not drift.
            if all(error["type"] != "json_invalid" for error in exc.errors()):
                found.drifted += 1
            continue
        when = record.timestamp
        if when is not None and when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        found.texts.extend((when, text) for text in texts)
        found.branch = record.git_branch or found.branch
        found.entrypoint = record.entrypoint or found.entrypoint
    return found


def _rank(
    tallies: Counter[str], seen: dict[str, datetime], order: dict[str, int]
) -> list[TermCount]:
    terms = sorted(tallies, key=lambda term: (seen[term], tallies[term], order[term]), reverse=True)
    return [
        TermCount(term=term, count=tallies[term], last_seen=seen[term])
        for term in terms[:MAX_KEYTERMS]
    ]


def _transcript(hook: HookInput, terms_dir: Path, now: datetime) -> _Transcript:
    if not hook.transcript_path:
        return _Transcript()
    path = Path(hook.transcript_path)
    try:
        if not path.is_file():
            # Claude Code may not have written the transcript yet when a session starts.
            if hook.hook_event_name != "SessionStart" or path.exists():
                log_failure(terms_dir, now, f"transcript unreadable: no file at {path}")
            return _Transcript()
        found = _read_transcript(read_tail(path))
    except OSError as exc:
        log_failure(terms_dir, now, f"transcript unreadable: {exc}")
        return _Transcript()
    if found.drifted:
        log_failure(terms_dir, now, f"{found.drifted} transcript records had an unknown shape")
    return found


def _previous(path: Path) -> SessionRecord | None:
    try:
        return SessionRecord.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None


def _session_record(
    hook: HookInput, terms_dir: Path, now: datetime, previous: SessionRecord | None
) -> SessionRecord:
    found = _transcript(hook, terms_dir, now)
    known = [when for when, _ in found.texts if when is not None]
    current = known[0] if known else now
    sources: list[tuple[datetime, list[str]]] = []
    for when, text in found.texts:
        current = when if when is not None else current
        sources.append((current, extract_terms(text)))
    named = [repo_name(Path(hook.cwd)), *([found.branch] if found.branch else [])]
    sources.append(
        (now, [term for term in named if _admissible(term) and _identifier_shaped(term)])
    )
    sources.append((now, extract_terms(hook.prompt or "")))

    tallies: Counter[str] = Counter()
    seen: dict[str, datetime] = {}
    order: dict[str, int] = {}
    for position, (when, term) in enumerate((when, t) for when, terms in sources for t in terms):
        tallies[term] += 1
        seen[term] = max(when, seen.get(term, when))
        order[term] = position
    ranked = previous.ranked if previous is not None else None
    if hook.hook_event_name in _ACTIVE_EVENTS:
        ranked = now if ranked is None else max(ranked, now)
    return SessionRecord(
        session_id=hook.session_id,
        cwd=hook.cwd,
        updated=now,
        ranked=ranked,
        entrypoint=found.entrypoint or (previous.entrypoint if previous is not None else None),
        terms=_rank(tallies, seen, order),
    )


def _replace_atomically(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _append_line(path: Path, line: str) -> None:
    data = memoryview(f"{line}\n".encode(errors="backslashreplace"))
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        # The lock keeps two sessions' lines whole even if a write comes back short.
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        while data:
            data = data[os.write(descriptor, data) :]
    finally:
        os.close(descriptor)


def log_failure(terms_dir: Path, now: datetime, message: str) -> None:
    """Append one line naming a failure to the hook log; never raise."""
    log = terms_dir / "hook.log"
    try:
        terms_dir.mkdir(parents=True, exist_ok=True)
        if log.exists() and log.stat().st_size > LOG_LIMIT:
            log.unlink(missing_ok=True)
        _append_line(log, f"{now.isoformat()} {' '.join(message.split())}")
    except OSError:
        pass


@contextmanager
def _locked(path: Path) -> Generator[None]:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _mergeable(term: str) -> bool:
    try:
        check_keyterms([term])
    except InputValidationError:
        return False
    return _admissible(term)


def write_merged(terms_dir: Path, *, now: datetime) -> None:
    """Rewrite `current.txt` from the recently active sessions; drop week-old records.

    The window's interactive sessions, newest rank time first, take turns filling
    the slots, so one busy session cannot crowd out another.
    """
    sessions = terms_dir / "sessions"
    records: list[SessionRecord] = []
    for path in sorted(sessions.iterdir()) if sessions.is_dir() else []:
        try:
            modified = path.stat().st_mtime
        except FileNotFoundError:
            continue  # removed by a concurrent run of the hook
        if modified < (now - RETENTION).timestamp():
            path.unlink(missing_ok=True)
        # A record is written when it is updated, after its rank time, so an old
        # file cannot be in the window and need not be parsed.
        elif path.suffix == ".json" and modified >= (now - WINDOW).timestamp():
            try:
                records.append(SessionRecord.model_validate_json(path.read_bytes()))
            except ValidationError as exc:
                log_failure(terms_dir, now, f"{path.name} unreadable: {exc.error_count()} errors")
    live = sorted(
        (
            (record.ranked, record.terms)
            for record in records
            if record.ranked is not None
            and record.ranked >= now - WINDOW
            and record.entrypoint in {None, _INTERACTIVE}
        ),
        key=lambda live: live[0],
        reverse=True,
    )
    queues = [
        iter(sorted(terms, key=lambda t: (t.last_seen, t.count), reverse=True)) for _, terms in live
    ]
    merged: list[str] = []
    while queues and len(merged) < MERGED_CAP:
        for queue in list(queues):
            term = next(
                (t.term for t in queue if t.term not in merged and _mergeable(t.term)), None
            )
            if term is None:
                queues.remove(queue)
            elif len(merged) < MERGED_CAP:
                merged.append(term)
    terms_dir.mkdir(parents=True, exist_ok=True)
    _replace_atomically(terms_dir / "current.txt", "".join(f"{term}\n" for term in merged))


def _refusal(exc: ValidationError) -> str:
    # Field locations and rules only: the input itself may be a private prompt.
    causes = (
        f"{'.'.join(str(part) for part in error['loc']) or 'input'}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )
    return "hook input refused: " + "; ".join(causes)


def run_hook(raw: bytes, *, state: Path, now: datetime, host: str) -> None:
    """Log the prompt, rebuild the session's terms and the merged list; never raise."""
    terms_dir = state / "terms"
    try:
        hook = HookInput.model_validate_json(raw)
    except ValidationError as exc:
        log_failure(terms_dir, now, _refusal(exc))
        return
    # The hook must exit 0 with nothing on stdout whatever happens; the log is its only voice.
    try:
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        log_failure(terms_dir, now, f"state directory unusable: {exc}")
        return
    if hook.hook_event_name == "UserPromptSubmit" and hook.prompt is not None:
        entry = {
            "ts": now.isoformat(),
            "host": host,
            "session_id": hook.session_id,
            "cwd": hook.cwd,
            "prompt": hook.prompt,
        }
        try:
            _append_line(state / "prompts.jsonl", json.dumps(entry))
        except Exception as exc:  # noqa: BLE001 - a lost prompt line must not cost the terms
            log_failure(terms_dir, now, f"prompt log: {type(exc).__name__}: {exc}")
    try:
        (terms_dir / "sessions").mkdir(parents=True, exist_ok=True)
        record_path = terms_dir / "sessions" / f"{hook.session_id}.json"
        # One writer at a time, so a slower merge cannot overwrite a newer one.
        with _locked(terms_dir / ".lock"):
            previous = _previous(record_path)
            # Events can arrive out of order; an older one would erase newer terms.
            if previous is not None and now < previous.updated:
                return
            record = _session_record(hook, terms_dir, now, previous)
            _replace_atomically(record_path, record.model_dump_json())
            write_merged(terms_dir, now=now)
    except Exception as exc:  # noqa: BLE001 - any failure becomes one log line, never an exit
        log_failure(terms_dir, now, f"{type(exc).__name__}: {exc}")


def main(stdin: BinaryIO, environ: Mapping[str, str]) -> None:
    """Run the hook on one JSON object read from `stdin`."""
    state = state_dir(environ, home=Path.home())
    now = datetime.now(UTC)
    try:
        raw = stdin.read()
    except OSError as exc:
        log_failure(state / "terms", now, f"stdin unreadable: {exc}")
        return
    run_hook(raw, state=state, now=now, host=socket.gethostname().split(".")[0])
