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
import unicodedata
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, ValidationError

from scribe.errors import InputValidationError
from scribe.xai_stt import MAX_KEYTERM_CHARS, MAX_KEYTERMS, check_keyterms

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import BinaryIO

TAIL_BYTES = 256 * 1024
MERGED_CAP = 60
MIN_TERM_CHARS = 3
WINDOW = timedelta(minutes=30)
RETENTION = timedelta(days=7)

# Em and en dashes join prose words, never identifiers.
_SPLIT = re.compile(r"[\s\u2013\u2014]+")
_LEAD = "\"'`([{<*\u201c\u2018"
_TRAIL = "\"'`)]}>*\u201d\u2019.,;:!?"
_APOSTROPHES = str.maketrans("", "", "'\u2019")
# "xAI's" would otherwise sit beside "xAI" as a term of its own.
_POSSESSIVE = re.compile(r"['\u2019]s$")
# Case-uniform, as a hash or id prints; a digit is required so that words
# spelled only in a-f, such as "defaced", are not taken for one.
_HEX_RUN = re.compile(r"(?=[a-f]*\d)[\da-f]{7,}|(?=[A-F]*\d)[\dA-F]{7,}")
_UNREADABLE = frozenset({"Cc", "Cf"})


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


class _Record(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str = ""
    is_meta: bool = Field(default=False, alias="isMeta")
    git_branch: str | None = Field(default=None, alias="gitBranch")
    timestamp: datetime | None = None
    message: JsonValue = None


def state_dir(environ: Mapping[str, str], *, home: Path) -> Path:
    """Return scribe's state directory under the XDG base directory rules."""
    configured = environ.get("XDG_STATE_HOME", "")
    base = Path(configured) if Path(configured).is_absolute() else home / ".local" / "state"
    return base / "scribe"


def _admissible(term: str) -> bool:
    return (
        MIN_TERM_CHARS <= len(term) <= MAX_KEYTERM_CHARS
        and any(char.isalpha() for char in term)
        and "://" not in term
        and not any(unicodedata.category(char) in _UNREADABLE for char in term)
        and _HEX_RUN.search(term) is None
    )


def _identifier_shaped(term: str) -> bool:
    # Apostrophes are set aside so that contractions read as the words they are.
    core = term.translate(_APOSTROPHES)
    return any(not char.isalpha() for char in core) or any(char.isupper() for char in core[1:])


def extract_terms(text: str) -> list[str]:
    """Return the identifier-shaped terms of `text`, in order, repeats included."""
    terms: list[str] = []
    for raw in _SPLIT.split(text):
        token = _POSSESSIVE.sub("", raw.lstrip(_LEAD).rstrip(_TRAIL)).rstrip(_TRAIL)
        if "://" in token or token.lower().startswith("www."):
            continue
        candidates = [token.rstrip("/").rsplit("/", 1)[-1], token] if "/" in token else [token]
        terms.extend(term for term in candidates if _admissible(term) and _identifier_shaped(term))
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


def _transcript_texts(
    lines: list[str],
) -> tuple[list[tuple[datetime | None, str]], str | None, int]:
    """Return (when, text) pairs, the latest branch, and how many records failed to parse."""
    texts: list[tuple[datetime | None, str]] = []
    branch: str | None = None
    drifted = 0
    for line in lines:
        try:
            record = _Record.model_validate_json(line)
            found = _record_texts(record)
        except ValidationError as exc:
            # The newest line may still be half written; that is not drift.
            if all(error["type"] != "json_invalid" for error in exc.errors()):
                drifted += 1
            continue
        when = record.timestamp
        if when is not None and when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        texts.extend((when, text) for text in found)
        branch = record.git_branch or branch
    return texts, branch, drifted


def _rank(
    tallies: Counter[str], seen: dict[str, datetime], order: dict[str, int]
) -> list[TermCount]:
    terms = sorted(tallies, key=lambda term: (seen[term], tallies[term], order[term]), reverse=True)
    return [
        TermCount(term=term, count=tallies[term], last_seen=seen[term])
        for term in terms[:MAX_KEYTERMS]
    ]


def _session_record(hook: HookInput, terms_dir: Path, now: datetime) -> SessionRecord:
    texts: list[tuple[datetime | None, str]] = []
    branch: str | None = None
    if hook.transcript_path:
        transcript = Path(hook.transcript_path)
        try:
            if not transcript.is_file():
                raise FileNotFoundError(f"no transcript file at {transcript}")
            texts, branch, drifted = _transcript_texts(read_tail(transcript))
            if drifted:
                log_failure(terms_dir, now, f"{drifted} transcript records had an unknown shape")
        except OSError as exc:
            log_failure(terms_dir, now, f"transcript unreadable: {exc}")
    known = [when for when, _ in texts if when is not None]
    current = known[0] if known else now
    sources: list[tuple[datetime, list[str]]] = []
    for when, text in texts:
        current = when if when is not None else current
        sources.append((current, extract_terms(text)))
    named = [repo_name(Path(hook.cwd)), *([branch] if branch else [])]
    sources.append((now, [term for term in named if _admissible(term)]))
    sources.append((now, extract_terms(hook.prompt or "")))

    tallies: Counter[str] = Counter()
    seen: dict[str, datetime] = {}
    order: dict[str, int] = {}
    for position, (when, term) in enumerate((when, t) for when, terms in sources for t in terms):
        tallies[term] += 1
        seen[term] = max(when, seen.get(term, when))
        order[term] = position
    return SessionRecord(
        session_id=hook.session_id, cwd=hook.cwd, updated=now, terms=_rank(tallies, seen, order)
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
    data = memoryview(f"{line}\n".encode())
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
    try:
        terms_dir.mkdir(parents=True, exist_ok=True)
        _append_line(terms_dir / "hook.log", f"{now.isoformat()} {' '.join(message.split())}")
    except OSError:
        pass


def write_merged(terms_dir: Path, *, now: datetime) -> None:
    """Rewrite `current.txt` from the recently updated sessions; drop week-old records."""
    sessions = terms_dir / "sessions"
    records: list[SessionRecord] = []
    for path in sorted(sessions.iterdir()) if sessions.is_dir() else []:
        try:
            modified = path.stat().st_mtime
        except FileNotFoundError:
            continue  # removed by a concurrent run of the hook
        if modified < (now - RETENTION).timestamp():
            path.unlink(missing_ok=True)
        elif path.suffix == ".json":
            try:
                records.append(SessionRecord.model_validate_json(path.read_bytes()))
            except ValidationError as exc:
                log_failure(terms_dir, now, f"{path.name} unreadable: {exc.error_count()} errors")
    merged: list[str] = []
    for record in sorted(records, key=lambda record: record.updated, reverse=True):
        if record.updated < now - WINDOW:
            break
        ranked = sorted(record.terms, key=lambda t: (t.last_seen, t.count), reverse=True)
        for term in (entry.term for entry in ranked):
            if term in merged or len(merged) == MERGED_CAP:
                continue
            try:
                check_keyterms([term])
            except InputValidationError:
                continue
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
    try:
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        if hook.hook_event_name == "UserPromptSubmit" and hook.prompt is not None:
            entry = {
                "ts": now.isoformat(),
                "host": host,
                "session_id": hook.session_id,
                "cwd": hook.cwd,
                "prompt": hook.prompt,
            }
            _append_line(state / "prompts.jsonl", json.dumps(entry))
        record = _session_record(hook, terms_dir, now)
        (terms_dir / "sessions").mkdir(parents=True, exist_ok=True)
        _replace_atomically(
            terms_dir / "sessions" / f"{hook.session_id}.json", record.model_dump_json()
        )
        write_merged(terms_dir, now=now)
    # The hook must exit 0 with nothing on stdout whatever happens; the log is its only voice.
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
