"""Move a finished call's two stems into the folder a sync client uploads, with a sidecar.

A sync client watching the recorder's own folder uploads partial copies while
the call runs, so the stems are recorded elsewhere and arrive here whole. The
sidecar names the pair and each stem's size and md5, which a reader of the
synced folder compares with the copies it sees.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scribe.errors import (
    AppError,
    InputValidationError,
    NotFoundError,
    PublishBusyError,
    PublishError,
)
from scribe.gemini_stt import probe_duration
from scribe.session_terms import state_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from structlog.stdlib import BoundLogger

    from scribe.gemini_stt import Runner

SCHEMA = "scribe.call/1"
# Audio Hijack names a second run in the same minute "... mic 2" and "... app 2".
STEM = re.compile(r"(Call [0-9]{8} [0-9]{4}) (mic|app)( [0-9]+)?\.mp4")
PART = ".part"
LOCK_SECONDS = 300.0
_RENAME_EXCL = 0x4

Side = Literal["mic", "app"]
SIDES: tuple[Side, Side] = ("mic", "app")
# What a stem needs: a copy, only its source deleted (a publish that stopped
# after its renames), or nothing.
Step = Literal["copy", "drop", "done"]

default_run: Runner = subprocess.run


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


class StemRecord(BaseModel):
    """One stem as the sidecar describes it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    size: int = Field(ge=0)
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    duration_ms: int = Field(gt=0)


class Stems(BaseModel):
    """The pair, by side."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mic: StemRecord
    app: StemRecord


class Sidecar(BaseModel):
    """The JSON written beside a published pair; its shape is fixed by its readers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # `schema` is taken on BaseModel, so the field carries it as its alias only.
    schema_id: Literal["scribe.call/1"] = Field(default=SCHEMA, alias="schema")
    # ISO 8601 text with a numeric offset; pydantic would write a UTC time with Z.
    start: str
    end: str
    host: str
    stems: Stems


class ReceiptStem(BaseModel):
    """Where one stem was published, and the md5 it had there."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")


class Receipt(BaseModel):
    """Left in the recorder folder before a publish deletes its sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mic: ReceiptStem
    app: ReceiptStem


@dataclass(frozen=True)
class _Measured:
    record: StemRecord
    birth: float | None
    atime_ns: int
    mtime_ns: int


@dataclass(frozen=True)
class _Slot:
    """Where a pair is published: its numbered names in the folder, and what each side needs."""

    finals: Mapping[Side, Path]
    sidecar: Path
    steps: Mapping[Side, Step]


def _split(mic: Path, far: Path) -> tuple[str, int]:
    mine, theirs = STEM.fullmatch(mic.name), STEM.fullmatch(far.name)
    if (
        mine is None
        or theirs is None
        or (mine[2], theirs[2]) != ("mic", "app")
        or (mine[1], mine[3]) != (theirs[1], theirs[3])
    ):
        raise InputValidationError(
            f"not the mic and app pair of one call: {mic.name!r} and {far.name!r}"
        )
    return mine[1], int(mine[3]) if mine[3] else 1


def _suffix(number: int) -> str:
    return "" if number == 1 else f" {number}"


def pair_name(mic: Path, far: Path) -> str:
    """Return the name a mic and app stem of one call share, as the sidecar is named."""
    base, number = _split(mic, far)
    return base + _suffix(number)


def birth_time(info: os.stat_result) -> float | None:
    """Return a file's birth time where the platform records one."""
    if sys.platform == "darwin":
        return info.st_birthtime
    return None


def _digest(path: Path) -> tuple[int, str]:
    # md5 because the synced folder's API reports md5Checksum, not for security.
    digest = hashlib.new("md5", usedforsecurity=False)
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
            size += len(block)
        # The source is deleted on the strength of this copy, so it must be on disk first.
        os.fsync(handle.fileno())
    return size, digest.hexdigest()


def _measure(path: Path, run: Runner) -> _Measured:
    try:
        info = path.stat()
        size, md5 = _digest(path)
    except OSError as exc:
        raise PublishError(f"cannot read {path}: {exc.strerror or exc}") from exc
    if size != info.st_size:
        raise PublishError(f"{path} changed while it was read")
    duration_ms = round(probe_duration(path, run) * 1000)
    if duration_ms < 1:
        raise PublishError(f"{path} lasts under a millisecond")
    return _Measured(
        record=StemRecord(name=path.name, size=size, md5=md5, duration_ms=duration_ms),
        birth=birth_time(info),
        atime_ns=info.st_atime_ns,
        mtime_ns=info.st_mtime_ns,
    )


def _staging(final: Path) -> Path:
    # Per run and created exclusively, so no other run's staged bytes can be
    # verified, overwritten or installed as this run's.
    part = final.with_name(f"{final.name}.{os.getpid()}-{secrets.token_hex(4)}{PART}")
    os.close(os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
    return part


def _write_json(path: Path, text: str) -> None:
    part: Path | None = None
    try:
        part = _staging(path)
        with part.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        part.replace(path)
    except OSError as exc:
        if part is not None:
            part.unlink(missing_ok=True)
        raise PublishError(f"cannot write {path}: {exc.strerror or exc}") from exc


def copy_stem(source: Path, part: Path) -> None:
    """Copy a stem's bytes to its .part name in the publish folder."""
    shutil.copyfile(source, part)


def _stamp_times(part: Path, source: _Measured) -> None:
    if source.birth is not None:
        # macOS moves a birth time back to an mtime set earlier than it, so the
        # birth time goes in first, through the mtime, and the mtime after it.
        birth_ns = round(source.birth * 1e9)
        os.utime(part, ns=(source.atime_ns, birth_ns))
    os.utime(part, ns=(source.atime_ns, source.mtime_ns))


def _local(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).astimezone().isoformat(timespec="seconds")


def _write_sidecar(path: Path, measured: Mapping[Side, _Measured], host: str) -> None:
    mic, far = measured["mic"], measured["app"]
    start = (
        mic.birth if mic.birth is not None else mic.mtime_ns / 1e9 - mic.record.duration_ms / 1000
    )
    car = Sidecar(
        start=_local(int(start)),
        end=_local(max(mic.mtime_ns, far.mtime_ns) // 10**9),
        host=host,
        stems=Stems(mic=mic.record, app=far.record),
    )
    _write_json(path, car.model_dump_json(by_alias=True, indent=2) + "\n")


def _sync_folder(folder: Path) -> None:
    # Makes the renames durable before a source is deleted; a platform that
    # cannot sync a folder this way leaves only the files' own sync.
    with contextlib.suppress(OSError):
        descriptor = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _same_folder(first: Path, second: Path) -> bool:
    if first.exists() and second.exists():
        return first.samefile(second)
    return first == second


@contextlib.contextmanager
def _locked(path: Path, timeout: float) -> Generator[None]:
    # One lock for every publish this user runs, so two of them never choose a
    # slot or install into one at once.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise PublishError(f"cannot open {path}: {exc.strerror or exc}") from exc
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise PublishBusyError(
                        f"another publish held {path} for {timeout:g} s"
                    ) from None
                time.sleep(0.1)
        yield
    finally:
        os.close(descriptor)


@cache
def _renamex_np() -> Callable[[bytes, bytes, int], int]:
    function = ctypes.CDLL(None, use_errno=True).renamex_np
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    renamex: Callable[[bytes, bytes, int], int] = function
    return renamex


def install_final(part: Path, final: Path, platform: str = sys.platform) -> None:
    """Give a staged file its final name, failing with FileExistsError if the name is taken.

    A rename would replace a final another run installed after this run checked.
    """
    if platform == "darwin":
        if _renamex_np()(os.fsencode(part), os.fsencode(final), _RENAME_EXCL) != 0:
            code = ctypes.get_errno()
            if code == errno.EEXIST:
                raise FileExistsError(code, os.strerror(code), str(final))
            raise OSError(code, os.strerror(code), str(final))
        return
    os.link(part, final)
    part.unlink()


def short_host() -> str:
    """Return this machine's host name up to its first dot."""
    return socket.gethostname().split(".")[0]


def publish(
    mic: Path,
    far: Path,
    to: Path | None = None,
    *,
    run: Runner | None = None,
    host: str | None = None,
    state: Path | None = None,
    lock_timeout: float | None = None,
) -> tuple[Path, Path]:
    """Put a call's mic and app stems in `to` with their sidecar; return the two final paths.

    Without `to`, or with the stems' own folder, nothing moves and the sidecar
    is written beside them. Otherwise both stems arrive whole or neither does,
    under the first number from the stems' own that holds no other recording,
    and a source is deleted only once both copies and the sidecar are in place.
    A run after one that stopped partway finishes its work. Runs wait on one
    lock in `state` (scribe's state directory) and give up after `lock_timeout`.
    """
    base, number = _split(mic, far)
    sources: dict[Side, Path] = {"mic": mic.absolute(), "app": far.absolute()}
    folder = sources["mic"].parent.resolve()
    if not _same_folder(sources["app"].parent.resolve(), folder):
        raise InputValidationError(f"{mic.name!r} and {far.name!r} are in different folders")
    probe = run if run is not None else default_run
    machine = host if host is not None else short_host()
    target = folder if to is None else to.expanduser().resolve()
    home = state if state is not None else state_dir(os.environ, home=Path.home())
    wait = lock_timeout if lock_timeout is not None else LOCK_SECONDS
    with _locked(home / "call-publish.lock", wait):
        if _same_folder(target, folder):
            measured: dict[Side, _Measured] = {s: _measure(sources[s], probe) for s in SIDES}
            _write_sidecar(folder / f"{base}{_suffix(number)}.json", measured, machine)
            _logger().info("call sidecar written", call=base, folder=str(folder), moved=False)
            return folder / mic.name, folder / far.name
        receipt = folder / f"{base}{_suffix(number)}.published.json"
        return _publish_into(target, base, number, sources, receipt, probe, machine)


def _publish_into(
    target: Path,
    base: str,
    number: int,
    sources: Mapping[Side, Path],
    receipt: Path,
    probe: Runner,
    machine: str,
) -> tuple[Path, Path]:
    if not target.is_dir():
        raise NotFoundError(f"no folder {target} to publish into")
    measured: dict[Side, _Measured] = {
        side: _measure(sources[side], probe) for side in SIDES if sources[side].exists()
    }
    present: dict[Side, StemRecord] = {side: found.record for side, found in measured.items()}
    if len(present) == len(SIDES):
        slot = _choose_slot(target, base, number, present)
    else:
        slot = _receipt_slot(receipt, target, base, present)
    finals = slot.finals
    log = _logger().bind(call=slot.sidecar.stem, folder=str(target))
    if all(step == "done" for step in slot.steps.values()) and slot.sidecar.exists():
        log.info("call already published")
        return finals["mic"], finals["app"]
    for side in SIDES:
        if slot.steps[side] == "done":
            measured[side] = _measure(finals[side], probe)
        # The sidecar names the published file, which a taken name renumbers.
        named = measured[side].record.model_copy(update={"name": finals[side].name})
        measured[side] = replace(measured[side], record=named)

    copying: list[Side] = [side for side in SIDES if slot.steps[side] == "copy"]
    _move(copying, sources, finals, measured)
    _write_sidecar(slot.sidecar, measured, machine)
    _sync_folder(target)
    leaving: list[Side] = [side for side in SIDES if slot.steps[side] != "done"]
    if not leaving:
        log.info("call sidecar written", moved=False)
        return finals["mic"], finals["app"]
    # Another run can install its own pair between this run's check and rename,
    # so a source goes only while its own bytes sit at the final it names.
    ours = all(_md5_of(finals[side]) == measured[side].record.md5 for side in leaving)
    if not (ours and slot.sidecar.exists()):
        raise PublishError(f"the stems in {target} are not this call's; kept the sources")
    ticket = Receipt(
        mic=ReceiptStem(path=str(finals["mic"]), md5=measured["mic"].record.md5),
        app=ReceiptStem(path=str(finals["app"]), md5=measured["app"].record.md5),
    )
    _write_json(receipt, ticket.model_dump_json(indent=2) + "\n")
    _sync_folder(receipt.parent)
    for side in leaving:
        # Both copies, the sidecar and the receipt are in place, so a source
        # left here is a leftover, which the next run's equal-content step removes.
        try:
            sources[side].unlink()
        except OSError as exc:
            # TRY400: an expected, recovered failure; a traceback would add nothing.
            log.error("cannot delete a published source", path=str(sources[side]), error=str(exc))  # noqa: TRY400
    log.info("call published", copied=copying)
    return finals["mic"], finals["app"]


def _choose_slot(target: Path, base: str, number: int, present: Mapping[Side, StemRecord]) -> _Slot:
    # A final name is never overwritten: a pair whose names are taken by other
    # recordings goes under the next number, as Audio Hijack would name it.
    while True:
        suffix = _suffix(number)
        finals: dict[Side, Path] = {side: target / f"{base} {side}{suffix}.mp4" for side in SIDES}
        steps: dict[Side, Step] = {}
        for side in SIDES:
            if not finals[side].exists():
                steps[side] = "copy"
            elif _same_content(finals[side], present[side]):
                steps[side] = "drop"
        if len(steps) == len(SIDES):
            return _Slot(finals=finals, sidecar=target / f"{base}{suffix}.json", steps=steps)
        number += 1


def _receipt_slot(
    receipt: Path, target: Path, base: str, present: Mapping[Side, StemRecord]
) -> _Slot:
    # A missing source was published only where its receipt says, with the md5
    # it had: a file that merely carries its name may be another call's.
    ticket = _read_receipt(receipt)
    finals: dict[Side, Path] = {"mic": Path(ticket.mic.path), "app": Path(ticket.app.path)}
    md5s: dict[Side, str] = {"mic": ticket.mic.md5, "app": ticket.app.md5}
    try:
        published, number = _split(finals["mic"], finals["app"])
    except InputValidationError as exc:
        raise PublishError(f"published copy not found: {receipt} names no pair") from exc
    if published != base or not all(_same_folder(f.parent, target) for f in finals.values()):
        raise PublishError(f"published copy not found: {receipt} names no pair in {target}")
    steps: dict[Side, Step] = {}
    for side in SIDES:
        want = present.get(side)
        if want is None:
            if _md5_of(finals[side]) != md5s[side]:
                raise PublishError(f"published copy not found: {finals[side]} is gone or changed")
            steps[side] = "done"
        elif want.md5 != md5s[side]:
            raise PublishError(f"{receipt} names another recording than {want.name}")
        elif not finals[side].exists():
            steps[side] = "copy"
        elif _same_content(finals[side], want):
            steps[side] = "drop"
        else:
            raise PublishError(f"{finals[side]} differs from its source")
    return _Slot(finals=finals, sidecar=target / f"{base}{_suffix(number)}.json", steps=steps)


def _read_receipt(receipt: Path) -> Receipt:
    try:
        return Receipt.model_validate_json(receipt.read_bytes())
    except FileNotFoundError as exc:
        raise PublishError(f"published copy not found: no receipt at {receipt}") from exc
    except (OSError, ValidationError) as exc:
        reason = " ".join(str(exc).split())
        raise PublishError(f"published copy not found: unreadable {receipt}: {reason}") from exc


def _md5_of(final: Path) -> str | None:
    try:
        return _digest(final)[1]
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PublishError(f"cannot read {final}: {exc.strerror or exc}") from exc


def _same_content(final: Path, source: StemRecord) -> bool:
    try:
        if final.stat().st_size != source.size:
            return False
        return _digest(final) == (source.size, source.md5)
    except OSError as exc:
        raise PublishError(f"cannot read {final}: {exc.strerror or exc}") from exc


def _move(
    copying: list[Side],
    sources: Mapping[Side, Path],
    finals: Mapping[Side, Path],
    measured: Mapping[Side, _Measured],
) -> None:
    parts: dict[Side, Path] = {}
    installed: dict[Side, int] = {}
    try:
        for side in copying:
            parts[side] = _staging(finals[side])
            copy_stem(sources[side], parts[side])
            if _digest(parts[side]) != (measured[side].record.size, measured[side].record.md5):
                raise PublishError(f"the copy of {sources[side]} does not match it")
        for side in copying:
            _stamp_times(parts[side], measured[side])
        for side in copying:
            if finals[side].exists():
                raise InputValidationError(f"{finals[side]} appeared while publishing")
            # Both ways of installing keep the staged file's inode.
            inode = parts[side].stat().st_ino
            try:
                install_final(parts[side], finals[side])
            except FileExistsError as exc:
                raise InputValidationError(f"{finals[side]} appeared while publishing") from exc
            installed[side] = inode
    except (OSError, AppError) as exc:
        # A final name is never left for one side only. Only a final this run
        # installed goes, and only while it is still that file; its source is
        # still in place.
        for side, inode in installed.items():
            try:
                if finals[side].stat().st_ino == inode:
                    finals[side].unlink()
            except OSError as undo:
                _logger().error("cannot take a stem back", path=str(finals[side]), error=str(undo))
        # Only this run's own staged files; another run's are not ours to remove.
        for part in parts.values():
            with contextlib.suppress(OSError):
                part.unlink(missing_ok=True)
        if isinstance(exc, AppError):
            raise
        folder = next(iter(finals.values())).parent
        raise PublishError(f"cannot publish into {folder}: {exc.strerror or exc}") from exc
