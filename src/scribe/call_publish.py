"""Move a finished call's two stems into the folder a sync client uploads, with a sidecar.

A sync client watching the recorder's own folder uploads partial copies while
the call runs, so the stems are recorded elsewhere and arrive here whole. The
sidecar names the pair and each stem's size and md5, which a reader of the
synced folder compares with the copies it sees.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from scribe.errors import AppError, InputValidationError, NotFoundError, PublishError
from scribe.gemini_stt import probe_duration

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from structlog.stdlib import BoundLogger

    from scribe.gemini_stt import Runner

SCHEMA = "scribe.call/1"
# Audio Hijack names a second run in the same minute "... mic 2" and "... app 2".
STEM = re.compile(r"(Call [0-9]{8} [0-9]{4}) (mic|app)( [0-9]+)?\.mp4")
PART = ".part"

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


@dataclass(frozen=True)
class _Measured:
    record: StemRecord
    birth: float | None
    atime_ns: int
    mtime_ns: int


def pair_name(mic: Path, far: Path) -> str:
    """Return the name a mic and app stem of one call share, as the sidecar is named."""
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
    return mine[1] + (mine[3] or "")


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
    seconds = probe_duration(path, run)
    return _Measured(
        record=StemRecord(name=path.name, size=size, md5=md5, duration_ms=round(seconds * 1000)),
        birth=birth_time(info),
        atime_ns=info.st_atime_ns,
        mtime_ns=info.st_mtime_ns,
    )


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
    part = path.with_name(path.name + PART)
    try:
        with part.open("w", encoding="utf-8") as handle:
            handle.write(car.model_dump_json(by_alias=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        part.replace(path)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise PublishError(f"cannot write {path}: {exc.strerror or exc}") from exc


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
) -> tuple[Path, Path]:
    """Put a call's mic and app stems in `to` with their sidecar; return the two final paths.

    Without `to`, or with the stems' own folder, nothing moves and the sidecar
    is written beside them. Otherwise both stems arrive whole or neither does,
    and a source is deleted only once its copy and the sidecar are in place.
    A run after one that stopped partway finishes its work.
    """
    name = pair_name(mic, far)
    sources: dict[Side, Path] = {"mic": mic.absolute(), "app": far.absolute()}
    folder = sources["mic"].parent.resolve()
    if not _same_folder(sources["app"].parent.resolve(), folder):
        raise InputValidationError(f"{mic.name!r} and {far.name!r} are in different folders")
    probe = run if run is not None else default_run
    machine = host if host is not None else short_host()
    target = folder if to is None else to.expanduser().resolve()
    log = _logger().bind(call=name, folder=str(target))

    if _same_folder(target, folder):
        finals = (folder / mic.name, folder / far.name)
        measured = {side: _measure(sources[side], probe) for side in SIDES}
        _write_sidecar(folder / f"{name}.json", measured, machine)
        log.info("call sidecar written", moved=False)
        return finals

    if not target.is_dir():
        raise NotFoundError(f"no folder {target} to publish into")
    finals_by: dict[Side, Path] = {side: target / sources[side].name for side in SIDES}
    car = target / f"{name}.json"
    steps = {side: _plan(sources[side], finals_by[side]) for side in SIDES}
    if all(step == "done" for step in steps.values()) and car.exists():
        log.info("call already published")
        return finals_by["mic"], finals_by["app"]

    measured: dict[Side, _Measured] = {}
    for side in SIDES:
        held = finals_by[side] if steps[side] == "done" else sources[side]
        measured[side] = _measure(held, probe)
        if steps[side] == "drop":
            _refuse_different(finals_by[side], measured[side].record)

    copying: list[Side] = [side for side in SIDES if steps[side] == "copy"]
    _move(copying, sources, finals_by, measured)
    _write_sidecar(car, measured, machine)
    _sync_folder(target)
    for side in SIDES:
        if steps[side] == "done":
            continue
        try:
            if finals_by[side].stat().st_size != measured[side].record.size:
                raise PublishError(f"{finals_by[side]} is not its source's size; kept the source")
            sources[side].unlink()
        except OSError as exc:
            raise PublishError(f"cannot delete {sources[side]}: {exc.strerror or exc}") from exc
    log.info("call published", copied=copying)
    return finals_by["mic"], finals_by["app"]


def _plan(source: Path, final: Path) -> Step:
    if source.exists():
        return "drop" if final.exists() else "copy"
    if final.exists():
        return "done"
    raise NotFoundError(f"no recording at {source}, and none published at {final}")


def _refuse_different(final: Path, source: StemRecord) -> None:
    try:
        size = final.stat().st_size
        same = size == source.size and _digest(final) == (source.size, source.md5)
    except OSError as exc:
        raise PublishError(f"cannot read {final}: {exc.strerror or exc}") from exc
    if not same:
        raise InputValidationError(f"{final} already exists and differs from its source")


def _move(
    copying: list[Side],
    sources: Mapping[Side, Path],
    finals: Mapping[Side, Path],
    measured: Mapping[Side, _Measured],
) -> None:
    parts = {side: finals[side].with_name(finals[side].name + PART) for side in copying}
    renamed: list[Side] = []
    try:
        for side in copying:
            copy_stem(sources[side], parts[side])
            if _digest(parts[side]) != (measured[side].record.size, measured[side].record.md5):
                raise PublishError(f"the copy of {sources[side]} does not match it")
        for side in copying:
            _stamp_times(parts[side], measured[side])
        for side in copying:
            if finals[side].exists():
                raise InputValidationError(f"{finals[side]} appeared while publishing")
            parts[side].rename(finals[side])
            renamed.append(side)
    except (OSError, AppError) as exc:
        # A final name is never left for one side only.
        for side in reversed(renamed):
            try:
                finals[side].rename(parts[side])
            except OSError as undo:
                _logger().error("cannot take a stem back", path=str(finals[side]), error=str(undo))
        for part in parts.values():
            with contextlib.suppress(OSError):
                part.unlink(missing_ok=True)
        if isinstance(exc, AppError):
            raise
        folder = next(iter(finals.values())).parent
        raise PublishError(f"cannot publish into {folder}: {exc.strerror or exc}") from exc
