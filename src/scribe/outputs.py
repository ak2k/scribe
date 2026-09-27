"""Plan a command's output paths and refuse the unusable ones before any write."""

from __future__ import annotations

import os
import tempfile
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from scribe.errors import InputValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping


def _same_file(first: Path, second: Path) -> bool:
    # Resolved first: an output named through a directory not yet created, as
    # in `new/..`, exists once that detour is gone.
    first, second = first.resolve(), second.resolve()
    # Existing paths ask the filesystem, which sees through a hard link and a
    # case-only difference on a case-insensitive volume.
    if first.exists() and second.exists():
        return first.samefile(second)
    # A path not yet created can only be compared by name. A case- or
    # normalization-insensitive volume, the macOS default, takes names that
    # differ only in case or Unicode form for one file, so those count as one:
    # on a case-sensitive volume that refuses a pair it could have written,
    # which costs a rename, where missing it loses an output.
    same_name = _folded(first.name) == _folded(second.name)
    return same_name and _same_directory(first.parent, second.parent)


def _same_directory(first: Path, second: Path) -> bool:
    if first.exists() and second.exists():
        return first.samefile(second)
    return first == second


def _folded(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


@dataclass(frozen=True)
class OutputPlan:
    """Every path a command will write, by label, checked together."""

    paths: Mapping[str, Path]

    def __getitem__(self, label: str) -> Path:
        """Return the planned path for `label`."""
        return self.paths[label]


def sibling(path: Path, suffix: str) -> Path:
    """Name a file beside `path`: its name less .transcript.json or .json, plus `suffix`."""
    for ending in (".transcript.json", ".json"):
        if path.name.endswith(ending):
            return path.with_name(f"{path.name.removesuffix(ending)}{suffix}")
    return path.with_name(f"{path.name}{suffix}")


def plan_outputs(
    outputs: Mapping[str, Path], inputs: Mapping[str, Path], *, missing_parent_ok: bool = False
) -> OutputPlan:
    """Check a command's outputs against each other and its inputs, all at once.

    Args:
        outputs: Paths the command will write, keyed by the label a refusal names.
        inputs: Paths the command reads, keyed by how a refusal describes them.
        missing_parent_ok: Accept an output directory that does not exist yet,
            for a command that creates it once the plan stands.

    Returns:
        The plan the command writes through.

    Raises:
        InputValidationError: An output is a directory, is an input, is another
            output, or has no directory to land in.

    """
    for label, path in outputs.items():
        if path.is_dir():
            raise InputValidationError(f"{label} {path} is a directory")
    for label, path in outputs.items():
        for described, source in inputs.items():
            if _same_file(path, source):
                raise InputValidationError(f"{label} {path} is {described}")
    # A link can make two outputs one file, the later write replacing the earlier.
    for (first_label, first), (second_label, second) in combinations(outputs.items(), 2):
        if _same_file(first, second):
            raise InputValidationError(
                f"{second_label} {second} is the same file as {first_label} {first}"
            )
    for path in outputs.values():
        parent = path.parent
        # Required to exist unless the caller creates it: a typo in the directory
        # part of a path is an error, not a new tree left behind by a failed run.
        if parent.is_dir() or (missing_parent_ok and not parent.exists()):
            continue
        raise InputValidationError(f"cannot write to {parent}: not a directory")
    return OutputPlan(MappingProxyType(dict(outputs)))


def prove_writable(plan: OutputPlan) -> None:
    """Show that every planned output can be written, changing none of them.

    For a command about to pay for model calls: finding out at the first write
    throws their answers away. Every output is written in place, so an existing
    one needs only to open for writing, and a directory only has to take a new
    file where an output does not exist yet.

    Raises:
        InputValidationError: An output, or the directory a new one goes in,
            is not writable.

    """
    for path in plan.paths.values():
        if not path.exists():
            continue
        try:
            # Non-blocking: opened for writing, a FIFO nobody reads waits forever.
            os.close(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK))
        except OSError as exc:
            raise InputValidationError(f"cannot write {path}: {exc}") from exc
    # Resolved: a write through a dangling link creates the file beside its target.
    for directory in dict.fromkeys(
        path.resolve().parent for path in plan.paths.values() if not path.exists()
    ):
        try:
            handle, probe = tempfile.mkstemp(prefix=".scribe-probe-", dir=directory)
            os.close(handle)
            Path(probe).unlink()
        except OSError as exc:
            # The error names the probe, a file the caller never asked for.
            raise InputValidationError(
                f"cannot write artifacts to {directory}: {exc.strerror}"
            ) from exc
