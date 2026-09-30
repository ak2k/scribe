"""Render the reading copy of a cleaned transcript.

Each turn the cleaned text keeps is a block under its speaker and the time it
started, in the words the cleaned text has, with no provenance header: that
stays in the cleaned markdown and its sidecar. A turn holding speech a second
transcription pass recovered says so, and an optional front section, copied as
written, opens the copy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated

from pydantic import AfterValidator, ConfigDict, Json, TypeAdapter, ValidationError

from scribe.cleanup import turn_label
from scribe.errors import InputValidationError
from scribe.schema import FiniteFloat
from scribe.writers import hms

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from scribe.cleanup import CleanupRequest
    from scribe.schema import Engine, Turn


def _ordered(span: tuple[float, float]) -> tuple[float, float]:
    if span[0] > span[1]:
        raise ValueError("a range cannot end before it starts")
    return span


# Strict: a quoted number or a boolean in the record is drift, not a time.
_FILL_RANGES = TypeAdapter(
    Json[list[Annotated[tuple[FiniteFloat, FiniteFloat], AfterValidator(_ordered)]]],
    config=ConfigDict(strict=True),
)


def fill_ranges(
    engine: Engine, source: Path, *, key: str = "fill_ranges"
) -> list[tuple[float, float]]:
    """The spans, in seconds, a second transcription pass inserted words into.

    Args:
        engine: Engine whose params may record a fill.
        source: File the transcript was read from, to name in an error.
        key: The param the spans are recorded under, such as
            fill_unresolved_ranges for the spans the fill could not repair.

    Returns:
        Each recorded span as (start, end); none when no fill ran.

    Raises:
        InputValidationError: the record is not a JSON list of [start, end]
            pairs of finite numbers, each start at most its end.

    """
    recorded = engine.params.get(key)
    if recorded is None:
        return []
    try:
        return _FILL_RANGES.validate_python(recorded)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise InputValidationError(
            f"{source} has a malformed {key}: {first['msg']} (at {where or 'top level'})"
        ) from exc


def read_front(path: Path) -> str:
    """Read the text that opens the reading copy, exactly as the file holds it.

    Args:
        path: The --front file.

    Returns:
        Its text, a byte order mark and carriage returns included.

    Raises:
        InputValidationError: the file is unreadable or not UTF-8.

    """
    try:
        # Decoded from bytes: read as text, a byte order mark would go and a
        # CRLF would turn into LF.
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputValidationError(f"cannot read --front {path}: {exc}") from exc


def _touches(turn: Turn, start: float, end: float) -> bool:
    # A fill of words timed with no length is a point. On the boundary of two
    # turns it marks both: a note too many costs a reader less than recovered
    # speech left unmarked.
    if start == end:
        return turn.start <= start <= turn.end
    return start < turn.end and turn.start < end


def _note(turn: Turn, ranges: Sequence[tuple[float, float]]) -> str:
    touched = [(start, end) for start, end in ranges if _touches(turn, start, end)]
    if not touched:
        return ""
    # The note covers the whole turn, not a place in it: cleanup rewrites
    # words, so no position inside a turn survives it.
    low = max(min(start for start, _ in touched), turn.start)
    high = min(max(end for _, end in touched), turn.end)
    return (
        "[Includes speech recovered by a second transcription pass, "
        f"{hms(math.floor(low))}\N{EN DASH}{hms(math.ceil(high))}.] "
    )


def render_curated(
    request: CleanupRequest,
    kept: Sequence[tuple[Turn, str]],
    ranges: Sequence[tuple[float, float]],
    front: str | None = None,
) -> str:
    """Render the reading copy of a cleaned transcript.

    Args:
        request: The request the text was cleaned from, for its labels.
        kept: Each turn that left a line in the cleaned text, with that
            line's text, in order.
        ranges: Spans, in seconds, a second transcription pass recovered
            speech in.
        front: Text to open the copy with, copied as written and ruled off
            from the turns.

    Returns:
        One `**Label | HH:MM:SS**` block per kept turn, its note if it
        touches a range, then its text; blocks separated by a blank line,
        ending in a newline.

    """
    blocks = [
        f"**{turn_label(request, turn)} | {hms(math.floor(turn.start))}**\n"
        f"{_note(turn, ranges)}{text}"
        for turn, text in kept
    ]
    body = "\n\n".join(blocks) + "\n"
    if front is None:
        return body
    ended = front if front.endswith("\n") else f"{front}\n"
    return f"{ended}\n---\n\n{body}"
