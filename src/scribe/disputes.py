"""List the places the pick and the fill recorded two recognizers disagreeing, ranked for a reader.

The pick records each spot it compared, with both readings and the side it
took; the fill records each span it put the reference's words in, or found
speech in that it could not repair. Each becomes one entry, given as a
critical apparatus gives a variant: where, the reading the transcript holds
and its engine, the reading set aside and its engine. No recorded spot is
left out, as a filter drops many of the pick's errors along with its noise;
bands rank the entries instead, missed speech, the costliest error, first. A
short stretch only one recognizer heard is in neither record, so it is not
listed unless the fill filled or flagged it.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import AfterValidator, ConfigDict, Json, NonNegativeInt, TypeAdapter, ValidationError

from scribe.curated import fill_ranges
from scribe.errors import InputValidationError, ToolMissingError
from scribe.gaps import run_ffmpeg
from scribe.pick import GUARDED_DROP, Side, normalized, surplus
from scribe.schema import FiniteFloat
from scribe.writers import hms

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from scribe.gaps import Runner
    from scribe.schema import Transcript, Word

# A starting value taken from a simpler word count on one meeting's truth
# sample, to be re-measured per band: the pick's token folding differs from
# that count.
MANY_WORDS = 4
# Seconds of audio a clip holds on each side of its entry, so its words are
# heard with the words around them.
CLIP_MARGIN = 3.0

Band = Literal["A", "B", "C", "D", "E"]
Kind = Literal["spot", "fill", "unresolved"]
BANDS: dict[Band, str] = {
    "A": "Words missing on one side",
    "B": "The pick was unsure or gave no answer",
    "C": f"{MANY_WORDS} or more words differ",
    "D": f"1 to {MANY_WORDS - 1} words differ",
    "E": (
        "Same words once folded; the difference is with the words around the spot "
        "(for example a number split differently)"
    ),
}
LISTED = (
    "Listed: every spot where the pick compared the two recognizers' readings, "
    "and every span the fill filled or flagged."
)
UNLISTED = (
    "Not listed: short stretches only one recognizer heard (unless the fill filled or flagged "
    "them), and words both got wrong the same way. Unlisted text is unverified."
)
QUOTES = "Quotes are the words before cleanup; the reading copy may word them differently."
FILL_TIMED = (
    "Fill quotes are bounded by time only: this record does not say how many words each fill "
    "put in, so a quote may hold a word the transcript already had."
)

_Row = tuple[FiniteFloat, FiniteFloat, str, str, Side]


def _ordered(row: _Row) -> _Row:
    # Before 0 is outside the recording: a clip of it would hold the opening instead.
    if row[0] < 0:
        raise ValueError("a spot cannot start before the recording")
    if row[1] < row[0]:
        raise ValueError("a spot cannot end before it starts")
    return row


# Strict: a quoted number or a boolean in the record is drift, not a time.
_PICK_RECORD = TypeAdapter(
    Json[list[Annotated[_Row, AfterValidator(_ordered)]]],
    config=ConfigDict(strict=True),
)
_FILL_COUNTS = TypeAdapter(Json[list[NonNegativeInt]], config=ConfigDict(strict=True))


@dataclass(frozen=True)
class Dispute:
    """One entry: a place the two engines disagreed, and what the transcript holds there."""

    # Rank, from 1, running on across bands.
    number: int
    band: Band
    kind: Kind
    start: float
    end: float
    # The pick's side at a spot; None for a fill or unresolved span.
    side: Side | None
    # The words the transcript holds, and the name of the engine they came from:
    # blank for an unresolved span, whose words may be either engine's.
    delivered: str
    delivered_by: str
    # The reading set aside and its engine's name: at a fill, the engine that
    # heard nothing there.
    other: str
    other_by: str
    # The words quoted at a fill, or the transcript's words starting in an
    # unresolved span, ends included; 0 at a spot.
    words: int


@dataclass(frozen=True)
class Disputes:
    """A transcript's entries in rank order, and what they were made from."""

    # The source audio's file name.
    recording: str
    # Each engine's name, then its model where one is recorded.
    transcript_engine: str
    reference_engine: str
    fill_engine: str
    entries: tuple[Dispute, ...]
    # The record has fill spans but no fill_counts, so each fill quotes the
    # words inside its span's times.
    fills_by_time: bool


def pick_record(transcript: Transcript, path: Path) -> list[_Row]:
    """Read the spots the pick recorded in `transcript`, read from `path`."""
    recorded = transcript.engine.params.get("pick_record")
    if recorded is None:
        raise InputValidationError(
            f"{path} has no pick_record; run `scribe transcribe` with the pick on, "
            "or `scribe pick`, first"
        )
    try:
        return _PICK_RECORD.validate_python(recorded)
    except ValidationError as exc:
        first = exc.errors()[0]
        row = f" row {first['loc'][0]}" if first["loc"] else ""
        raise InputValidationError(
            f"{path} has a malformed pick_record{row}: {first['msg']}"
        ) from exc


def _engine_param(transcript: Transcript, path: Path, key: str, default: str | None) -> str:
    named = transcript.engine.params.get(key, default)
    if not isinstance(named, str) or not named.strip():
        raise InputValidationError(f"{path} has a pick_record but no {key} naming its engine")
    return named


def _ranges(transcript: Transcript, path: Path, key: str) -> list[tuple[float, float]]:
    ranges = fill_ranges(transcript.engine, path, key=key)
    for index, (start, _) in enumerate(ranges):
        if start < 0:
            raise InputValidationError(
                f"{path} has a malformed {key}: range {index} starts before the recording"
            )
    return ranges


def _counts(transcript: Transcript, path: Path, ranges: int) -> list[int] | None:
    recorded = transcript.engine.params.get("fill_counts")
    if recorded is None:
        return None
    try:
        counts = _FILL_COUNTS.validate_python(recorded)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise InputValidationError(
            f"{path} has a malformed fill_counts: {first['msg']} (at {where or 'top level'})"
        ) from exc
    if len(counts) != ranges:
        raise InputValidationError(
            f"{path} has a malformed fill_counts: not one count per range of fill_ranges "
            f"({len(counts)} against {ranges})"
        )
    return counts


def _filled(
    words: Sequence[Word], path: Path, index: int, span: tuple[float, float], count: int | None
) -> list[str]:
    start, end = span
    if count is None:
        # Time alone cannot tell the word that closed the hole from the fill's
        # own words when it lies inside the span, or at its end with no length.
        return [word.text for word in words if start <= word.start and word.end <= end]
    # The fill put its words in as one run, just before the word that closed the
    # hole, the first starting at the span's start.
    first = next((at for at, word in enumerate(words) if word.start >= start), len(words))
    held = words[first : first + count]
    if len(held) < count:
        raise InputValidationError(
            f"{path} has a fill_counts of {count} for range {index} of fill_ranges, but only "
            f"{len(held)} words start at or after it"
        )
    return [word.text for word in held]


def _started(words: Sequence[Word], span: tuple[float, float]) -> list[str]:
    # An unresolved span's ends are window bounds, so a word starting on one is counted.
    start, end = span
    return [word.text for word in words if start <= word.start <= end]


def _name(engine: str) -> str:
    return engine.split(maxsplit=1)[0]


def _difference(mine: str, theirs: str) -> int:
    """Count the words, as the pick compares them, one reading holds beyond the other's."""
    first, second = Counter(normalized(mine)), Counter(normalized(theirs))
    return max((first - second).total(), (second - first).total())


def _band(side: Side, mine: str, theirs: str) -> Band:
    # A reading this much shorter than the other may have lost speech, whatever
    # the pick took.
    if side in {"guarded", "restored"} or abs(surplus(mine, theirs)) >= GUARDED_DROP:
        return "A"
    if side in {"unsure", "failed"}:
        return "B"
    differ = _difference(mine, theirs)
    if differ >= MANY_WORDS:
        return "C"
    # No word apart: the pick kept the spot because its readings differ with the
    # words around them, as "five" against "5" after "twenty".
    return "D" if differ else "E"


def _spot(row: _Row, said: str, heard: str) -> Dispute:
    start, end, mine, theirs, side = row
    # The pick's own rule: only these sides put the reference's words in.
    put_in = side in {"reference", "restored"}
    return Dispute(
        number=0,
        band=_band(side, mine, theirs),
        kind="spot",
        start=start,
        end=end,
        side=side,
        delivered=theirs if put_in else mine,
        delivered_by=heard if put_in else said,
        other=mine if put_in else theirs,
        other_by=said if put_in else heard,
        words=0,
    )


def _span(
    kind: Kind, held: Sequence[str], span: tuple[float, float], by: str, other: str
) -> Dispute:
    start, end = span
    return Dispute(
        number=0,
        band="A",
        kind=kind,
        start=start,
        end=end,
        side=None,
        delivered=" ".join(held),
        delivered_by=by,
        other="",
        other_by=other,
        words=len(held),
    )


def find_disputes(transcript: Transcript, path: Path) -> Disputes:
    """List every spot the pick recorded and every span the fill did, in rank order.

    Args:
        transcript: A transcript the pick ran on.
        path: File it was read from, to name in an error.

    Returns:
        Its entries numbered from 1 by band, A first, then by start and end.
        A: a reading GUARDED_DROP or more words shorter than the other, as
        the pick counts them, a spot guarded or restored, and every fill and
        unresolved span. B: the pick unsure or failed. C: readings MANY_WORDS
        or more words apart. D: fewer. E: none, the readings differing only
        with the words around them.

    Raises:
        InputValidationError: there is no pick_record, or it, the engine it
            names, the fill's ranges or its counts are malformed, a time
            before 0 among them, or a count runs past the transcript's words.

    """
    rows = pick_record(transcript, path)
    engine = transcript.engine
    reference = _engine_param(transcript, path, "pick_reference", None)
    filler = _engine_param(transcript, path, "fill_reference", reference)
    said, heard = engine.name, _name(reference)
    words = transcript.words
    fills = _ranges(transcript, path, "fill_ranges")
    counts = _counts(transcript, path, len(fills))
    entries = [_spot(row, said, heard) for row in rows]
    entries += [
        _span(
            "fill",
            _filled(words, path, index, span, None if counts is None else counts[index]),
            span,
            _name(filler),
            said,
        )
        for index, span in enumerate(fills)
    ]
    entries += [
        _span("unresolved", _started(words, span), span, "", "")
        for span in _ranges(transcript, path, "fill_unresolved_ranges")
    ]
    ranked = sorted(entries, key=lambda entry: (entry.band, entry.start, entry.end))
    return Disputes(
        recording=Path(transcript.source.ref).name,
        transcript_engine=said if engine.model is None else f"{said} {engine.model}",
        reference_engine=reference,
        fill_engine=filler,
        entries=tuple(replace(entry, number=number) for number, entry in enumerate(ranked, 1)),
        fills_by_time=bool(fills) and counts is None,
    )


def _quoted(reading: str) -> str:
    collapsed = " ".join(reading.split())
    return f'"{collapsed}"' if collapsed else "(nothing)"


def _times(entry: Dispute) -> str:
    # Rounded outward, so the span shown holds every word of the entry.
    return f"{hms(math.floor(entry.start))}\N{EN DASH}{hms(math.ceil(entry.end))}"


def _line(entry: Dispute, clip: Path | None) -> str:
    if entry.kind == "fill":
        body = (
            f"{entry.other_by} heard nothing; filled from {entry.delivered_by}: "
            f"{_quoted(entry.delivered)}"
        )
    elif entry.kind == "unresolved":
        words = f"{entry.words} {'word' if entry.words == 1 else 'words'}"
        body = f"possible dropped speech the fill could not repair; {words} delivered here"
    else:
        body = (
            f"{entry.delivered_by} ({entry.side}): {_quoted(entry.delivered)} ] "
            f"{entry.other_by}: {_quoted(entry.other)}"
        )
    linked = "" if clip is None else f" \N{MIDDLE DOT} clip: {clip.as_posix()}"
    return f"{entry.number}. {_times(entry)} \N{MIDDLE DOT} {body}{linked}"


def render_disputes(disputes: Disputes, clips: Mapping[int, Path] | None = None) -> str:
    """Render the entries as Markdown: a header, then one section per band that holds any.

    Args:
        disputes: What `find_disputes` found.
        clips: Each entry's clip by its number, as a path relative to the
            Markdown file; an entry without one is not linked.

    Returns:
        The Markdown, ending in a newline.

    """
    entries, linked = disputes.entries, clips or {}
    bands = Counter(entry.band for entry in entries)
    kinds = Counter(entry.kind for entry in entries)
    lines = [
        f"# Disputes: {disputes.recording}",
        "",
        f"- Transcript: {disputes.transcript_engine}",
        f"- Reference: {disputes.reference_engine}",
    ]
    if disputes.fill_engine != disputes.reference_engine:
        lines.append(f"- Fill reference: {disputes.fill_engine}")
    lines += [
        f"- Fill spans: {kinds['fill']}; unresolved spans: {kinds['unresolved']} "
        "(both listed in A)",
        "",
        LISTED,
        "",
        UNLISTED,
        "",
        QUOTES,
        "",
        *([FILL_TIMED, ""] if disputes.fills_by_time else []),
        "Entries by band, likeliest errors first:",
        "",
        *(f"- {band}. {label}: {bands[band]}" for band, label in BANDS.items()),
    ]
    for band, label in BANDS.items():
        held = [entry for entry in entries if entry.band == band]
        if held:
            lines += ["", f"## {band}. {label} ({len(held)})", ""]
            lines += [_line(entry, linked.get(entry.number)) for entry in held]
    return "\n".join(lines) + "\n"


def summarize(disputes: Disputes) -> str:
    """Count the entries per band, and the fill's spans among them, for a progress line."""
    entries = disputes.entries
    bands = Counter(entry.band for entry in entries)
    kinds = Counter(entry.kind for entry in entries)
    counts = ", ".join(f"{band} {bands[band]}" for band in BANDS)
    listed = f"{len(entries)} {'entry' if len(entries) == 1 else 'entries'}"
    fills = f"{kinds['fill']} fill {'span' if kinds['fill'] == 1 else 'spans'}"
    return f"{listed}: {counts}; {fills}, {kinds['unresolved']} unresolved"


def clips_directory(markdown: Path) -> Path:
    """Name the directory beside `markdown` for its clips: .md becomes .clips, else it is added."""
    if markdown.suffix == ".md":
        return markdown.with_suffix(".clips")
    return markdown.with_name(f"{markdown.name}.clips")


def _window(entry: Dispute, duration: float | None) -> tuple[float, float]:
    """Return where an entry's clip starts and how long it runs, in seconds."""
    start, end = max(entry.start - CLIP_MARGIN, 0.0), entry.end + CLIP_MARGIN
    if duration is not None:
        # The entry's own start, not its clip's: a clip of an entry just past the
        # end would hold only the lead-in, none of the entry.
        if entry.start >= duration:
            raise InputValidationError(
                f"entry {entry.number} starts at {hms(entry.start)}, at or past the "
                f"recording's end at {hms(duration)}"
            )
        end = min(end, duration)
    return start, end - start


def _check_directory(directory: Path) -> None:
    if not directory.is_dir():
        # A dangling link too: making the directory would fail on it.
        if directory.exists() or directory.is_symlink():
            raise InputValidationError(f"{directory}, where the clips go, is not a directory")
        return
    try:
        held = next(directory.iterdir(), None)
    except OSError as exc:
        raise InputValidationError(f"cannot read the clips directory {directory}: {exc}") from exc
    # Never emptied or written over: they may be the only copy of an earlier run's clips.
    if held is not None:
        raise InputValidationError(
            f"the clips directory {directory} is not empty; clips an earlier run left, whole "
            "or partial, block this one until the directory is removed"
        )


def cut_clips(
    entries: Sequence[Dispute],
    audio: Path,
    directory: Path,
    *,
    duration: float | None,
    run: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[int, Path]:
    """Cut each entry's span of `audio`, CLIP_MARGIN s on each side, into an AAC file.

    A clip starts no earlier than 0 and, when `duration` is known, ends no
    later than it. It is named for its entry's number, padded to 3 digits,
    and start: 001-000840.m4a. Everything is checked before `directory` is
    made; an existing one is used only if it is empty.

    Args:
        entries: The entries to cut a clip for.
        audio: The recording, already checked to be the transcript's.
        directory: Where the clips go.
        duration: The recording's length in seconds, where known.
        run: Runs ffmpeg.
        which: Finds ffmpeg on PATH.

    Returns:
        Each entry's number and the clip cut for it.

    Raises:
        ToolMissingError: ffmpeg is not on PATH.
        InputValidationError: `directory` is not a directory or is not empty,
            or an entry starts at or past the recording's end.
        ExternalServiceError: ffmpeg failed; the clips cut before it stay.

    """
    if which("ffmpeg") is None:
        raise ToolMissingError("ffmpeg is not on PATH; the clips are cut with it")
    cuts = [
        (entry.number, directory / f"{entry.number:03d}-{hms(entry.start).replace(':', '')}.m4a")
        for entry in entries
    ]
    windows = [_window(entry, duration) for entry in entries]
    _check_directory(directory)
    try:
        directory.mkdir(exist_ok=True)
    except OSError as exc:
        raise InputValidationError(f"cannot make the clips directory {directory}: {exc}") from exc
    # Absolute, so a colon in a relative name is not read as a protocol; -vn, so
    # the picture of a video container is not decoded for nothing; -ss before
    # -i, so ffmpeg seeks rather than decodes its way to the start.
    source = str(audio.absolute())
    for (_, clip), (start, length) in zip(cuts, windows, strict=True):
        timing = ["-ss", f"{start:.3f}", "-t", f"{length:.3f}"]
        argv = ["ffmpeg", "-nostdin", "-v", "error", *timing, "-i", source, "-vn", "-c:a", "aac"]
        run_ffmpeg(run, [*argv, str(clip.absolute())])
    return dict(cuts)
