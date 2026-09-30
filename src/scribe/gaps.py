"""Holes in a transcript's words where the audio is still about as loud as speech."""

from __future__ import annotations

import math
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scribe.errors import ExternalServiceError, InputValidationError, ToolMissingError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from scribe.schema import Transcript, Word

    Runner = Callable[..., subprocess.CompletedProcess[str]]

# The four thresholds below were calibrated together on 4 meetings (4.4 h).
# 50 ms frames: short enough that a burst of speech in a quiet hole keeps its own level.
FRAMES_PER_SECOND = 20
# Ordinary talk has many shorter pauses; flagging them would bury the dropped passages.
MIN_HOLE_SECONDS = 6.0
# The engine can stretch one word across a hole, which would otherwise hide the hole.
MAX_WORD_SECONDS = 2.0
# A hole's loudest tenth of frames this close to the speech level is taken to be speech.
FLAG_DB = -25.0
# astats reports digital silence as -inf, which no percentile or median can use.
SILENCE_DB = -120.0
# Measured with coverage's constants on 5 transcripts (6.2 h): a shorter hole
# is ordinary talk, not a dropped passage.
MIN_DROP_SECONDS = 2.0
# Measured with them too: fewer words than this in a hole, fillers aside, are the
# engines disagreeing, or too little speech for its speaker to matter.
MIN_DROP_WORDS = 3

_RATE = 16000
_KEY = "lavfi.astats.Overall.RMS_level"
# Resampled inside the graph: as an output option, astats would frame at the input rate.
_FILTER = ",".join(
    [
        f"aresample={_RATE}",
        "aformat=channel_layouts=mono",
        f"asetnsamples=n={_RATE // FRAMES_PER_SECOND}:p=0",
        "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level",
        f"ametadata=mode=print:key={_KEY}:file=-",
    ]
)
_EXCERPT_CHARS = 300


@dataclass(frozen=True)
class Gap:
    """A flagged hole in seconds, and its loudest tenth's level relative to speech in dB."""

    start: float
    end: float
    # None when the transcript has no words, and so no speech level.
    relative_db: float | None


@dataclass(frozen=True)
class Hole:
    """A stretch with no word in it, in seconds."""

    start: float
    end: float
    # The list index of the word whose start ends the hole; None for the hole
    # that runs to the audio's end.
    before: int | None


@dataclass(frozen=True)
class GapCheck:
    """The flagged holes and the speech level they were judged against."""

    gaps: list[Gap]
    speech_db: float | None


def run_ffmpeg(run: Runner, argv: list[str]) -> str:
    """Run ffmpeg's `argv` through `run` and return what it printed to stdout.

    Raises:
        ExternalServiceError: it could not start, or exited nonzero; the
            start of its stderr is quoted.

    """
    try:
        done = run(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ExternalServiceError(f"cannot run {argv[0]}: {exc.strerror or exc}") from exc
    if done.returncode != 0:
        excerpt = " ".join(done.stderr[:_EXCERPT_CHARS].split())
        raise ExternalServiceError(f"{argv[0]} exited {done.returncode}: {excerpt}")
    return done.stdout


def _level(value: str) -> float:
    try:
        level = float(value)
    except ValueError:
        level = math.nan
    if level == -math.inf:
        return SILENCE_DB
    if not math.isfinite(level):
        raise ExternalServiceError(f"ffmpeg printed a frame level that is not a number: {value!r}")
    return level


def frame_levels(audio: Path, run: Runner) -> list[float]:
    """Return the RMS level in dB of each 50 ms frame of `audio`, mixed to mono.

    Frame i covers [i/20, (i+1)/20) s. Digital silence is SILENCE_DB.

    Raises:
        ExternalServiceError: ffmpeg failed, or printed no frames, a frame
            without its level, or a line that is not the next frame or level.

    """
    # Absolute, so a colon in a relative name is not read as a protocol; -vn, so
    # the picture of a video container is not decoded for nothing.
    argv = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio.absolute()), "-vn"]
    levels: list[float] = []
    # Each numbered header must be followed by its level: one missing line
    # would move every later frame 50 ms earlier.
    awaiting = False
    for line in run_ffmpeg(run, [*argv, "-af", _FILTER, "-f", "null", "-"]).splitlines():
        key, _, value = line.partition("=")
        if not awaiting and line.split(maxsplit=1)[:1] == [f"frame:{len(levels)}"]:
            awaiting = True
        elif awaiting and key == _KEY:
            levels.append(_level(value))
            awaiting = False
        else:
            raise ExternalServiceError(f"ffmpeg printed an unexpected line: {line[:80]!r}")
    if awaiting:
        raise ExternalServiceError(f"ffmpeg printed frame {len(levels)} without its level")
    if not levels:
        raise ExternalServiceError(f"ffmpeg found no audio frames in {audio}")
    return levels


def _frames(first: int, stop: int, count: int) -> range:
    return range(max(first, 0), min(stop, count))


def _capped_end(word: Word) -> float:
    # Never before the start: an end before it would reopen a hole the word sits in.
    return max(word.start, min(word.end, word.start + MAX_WORD_SECONDS))


def _speech_level(words: Sequence[Word], levels: Sequence[float]) -> float:
    # A set, so a frame two words share counts once.
    covered: set[int] = set()
    for word in words:
        first = int(word.start * FRAMES_PER_SECOND)
        stop = max(int(_capped_end(word) * FRAMES_PER_SECOND), first + 1)
        covered.update(_frames(first, stop, len(levels)))
    if not covered:
        raise InputValidationError(
            f"no word falls within the audio's {len(levels) / FRAMES_PER_SECOND:.1f} s; "
            "the transcript is not of this audio"
        )
    return statistics.median(levels[index] for index in covered)


def find_holes(words: Sequence[Word], audio_end: float, min_seconds: float) -> list[Hole]:
    """Return the holes of at least `min_seconds` between `words`, in time order.

    A word counts until its end, or MAX_WORD_SECONDS after its start if that is
    sooner, and at least until its start. A hole runs from the latest such end
    of the words before it to the next word's start; the first runs from 0 and
    the last to `audio_end`, and none past it.
    """
    holes: list[Hole] = []
    edge = 0.0
    for index in [*sorted(range(len(words)), key=lambda index: words[index].start), None]:
        end = audio_end if index is None else min(words[index].start, audio_end)
        # Rounded to the microsecond: subtraction can leave a hole of exactly
        # `min_seconds` just short.
        if round(end - edge, 6) >= min_seconds:
            holes.append(Hole(edge, end, index))
        if index is not None:
            edge = max(edge, _capped_end(words[index]))
    return holes


def find_gaps(words: Sequence[Word], levels: Sequence[float]) -> list[Gap]:
    """Flag the holes between `words` where the audio's `levels` stay near speech.

    A hole, as `find_holes` finds it up to the audio's end, of at least
    MIN_HOLE_SECONDS is flagged when the 90th percentile of its frame levels
    is at least FLAG_DB relative to the speech level, the median level of the
    frames the words cover. With no words the whole audio is one hole,
    flagged on its length alone.

    Args:
        words: The transcript's words, in any order.
        levels: The audio's `frame_levels`.

    Returns:
        The flagged holes in time order.

    Raises:
        InputValidationError: there are words, but none falls within the audio.

    """
    speech = _speech_level(words, levels) if words else None
    flagged: list[Gap] = []
    for hole in find_holes(words, len(levels) / FRAMES_PER_SECOND, MIN_HOLE_SECONDS):
        start, end = hole.start, hole.end
        if speech is None:
            flagged.append(Gap(start, end, None))
            continue
        span = _frames(int(start * FRAMES_PER_SECOND), int(end * FRAMES_PER_SECOND), len(levels))
        loud = statistics.quantiles([levels[i] for i in span], n=10, method="inclusive")[8]
        if loud >= speech + FLAG_DB:
            flagged.append(Gap(start, end, loud - speech))
    return flagged


def check_gaps(
    transcript: Transcript,
    audio: Path,
    *,
    run: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> GapCheck:
    """Measure `audio` with ffmpeg and flag the holes in `transcript`'s words.

    Raises:
        InputValidationError: the transcript has turns but no words, or no word
            falls within the audio.
        ToolMissingError: ffmpeg is not on PATH.
        ExternalServiceError: ffmpeg failed.

    """
    words = transcript.words
    if transcript.turns and not words:
        raise InputValidationError("the transcript has turns but no words to find holes between")
    if which("ffmpeg") is None:
        raise ToolMissingError("ffmpeg is not on PATH; the check measures the audio with it")
    levels = frame_levels(audio, run)
    return GapCheck(find_gaps(words, levels), _speech_level(words, levels) if words else None)


def clock(seconds: float) -> str:
    """Format `seconds` as HH:MM:SS.s, rounded to a tenth before it is split."""
    tenths = round(seconds * 10)
    return f"{tenths // 36000:02d}:{tenths // 600 % 60:02d}:{tenths % 600 // 10:02d}.{tenths % 10}"


def format_gap(gap: Gap) -> str:
    """Render `gap` as its span, its length and its relative level, tab-separated."""
    relative = "-" if gap.relative_db is None else f"{gap.relative_db:.1f}"
    # From the rounded ends, so the length agrees with the span printed beside it.
    tenths = round(gap.end * 10) - round(gap.start * 10)
    return f"{clock(gap.start)}-{clock(gap.end)}\t{tenths / 10:.1f} s\t{relative} dB"
