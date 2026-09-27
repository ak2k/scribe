"""Render a `Transcript` as markdown, SRT or WebVTT.

Subtitle formats need short cues, so turns are re-split on word boundaries;
markdown keeps whole turns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scribe.schema import Turn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scribe.schema import Transcript, Word

_SECONDS_PER_HOUR = 3600
_SECONDS_PER_MINUTE = 60
_MS_PER_HOUR = 3_600_000
_MS_PER_MINUTE = 60_000
_MS_PER_SECOND = 1000


def _clock(seconds: float) -> str:
    whole = int(seconds)
    hours, rest = divmod(whole, _SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, _SECONDS_PER_MINUTE)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stamp(seconds: float, separator: str) -> str:
    total_ms = round(seconds * _MS_PER_SECOND)
    hours, rest = divmod(total_ms, _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    secs, millis = divmod(rest, _MS_PER_SECOND)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def to_markdown(transcript: Transcript) -> str:
    """Render turns as a speaker-labeled markdown document.

    Args:
        transcript: A transcript whose `turns` are already built.

    Returns:
        Markdown text; empty when the transcript has no turns.

    """
    return "".join(
        f"**{turn.speaker}** [{_clock(turn.start)}]\n{turn.text}\n\n" for turn in transcript.turns
    )


def _cue(speaker: str, words: Sequence[Word]) -> Turn:
    return Turn(
        speaker=speaker,
        start=words[0].start,
        # Overlapping speech leaves words in start order, not end order.
        end=max(word.end for word in words),
        text=" ".join(word.text for word in words),
    )


def _split(speaker: str, words: Sequence[Word], max_seconds: float, max_chars: int) -> list[Turn]:
    groups: list[list[Word]] = [[words[0]]]
    for word in words[1:]:
        current = groups[-1]
        width = len(" ".join(w.text for w in current)) + 1 + len(word.text)
        if width > max_chars or (word.end - current[0].start) > max_seconds:
            groups.append([word])
        else:
            current.append(word)
    return [_cue(speaker, group) for group in groups]


def _own_words(transcript: Transcript) -> list[list[Word]] | None:
    """Each turn's own words, where the turns are the words cut into runs in order.

    None where they are not: a turns-only transcript, or turns not built from
    these words. A word with no text shows nothing and is left out.
    """
    words = [word for word in transcript.words if word.text.split()]
    owned: list[list[Word]] = []
    index = 0
    for turn in transcript.turns:
        expected = turn.text.split()
        held: list[Word] = []
        tokens: list[str] = []
        while len(tokens) < len(expected) and index < len(words):
            held.append(words[index])
            tokens.extend(words[index].text.split())
            index += 1
        if tokens != expected:
            return None
        owned.append(held)
    return owned if index == len(words) else None


def build_cues(
    transcript: Transcript, *, max_seconds: float = 7.0, max_chars: int = 84
) -> list[Turn]:
    """Split turns into subtitle-sized cues on word boundaries.

    A cue closes when the next word would push it past either limit; a single
    word longer than `max_chars` still forms its own cue.

    Args:
        transcript: A transcript whose `turns` are already built.
        max_seconds: Longest cue duration before a split.
        max_chars: Longest cue text before a split.

    Returns:
        Cues in transcript order, each carrying its turn's speaker label and
        only that turn's own words, so every word is in one cue under the
        speaker markdown shows it under. Where the turns are not the words cut
        in order (a turns-only transcript has no word timings), each turn is
        one unsplit cue.

    """
    owned = _own_words(transcript)
    if owned is None:
        return list(transcript.turns)
    # Ownership is membership, not time: overlapping speech puts one speaker's
    # word inside another's span, and a span test would hand it to the wrong one.
    cues: list[Turn] = []
    for turn, words in zip(transcript.turns, owned, strict=True):
        cues.extend(_split(turn.speaker, words, max_seconds, max_chars) if words else [turn])
    return cues


def to_srt(transcript: Transcript) -> str:
    """Render cues as SubRip (`.srt`) text."""
    return "".join(
        f"{index}\n{_stamp(cue.start, ',')} --> {_stamp(cue.end, ',')}\n"
        f"{cue.speaker}: {cue.text}\n\n"
        for index, cue in enumerate(build_cues(transcript), start=1)
    )


def to_vtt(transcript: Transcript) -> str:
    """Render cues as WebVTT (`.vtt`) text."""
    body = "".join(
        f"{_stamp(cue.start, '.')} --> {_stamp(cue.end, '.')}\n{cue.speaker}: {cue.text}\n\n"
        for cue in build_cues(transcript)
    )
    return "WEBVTT\n\n" + body
