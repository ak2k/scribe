"""Replies a fake pick backend gives, read from the spots a chunk's target marks."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from scribe.schema import Engine, Source, Transcript, Word

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

_MARK = re.compile(r"\[#(\d+) A: (.*?) \| B: (.*?)\]")


def marks(target: str) -> list[tuple[int, str, str]]:
    """(id, A's reading, B's reading) for every spot a target marks, in order."""
    return [(int(found[1]), found[2], found[3]) for found in _MARK.finditer(target)]


def answering(choose: Callable[[int, str, str], str]) -> Callable[[str], str]:
    """A reply picking `choose(id, a, b)` at every spot the target marks."""

    def reply(target: str) -> str:
        picks = [
            {"id": number, "pick": choose(number, a, b), "reason": "fits the talk"}
            for number, a, b in marks(target)
        ]
        return json.dumps({"picks": picks})

    return reply


def unsure(_number: int, _a: str, _b: str) -> str:
    return "unsure"


def choosing(readings: Collection[str]) -> Callable[[int, str, str], str]:
    """Pick whichever label shows one of `readings`, B when neither does."""
    return lambda _number, a, _b: "A" if a in readings else "B"


def transcript(
    words: list[Word], *, engine: str = "xai-stt", model: str | None = None
) -> Transcript:
    return Transcript(
        source=Source(kind="audio", ref="meeting.mp3"),
        engine=Engine(name=engine, model=model),
        text=" ".join(word.text for word in words),
        words=words,
    )


def heard_late(lag: float) -> tuple[list[Word], list[Word]]:
    """Twelve words two a second, and the same words heard `lag` s later, "cat" (6) as "hat"."""
    texts = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "cat",
        "lazy",
        "dog",
        "and",
        "runs",
        "away",
    ]
    said = [
        Word(text=text, start=index * 0.5, end=index * 0.5 + 0.2, speaker=index // 3 % 2)
        for index, text in enumerate(texts)
    ]
    heard = [
        Word(
            text="hat" if word.text == "cat" else word.text,
            start=word.start + lag,
            end=word.end + lag,
        )
        for word in said
    ]
    return said, heard


def heard_early_at_an_edge() -> tuple[list[Word], list[Word]]:
    """Sixteen words a second apart, a new speaker from w10; heard with w10 as "SUB".

    The reference hears w8 to w11 a second early, held back so no start steps back.
    """
    said = [
        Word(text=f"w{index}", start=float(index), end=index + 0.3, speaker=int(index >= 10))
        for index in range(16)
    ]
    heard: list[Word] = []
    for index, word in enumerate(said):
        early = word.start - 1.0 if 8 <= index <= 11 else word.start
        start = max(early, heard[-1].start) if heard else early
        heard.append(Word(text="SUB" if index == 10 else word.text, start=start, end=start + 0.3))
    return said, heard


def numbered(count: int, changed: dict[int, str] | None = None) -> list[Word]:
    """`count` words a second, w0 to w{count-1}, every tenth ending a sentence; some replaced."""
    swapped = changed or {}
    return [
        Word(
            text=swapped.get(index, f"w{index}." if index % 10 == 9 else f"w{index}"),
            start=float(index),
            end=index + 0.5,
            speaker=index // 100 % 2,
        )
        for index in range(count)
    ]
