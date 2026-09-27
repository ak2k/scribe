"""Rewrite spoken English numbers as digits, so the number check can see them.

xAI's word-level text spells numbers out ("forty two") where its top-level text
has digits, and turns are built from the words. Both sides of the check pass
through `digitize`, so "forty two" in and "42" out compare equal.

Hand-written rather than `text2num`: that library ships no macOS x86_64 wheel,
one of the four systems the flake builds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_HUNDRED = 100
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_ORDINALS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
        "thirtieth",
        "fortieth",
        "fiftieth",
        "sixtieth",
        "seventieth",
        "eightieth",
        "ninetieth",
        "hundredth",
        "thousandth",
        "millionth",
        "billionth",
    }
)
_ARTICLES = frozenset({"a", "an"})
_SIGNS = frozenset({"minus", "negative"})
_LAST_HOUR = 12
_FIRST_MINUTE = 10
_LAST_MINUTE = 59
# A century is spoken as one word; a first half that could be an hour is left
# to the clock ("twelve fifteen" is far more often 12:15 than the year 1215),
# and one past twenty is a range ("fifty fifty", "forty five fifty"), not a year.
_CENTURIES = {word: value for word, value in _TEENS.items() if value > _LAST_HOUR} | {"twenty": 20}

_WORD = re.compile(r"[A-Za-z]+")
# Words join into one number across spaces or one hyphen ("forty-two"), never
# across punctuation: "five, six" is two numbers.
_JOIN = re.compile(r"[ \t]+|[ \t]*-[ \t]*")


@dataclass(frozen=True)
class _Word:
    text: str
    start: int
    end: int


def _chains(text: str) -> list[list[_Word]]:
    chains: list[list[_Word]] = []
    for match in _WORD.finditer(text):
        word = _Word(match.group().lower(), match.start(), match.end())
        if chains and _JOIN.fullmatch(text, chains[-1][-1].end, word.start):
            chains[-1].append(word)
        else:
            chains.append([word])
    return chains


def _at(words: list[_Word], index: int) -> str:
    return words[index].text if index < len(words) else ""


def _tens_part(words: list[_Word], index: int) -> tuple[int, int] | None:
    """Parse 0..99 at `index`; return the value and the index after it."""
    word = _at(words, index)
    if word in _TENS:
        unit = _UNITS.get(_at(words, index + 1), 0)
        return (_TENS[word] + unit, index + 2) if unit else (_TENS[word], index + 1)
    if word in _TEENS:
        return _TEENS[word], index + 1
    if word in _UNITS:
        return _UNITS[word], index + 1
    return None


def _below_thousand(words: list[_Word], index: int) -> tuple[int, int] | None:
    """Parse 1..9,999 in the "fifteen hundred and five" shape; zero is not a group."""
    if _at(words, index) in _ARTICLES:
        # "a hundred", "a thousand"; alone it is a lone one, which never counts.
        value, index = 1, index + 1
    elif _at(words, index) == "hundred":
        # Speech recognition drops the article, so "a hundred plus" arrives bare.
        value = 1
    else:
        parsed = _tens_part(words, index)
        if parsed is None or parsed[0] == 0:
            return None
        value, index = parsed
    if _at(words, index) != "hundred":
        return value, index
    value, index = value * _HUNDRED, index + 1
    rest = _after_and(words, index)
    if rest is not None and rest[0] > 0:
        return value + rest[0], rest[1]
    return value, index


def _after_and(words: list[_Word], index: int) -> tuple[int, int] | None:
    """Parse the 0..99 that follows "hundred", "and" allowed first."""
    if _at(words, index) == "and":
        index += 1
    return _tens_part(words, index)


def _cardinal(words: list[_Word], index: int) -> tuple[int, int] | None:
    """Parse a whole cardinal, scales descending ("two million three hundred thousand")."""
    if _at(words, index) == "zero":
        return 0, index + 1
    first = _below_thousand(words, index)
    if first is None:
        return None
    total, group, index = 0, first[0], first[1]
    while (scale := _SCALES.get(_at(words, index))) is not None:
        total, index = total + group * scale, index + 1
        group = 0
        start = index + 1 if _at(words, index) == "and" else index
        following = _below_thousand(words, start)
        # A group followed by a scale no smaller than this one starts a new number.
        if following is None or _SCALES.get(_at(words, following[1]), 0) >= scale:
            break
        group, index = following
    return total + group, index


def _decimal(words: list[_Word], index: int) -> tuple[str, int]:
    """Read "point" and its digit words at `index`; empty when there are none."""
    if _at(words, index) != "point":
        return "", index
    digits = ""
    cursor = index + 1
    while _at(words, cursor) in _UNITS:
        digits += str(_UNITS[_at(words, cursor)])
        cursor += 1
    return (f".{digits}", cursor) if digits else ("", index)


def _second_half(words: list[_Word], index: int) -> tuple[int, int] | None:
    """Read the "thirty" of "six thirty" or the "oh five" of "twenty oh five"."""
    if _at(words, index) == "oh":
        value = _UNITS.get(_at(words, index + 1), 0)
        end = index + 2
    else:
        parsed = _tens_part(words, index)
        if parsed is None or parsed[0] < _FIRST_MINUTE:
            return None
        value, end = parsed
    if value == 0:
        return None
    # "six twenty thousand" is a count that happens to open like a time or year.
    if _at(words, end) == "hundred" or _at(words, end) in _SCALES:
        return None
    return value, end


def _clock(words: list[_Word], index: int) -> tuple[str, int] | None:
    """Read "six thirty" or "six oh five" at `index` as one "H:MM" token."""
    hour = _UNITS.get(_at(words, index)) or _TEENS.get(_at(words, index))
    if hour is None or hour > _LAST_HOUR:
        return None
    minute = _second_half(words, index + 1)
    if minute is None or minute[0] > _LAST_MINUTE:
        return None
    return f"{hour}:{minute[0]:02d}", minute[1]


def _year(words: list[_Word], index: int) -> tuple[str, int] | None:
    """Read "twenty twenty-six" or "nineteen oh five" at `index` as one year."""
    century = _CENTURIES.get(_at(words, index))
    if century is None:
        return None
    second = _second_half(words, index + 1)
    return None if second is None else (f"{century}{second[0]:02d}", second[1])


def _is_ordinal_after(words: list[_Word], index: int) -> bool:
    if _at(words, index) == "and":
        index += 1
    return _at(words, index) in _ORDINALS


def _replacements(words: list[_Word]) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    index = 0
    while index < len(words):
        paired = _clock(words, index) or _year(words, index)
        if paired is not None:
            found.append((words[index].start, words[paired[1] - 1].end, paired[0]))
            index = paired[1]
            continue
        parsed = _cardinal(words, index)
        if parsed is None:
            index += 1
            continue
        value, value_end = parsed
        fraction, end = _decimal(words, value_end)
        # A lone "one" is far more often a pronoun or "one of them" than a
        # count, and a count spelled that way is rarely what a number check is
        # for; "one hundred", "one point five" or "minus one" still count.
        signed = index > 0 and words[index - 1].text in _SIGNS
        lone_one = value == 1 and value_end == index + 1 and not fraction and not signed
        # The tail of an ordinal ("twenty-first") is a position, not a count.
        if not lone_one and not _is_ordinal_after(words, end):
            found.append((words[index].start, words[end - 1].end, f"{value}{fraction}"))
        index = end
    return found


def digitize(text: str) -> str:
    """Rewrite every counted spoken number in `text` as digits.

    Counted: "zero", "two" through "nine" on their own ("six PM"), any larger
    cardinal or decimal, a clock time ("six thirty" becomes "6:30", one
    token), and a year spoken in halves ("twenty twenty-six" becomes "2026").
    A "minus" or "negative" stays a word ("minus five" is "minus 5"); the
    number check decides whether it is a sign.
    Not counted: a lone "one", "a" or "an" ("the one who", "a lot"), a lone
    "oh", and ordinals ("first", "twenty-first"). The rule only has to be
    the same on both sides of the check; it is deliberately narrow where a
    word is more often prose than a count.

    Args:
        text: Any text.

    Returns:
        The text with each counted number's words replaced by its digit form,
        everything else untouched.

    """
    pieces: list[str] = []
    cursor = 0
    for chain in _chains(text):
        for start, end, digits in _replacements(chain):
            pieces.extend([text[cursor:start], digits])
            cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
