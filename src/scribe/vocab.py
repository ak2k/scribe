"""The dictation vocabulary: a terms file, its aliases, and the rewrites they drive."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import structlog

from scribe.errors import InputValidationError
from scribe.xai_stt import check_keyterms

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from structlog.stdlib import BoundLogger

# The most words one alias or snap can rewrite at once.
MAX_RUN_WORDS = 4
# Shorter keys spell ordinary words ("c", "us", "it") too often to be snapped to.
MIN_SNAP_KEY = 3
# `#` after whitespace, not inside a term: "C#" is a term, "herdr # mine" a comment.
_COMMENT = re.compile(r"(?:^|\s)#.*")


@dataclass(frozen=True)
class Alias:
    """An operator's rewrite of what the engine hears into what was meant."""

    heard: str
    written: str


@dataclass(frozen=True)
class Vocab:
    """The terms sent to the engine as keyterms, and the aliases applied after it."""

    terms: tuple[str, ...]
    aliases: tuple[Alias, ...]


@dataclass(frozen=True)
class Edit:
    """One rewrite of a run of delivered words."""

    rule: Literal["alias", "snap"]
    before: str
    after: str


def _logger() -> BoundLogger:
    return structlog.get_logger(__name__)  # pyright: ignore[reportAny]  # structlog.get_logger is Any


def parse_terms(text: str, *, where: str) -> Vocab:
    """Parse a terms file: one term per line, `#` comments, `heard => written` aliases.

    A comment is a `#` that starts a line or follows whitespace, to the line's end.

    Args:
        text: The file's contents.
        where: How errors name the file.

    Returns:
        The terms, repeats dropped, and the aliases, in file order.

    Raises:
        InputValidationError: an alias line is malformed, a term is past xAI's
            keyterm limits, or there are too many terms; named by line number.

    """
    terms: list[str] = []
    aliases: list[Alias] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _COMMENT.sub("", raw).strip()
        if not line:
            continue
        try:
            if "=>" in line:
                aliases.append(_parse_alias(line))
            else:
                check_keyterms([line])
                terms.append(line)
        except InputValidationError as exc:
            raise InputValidationError(f"{where}:{number}: {exc}") from exc
    unique = tuple(dict.fromkeys(terms))
    try:
        check_keyterms(unique)
    except InputValidationError as exc:
        raise InputValidationError(f"{where}: {exc}") from exc
    return Vocab(terms=unique, aliases=tuple(aliases))


def _parse_alias(line: str) -> Alias:
    sides = line.split("=>")
    if len(sides) != 2:  # noqa: PLR2004  # an alias has exactly two sides
        raise InputValidationError("an alias line holds exactly one '=>'")
    heard, written = (" ".join(side.split()) for side in sides)
    if not heard or not written:
        raise InputValidationError("an alias needs words on both sides of '=>'")
    if len(heard.split()) > MAX_RUN_WORDS:
        raise InputValidationError(f"an alias hears at most {MAX_RUN_WORDS} words")
    return Alias(heard=heard, written=written)


def deliver(words: Sequence[str], vocab: Vocab) -> tuple[list[str], list[Edit]]:
    """Apply the aliases, then snap the result to the terms.

    Returns:
        The delivered words and every edit made, in order.

    """
    heard: dict[str, list[tuple[str, str, Alias]]] = {}
    for alias in vocab.aliases:
        lead, core, trail = _edges(alias.heard)
        heard.setdefault(core.lower(), []).append((lead, trail, alias))
    aliased, alias_edits = _rewrite(words, "alias", lambda run: _alias(run, heard))
    snapped, snap_edits = snap(aliased, vocab.terms)
    return snapped, alias_edits + snap_edits


def snap(words: Sequence[str], terms: Sequence[str]) -> tuple[list[str], list[Edit]]:
    """Respell each run of 1-4 words whose letters and digits spell an identifier term.

    Only a term ordinary speech cannot spell is a target (see `_target`), and a
    run is only ever replaced by a term with its exact letters and digits, and
    only when the term holds, in order, the punctuation inside the run.

    Returns:
        The snapped words and the edits made, in order.

    """
    targets: dict[str, str] = {}
    for term in terms:
        if _target(term):
            targets.setdefault(_key(term), term)

    def respell(run: str) -> str | None:
        term = targets.get(_key(run))
        # Punctuation the term lacks is a boundary the speaker made: "my voice. Ink".
        if term is None or not _in_order(_inner_punctuation(run), _inner_punctuation(term)):
            return None
        return _wrap(run, term, term)

    return _rewrite(words, "snap", respell)


def _spelled(char: str) -> bool:
    """True for a letter, a digit, or a combining mark: what a word is spelled with."""
    return char.isalnum() or unicodedata.category(char).startswith("M")


def _key(text: str) -> str:
    # NFC, and marks kept, so "café" never keys like "cafe", nor "İ" like "i".
    return "".join(char for char in unicodedata.normalize("NFC", text).lower() if _spelled(char))


def _target(term: str) -> bool:
    """True for a term shaped like an identifier, which ordinary speech does not spell.

    Its core (edge punctuation aside) holds a digit, a capital right after a
    lowercase letter, or punctuation; it has a letter; and its key is at least
    `MIN_SNAP_KEY` long. "TODO", "c++", ".NET", "pull request" and "2.0" are not.
    """
    _, core, _ = _edges(term)
    joint = (
        any(char.isdigit() for char in core)
        or any(left.islower() and right.isupper() for left, right in pairwise(core))
        or bool(_inner_punctuation(term))
    )
    return joint and any(char.isalpha() for char in core) and len(_key(term)) >= MIN_SNAP_KEY


def _inner_punctuation(text: str) -> list[str]:
    _, core, _ = _edges(text)
    return [char for char in core if not _spelled(char) and not char.isspace()]


def _in_order(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    rest = iter(haystack)
    return all(char in rest for char in needle)


def _edges(text: str) -> tuple[str, str, str]:
    """Split `text` into leading punctuation, a core, and trailing punctuation."""
    spelled = [index for index, char in enumerate(text) if _spelled(char)]
    if not spelled:
        return text, "", ""
    return text[: spelled[0]], text[spelled[0] : spelled[-1] + 1], text[spelled[-1] + 1 :]


def _wrap(run: str, matched: str, replacement: str) -> str:
    """Put the run's edge punctuation, less what `matched` itself carries, around `replacement`."""
    lead, _, trail = _edges(run)
    matched_lead, _, matched_trail = _edges(matched)
    if lead.endswith(matched_lead):
        lead = lead[: len(lead) - len(matched_lead)]
    if trail.startswith(matched_trail):
        trail = trail[len(matched_trail) :]
    return lead + replacement + trail


def _alias(run: str, heard: Mapping[str, Sequence[tuple[str, str, Alias]]]) -> str | None:
    """Rewrite `run` by the first alias heard as it, keyed by lowercased core."""
    lead, core, trail = _edges(run)
    for heard_lead, heard_trail, alias in heard.get(core.lower(), ()):
        if lead.endswith(heard_lead) and trail.startswith(heard_trail):
            return _wrap(run, alias.heard, alias.written)
    return None


def _rewrite(
    words: Sequence[str],
    rule: Literal["alias", "snap"],
    replace: Callable[[str], str | None],
) -> tuple[list[str], list[Edit]]:
    """Replace runs leftmost-longest, never overlapping, wherever `replace` matches one."""
    out: list[str] = []
    edits: list[Edit] = []
    start = 0
    while start < len(words):
        for size in range(min(MAX_RUN_WORDS, len(words) - start), 0, -1):
            run = words[start : start + size]
            before = " ".join(run)
            # A punctuation-only run has no letters to match, so any match would invent them.
            if not _key(before):
                continue
            after = replace(before)
            if after is not None:
                out.extend(after.split())
                if after != before:
                    edits.append(Edit(rule=rule, before=before, after=after))
                start += size
                break
        else:
            out.append(words[start])
            start += 1
    return out, edits


class TermsFile:
    """A terms file, re-read whenever its modification time or size changes."""

    def __init__(self, path: Path, *, required: bool) -> None:
        """Read the file now.

        Args:
            path: The terms file.
            required: Whether a missing file is an error rather than no terms.

        Raises:
            InputValidationError: the file is required and missing, unreadable,
                or malformed.

        """
        self.path = path
        self._required = required
        self._signature = self._stat()
        self._vocab = self._read()

    def _stat(self) -> tuple[int, int] | None:
        try:
            info = self.path.stat()
        except OSError:
            return None
        return info.st_mtime_ns, info.st_size

    def _read(self) -> Vocab:
        try:
            # utf-8-sig: a byte order mark some editors write is not part of a term.
            text = self.path.read_text(encoding="utf-8-sig")
        except FileNotFoundError as exc:
            if not self._required:
                return Vocab(terms=(), aliases=())
            raise InputValidationError(f"terms file {self.path} does not exist") from exc
        # ValueError too: a non-UTF-8 file raises UnicodeDecodeError, not OSError.
        except (OSError, ValueError) as exc:
            raise InputValidationError(f"cannot read terms file {self.path}: {exc}") from exc
        return parse_terms(text, where=str(self.path))

    def current(self) -> Vocab:
        """Return the vocabulary, re-reading the file first if it changed.

        A bad re-read logs one error and keeps the previous vocabulary.
        """
        signature = self._stat()
        if signature != self._signature:
            self._signature = signature
            try:
                self._vocab = self._read()
            except InputValidationError as exc:
                _logger().error("serve.terms_reload_failed", error=str(exc))
        return self._vocab
