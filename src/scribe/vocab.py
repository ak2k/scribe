"""The dictation vocabulary: a terms file, its aliases, and the rewrites they drive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import structlog

from scribe.errors import InputValidationError
from scribe.xai_stt import check_keyterms

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from structlog.stdlib import BoundLogger

# The most words one alias or snap can rewrite at once.
MAX_RUN_WORDS = 4


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
        line = raw.strip()
        if not line or line.startswith("#"):
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
    aliased, alias_edits = _rewrite(words, "alias", lambda run: _alias(run, vocab.aliases))
    snapped, snap_edits = snap(aliased, vocab.terms)
    return snapped, alias_edits + snap_edits


def snap(words: Sequence[str], terms: Sequence[str]) -> tuple[list[str], list[Edit]]:
    """Respell each run of 1-4 words whose letters and digits spell an identifier term.

    A plain-word term is never a target, so ordinary speech is never re-cased
    or re-spaced; and a run is only ever replaced by a term with its exact
    letters and digits.

    Returns:
        The snapped words and the edits made, in order.

    """
    targets: dict[str, str] = {}
    for term in terms:
        if _key(term) and not _plain(term):
            targets.setdefault(_key(term), term)

    def respell(run: str) -> str | None:
        term = targets.get(_key(run))
        return None if term is None else _wrap(run, term, term)

    return _rewrite(words, "snap", respell)


def _key(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _plain(term: str) -> bool:
    """True for a dictionary-shaped word: all lowercase, or one capital then lowercase."""
    rest = term[1:]
    return term.isalpha() and (term.islower() or (term[0].isupper() and rest.islower()))


def _edges(text: str) -> tuple[str, str, str]:
    """Split `text` into leading punctuation, a core, and trailing punctuation."""
    alnum = [index for index, char in enumerate(text) if char.isalnum()]
    if not alnum:
        return text, "", ""
    return text[: alnum[0]], text[alnum[0] : alnum[-1] + 1], text[alnum[-1] + 1 :]


def _wrap(run: str, matched: str, replacement: str) -> str:
    """Put the run's edge punctuation, less what `matched` itself carries, around `replacement`."""
    lead, _, trail = _edges(run)
    matched_lead, _, matched_trail = _edges(matched)
    if lead.endswith(matched_lead):
        lead = lead[: len(lead) - len(matched_lead)]
    if trail.startswith(matched_trail):
        trail = trail[len(matched_trail) :]
    return lead + replacement + trail


def _alias(run: str, aliases: Sequence[Alias]) -> str | None:
    lead, core, trail = _edges(run)
    for alias in aliases:
        heard_lead, heard_core, heard_trail = _edges(alias.heard)
        if (
            core.lower() == heard_core.lower()
            and lead.endswith(heard_lead)
            and trail.startswith(heard_trail)
        ):
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
            # A run edged by a bare punctuation word would swallow that word into the match.
            if not (_key(run[0]) and _key(run[-1])):
                continue
            before = " ".join(run)
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
