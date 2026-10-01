"""Name speaker labels from the people a meeting's words address.

The speaker pass's model lists each place a chunk names an attendee, and how;
this module checks every such place against the words and decides, by count,
which label is which attendee.
"""

from __future__ import annotations

import re

from scribe.errors import InputValidationError

# A name shaped like a label could not be told from an unnamed one downstream.
_LABEL = re.compile(r"speaker\s+(\d+|\?)", re.IGNORECASE)
# The names block is one `|`-separated line per mention, and the list is shown
# to the model inside the tagged prompt.
_FORBIDDEN = ("|", "<", ">")


def parse_attendees(text: str) -> tuple[str, ...]:
    """Read a comma-separated attendee list, each name trimmed, in the order given.

    Raises:
        InputValidationError: the list names nobody, an item is empty, a name
            repeats in any case, holds a line break, `|`, `<` or `>`, or reads
            like a speaker label.

    """
    if not text.strip():
        raise InputValidationError("--attendees names nobody")
    names = tuple(item.strip() for item in text.split(","))
    seen: set[str] = set()
    for name in names:
        if not name:
            raise InputValidationError(f"--attendees has an empty name in {text!r}")
        if any(mark in name for mark in _FORBIDDEN) or name.splitlines() != [name]:
            raise InputValidationError(
                f"--attendees name {name!r} may not hold a line break, '|', '<' or '>'"
            )
        if _LABEL.fullmatch(name):
            raise InputValidationError(f"--attendees name {name!r} looks like a speaker label")
        if name.casefold() in seen:
            raise InputValidationError(f"--attendees lists {name!r} twice")
        seen.add(name.casefold())
    return names
