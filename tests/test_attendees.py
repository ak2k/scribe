from __future__ import annotations

import pytest

from scribe.attendees import parse_attendees
from scribe.errors import InputValidationError


def test_attendees_are_read_in_order_with_their_spacing_trimmed() -> None:
    assert parse_attendees("Connor, Jose,Adam ,  Keigo") == ("Connor", "Jose", "Adam", "Keigo")


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        pytest.param("Connor,,Jose", "empty name", id="empty-item"),
        pytest.param("Connor, Jose,", "empty name", id="trailing-comma"),
        pytest.param("", "names nobody", id="empty-list"),
        pytest.param("  ", "names nobody", id="blank-list"),
        pytest.param("Jose, Connor, jose", "twice", id="duplicate"),
        pytest.param("Connor | Jose", "may not hold", id="pipe"),
        pytest.param("Con\nnor, Jose", "may not hold", id="newline"),
        pytest.param("Connor <spk:1>", "may not hold", id="angle-bracket"),
        pytest.param("Jose, Speaker 2", "looks like a speaker label", id="numbered-label"),
        pytest.param("Speaker ?, Jose", "looks like a speaker label", id="unattributed-label"),
        pytest.param("Jose, sPEAKER  12", "looks like a speaker label", id="label-any-case"),
    ],
)
def test_a_malformed_attendee_list_is_refused(text: str, complaint: str) -> None:
    with pytest.raises(InputValidationError, match=complaint):
        parse_attendees(text)
