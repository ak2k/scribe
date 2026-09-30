"""Guard: a property test's verdict never depends on how long one example ran."""

from __future__ import annotations

import time

from hypothesis import given, settings
from hypothesis import strategies as st

# Longer than the default 200 ms deadline and the 250 ms it allows while generating.
SLOW_EXAMPLE_SECONDS = 0.3


@given(st.none())
def test_a_property_without_settings_has_no_deadline(_: None) -> None:
    time.sleep(SLOW_EXAMPLE_SECONDS)


@settings(max_examples=1)
@given(st.none())
def test_a_property_with_its_own_settings_has_no_deadline(_: None) -> None:
    time.sleep(SLOW_EXAMPLE_SECONDS)
