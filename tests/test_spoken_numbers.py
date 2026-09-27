from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scribe.cleanup import NumberDiff, verify_numbers

_ONES = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _below_hundred(n: int, joiner: str) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (joiner + _ONES[ones] if ones else "")


def _below_thousand(n: int, joiner: str, *, with_and: bool) -> str:
    hundreds, rest = divmod(n, 100)
    words = [f"{_ONES[hundreds]} hundred"] if hundreds else []
    if rest:
        if hundreds and with_and:
            words.append("and")
        words.append(_below_hundred(rest, joiner))
    return " ".join(words)


def _spell(n: int, *, hyphen: bool, with_and: bool) -> str:
    """Spell 1..999,999 in English words, independently of the code under test."""
    joiner = "-" if hyphen else " "
    thousands, rest = divmod(n, 1000)
    words = (
        [f"{_below_thousand(thousands, joiner, with_and=with_and)} thousand"] if thousands else []
    )
    if rest:
        words.append(_below_thousand(rest, joiner, with_and=with_and))
    return " ".join(words)


@given(n=st.integers(min_value=2, max_value=999_999), hyphen=st.booleans(), with_and=st.booleans())
def test_a_spoken_number_matches_its_digits(n: int, *, hyphen: bool, with_and: bool) -> None:
    spoken = _spell(n, hyphen=hyphen, with_and=with_and)

    diff = verify_numbers(f"we counted {spoken} of them", f"we counted {n} of them")

    assert diff == NumberDiff(checked=1), spoken


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("we ordered forty two jars", "we ordered 42 jars"),
        ("we ordered forty-two jars", "we ordered 42 jars"),
        ("seventeen more", "17 more"),
        ("Seventeen more", "seventeen more"),
        ("two hundred and five people", "205 people"),
        ("fifteen hundred people", "1,500 people"),
        ("a hundred people", "100 people"),
        ("a thousand people", "1000 people"),
        ("one hundred people", "100 people"),
        # Speech recognition drops the article: "a hundred plus tests" heard bare.
        ("hundred plus tests", "100-plus tests"),
        ("hundred and five people", "105 people"),
        ("five hundred people", "500 people"),
        ("two million three hundred thousand", "2,300,000"),
        ("a rate of three point five", "a rate of 3.5"),
        ("a rate of one point five", "a rate of 1.5"),
        ("a rate of zero point two five", "a rate of 0.25"),
        ("see you at six PM", "see you at 6 PM"),
        ("the score was zero", "the score was 0"),
        ("meet at six thirty", "meet at 6:30"),
        ("meet at six-thirty", "meet at 6:30"),
        ("meet at six oh five", "meet at 6:05"),
        ("meet at eleven forty five", "meet at 11:45"),
    ],
)
def test_a_spoken_number_and_its_digits_are_the_same_number(before: str, after: str) -> None:
    assert verify_numbers(before, after) == NumberDiff(checked=1)


def test_a_spoken_number_dropped_in_cleanup_is_missing_in_digit_form() -> None:
    assert verify_numbers("we ordered forty two jars", "we ordered jars") == NumberDiff(
        missing=["42"], checked=1
    )


def test_a_spoken_number_changed_in_cleanup_is_missing_and_added() -> None:
    assert verify_numbers("seventeen more", "70 more") == NumberDiff(
        missing=["17"], added=["70"], checked=1
    )


@pytest.mark.parametrize(
    ("before", "tokens"),
    [
        # A scale word after the minutes makes it a count, not a clock time.
        ("six twenty thousand", ["6", "20000"]),
        ("twelve thousand", ["12000"]),
        # Past twelve there is no hour to read, so it is a year.
        ("thirteen thirty", ["1330"]),
        ("twenty thirty", ["2030"]),
        ("fourteen ninety two", ["1492"]),
        # No year opens with a composite or a tens word past twenty.
        ("forty five fifty", ["45", "50"]),
        ("twenty one twenty two", ["21", "22"]),
        ("thirty forty", ["30", "40"]),
        # Sixty is no minute, and no year opens with an hour.
        ("ten sixty", ["10", "60"]),
        # A word that only opens with a scale or unit is neither.
        ("2 thousandths", ["2"]),
        ("5 percentage points", ["5"]),
        # A scale word after the second half makes it a count, not a year.
        ("twenty twenty thousand", ["20", "20000"]),
        ("twenty oh", ["20"]),
        # Sixty is no minute.
        ("six sixty", ["6", "60"]),
        ("six hundred thirty", ["630"]),
        # "oh" makes a time only before a digit word one to nine.
        ("six oh", ["6"]),
        ("six oh zero", ["6", "0"]),
        ("five six seven", ["5", "6", "7"]),
        ("five, six", ["5", "6"]),
        ("twenty, five", ["20", "5"]),
        ("two thousand three thousand", ["2000", "3000"]),
        ("two thousand zero", ["2000", "0"]),
        ("a thousand and one nights", ["1001"]),
    ],
)
def test_how_a_run_of_number_words_splits(before: str, tokens: list[str]) -> None:
    assert verify_numbers(before, "") == NumberDiff(missing=tokens, checked=len(tokens))


@pytest.mark.parametrize(
    "text",
    [
        "the one who left",
        "one of them",
        "a lot of them",
        "an hour or so",
        "oh well",
        "the first and second items",
        "the twenty-first century",
        "on the one hundred and first day",
        "hundreds of people",
        "the hundredth time",
    ],
)
def test_words_that_are_rarely_a_count_are_not_checked(text: str) -> None:
    assert verify_numbers(text, text) == NumberDiff(checked=0)


def test_a_lone_one_is_not_checked_even_when_cleanup_drops_it() -> None:
    assert verify_numbers("the one who left", "who left") == NumberDiff(checked=0)


def test_zero_is_checked_on_its_own() -> None:
    assert verify_numbers("zero", "") == NumberDiff(missing=["0"], checked=1)


def test_a_digit_time_is_still_one_time() -> None:
    assert verify_numbers("at 6:30", "at 6:30") == NumberDiff(checked=1)


def test_a_spoken_time_rewritten_in_digits_is_the_same_time() -> None:
    assert verify_numbers("at six thirty", "at 6:30") == NumberDiff(checked=1)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("like six, six, six, six PM", "6 PM", NumberDiff(checked=1)),
        ("42 42", "42", NumberDiff(checked=1)),
        ("42, 43", "42", NumberDiff(missing=["43"], checked=2)),
        # A word between two equal values makes them two values.
        ("10 and 10", "10", NumberDiff(reduced=["10"], checked=2)),
        ("6. 6", "6", NumberDiff(reduced=["6"], checked=2)),
        # Turns are joined by a blank line; a repeat across one is not a stutter.
        (
            "it came to six\n\nsix people came",
            "it came to six\n\npeople came",
            NumberDiff(reduced=["6"], checked=2),
        ),
    ],
)
def test_a_stuttered_repeat_is_one_number(before: str, after: str, expected: NumberDiff) -> None:
    assert verify_numbers(before, after) == expected


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("two million", "2 million"),
        ("two million dollars", "$2 million"),
        ("one point five million", "1.5 million"),
        ("one point five million", "1,500,000"),
        ("a million", "1 million"),
        ("three hundred million", "300 million"),
        ("fifty thousand", "50K"),
        ("fifty thousand", "50k"),
        ("fifty thousand", "50,000"),
        ("we raised 2 million", "we raised two million"),
        ("twenty twenty-six", "2026"),
        ("nineteen eighty-four", "1984"),
        ("twenty oh five", "2005"),
        ("ten percent", "10%"),
        ("ten percent", "10 percent"),
        ("ten per cent", "10%"),
        ("1.2345 thousand", "1,234.5"),
        # A unit that opens with a suffix letter is not a scale.
        ("a 5km run", "a 5 km run"),
        ("five dollars", "$5"),
        ("twenty bucks", "$20"),
        ("five dollars", "$5.00"),
        ("5%", "5.0%"),
        ("€5.00", "€5"),
        ("£5.00", "£5"),
        ("two million dollars", "$2M"),
        ("three billion", "3B"),
        ("one point five billion", "1.5bn"),
        ("2 Million", "2,000,000"),
        ("5 Dollars", "$5"),
        ("a two dollar fee", "a $2 fee"),
    ],
)
def test_a_faithful_rewrite_of_a_number_is_no_difference(before: str, after: str) -> None:
    assert verify_numbers(before, after) == NumberDiff(checked=1)


@pytest.mark.parametrize(
    ("before", "after", "missing", "added"),
    [
        ("two million", "3 million", "2000000", "3000000"),
        ("two million dollars", "$3 million", "$2000000", "$3000000"),
        ("one point five million", "2.5 million", "1500000", "2500000"),
        ("a million", "2 million", "1000000", "2000000"),
        ("fifty thousand", "60K", "50000", "60000"),
        ("twenty twenty-six", "2025", "2026", "2025"),
        ("nineteen eighty-four", "1985", "1984", "1985"),
        ("twenty oh five", "2006", "2005", "2006"),
        ("ten percent", "20%", "10%", "20%"),
        ("five dollars", "$6", "$5", "$6"),
        ("twenty bucks", "$30", "$20", "$30"),
        ("$5", "$5.50", "$5", "$5.5"),
        ("$5.05", "$5.5", "$5.05", "$5.5"),
    ],
)
def test_a_changed_value_is_reported_whatever_its_form(
    before: str, after: str, missing: str, added: str
) -> None:
    assert verify_numbers(before, after) == NumberDiff(missing=[missing], added=[added], checked=1)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("it is fifty fifty", "it is 50-50"),
        ("it is fifty fifty", "it is 50/50"),
        ("forty five fifty", "45, 50"),
        ("twenty five thirty people", "25, 30 people"),
        ("twenty one twenty two", "21, 22"),
        ("thirty forty percent", "30 to 40%"),
    ],
)
def test_a_spoken_range_written_in_digits_is_no_difference(before: str, after: str) -> None:
    diff = verify_numbers(before, after)

    assert (diff.missing, diff.added) == ([], [])


_SCALE_VALUES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


@given(
    n=st.integers(min_value=1, max_value=999),
    scale=st.sampled_from(sorted(_SCALE_VALUES)),
    hyphen=st.booleans(),
    with_and=st.booleans(),
)
def test_a_scaled_amount_matches_in_every_form(
    n: int, scale: str, *, hyphen: bool, with_and: bool
) -> None:
    spoken = f"{_spell(n, hyphen=hyphen, with_and=with_and)} {scale}"

    for written in (f"{n} {scale}", f"{n * _SCALE_VALUES[scale]:,}"):
        assert verify_numbers(spoken, written) == NumberDiff(checked=1), (spoken, written)


@pytest.mark.parametrize(
    ("before", "after", "missing", "added"),
    [
        ("-5", "5", "-5", "5"),
        ("\N{MINUS SIGN}5", "5", "-5", "5"),
        ("the rate was minus five percent", "the rate was five percent", "-5%", "5%"),
        ("negative five dollars", "$5", "-$5", "$5"),
        ("down -3.2% year on year", "down 3.2% year on year", "-3.2%", "3.2%"),
        ("(-12)", "(12)", "-12", "12"),
        ("C$5", "$5", "C$5", "$5"),
        ("A$5", "C$5", "A$5", "C$5"),
        ("negative-five percent", "5 percent", "-5%", "5%"),
    ],
)
def test_a_dropped_sign_or_currency_prefix_is_reported(
    before: str, after: str, missing: str, added: str
) -> None:
    assert verify_numbers(before, after) == NumberDiff(missing=[missing], added=[added], checked=1)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("minus five", "-5"),
        ("negative five", "\N{MINUS SIGN}5"),
        ("minus one", "-1"),
        ("minus five percent", "-5%"),
        ("negative two million dollars", "-$2M"),
        ("it fell to negative two point five", "it fell to -2.5"),
        ("the rate was minus 5%", "the rate was -5%"),
        ("negative $12", "-$12"),
        ("US$5", "$5"),
        ("C$5.00", "C$5"),
    ],
)
def test_a_kept_sign_or_currency_prefix_is_no_difference(before: str, after: str) -> None:
    assert verify_numbers(before, after) == NumberDiff(checked=1)


@pytest.mark.parametrize(
    ("before", "after", "tokens"),
    [
        ("zero or negative five", "0 or -5", ["0", "-5"]),
        ("ten plus negative three", "10 + -3", ["10", "-3"]),
        ("negative-five percent", "-5%", ["-5%"]),
    ],
)
def test_a_sign_after_any_word_but_a_tolerance_is_kept(
    before: str, after: str, tokens: list[str]
) -> None:
    assert verify_numbers(before, after) == NumberDiff(checked=len(tokens))
    assert verify_numbers(before, "") == NumberDiff(missing=tokens, checked=len(tokens))


@pytest.mark.parametrize(
    ("before", "after", "missing", "added"),
    [
        ("zero or negative five", "0 or 5", "-5", "5"),
        ("ten plus negative three", "10 + 3", "-3", "3"),
    ],
)
def test_a_sign_dropped_after_any_word_but_a_tolerance_is_reported(
    before: str, after: str, missing: str, added: str
) -> None:
    assert verify_numbers(before, after) == NumberDiff(missing=[missing], added=[added], checked=2)


@pytest.mark.parametrize(
    ("before", "after", "tokens"),
    [
        # A hyphen between two numbers joins a range, a score or a date.
        ("5-10", "5-10", ["5", "10"]),
        ("3-2", "3-2", ["3", "2"]),
        ("2026-09-09", "2026-09-09", ["2026", "09"]),
        # "minus" after a number is arithmetic, not a sign.
        ("five minus three", "5 - 3", ["5", "3"]),
        ("5 minus 3", "5 - 3", ["5", "3"]),
        ("ten dollars minus five dollars", "$10 minus $5", ["$10", "$5"]),
        ("ten dollars minus five dollars", "$10 - $5", ["$10", "$5"]),
        # "plus or minus" is a tolerance, not a negative value.
        ("plus or minus five percent", "\N{PLUS-MINUS SIGN}5%", ["5%"]),
        ("plus or minus five percent", "+/- 5%", ["5%"]),
        ("plus minus five percent", "\N{PLUS-MINUS SIGN}5%", ["5%"]),
        ("plus  or\tminus five percent", "\N{PLUS-MINUS SIGN}5%", ["5%"]),
    ],
)
def test_a_hyphen_between_numbers_is_not_a_sign(before: str, after: str, tokens: list[str]) -> None:
    assert verify_numbers(before, after) == NumberDiff(checked=len(tokens))
    assert verify_numbers(before, "") == NumberDiff(missing=tokens, checked=len(tokens))
