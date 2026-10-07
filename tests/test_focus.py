"""The focused Ghostty tab: asked for by each request, matched to one local session block."""

from __future__ import annotations

import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import anyio
import pytest
from hypothesis import given
from hypothesis import strategies as st
from structlog.testing import capture_logs

from scribe.errors import ExternalServiceError, InputValidationError
from scribe.focus import (
    ARGV,
    QUERY_CAP_SECONDS,
    QUERY_TIMEOUT_SECONDS,
    Focus,
    Tab,
    Verdict,
    match,
    parse_answer,
    title_key,
)
from scribe.session_terms import SessionBlock

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
TITLE = "Fix the widget panel"


def _block(
    session: str,
    *,
    ago: timedelta = timedelta(minutes=1),
    titles: Sequence[str] = (TITLE,),
    cwd: str = "/w",
    terms: Sequence[str] = ("a_term",),
) -> SessionBlock:
    ranked = NOW - ago
    return SessionBlock(
        session, ranked, ranked + timedelta(minutes=30), tuple(terms), cwd, tuple(titles)
    )


class Osascript:
    """A fake osascript: answers from a queue, recording each call."""

    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> str:
        self.calls.append((list(argv), timeout))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _front(title: str = TITLE, cwd: str = "/w") -> str:
    return f"front\n{title}\n{cwd}\n"


def _pick(focus: Focus, blocks: Sequence[SessionBlock]) -> tuple[Verdict, SessionBlock | None]:
    return anyio.run(focus.pick, blocks)


def _asked(answer: str | Exception) -> Focus:
    focus = Focus(Osascript(answer))
    _pick(focus, [_block("s")])
    return focus


def test_the_query_runs_osascript_by_absolute_path_with_a_2_s_timeout() -> None:
    osascript = Osascript(_front())

    _pick(Focus(osascript), [_block("s")])

    assert osascript.calls == [(list(ARGV), QUERY_TIMEOUT_SECONDS)]
    assert ARGV[:2] == ("/usr/bin/osascript", "-e")
    assert QUERY_TIMEOUT_SECONDS == 2.0
    assert 'if application id "com.mitchellh.ghostty" is not running' in ARGV[2]
    assert "linefeed" in ARGV[2]


def test_a_request_waits_at_most_a_quarter_second_for_the_answer() -> None:
    assert Focus(Osascript(_front())).cap == QUERY_CAP_SECONDS == 0.25


@pytest.mark.parametrize(
    "title",
    [
        f"\u25d0 {TITLE}",
        f"\u2733 {TITLE}",
        f"\u280b {TITLE}",
        f"\U0001f514\ufe0f {TITLE}",
        f"\U0001f514\ufe0f \u2733  {TITLE.upper()}",
    ],
)
def test_the_title_key_drops_status_glyphs_and_spaces_and_folds_case(title: str) -> None:
    assert title_key(title) == TITLE.lower()


def test_the_title_key_keeps_punctuation() -> None:
    assert title_key("\u2733 (WIP) fix: the panel.") == "(wip) fix: the panel."


@given(prefix=st.text(st.characters(categories=["Sm", "Sc", "Sk", "So", "Zs"])))
def test_the_title_key_ignores_any_symbol_prefix(prefix: str) -> None:
    assert title_key(prefix + TITLE) == TITLE.lower()


@pytest.mark.parametrize(("answer", "tab"), [("absent\n", None), ("back\n", None), ("back", None)])
def test_ghostty_not_running_or_not_frontmost_is_away(answer: str, tab: Tab | None) -> None:
    assert parse_answer(answer) == tab


def test_a_front_answer_is_the_tabs_title_and_cwd() -> None:
    assert parse_answer(_front()) == Tab(TITLE, "/w")
    assert parse_answer(f"front\n{TITLE}\n/w") == Tab(TITLE, "/w")


def test_an_empty_cwd_is_kept_empty() -> None:
    assert parse_answer(f"front\n{TITLE}\n\n") == Tab(TITLE, "")


@pytest.mark.parametrize(
    "answer", ["", "front\n", f"front\n{TITLE}\n", "front\na\nb\n/w\n", "side\na\n/w\n", "away"]
)
def test_any_other_shape_is_unparseable(answer: str) -> None:
    with pytest.raises(InputValidationError):
        parse_answer(answer)


@given(st.text())
def test_any_text_parses_to_a_sample_or_is_refused(text: str) -> None:
    try:
        answer = parse_answer(text)
    except InputValidationError:
        return
    assert answer is None or isinstance(answer, Tab)


_LINE = st.text(st.characters(exclude_characters="\n"))


@given(st.one_of(st.text(), st.builds(_front, _LINE, _LINE)))
def test_any_query_output_is_a_verdict(text: str) -> None:
    assert _pick(Focus(Osascript(text)), [_block("s")])[0] in {
        "error",
        "away",
        "miss",
        "hit",
        "ambiguous",
        "elsewhere",
    }


def test_a_tab_cwd_holding_a_nul_is_matched_as_reported() -> None:
    mine = _block("mine", cwd="/w\x00x")

    assert match(Tab(TITLE, "/w\x00x"), [mine, _block("other", titles=("x",))]) == ("hit", mine)


# An untitled tab picks nothing: a new session has no block yet, so its directory finds a sibling's.


def test_an_untitled_tab_is_a_miss_even_when_one_block_has_its_cwd() -> None:
    blocks = [_block("mine", titles=()), _block("named", titles=("Claude Code",)), _block("titled")]
    focus = Focus(Osascript(_front("\u2733 Claude Code", "/w")))

    assert _pick(focus, blocks) == ("miss", None)


# A titled tab: the one block, ranked within 24 h, holding its title now or before, in its cwd.


def test_a_tab_lagging_a_retitle_is_ambiguous_when_the_earlier_title_shares_its_cwd() -> None:
    renamed = _block("a", titles=("Fix billing", "Fix auth"), cwd="/proj-a")
    other = _block("b", titles=("Fix auth",), cwd="/proj-b")

    assert match(Tab("Fix auth", "/proj-a"), [renamed, other]) == ("ambiguous", None)


def test_a_lagging_tab_without_a_cwd_never_hits_the_session_now_holding_its_title() -> None:
    renamed = _block("a", titles=("Fix billing", "Fix auth"), cwd="/proj-a")
    other = _block("b", titles=("Fix auth",), cwd="/proj-b")

    assert match(Tab("Fix auth", ""), [renamed, other])[1] is None


def test_a_tab_lagging_a_retitle_is_ambiguous_when_both_sessions_share_its_cwd() -> None:
    renamed = _block("a", titles=("Fix billing", "Fix auth"), cwd="/proj")
    sibling = _block("b", titles=("Fix auth",), cwd="/proj")

    assert match(Tab("Fix auth", "/proj"), [renamed, sibling]) == ("ambiguous", None)


def test_a_tab_cwd_differing_in_case_still_finds_the_earlier_holder() -> None:
    renamed = _block("a", titles=("Fix billing", "Fix auth"), cwd="/Users/someone/proj")
    other = _block("b", titles=("Fix auth",), cwd="/elsewhere")

    assert match(Tab("Fix auth", "/USERS/someone/proj"), [renamed, other]) == ("ambiguous", None)


def test_the_one_holder_is_hit_from_a_tab_cwd_differing_in_case() -> None:
    mine = _block("mine", cwd="/Users/someone/Proj")

    assert match(Tab(TITLE, "/USERS/someone/proj"), [mine]) == ("hit", mine)


def test_the_one_holder_in_another_cwd_is_not_hit() -> None:
    there = _block("there", cwd="/elsewhere")

    assert match(Tab(TITLE, "/w"), [there, _block("other", titles=("x",))]) == ("elsewhere", None)


def test_a_titled_tab_hits_a_block_past_its_expiry_by_title() -> None:
    old = _block("old", ago=timedelta(hours=23))

    assert match(Tab(f"\u2733 {TITLE}", "/w"), [old, _block("other", titles=("x",))]) == (
        "hit",
        old,
    )


def test_an_earlier_title_hits_when_no_other_block_holds_it() -> None:
    renamed = _block("renamed", titles=("Newer name", TITLE))

    assert match(Tab(TITLE, "/w"), [renamed, _block("other", titles=("x",))]) == (
        "hit",
        renamed,
    )


def test_two_title_holders_are_ambiguous_whatever_their_cwds() -> None:
    here, there = _block("here"), _block("there", cwd="/v")
    renamed = _block("renamed", titles=("Newer name", TITLE))

    assert match(Tab(TITLE, "/w"), [here, there]) == ("ambiguous", None)
    assert match(Tab(TITLE, "/w"), [here, _block("twin")]) == ("ambiguous", None)
    assert match(Tab(TITLE, "/w"), [renamed, here]) == ("ambiguous", None)


_TITLES = st.sampled_from(["Fix auth", "\u2733 Fix auth", "Fix billing", "Claude Code", ""])
_CWDS = st.sampled_from(["/proj", "/PROJ", "/other", ""])


@given(
    tab=st.builds(Tab, _TITLES, _CWDS),
    held=st.lists(st.tuples(st.lists(_TITLES, max_size=3), _CWDS), max_size=4),
)
def test_a_hit_is_the_only_block_holding_the_title_and_shares_the_tabs_cwd(
    tab: Tab, held: list[tuple[list[str], str]]
) -> None:
    blocks = [_block(f"s{n}", titles=titles, cwd=cwd) for n, (titles, cwd) in enumerate(held)]

    verdict, block = match(tab, blocks)

    if verdict == "hit":
        assert block is not None
        holders = [b for b in blocks if title_key(tab.title) in map(title_key, b.titles)]
        assert holders == [block]
        assert tab.cwd
        assert block.cwd.casefold() == tab.cwd.casefold()
    else:
        assert block is None


def test_a_titled_tab_never_falls_back_to_the_cwd() -> None:
    blocks = [_block("same_cwd", titles=("Something else",)), _block("untitled", titles=())]

    assert match(Tab(TITLE, "/w"), blocks) == ("miss", None)


def test_a_tab_whose_title_is_only_glyphs_matches_nothing() -> None:
    assert match(Tab("\u2733 ", "/w"), [_block("glyphs", titles=("\u2733",))]) == (
        "miss",
        None,
    )


def test_a_symlinked_tab_cwd_is_resolved_before_matching(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    mine = _block("mine", cwd=str(real))

    assert match(Tab(TITLE, str(tmp_path / "link")), [mine]) == ("hit", mine)


def test_a_tab_cwd_that_cannot_be_resolved_is_matched_as_reported(tmp_path: Path) -> None:
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    looped = str(tmp_path / "a")
    mine = _block("mine", cwd=looped)

    assert match(Tab(TITLE, looped), [mine]) == ("hit", mine)


# The query, as each request runs it.


def test_no_local_block_within_24_h_runs_no_query_and_is_a_miss() -> None:
    osascript = Osascript(AssertionError("osascript ran with no local block"))
    focus = Focus(osascript)

    assert _pick(focus, []) == ("miss", None)
    assert osascript.calls == []
    assert focus.health() == {"state": "never"}


def test_each_request_asks_for_the_tab_it_is_dictated_into() -> None:
    first, second = _block("first", titles=("First tab",)), _block("second", titles=("Second",))
    osascript = Osascript(_front("First tab"), _front("Second"))
    focus = Focus(osascript)

    assert _pick(focus, [first, second]) == ("hit", first)
    assert _pick(focus, [first, second]) == ("hit", second)
    assert len(osascript.calls) == 2


def test_a_query_hanging_past_the_cap_is_slow_and_focuses_nothing() -> None:
    release = threading.Event()

    def hung(_argv: Sequence[str], _timeout: float) -> str:
        release.wait(3)
        return _front()

    focus = Focus(hung)
    started = time.monotonic()
    try:
        with capture_logs() as logs:
            assert _pick(focus, [_block("s")]) == ("slow", None)
    finally:
        release.set()
    assert time.monotonic() - started < QUERY_CAP_SECONDS + 0.5
    assert logs == [{"event": "serve.focus_failed", "error": "slow", "log_level": "warning"}]
    assert focus.health() == {"state": "slow"}


def test_before_any_request_the_state_is_never() -> None:
    assert Focus(Osascript(_front())).health() == {"state": "never"}


def test_ghostty_away_is_away() -> None:
    focus = _asked("back\n")

    assert _pick(focus, [_block("s")]) == ("away", None)
    assert focus.health() == {"state": "away"}


def test_a_front_answer_shows_front_in_health() -> None:
    assert _asked(_front()).health() == {"state": "front"}


def _failure(message: str, cause: BaseException | None = None) -> ExternalServiceError:
    error = ExternalServiceError(message)
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        (_failure(f"exit 1: 0:12: execution error: {TITLE} not authorized. (-1743)"), "-1743"),
        (_failure(f"exit 1: execution error: Ghostty got an error: {TITLE} (-1728)"), "-1728"),
        (_failure("exit 1: execution error: (-600)"), "-600"),
        (_failure("no answer within 2 s", subprocess.TimeoutExpired("osascript", 2)), "timeout"),
        (_failure("cannot run /usr/bin/osascript: gone", FileNotFoundError()), "missing"),
        (_failure(f"exit 1: {TITLE}"), "unparseable"),
        (RuntimeError(TITLE), "unparseable"),
    ],
)
def test_a_failed_query_logs_only_its_cause_and_is_an_error(error: Exception, cause: str) -> None:
    focus = Focus(Osascript(error))

    with capture_logs() as logs:
        assert _pick(focus, [_block("s")]) == ("error", None)

    assert logs == [{"event": "serve.focus_failed", "error": cause, "log_level": "warning"}]
    assert focus.health() == {"state": "error"}


def test_an_unparseable_answer_is_an_error_naming_no_title() -> None:
    focus = Focus(Osascript(f"front\n{TITLE}\nline\n/w\n"))

    with capture_logs() as logs:
        _pick(focus, [_block("s")])

    assert logs == [{"event": "serve.focus_failed", "error": "unparseable", "log_level": "warning"}]


def test_failures_log_once_per_cause_and_recovery_once() -> None:
    denied = _failure(f"exit 1: {TITLE} (-1743)")
    timeout = _failure("no answer within 2 s", subprocess.TimeoutExpired("osascript", 2))
    focus = Focus(Osascript(denied, denied, timeout, timeout, _front(), _front()))

    with capture_logs() as logs:
        for _ in range(6):
            _pick(focus, [_block("s")])

    assert [(entry["event"], entry.get("error")) for entry in logs] == [
        ("serve.focus_failed", "-1743"),
        ("serve.focus_failed", "timeout"),
        ("serve.focus_restored", None),
    ]
    assert TITLE not in repr(logs)
