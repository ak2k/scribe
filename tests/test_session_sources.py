"""Session terms for serve: sources polled off the request path, merged per dictation."""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, override

import anyio
import anyio.from_thread
import anyio.to_thread
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from structlog.testing import capture_logs

from scribe.errors import ExternalServiceError, InputValidationError
from scribe.focus import Focus
from scribe.session_sources import (
    FETCH_TIMEOUT_SECONDS,
    LOCAL_NAME,
    FocusUse,
    SessionTerms,
    Source,
    check_host,
    local_source,
    merge,
    remote_source,
    run_program,
)
from scribe.session_terms import SessionBlock, format_block
from scribe.vocab import Alias, Vocab, deliver
from scribe.xai_stt import MAX_KEYTERMS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
SSH_FILE = ".local/state/scribe/terms/current.txt"


def _header(session: str, ranked: datetime, expires: datetime | None = None) -> str:
    leaves = ranked + timedelta(minutes=30) if expires is None else expires
    return (
        f"# session {session} ranked {ranked.isoformat()} expires {leaves.isoformat()}"
        ' {"cwd": "/w", "titles": ["A tab"]}\n'
    )


def _file(*blocks: tuple[str, datetime, list[str]]) -> str:
    return "".join(
        _header(s, ranked) + "".join(f"{t}\n" for t in terms) for s, ranked, terms in blocks
    )


def _block(session: str, ranked: datetime, terms: Sequence[str]) -> SessionBlock:
    return SessionBlock(
        session, ranked, ranked + timedelta(minutes=30), tuple(terms), cwd="/w", titles=("A tab",)
    )


class Runner:
    """A fake ssh: answers from a queue, recording each call."""

    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[list[str], float]] = []
        self.polled_twice = threading.Event()

    def __call__(self, argv: Sequence[str], timeout: float) -> str:
        self.calls.append((list(argv), timeout))
        if len(self.calls) == 2:
            self.polled_twice.set()
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_merge_puts_static_terms_first_then_sessions_newest_first_taking_turns() -> None:
    sources = [
        ("local", [_block("old", NOW - timedelta(minutes=20), ["old_a", "old_b"])]),
        ("box-a", [_block("new", NOW - timedelta(minutes=1), ["new_a", "new_b", "new_c"])]),
        ("box-b", [_block("mid", NOW - timedelta(minutes=5), ["mid_a"])]),
    ]

    merged = merge(["herdr", "VoiceInk"], sources)

    assert merged.terms == (
        "herdr",
        "VoiceInk",
        "new_a",
        "new_b",
        "new_c",
        "mid_a",
        "old_a",
        "old_b",
    )
    assert merged.counts == {"local": 2, "box-a": 3, "box-b": 1}


def test_merge_drops_repeats_and_a_bad_term_alone() -> None:
    sources = [
        ("local", [_block("a", NOW, ["herdr", "x" * 51, "shared_t", "a_only"])]),
        ("box-a", [_block("b", NOW - timedelta(minutes=1), ["shared_t", "b_only"])]),
    ]

    merged = merge(["herdr"], sources)

    assert merged.terms == ("herdr", "shared_t", "a_only", "b_only")
    assert merged.counts == {"local": 2, "box-a": 1}


def test_merge_cuts_at_100_keeping_every_static_term() -> None:
    static = [f"static_{n}" for n in range(90)]
    sources = [("local", [_block("a", NOW, [f"session_{n}" for n in range(60)])])]

    merged = merge(static, sources)

    assert len(merged.terms) == MAX_KEYTERMS
    assert merged.terms[:90] == tuple(static)
    assert merged.counts == {"local": 10}


def test_the_cut_holds_when_it_falls_mid_turn() -> None:
    static = [f"static_{n}" for n in range(MAX_KEYTERMS - 1)]
    sources = [
        ("local", [_block("a", NOW - timedelta(minutes=1), ["a_one", "a_two"])]),
        ("box-a", [_block("b", NOW - timedelta(minutes=2), ["b_one"])]),
    ]

    merged = merge(static, sources)

    assert merged.terms == (*static, "a_one")
    assert merged.counts == {"local": 1, "box-a": 0}


def test_the_newer_sessions_spelling_wins_a_snap_and_neither_spelling_is_respelled() -> None:
    sources = [
        ("box-a", [_block("new", NOW - timedelta(minutes=1), ["Unrelated_thing", "fooBar"])]),
        ("box-b", [_block("old", NOW - timedelta(minutes=5), ["foo_bar"])]),
    ]
    merged = merge([], sources)

    words, _ = deliver(
        ["see", "foo", "bar", "and", "fooBar", "and", "foo_bar"], Vocab(merged.terms, ())
    )

    assert words == ["see", "fooBar", "and", "fooBar", "and", "foo_bar"]


def test_a_static_spelling_beats_every_session_spelling_in_a_snap() -> None:
    sources = [("box-a", [_block("new", NOW, ["fooBar"])])]
    merged = merge(["foo_bar"], sources)

    words, _ = deliver(["foo", "bar", "and", "fooBar"], Vocab(merged.terms, ()))

    assert words == ["foo_bar", "and", "fooBar"]


def test_the_terms_taken_by_turns_are_listed_newest_block_first() -> None:
    static = [f"static_{n}" for n in range(MAX_KEYTERMS - 4)]
    sources = [
        ("box-b", [_block("old", NOW - timedelta(minutes=5), ["o_one", "o_two", "o_three"])]),
        ("box-a", [_block("new", NOW - timedelta(minutes=1), ["n_one", "n_two", "n_three"])]),
    ]

    merged = merge(static, sources)

    # Turns still decide which four make the cut: two from each block, not three newest.
    assert merged.terms == (*static, "n_one", "n_two", "o_one", "o_two")
    assert merged.counts == {"box-b": 2, "box-a": 2}


def test_an_expired_block_contributes_nothing_while_its_sibling_does() -> None:
    gone = _header("gone", NOW - timedelta(minutes=30), NOW) + "gone_term\n"
    # Ranked earlier, yet its own expiry is still ahead.
    here = _header("here", NOW - timedelta(minutes=40), NOW + timedelta(seconds=1)) + "here_t\n"
    source = remote_source("box-a", runner=Runner(gone + here))
    source.refresh(lambda: NOW)

    vocab, counts, _ = SessionTerms([source], clock=lambda: NOW).vocab(Vocab((), ()))

    assert vocab.terms == ("here_t",)
    assert counts == {"box-a": 1}


_TERM = st.text(alphabet="abcdefghij_", min_size=1, max_size=6)
_BAD_TERMS = ("x" * 51, " ")


# Short and near-full static lists alike, so the 100 cut is met as often as not.
@given(
    static=st.one_of(
        st.lists(_TERM, max_size=10, unique=True),
        st.lists(_TERM, min_size=90, max_size=MAX_KEYTERMS, unique=True),
    ),
    blocks=st.lists(
        st.tuples(
            st.sampled_from(["local", "box-a", "box-b"]),
            st.integers(min_value=0, max_value=40),
            st.lists(st.one_of(_TERM, st.sampled_from(_BAD_TERMS)), min_size=1, max_size=70),
        ),
        min_size=2,
        max_size=6,
    ),
    focused=st.one_of(
        st.none(),
        st.lists(st.one_of(_TERM, st.sampled_from(_BAD_TERMS)), min_size=1, max_size=70),
    ),
)
def test_merge_properties(
    static: list[str], blocks: list[tuple[str, int, list[str]]], focused: list[str] | None
) -> None:
    sources: dict[str, list[SessionBlock]] = {}
    for name, ranked, terms in blocks:
        sources.setdefault(name, []).append(_block("s", NOW - timedelta(minutes=ranked), terms))
    offered = set(static) | {t for _, _, terms in blocks for t in terms if t not in _BAD_TERMS}
    mine = None if focused is None else _block("mine", NOW - timedelta(hours=2), focused)
    if focused is not None:
        offered |= {t for t in focused if t not in _BAD_TERMS}
    if len(offered) > MAX_KEYTERMS:
        event("the cut falls while two or more blocks take turns")

    merged = merge(static, list(sources.items()), None if mine is None else ("local", mine))

    assert len(merged.terms) == len(set(merged.terms))
    assert len(merged.terms) == min(len(offered), MAX_KEYTERMS)
    assert merged.terms[: len(static)] == tuple(static)
    lead = [t for t in dict.fromkeys(focused or []) if t not in _BAD_TERMS and t not in static]
    lead = lead[: MAX_KEYTERMS - len(static)]
    assert merged.terms[len(static) : len(static) + len(lead)] == tuple(lead)
    assert merged.focused == len(lead)
    assert set(merged.terms) <= offered
    assert sum(merged.counts.values()) == len(merged.terms) - len(static)
    assert set(merged.counts) == set(sources) | ({"local"} if mine is not None else set())


def test_a_remote_source_runs_ssh_with_a_short_connect_timeout() -> None:
    runner = Runner(_file(("s1", NOW, ["remote_term"])))
    source = remote_source("box-a", runner=runner)

    source.refresh(lambda: NOW)

    assert runner.calls == [
        (
            ["ssh", "-o", "ConnectTimeout=2", "-o", "BatchMode=yes", "box-a", "cat", SSH_FILE],
            FETCH_TIMEOUT_SECONDS,
        )
    ]
    assert source.live(NOW) == (_block("s1", NOW, ["remote_term"]),)


def test_a_failing_host_keeps_its_last_good_blocks_and_logs_each_change_once() -> None:
    down = ExternalServiceError("exit 255: connection refused")
    runner = Runner(_file(("s1", NOW, ["kept_term"])), down, down, down, "garbage_line\n", "")
    source = remote_source("box-a", runner=runner)

    with capture_logs() as logs:
        for _ in range(6):
            source.refresh(lambda: NOW)
            if len(runner.calls) < 5:  # before the good, empty answer
                assert source.live(NOW) == (_block("s1", NOW, ["kept_term"]),)

    assert source.live(NOW) == ()
    assert [(entry["event"], entry.get("error")) for entry in logs] == [
        ("serve.session_terms_failed", "exit 255: connection refused"),
        ("serve.session_terms_failed", "line 1: a term before any session header"),
        ("serve.session_terms_restored", None),
    ]
    assert "kept_term" not in repr(logs)
    assert "garbage_line" not in repr(logs)


def test_a_header_field_this_serve_does_not_know_still_yields_the_terms() -> None:
    head = _header("s1", NOW).replace('"titles"', '"focus": "x", "titles"')
    source = remote_source("box-a", runner=Runner(head + "newer_term\n"))

    source.refresh(lambda: NOW)

    assert source.live(NOW) == (_block("s1", NOW, ["newer_term"]),)


def test_a_remote_answer_over_a_mebibyte_is_a_failure_that_keeps_the_last_good() -> None:
    good = _file(("s1", NOW, ["kept_term"]))
    huge = good + "filler_term\n" * (2**20 // len("filler_term\n"))
    source = remote_source("box-a", runner=Runner(good, huge, huge))

    with capture_logs() as logs:
        for _ in range(3):
            source.refresh(lambda: NOW)

    assert source.live(NOW) == (_block("s1", NOW, ["kept_term"]),)
    assert [entry["event"] for entry in logs] == ["serve.session_terms_failed"]


def test_one_long_block_parses_in_linear_time() -> None:
    text = _header("s1", NOW) + "".join(f"term_{n}\n" for n in range(60_000))
    source = remote_source("box-a", runner=Runner(text))

    started = time.monotonic()
    source.refresh(lambda: NOW)

    assert len(source.live(NOW)[0].terms) == 60_000
    assert time.monotonic() - started < 3  # a rebuild per term takes several times this


def test_a_source_never_fetched_contributes_nothing_and_says_never() -> None:
    source = remote_source("box-a", runner=Runner(ExternalServiceError("timed out")))
    sessions = SessionTerms([source], clock=lambda: NOW)

    source.refresh(lambda: NOW)

    assert source.live(NOW) == ()
    assert sessions.health() == [{"source": "box-a", "blocks": 0, "terms": 0, "age_seconds": None}]


def test_a_file_with_no_block_header_contributes_nothing_logged_once(tmp_path: Path) -> None:
    path = tmp_path / "current.txt"
    path.write_text("# expires 2026-10-05T12:30:00+00:00\nold_format_term\n")
    source = local_source(path)

    with capture_logs() as logs:
        source.refresh(lambda: NOW)
        source.refresh(lambda: NOW)

    assert source.live(NOW) == ()
    assert [entry["event"] for entry in logs] == ["serve.session_terms_failed"]


def test_the_local_file_is_read_again_only_when_it_changes(tmp_path: Path) -> None:
    path = tmp_path / "current.txt"
    path.write_text(_file(("s1", NOW, ["first_term"])))
    source = local_source(path)
    source.refresh(lambda: NOW)
    later = NOW + timedelta(seconds=5)

    path.chmod(0o000)
    source.refresh(lambda: later)

    assert source.live(later)[0].terms == ("first_term",)
    assert source.health(later)["age_seconds"] == 0
    path.chmod(0o600)
    path.write_text(_file(("s1", NOW, ["second_term", "more_term"])))
    stamp = time.time() + 10
    os.utime(path, (stamp, stamp))
    source.refresh(lambda: later)
    assert source.live(later)[0].terms == ("second_term", "more_term")


def test_a_local_file_failing_unchanged_keeps_aging(tmp_path: Path) -> None:
    path = tmp_path / "current.txt"
    path.write_text(_file(("s1", NOW, ["first_term"])))
    source = local_source(path)
    source.refresh(lambda: NOW)
    path.write_text("not_a_session_file\n")
    stamp = time.time() + 10
    os.utime(path, (stamp, stamp))

    source.refresh(lambda: NOW + timedelta(seconds=5))
    source.refresh(lambda: NOW + timedelta(seconds=10))

    assert source.health(NOW + timedelta(seconds=10))["age_seconds"] == 10


def test_a_local_file_back_unchanged_after_a_failed_read_is_read_again(tmp_path: Path) -> None:
    path = tmp_path / "current.txt"
    path.write_text(_file(("s1", NOW, ["first_term"])))
    source = local_source(path)
    source.refresh(lambda: NOW)
    aside = tmp_path / "aside.txt"
    path.rename(aside)
    later = NOW + timedelta(seconds=10)

    with capture_logs() as logs:
        source.refresh(lambda: NOW + timedelta(seconds=5))
        aside.rename(path)
        source.refresh(lambda: later)

    assert [entry["event"] for entry in logs] == [
        "serve.session_terms_failed",
        "serve.session_terms_restored",
    ]
    assert source.health(later)["age_seconds"] == 0


def test_a_missing_local_file_contributes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "absent.txt"
    source = local_source(path)

    with capture_logs() as logs:
        source.refresh(lambda: NOW)

    assert source.live(NOW) == ()
    assert [(entry["event"], entry["error"]) for entry in logs] == [
        ("serve.session_terms_failed", f"cannot read {path}: No such file or directory")
    ]


def test_a_local_file_that_is_not_utf8_contributes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "current.txt"
    path.write_bytes(b"\xff\n")
    source = local_source(path)

    with capture_logs() as logs:
        source.refresh(lambda: NOW)

    assert source.live(NOW) == ()
    cause = "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"
    assert [entry["error"] for entry in logs] == [f"cannot read {path}: {cause}"]


def test_a_blocks_expiry_ages_it_out_even_while_its_host_is_down() -> None:
    runner = Runner(_file(("s1", NOW, ["kept_term"])), ExternalServiceError("timed out"))
    source = remote_source("box-a", runner=runner)
    sessions = SessionTerms([source], clock=lambda: NOW)
    source.refresh(lambda: NOW)
    static = Vocab(terms=("herdr",), aliases=(Alias("her der", "herdr"),))

    vocab, counts, _ = sessions.vocab(static)
    assert vocab == Vocab(terms=("herdr", "kept_term"), aliases=static.aliases)
    assert counts == {"box-a": 1}

    gone = SessionTerms([source], clock=lambda: NOW + timedelta(minutes=30))
    source.refresh(lambda: NOW + timedelta(minutes=30))
    assert gone.vocab(static) == (static, {"box-a": 0}, FocusUse("off", 0))


def test_health_counts_live_blocks_and_terms_and_the_fetch_age() -> None:
    text = _file(("s1", NOW, ["a_term", "b_term"]), ("s2", NOW - timedelta(minutes=29), ["c_t"]))
    source = remote_source("box-a", runner=Runner(text))
    source.refresh(lambda: NOW)

    later = NOW + timedelta(minutes=1, seconds=30)
    assert source.health(later) == {"source": "box-a", "blocks": 1, "terms": 2, "age_seconds": 90}


def test_an_unexpected_error_in_a_fetch_is_a_failure_not_a_crash() -> None:
    source = remote_source("box-a", runner=Runner(RuntimeError("boom")))

    with capture_logs() as logs:
        source.refresh(lambda: NOW)

    assert [entry["error"] for entry in logs] == ["RuntimeError"]


@pytest.mark.anyio
async def test_the_poller_refreshes_every_source_until_cancelled() -> None:
    runner = Runner(_file(("s1", NOW, ["polled_term"])))
    sessions = SessionTerms(
        [remote_source("box-a", runner=runner, interval=0.01)], clock=lambda: NOW
    )

    async with anyio.create_task_group() as group:
        group.start_soon(sessions.poll)
        assert await anyio.to_thread.run_sync(runner.polled_twice.wait, 2)
        group.cancel_scope.cancel()

    assert sessions.vocab(Vocab((), ()))[0].terms == ("polled_term",)
    calls = len(runner.calls)
    await anyio.sleep(0.05)
    assert len(runner.calls) == calls


@pytest.mark.anyio
async def test_a_remote_source_is_polled_once_per_interval() -> None:
    runner = Runner(_file(("s1", NOW, ["polled_term"])))
    sessions = SessionTerms([remote_source("box-a", runner=runner)], clock=lambda: NOW)

    async with anyio.create_task_group() as group:
        group.start_soon(sessions.poll)
        await anyio.sleep(0.5)  # well inside the 3 s interval
        group.cancel_scope.cancel()

    assert len(runner.calls) == 1


@pytest.mark.anyio
async def test_shutdown_does_not_wait_for_a_fetch_in_flight() -> None:
    started, release = threading.Event(), threading.Event()

    def hung(_argv: Sequence[str], _timeout: float) -> str:
        started.set()
        release.wait(2)
        return ""

    sessions = SessionTerms([remote_source("box-a", runner=hung)], clock=lambda: NOW)
    cancelled: float | None = None
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(sessions.poll)
            assert await anyio.to_thread.run_sync(started.wait, 2)
            cancelled = time.monotonic()
            group.cancel_scope.cancel()
        assert cancelled is not None
        assert time.monotonic() - cancelled < 0.5
    finally:
        release.set()


@pytest.mark.anyio
async def test_fetches_in_flight_never_hold_a_requests_worker_thread() -> None:
    hosts = int(anyio.to_thread.current_default_thread_limiter().total_tokens)
    lock, release, all_started = threading.Lock(), threading.Event(), anyio.Event()
    started = 0

    def hung(_argv: Sequence[str], _timeout: float) -> str:
        nonlocal started
        with lock:
            started += 1
            last = started == hosts
        if last:
            anyio.from_thread.run_sync(all_started.set)
        release.wait(2)
        return ""

    sessions = SessionTerms(
        [remote_source(f"box-{n}", runner=hung) for n in range(hosts)], clock=lambda: NOW
    )
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(sessions.poll)
            await all_started.wait()
            with anyio.fail_after(0.5):
                assert await anyio.to_thread.run_sync(lambda: "served") == "served"
            group.cancel_scope.cancel()
    finally:
        release.set()


@pytest.mark.parametrize("host", ["", "-oProxyCommand=x", "box a", "box\ta", "box\x7f"])
def test_a_bad_host_is_refused(host: str) -> None:
    with pytest.raises(InputValidationError):
        check_host(host)


def test_a_plain_or_user_qualified_host_is_accepted() -> None:
    check_host("box-a")
    check_host("me@box-a.example")


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_the_runner_returns_a_programs_output() -> None:
    assert run_program(_python("print('caf\\u00e9')"), 5) == "café\n"


def test_the_runner_names_a_non_zero_exit_with_its_last_error_line() -> None:
    code = "import sys; sys.stderr.write('first\\nlast words\\n'); sys.exit(3)"
    with pytest.raises(ExternalServiceError, match=r"^exit 3: last words$"):
        run_program(_python(code), 5)


def test_the_runner_kills_a_program_past_its_timeout() -> None:
    started = time.monotonic()
    with pytest.raises(ExternalServiceError, match=r"no answer within 0\.3 s"):
        run_program(_python("import time; time.sleep(30)"), 0.3)
    assert time.monotonic() - started < 5  # far below the child's 30 s


def test_the_runner_gives_its_child_no_stdin() -> None:
    # Serve's own stdin is a pipe holding data here; the child must not read it.
    read, write = os.pipe()
    os.write(write, b"typed into the terminal")
    os.close(write)
    saved = os.dup(0)
    os.dup2(read, 0)
    try:
        out = run_program(_python("import sys; print(repr(sys.stdin.read()))"), 5)
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read)
    assert out == "''\n"


def test_the_runner_refuses_output_that_is_not_utf8() -> None:
    with pytest.raises(ExternalServiceError, match="not UTF-8"):
        run_program(_python("import sys; sys.stdout.buffer.write(b'\\xff')"), 5)


def test_the_runner_names_a_program_that_is_not_there(tmp_path: Path) -> None:
    with pytest.raises(ExternalServiceError, match="cannot run"):
        run_program([str(tmp_path / "no-such-program")], 5)


def _titled(session: str, ranked: datetime, terms: Sequence[str], title: str) -> SessionBlock:
    return SessionBlock(
        session, ranked, ranked + timedelta(minutes=30), tuple(terms), cwd="/w", titles=(title,)
    )


def _focused_on(title: str) -> Focus:
    focus = Focus(Runner(f"front\n{title}\n/w\n"))
    focus.refresh(lambda: NOW, lambda _now: [_block("any", NOW, [])])
    return focus


def _local(*blocks: SessionBlock) -> Source:
    text = "".join(format_block(block) for block in blocks)
    source = Source(LOCAL_NAME, lambda: text, interval=1.0)
    source.refresh(lambda: NOW)
    return source


def test_the_focused_block_follows_the_static_terms_whole_then_the_others_take_turns() -> None:
    focused = _block("mine", NOW - timedelta(minutes=10), ["herdr", "f_one", "x" * 51, "f_two"])
    sources = [
        ("local", [focused, _block("old", NOW - timedelta(minutes=20), ["old_a", "f_one"])]),
        ("box-a", [_block("new", NOW - timedelta(minutes=1), ["new_a", "new_b"])]),
    ]

    merged = merge(["herdr"], sources, focused=("local", focused))

    assert merged.terms == ("herdr", "f_one", "f_two", "new_a", "new_b", "old_a")
    assert merged.counts == {"local": 3, "box-a": 2}
    assert merged.focused == 2


def test_an_expired_focused_block_still_leads() -> None:
    focused = _block("mine", NOW - timedelta(hours=5), ["f_one"])
    sources = [("local", [_block("live", NOW, ["live_a"])])]

    merged = merge([], sources, focused=("local", focused))

    assert merged.terms == ("f_one", "live_a")
    assert merged.counts == {"local": 2}


def test_static_plus_focused_terms_are_cut_at_100() -> None:
    static = [f"static_{n}" for n in range(90)]
    focused = _block("mine", NOW, [f"focus_{n}" for n in range(20)])
    sources = [("local", [focused, _block("other", NOW, ["other_a"])])]

    merged = merge(static, sources, focused=("local", focused))

    assert merged.terms == (*static, *(f"focus_{n}" for n in range(10)))
    assert merged.focused == 10
    assert merged.counts == {"local": 10}


def test_the_focused_sessions_spelling_wins_a_snap() -> None:
    focused = _block("mine", NOW - timedelta(minutes=20), ["foo_bar"])
    sources = [("local", [_block("new", NOW, ["fooBar"]), focused])]
    merged = merge([], sources, focused=("local", focused))

    words, _ = deliver(["foo", "bar"], Vocab(merged.terms, ()))

    assert words == ["foo_bar"]


def test_recent_keeps_blocks_past_expiry_for_24_h() -> None:
    expired = _block("expired", NOW - timedelta(hours=24), ["e"])
    too_old = _block("old", NOW - timedelta(hours=24, seconds=1), ["o"])
    live = _block("live", NOW, ["l"])
    source = _local(live, expired, too_old)

    assert source.recent(NOW) == (live, expired)
    assert source.live(NOW) == (live,)


def test_a_focus_hit_puts_that_sessions_terms_first_and_says_so() -> None:
    mine = _titled("mine", NOW - timedelta(hours=3), ["mine_a", "mine_b"], "My tab")
    other = _titled("other", NOW - timedelta(minutes=1), ["other_a"], "Other tab")
    sessions = SessionTerms([_local(other, mine)], clock=lambda: NOW, focus=_focused_on("My tab"))

    vocab, counts, focus = sessions.vocab(Vocab(("static_t",), ()))

    assert vocab.terms == ("static_t", "mine_a", "mine_b", "other_a")
    assert counts == {LOCAL_NAME: 3}
    assert focus == FocusUse("hit", 2)


def test_a_request_reads_the_local_file_once_so_a_newer_read_cannot_double_the_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _titled("mine", NOW - timedelta(minutes=5), ["mine_a"], "My tab")
    new = _titled("mine", NOW - timedelta(seconds=1), ["mine_a", "mine_b"], "My tab")
    other = _titled("other", NOW - timedelta(minutes=2), ["other_a"], "Other tab")
    reads = iter([format_block(old) + format_block(other)])
    local = Source(
        LOCAL_NAME, lambda: next(reads, format_block(new) + format_block(other)), interval=1.0
    )
    local.refresh(lambda: NOW)
    # The poller swaps in the newer file after every read the request makes.
    reads_of_local: list[tuple[str, Callable[[datetime], object]]] = [
        ("live", local.live),
        ("recent", local.recent),
        ("live_and_recent", local.live_and_recent),
    ]
    for name, read in reads_of_local:

        def read_then_swap(now: datetime, read: Callable[[datetime], object] = read) -> object:
            blocks = read(now)
            local.refresh(lambda: NOW)
            return blocks

        monkeypatch.setattr(local, name, read_then_swap)
    sessions = SessionTerms([local], clock=lambda: NOW, focus=_focused_on("My tab"))

    vocab, _, focus = sessions.vocab(Vocab((), ()))

    assert vocab.terms == ("mine_a", "other_a")
    assert focus == FocusUse("hit", 1)


def test_an_untitled_tab_leaves_the_turns_as_they_were() -> None:
    untitled = SessionBlock(
        "mine", NOW - timedelta(minutes=5), NOW + timedelta(minutes=25), ("mine_a",), "/w", ()
    )
    newer = _titled("newer", NOW - timedelta(minutes=1), ["newer_a"], "Other tab")
    sessions = SessionTerms(
        [_local(newer, untitled)], clock=lambda: NOW, focus=_focused_on("Claude Code")
    )

    vocab, _, focus = sessions.vocab(Vocab((), ()))

    assert focus == FocusUse("miss", 0)
    assert vocab.terms == ("newer_a", "mine_a")


def test_a_remote_block_holding_the_tabs_title_is_never_focused() -> None:
    remote = remote_source(
        "box-a", runner=Runner(format_block(_titled("r", NOW, ["remote_t"], "My tab")))
    )
    remote.refresh(lambda: NOW)
    local = _local(_titled("l", NOW - timedelta(minutes=2), ["local_t"], "Other tab"))
    sessions = SessionTerms([local, remote], clock=lambda: NOW, focus=_focused_on("My tab"))

    vocab, _, focus = sessions.vocab(Vocab((), ()))

    assert focus == FocusUse("miss", 0)
    assert vocab.terms == ("remote_t", "local_t")


def test_without_a_recent_local_block_a_request_is_a_miss_that_never_queries() -> None:
    runner = Runner(AssertionError("osascript ran during a request"))
    remote = remote_source("box-a", runner=Runner(_file(("r", NOW, ["remote_t"]))))
    remote.refresh(lambda: NOW)
    sessions = SessionTerms([remote], clock=lambda: NOW, focus=Focus(runner))

    vocab, _, focus = sessions.vocab(Vocab((), ()))

    assert focus == FocusUse("miss", 0)
    assert vocab.terms == ("remote_t",)
    assert runner.calls == []


def test_a_request_never_reaches_the_focus_query() -> None:
    runner = Runner(AssertionError("osascript ran during a request"))
    sessions = SessionTerms(
        [_local(_block("s", NOW, ["t"]))], clock=lambda: NOW, focus=Focus(runner)
    )

    assert sessions.vocab(Vocab((), ()))[2] == FocusUse("stale", 0)
    assert runner.calls == []


def test_focus_off_is_reported_as_off() -> None:
    sessions = SessionTerms([_local(_block("s", NOW, ["t"]))], clock=lambda: NOW)

    assert sessions.vocab(Vocab((), ()))[2] == FocusUse("off", 0)
    assert sessions.focus_health() is None


@pytest.mark.anyio
async def test_the_focus_poller_takes_its_own_token_and_stops_at_shutdown() -> None:
    remote_started, local_started = threading.Event(), threading.Event()
    release = threading.Event()

    def hung(_argv: Sequence[str], _timeout: float) -> str:
        remote_started.set()
        release.wait(5)
        return ""

    reads = iter([format_block(_block("s", NOW, ["t"]))])

    def hung_local() -> str:
        # The first read gives focus a recent block to query for; every later one hangs.
        text = next(reads, None)
        if text is None:
            local_started.set()
            release.wait(5)
            return ""
        return text

    asked = threading.Event()

    def osascript(_argv: Sequence[str], _timeout: float) -> str:
        asked.set()
        return "back\n"

    local = Source(LOCAL_NAME, hung_local, interval=0.01)
    local.refresh(lambda: NOW)
    sessions = SessionTerms(
        [local, remote_source("box-a", runner=hung)],
        clock=lambda: NOW,
        focus=Focus(osascript, interval=0.01),
    )
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(sessions.poll)
            assert await anyio.to_thread.run_sync(remote_started.wait, 2)
            assert await anyio.to_thread.run_sync(local_started.wait, 2)
            assert await anyio.to_thread.run_sync(asked.wait, 2)
            group.cancel_scope.cancel()
    finally:
        release.set()
    asked.clear()
    await anyio.sleep(0.05)
    assert not asked.is_set()
    assert sessions.focus_health() == {"state": "away", "age_seconds": 0}


class _Raising(Focus):
    """A focus poller whose refresh raises each error in turn, then succeeds."""

    def __init__(self, *errors: Exception) -> None:
        super().__init__(Runner("back\n"), interval=0.01)
        self.errors = list(errors)
        self.recovered = threading.Event()

    @override
    def refresh(
        self, clock: Callable[[], datetime], recent: Callable[[datetime], Sequence[SessionBlock]]
    ) -> None:
        if self.errors:
            raise self.errors.pop(0)
        self.recovered.set()


@pytest.mark.anyio
async def test_a_poller_that_raises_logs_the_type_once_per_change_and_keeps_polling() -> None:
    quoted = "a title from the tab"
    focus = _Raising(KeyError(quoted), KeyError(quoted), ValueError(quoted))
    polled = threading.Event()

    def fetch() -> str:
        polled.set()
        return format_block(_block("s", NOW, ["t"]))

    sessions = SessionTerms(
        [Source(LOCAL_NAME, fetch, interval=0.01)], clock=lambda: NOW, focus=focus
    )
    with capture_logs() as logs:
        async with anyio.create_task_group() as group:
            group.start_soon(sessions.poll)
            assert await anyio.to_thread.run_sync(focus.recovered.wait, 2)
            polled.clear()
            assert await anyio.to_thread.run_sync(polled.wait, 2)
            group.cancel_scope.cancel()

    assert [(entry["event"], entry.get("error")) for entry in logs] == [
        ("serve.poller_failed", "KeyError"),
        ("serve.poller_failed", "ValueError"),
        ("serve.poller_restored", None),
    ]
    assert {entry["poller"] for entry in logs} == {"ghostty"}
    assert quoted not in repr(logs)


@given(
    static=st.one_of(
        st.lists(_TERM, max_size=10, unique=True),
        st.lists(_TERM, min_size=90, max_size=MAX_KEYTERMS, unique=True),
    ),
    blocks=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=120),
            st.lists(st.one_of(_TERM, st.sampled_from(_BAD_TERMS)), min_size=1, max_size=40),
        ),
        min_size=1,
        max_size=5,
    ),
    chosen=st.integers(min_value=0),
)
def test_a_focused_vocab_never_holds_an_expired_unfocused_blocks_term(
    static: list[str], blocks: list[tuple[int, list[str]]], chosen: int
) -> None:
    made = [
        _titled(f"s{n}", NOW - timedelta(minutes=ago), terms, f"tab {n}")
        for n, (ago, terms) in enumerate(blocks)
    ]
    focused = made[chosen % len(made)]
    sessions = SessionTerms(
        [_local(*made)], clock=lambda: NOW, focus=_focused_on(focused.titles[0])
    )

    vocab, _, focus = sessions.vocab(Vocab(tuple(static), ()))

    admissible = [t for t in dict.fromkeys(focused.terms) if t not in _BAD_TERMS]
    lead = [t for t in admissible if t not in static][: MAX_KEYTERMS - len(static)]
    assert focus == FocusUse("hit", len(lead))
    assert vocab.terms[: len(static) + len(lead)] == (*static, *lead)
    allowed = set(static) | set(focused.terms)
    allowed |= {t for block in made if block.expires > NOW for t in block.terms}
    assert set(vocab.terms) <= allowed
