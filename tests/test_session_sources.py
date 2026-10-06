"""Session terms for serve: sources polled off the request path, merged per dictation."""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import anyio
import anyio.from_thread
import anyio.to_thread
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from structlog.testing import capture_logs

from scribe.errors import ExternalServiceError, InputValidationError
from scribe.session_sources import (
    FETCH_TIMEOUT_SECONDS,
    SessionTerms,
    check_host,
    local_source,
    merge,
    remote_source,
    run_program,
)
from scribe.session_terms import SessionBlock
from scribe.vocab import Alias, Vocab, deliver
from scribe.xai_stt import MAX_KEYTERMS

if TYPE_CHECKING:
    from collections.abc import Sequence
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

    vocab, counts = SessionTerms([source], clock=lambda: NOW).vocab(Vocab((), ()))

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
)
def test_merge_properties(static: list[str], blocks: list[tuple[str, int, list[str]]]) -> None:
    sources: dict[str, list[SessionBlock]] = {}
    for name, ranked, terms in blocks:
        sources.setdefault(name, []).append(_block("s", NOW - timedelta(minutes=ranked), terms))
    offered = set(static) | {t for _, _, terms in blocks for t in terms if t not in _BAD_TERMS}
    if len(offered) > MAX_KEYTERMS:
        event("the cut falls while two or more blocks take turns")

    merged = merge(static, list(sources.items()))

    assert len(merged.terms) == len(set(merged.terms))
    assert len(merged.terms) == min(len(offered), MAX_KEYTERMS)
    assert merged.terms[: len(static)] == tuple(static)
    assert set(merged.terms) <= offered
    assert sum(merged.counts.values()) == len(merged.terms) - len(static)
    assert set(merged.counts) == set(sources)


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

    vocab, counts = sessions.vocab(static)
    assert vocab == Vocab(terms=("herdr", "kept_term"), aliases=static.aliases)
    assert counts == {"box-a": 1}

    gone = SessionTerms([source], clock=lambda: NOW + timedelta(minutes=30))
    source.refresh(lambda: NOW + timedelta(minutes=30))
    assert gone.vocab(static) == (static, {"box-a": 0})


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
