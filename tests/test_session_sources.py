"""Session terms for serve: sources polled off the request path, merged per dictation."""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread
import pytest
from hypothesis import given
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
from scribe.vocab import Alias, Vocab
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
    blocks = [
        ("local", _block("old", NOW - timedelta(minutes=20), ["old_a", "old_b"])),
        ("box-a", _block("new", NOW - timedelta(minutes=1), ["new_a", "new_b", "new_c"])),
        ("box-b", _block("mid", NOW - timedelta(minutes=5), ["mid_a"])),
    ]

    merged = merge(["herdr", "VoiceInk"], blocks, NOW)

    assert merged.terms == (
        "herdr",
        "VoiceInk",
        "new_a",
        "mid_a",
        "old_a",
        "new_b",
        "old_b",
        "new_c",
    )
    assert merged.counts == {"local": 2, "box-a": 3, "box-b": 1}


def test_merge_drops_repeats_and_a_bad_term_alone() -> None:
    blocks = [
        ("local", _block("a", NOW, ["herdr", "x" * 51, "shared_t", "a_only"])),
        ("box-a", _block("b", NOW - timedelta(minutes=1), ["shared_t", "b_only"])),
    ]

    merged = merge(["herdr"], blocks, NOW)

    assert merged.terms == ("herdr", "shared_t", "b_only", "a_only")
    assert merged.counts == {"local": 2, "box-a": 1}


def test_merge_cuts_at_100_keeping_every_static_term() -> None:
    static = [f"static_{n}" for n in range(90)]
    blocks = [("local", _block("a", NOW, [f"session_{n}" for n in range(60)]))]

    merged = merge(static, blocks, NOW)

    assert len(merged.terms) == MAX_KEYTERMS
    assert merged.terms[:90] == tuple(static)
    assert merged.counts == {"local": 10}


def test_an_expired_block_contributes_nothing_while_its_sibling_does() -> None:
    gone = SessionBlock(
        "gone", NOW - timedelta(minutes=30), NOW, ("gone_term",), cwd="/w", titles=()
    )
    # Ranked earlier, yet its own expiry is still ahead.
    here = SessionBlock(
        "here",
        NOW - timedelta(minutes=40),
        NOW + timedelta(seconds=1),
        ("here_t",),
        cwd="/w",
        titles=(),
    )

    merged = merge([], [("local", gone), ("local", here)], NOW)

    assert merged.terms == ("here_t",)


_TERM = st.text(alphabet="abcdefghij_", min_size=1, max_size=6)


@given(
    static=st.lists(_TERM, max_size=120, unique=True),
    blocks=st.lists(
        st.tuples(
            st.sampled_from(["local", "box-a", "box-b"]),
            st.integers(min_value=-40, max_value=40),
            st.integers(min_value=-5, max_value=30),
            st.lists(st.one_of(_TERM, st.just("x" * 51), st.just(" ")), max_size=70),
        ),
        max_size=6,
    ),
)
def test_merge_properties(static: list[str], blocks: list[tuple[str, int, int, list[str]]]) -> None:
    given_blocks = [
        (
            name,
            SessionBlock(
                "s",
                NOW - timedelta(minutes=ranked),
                NOW + timedelta(minutes=expires),
                tuple(terms),
                cwd="/w",
                titles=(),
            ),
        )
        for name, ranked, expires, terms in blocks
    ]

    merged = merge(static, given_blocks, NOW)

    assert len(merged.terms) == len(set(merged.terms))
    assert len(merged.terms) <= MAX_KEYTERMS
    if len(static) <= MAX_KEYTERMS:
        assert merged.terms[: len(static)] == tuple(static)
    live = {term for _, block in given_blocks if block.expires > NOW for term in block.terms}
    assert set(merged.terms) <= set(static) | live
    assert sum(merged.counts.values()) == len(merged.terms) - min(len(static), MAX_KEYTERMS)


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


def test_a_source_never_fetched_contributes_nothing_and_says_never() -> None:
    source = remote_source("box-a", runner=Runner(ExternalServiceError("timed out")))
    sessions = SessionTerms([source], clock=lambda: NOW)

    sessions.refresh()

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


def test_a_missing_local_file_contributes_nothing(tmp_path: Path) -> None:
    source = local_source(tmp_path / "absent.txt")

    with capture_logs() as logs:
        source.refresh(lambda: NOW)

    assert source.live(NOW) == ()
    assert [entry["event"] for entry in logs] == ["serve.session_terms_failed"]


def test_a_blocks_expiry_ages_it_out_even_while_its_host_is_down() -> None:
    runner = Runner(_file(("s1", NOW, ["kept_term"])), ExternalServiceError("timed out"))
    source = remote_source("box-a", runner=runner)
    sessions = SessionTerms([source], clock=lambda: NOW)
    sessions.refresh()
    static = Vocab(terms=("herdr",), aliases=(Alias("her der", "herdr"),))

    vocab, counts = sessions.vocab(static)
    assert vocab == Vocab(terms=("herdr", "kept_term"), aliases=static.aliases)
    assert counts == {"box-a": 1}

    gone = SessionTerms([source], clock=lambda: NOW + timedelta(minutes=30))
    gone.refresh()
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


def test_the_runner_refuses_output_that_is_not_utf8() -> None:
    with pytest.raises(ExternalServiceError, match="not UTF-8"):
        run_program(_python("import sys; sys.stdout.buffer.write(b'\\xff')"), 5)


def test_the_runner_names_a_program_that_is_not_there(tmp_path: Path) -> None:
    with pytest.raises(ExternalServiceError, match="cannot run"):
        run_program([str(tmp_path / "no-such-program")], 5)
