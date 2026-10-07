"""The terms hook: a session's vocabulary and the prompt log, from hook JSON on stdin."""

from __future__ import annotations

import json
import os
import string
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from scribe import session_terms
from scribe.cli import app
from scribe.errors import InputValidationError
from scribe.session_sources import MAX_FETCH_BYTES, remote_source
from scribe.session_terms import (
    LOG_LIMIT,
    MERGED_CAP,
    TAIL_BYTES,
    SessionBlock,
    SessionRecord,
    TermCount,
    extract_terms,
    format_block,
    log_failure,
    parse_blocks,
    read_tail,
    repo_name,
    run_hook,
    state_dir,
    write_merged,
)
from scribe.vocab import parse_terms
from scribe.xai_stt import check_keyterms

runner = CliRunner()
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
_HUMAN = {"kind": "human"}

_RECORDS: list[dict[str, object]] = [
    {
        "type": "user",
        "timestamp": "2026-10-05T11:00:00Z",
        "entrypoint": "cli",
        "gitBranch": "feature/old-branch",
        "origin": {"kind": "human"},
        "message": {"role": "user", "content": "rename widget_factory in GadgetPanel"},
    },
    {
        "type": "user",
        "timestamp": "2026-10-05T11:00:00.500Z",
        "message": {"role": "user", "content": "a peer session sent unlabeled_term"},
    },
    {
        "type": "user",
        "isMeta": True,
        "timestamp": "2026-10-05T11:00:01Z",
        "message": {"role": "user", "content": "meta_only_term should not count"},
    },
    {
        "type": "assistant",
        "timestamp": "2026-10-05T11:00:02Z",
        "gitBranch": "feature/new-branch",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "thinking_only_term stays private"},
                {"type": "text", "text": "I'll edit **`src/pkg/sprocket_io.py`** next."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "Bash",
                    "input": {"command": "make lint-all --fix-only", "timeout": 5},
                },
            ],
        },
    },
    {
        "type": "user",
        "timestamp": "2026-10-05T11:00:03Z",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "result_only_term"}
            ],
        },
    },
    {
        "type": "user",
        "timestamp": "2026-10-05T11:00:04Z",
        "origin": {"kind": "task-notification"},
        "message": {"role": "user", "content": "notified_term is not the operator's"},
    },
    {
        "type": "user",
        "timestamp": "2026-10-05T11:00:05Z",
        "origin": {"kind": "human"},
        "message": {"role": "user", "content": "and spoken_term too"},
    },
    {"type": "attachment", "timestamp": "2026-10-05T11:00:06Z", "attachment": {"x": 1}},
]


def _transcript(tmp_path: Path, records: list[dict[str, object]] = _RECORDS) -> Path:
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _cli_transcript(tmp_path: Path) -> Path:
    return _transcript(tmp_path, [{"type": "system", "entrypoint": "cli"}])


def _fields(transcript: Path, cwd: Path, **fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "session_id": "11111111-aaaa-4bbb-8ccc-222222222222",
        "transcript_path": str(transcript),
        "cwd": str(cwd),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "now check DictationBox please",
        "permission_mode": "default",
    }
    return base | fields


def _payload(transcript: Path, cwd: Path, **fields: object) -> str:
    return json.dumps(_fields(transcript, cwd, **fields))


def _terms(record_path: Path) -> list[str]:
    record = SessionRecord.model_validate_json(record_path.read_text())
    return [term.term for term in record.terms]


def test_state_dir_prefers_xdg_state_home(tmp_path: Path) -> None:
    assert state_dir({"XDG_STATE_HOME": str(tmp_path)}, home=tmp_path / "h") == (
        tmp_path / "scribe"
    )
    assert state_dir({}, home=tmp_path / "h") == tmp_path / "h" / ".local" / "state" / "scribe"


def test_hook_reads_each_part_type_it_should_and_skips_the_rest(tmp_path: Path) -> None:
    repo = tmp_path / "toy-repo"
    (repo / ".git").mkdir(parents=True)
    cwd = repo / "sub"
    cwd.mkdir()
    state = tmp_path / "state"

    run_hook(
        _payload(_transcript(tmp_path), cwd).encode(),
        state=state,
        now=NOW,
        clock=lambda: NOW,
        host="box",
    )

    terms = _terms(state / "terms" / "sessions" / "11111111-aaaa-4bbb-8ccc-222222222222.json")
    for wanted in (
        "widget_factory",
        "GadgetPanel",
        "spoken_term",
        "sprocket_io.py",
        "src/pkg/sprocket_io.py",
        "lint-all",
        "DictationBox",
        "toy-repo",
        "feature/new-branch",
    ):
        assert wanted in terms
    for unwanted in (
        "meta_only_term",
        "thinking_only_term",
        "result_only_term",
        "feature/old-branch",
        "I'll",
        "--fix-only",
        "notified_term",
        "unlabeled_term",
    ):
        assert unwanted not in terms
    # The prompt is the newest source, so its term leads the session.
    assert terms[0] == "DictationBox"


def test_hook_logs_the_prompt_as_one_private_line(tmp_path: Path) -> None:
    state = tmp_path / "state"
    payload = _payload(_transcript(tmp_path), tmp_path)

    run_hook(payload.encode(), state=state, now=NOW, clock=lambda: NOW, host="box")
    run_hook(payload.encode(), state=state, now=NOW, clock=lambda: NOW, host="box")

    log = state / "prompts.jsonl"
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "ts": "2026-10-05T12:00:00+00:00",
        "host": "box",
        "session_id": "11111111-aaaa-4bbb-8ccc-222222222222",
        "cwd": str(tmp_path),
        "prompt": "now check DictationBox please",
    }
    assert log.stat().st_mode & 0o777 == 0o600


def test_other_events_log_no_prompt_but_still_update_terms(tmp_path: Path) -> None:
    state = tmp_path / "state"
    payload = _fields(_transcript(tmp_path), tmp_path)
    del payload["prompt"]
    payload["hook_event_name"] = "SessionStart"

    run_hook(json.dumps(payload).encode(), state=state, now=NOW, clock=lambda: NOW, host="box")

    assert not (state / "prompts.jsonl").exists()
    assert "GadgetPanel" in (state / "terms" / "current.txt").read_text().split("\n")


def test_session_id_that_is_not_a_plain_name_is_refused(tmp_path: Path) -> None:
    state = tmp_path / "state"
    payload = _payload(_transcript(tmp_path), tmp_path, session_id="../escape")

    run_hook(payload.encode(), state=state, now=NOW, clock=lambda: NOW, host="box")

    assert not (state / "terms" / "sessions").exists()
    assert not (state / "escape.json").exists()
    assert "session_id" in (state / "terms" / "hook.log").read_text()


def test_hook_command_prints_nothing_and_exits_zero_even_on_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    garbage = runner.invoke(app, ["terms", "hook"], input="not json {")
    good = runner.invoke(app, ["terms", "hook"], input=_payload(_transcript(tmp_path), tmp_path))

    assert garbage.exit_code == 0
    assert garbage.stdout == ""
    assert good.exit_code == 0
    assert good.stdout == ""
    log = (tmp_path / "scribe" / "terms" / "hook.log").read_text().splitlines()
    assert len(log) == 1
    assert (tmp_path / "scribe" / "prompts.jsonl").exists()


def test_missing_transcript_is_logged_and_the_prompt_still_counts(tmp_path: Path) -> None:
    state = tmp_path / "state"
    payload = _payload(tmp_path / "gone.jsonl", tmp_path)

    run_hook(payload.encode(), state=state, now=NOW, clock=lambda: NOW, host="box")

    record_path = state / "terms" / "sessions" / "11111111-aaaa-4bbb-8ccc-222222222222.json"
    assert "DictationBox" in _terms(record_path)
    assert len((state / "terms" / "hook.log").read_text().splitlines()) == 1


def test_tail_starts_at_the_first_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "big.jsonl"
    filler = json.dumps({"type": "system", "pad": "x" * 1000}) + "\n"
    last = json.dumps({"type": "user", "message": {"content": "tail_marker_term"}}) + "\n"
    path.write_text(filler * (2 * TAIL_BYTES // len(filler)) + last)

    lines = read_tail(path)

    assert lines[-1] == last.rstrip("\n")
    assert all(json.loads(line) for line in lines)
    assert sum(len(line) + 1 for line in lines) <= TAIL_BYTES


def test_tail_keeps_a_line_that_starts_exactly_at_the_cut(tmp_path: Path) -> None:
    path = tmp_path / "exact.jsonl"
    head = "h" * 99 + "\n"
    body = "b" * (TAIL_BYTES - 1) + "\n"
    path.write_text(head + body)

    assert read_tail(path) == [body.rstrip("\n")]


def test_repo_name_walks_up_to_a_worktree_git_file(tmp_path: Path) -> None:
    tree = tmp_path / "unit-tree"
    (tree / "a" / "b").mkdir(parents=True)
    (tree / ".git").write_text("gitdir: /elsewhere\n")

    assert repo_name(tree / "a" / "b") == "unit-tree"
    assert repo_name(tmp_path / "loose") == "loose"


def test_extraction_trims_drops_and_splits_paths() -> None:
    text = (
        'Edit "modules/darwin/base.nix", then run `--keyterm` (see XaiStt). '
        "Commit 1454fb6 at https://example.com/a_b and 3.14, also kbx25; "
        "id 0b8a3c2e-1f4d-4e5a-9b6c-7d8e9f0a1b2c, ok_x, ctx-guard-100! "
        "words like Hello and don't and **bold** stay out, and XaiStt's loses its 's. "
        "/a/very/long/directory/path/that/runs/past/fifty/chars/leaf_file.txt"
    )

    assert extract_terms(text) == [
        "base.nix",
        "modules/darwin/base.nix",
        "XaiStt",
        "kbx25",
        "ok_x",
        "ctx-guard-100",
        "XaiStt",
        "leaf_file.txt",
    ]


_TOKEN = st.text(alphabet=st.characters(codec="utf-8", exclude_categories=("Cs",)), max_size=60)


@given(st.lists(_TOKEN, max_size=12).map(" ".join))
def test_every_extracted_term_is_verbatim_and_a_valid_keyterm(text: str) -> None:
    terms = extract_terms(text)

    for term in terms:
        assert term in text
    check_keyterms(terms[:100])


def _record(session: str, updated: datetime, terms: list[str]) -> SessionRecord:
    return SessionRecord(
        session_id=session,
        cwd="/w",
        updated=updated,
        ranked=updated,
        entrypoint="cli",
        terms=[TermCount(term=term, count=1, last_seen=updated) for term in terms],
    )


def _store(sessions: Path, record: SessionRecord) -> Path:
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{record.session_id}.json"
    path.write_text(record.model_dump_json())
    return path


def test_each_session_block_is_newest_first_and_capped(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("old", NOW - timedelta(hours=24, minutes=1), ["stale_term"]))
    _store(sessions, _record("mid", NOW - timedelta(minutes=10), ["mid_a", "shared_x"]))
    _store(
        sessions,
        _record("new", NOW - timedelta(minutes=1), ["shared_x", *[f"t_{n}" for n in range(70)]]),
    )

    write_merged(tmp_path / "terms", now=NOW)

    new, mid = _blocks(tmp_path / "terms")
    assert new.session_id == "new"
    assert new.terms[:3] == ("shared_x", "t_0", "t_1")
    assert len(new.terms) == MERGED_CAP
    assert (mid.session_id, mid.terms) == ("mid", ("mid_a", "shared_x"))
    check_keyterms(new.terms)


def _letters(n: int) -> str:
    return "".join(string.ascii_letters[int(digit)] for digit in str(n))


def test_the_merged_file_stays_within_what_a_remote_host_reads(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("fresh", NOW, ["fresh_term"]))
    # Each block holds 60 terms of 50 characters, so 400 of them pass 1 MiB.
    for n in range(400):
        terms = [f"Big{_letters(n)}Term{_letters(i)}".ljust(50, "x") for i in range(MERGED_CAP)]
        _store(sessions, _record(f"big-{n:03}", NOW - timedelta(hours=2, seconds=n), terms))
    _store(sessions, _record("small", NOW - timedelta(hours=3), ["small_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    text = (tmp_path / "terms" / "current.txt").read_text()
    assert len(text.encode()) <= MAX_FETCH_BYTES
    ids = [block.session_id for block in parse_blocks(text)]
    assert ids[0] == "fresh"
    assert ids[-1] == "small"
    assert len(ids) < 402
    remote = remote_source("box", runner=lambda _argv, _timeout: text)
    remote.refresh(lambda: NOW)
    assert "fresh_term" in {term for block in remote.live(NOW) for term in block.terms}


def test_merged_file_ranks_within_a_session_by_last_seen_then_count(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    earlier = NOW - timedelta(minutes=5)
    record = SessionRecord(
        session_id="s",
        cwd="/w",
        updated=NOW,
        ranked=NOW,
        entrypoint="cli",
        terms=[
            TermCount(term="early_many", count=9, last_seen=earlier),
            TermCount(term="late_few", count=1, last_seen=NOW),
            TermCount(term="late_many", count=4, last_seen=NOW),
        ],
    )
    _store(sessions, record)
    _store(sessions, _record("other", NOW - timedelta(minutes=20), ["other_t"]))

    write_merged(tmp_path / "terms", now=NOW)

    assert _current(tmp_path / "terms") == ["late_many", "late_few", "early_many", "other_t"]


def test_records_older_than_a_week_are_deleted(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    old = _store(sessions, _record("old", NOW - timedelta(days=8), ["x_old"]))
    week_ago = (NOW - timedelta(days=7, seconds=1)).timestamp()
    os.utime(old, (week_ago, week_ago))
    kept = _store(sessions, _record("kept", NOW - timedelta(days=1), ["x_kept"]))

    write_merged(tmp_path / "terms", now=NOW)

    assert not old.exists()
    assert kept.exists()


def test_shape_drift_and_a_corrupt_record_are_logged_not_fatal(tmp_path: Path) -> None:
    state = tmp_path / "state"
    records: list[dict[str, object]] = [
        {"type": "user", "origin": _HUMAN, "message": {"content": 7}},
        {
            "type": "user",
            "entrypoint": "cli",
            "origin": _HUMAN,
            "message": {"content": "keep shaped_term"},
        },
    ]
    transcript = _transcript(tmp_path, records)
    with transcript.open("a") as handle:
        handle.write('{"type": "user", "mess')
    (state / "terms" / "sessions").mkdir(parents=True)
    (state / "terms" / "sessions" / "broken.json").write_text("{}")

    run_hook(
        _payload(transcript, tmp_path).encode(), state=state, now=NOW, clock=lambda: NOW, host="box"
    )

    log = (state / "terms" / "hook.log").read_text().splitlines()
    assert len(log) == 2
    assert "1 transcript records" in log[0]
    assert "broken.json" in log[1]
    assert "shaped_term" in (state / "terms" / "current.txt").read_text().split("\n")


def _event(
    state: Path, session: str, event: str, *, transcript: Path | None = None, **extra: object
) -> None:
    payload: dict[str, object] = {
        "session_id": session,
        "hook_event_name": event,
        "cwd": "/x",
        "transcript_path": str(transcript) if transcript else None,
    }
    run_hook(
        json.dumps(payload | extra).encode(), state=state, now=NOW, clock=lambda: NOW, host="box"
    )


def _blocks(terms_dir: Path) -> list[SessionBlock]:
    return parse_blocks((terms_dir / "current.txt").read_text())


def _current(terms_dir: Path) -> list[str]:
    return [term for block in _blocks(terms_dir) for term in block.terms]


def test_each_session_block_says_when_that_session_leaves_the_window(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("older", NOW - timedelta(minutes=10), ["older_term"]))
    _store(sessions, _record("newer", NOW - timedelta(minutes=1), ["newer_term"]))
    _store(sessions, _record("gone", NOW - timedelta(hours=24, minutes=1), ["gone_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    assert (tmp_path / "terms" / "current.txt").read_text().splitlines() == [
        "# session newer ranked 2026-10-05T11:59:00+00:00 expires 2026-10-05T12:29:00+00:00"
        ' {"cwd": "/w", "titles": []}',
        "newer_term",
        "# session older ranked 2026-10-05T11:50:00+00:00 expires 2026-10-05T12:20:00+00:00"
        ' {"cwd": "/w", "titles": []}',
        "older_term",
    ]
    assert parse_blocks((tmp_path / "terms" / "current.txt").read_text()) == [
        SessionBlock(
            "newer",
            NOW - timedelta(minutes=1),
            NOW + timedelta(minutes=29),
            ("newer_term",),
            cwd="/w",
            titles=(),
        ),
        SessionBlock(
            "older",
            NOW - timedelta(minutes=10),
            NOW + timedelta(minutes=20),
            ("older_term",),
            cwd="/w",
            titles=(),
        ),
    ]


def test_no_session_in_the_window_is_an_empty_file(tmp_path: Path) -> None:
    write_merged(tmp_path / "terms", now=NOW)

    assert (tmp_path / "terms" / "current.txt").read_text() == ""
    assert _blocks(tmp_path / "terms") == []


def test_every_terms_file_reader_takes_each_block_header_as_a_comment(tmp_path: Path) -> None:
    _store(tmp_path / "terms" / "sessions", _record("s", NOW, ["one_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    vocab = parse_terms((tmp_path / "terms" / "current.txt").read_text(), where="current.txt")
    assert vocab.terms == ("one_term",)


_HEAD = "# session s ranked 2026-10-05T12:00:00+00:00 expires 2026-10-05T12:30:00+00:00"


@pytest.mark.parametrize(
    "text",
    [
        "# expires 2026-10-05T12:00:00+00:00\nold_term\n",
        "plain_term\n",
        "# session s ranked 2026-10-05T12:00:00 expires 2026-10-05T12:30:00+00:00"
        ' {"cwd": "/w", "titles": []}\n',
        '# session s ranked 2026-10-05T12:00:00+00:00 expires soon {"cwd": "/w", "titles": []}\n',
        "# session s/../x ranked 2026-10-05T12:00:00+00:00 expires 2026-10-05T12:30:00+00:00"
        ' {"cwd": "/w", "titles": []}\n',
        _HEAD + "\n",
        _HEAD + ' {"cwd": "/w"}\n',
        _HEAD + ' {"cwd": "/w", "titles": [1]}\n',
        _HEAD + ' {"cwd": "/w", "titles": []} trailing\n',
    ],
)
def test_text_that_is_not_session_blocks_is_refused(text: str) -> None:
    with pytest.raises(InputValidationError):
        parse_blocks(text)


def test_blank_lines_and_other_comments_are_skipped() -> None:
    text = (
        "# written by a test\n\n"
        "# session s ranked 2026-10-05T12:00:00+00:00 expires 2026-10-05T12:30:00+00:00"
        ' {"cwd": "/w", "titles": []}\n'
        "  a_term  \n\n# aside\nb_term\n"
    )

    (block,) = parse_blocks(text)

    assert block.terms == ("a_term", "b_term")


def test_the_reach_alone_drops_a_session_ranked_24_h_and_a_minute_ago(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("old", NOW - timedelta(hours=24, minutes=1), ["stale_term"]))
    _store(sessions, _record("new", NOW - timedelta(minutes=1), ["fresh_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    assert _current(tmp_path / "terms") == ["fresh_term"]


def test_a_session_ranked_31_minutes_ago_keeps_its_block_with_its_own_past_expiry(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("expired", NOW - timedelta(minutes=31), ["expired_term"]))
    _store(sessions, _record("new", NOW - timedelta(minutes=1), ["fresh_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    new, expired = _blocks(tmp_path / "terms")
    assert (new.session_id, expired.session_id) == ("new", "expired")
    assert expired.expires == NOW - timedelta(minutes=1)
    assert expired.terms == ("expired_term",)


def test_a_session_file_updated_within_24_h_is_parsed(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    path = _store(sessions, _record("day_old", NOW - timedelta(hours=23), ["day_term"]))
    old = (NOW - timedelta(hours=23)).timestamp()
    os.utime(path, (old, old))

    write_merged(tmp_path / "terms", now=NOW)

    assert _current(tmp_path / "terms") == ["day_term"]


def test_no_merged_line_reads_as_an_alias_or_a_comment(tmp_path: Path) -> None:
    prompt = "then rows.map(id=>id.name) and #fix_it, curry f=>g=>h"
    _event(tmp_path, "s1", "UserPromptSubmit", transcript=_cli_transcript(tmp_path), prompt=prompt)
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("stored", NOW, ["heard=>written", "#comment_x", "kept_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    lines = _current(tmp_path / "terms")
    assert "kept_term" in lines
    assert [line for line in lines if "=>" in line or line.startswith("#")] == []


def test_a_busier_session_does_not_shorten_another_sessions_block(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    mine = [f"mine_{n}" for n in range(100)]
    _store(sessions, _record("dictated", NOW - timedelta(minutes=2), mine))
    _store(sessions, _record("worker", NOW, [f"worker_{n}" for n in range(100)]))

    write_merged(tmp_path / "terms", now=NOW)

    worker, dictated = _blocks(tmp_path / "terms")
    assert worker.terms[:2] == ("worker_0", "worker_1")
    assert dictated.terms == tuple(mine[:MERGED_CAP])


def test_stop_updates_terms_but_not_the_rank_time(tmp_path: Path) -> None:
    transcript = _transcript(tmp_path)
    _event(tmp_path, "s1", "SessionStart", transcript=transcript)
    first = SessionRecord.model_validate_json(
        (tmp_path / "terms" / "sessions" / "s1.json").read_text()
    )
    later = NOW + timedelta(minutes=20)
    payload = {"session_id": "s1", "hook_event_name": "Stop", "cwd": "/x", "prompt": None}
    run_hook(
        json.dumps(payload).encode(), state=tmp_path, now=later, clock=lambda: later, host="box"
    )

    record = SessionRecord.model_validate_json(
        (tmp_path / "terms" / "sessions" / "s1.json").read_text()
    )
    assert first.ranked == NOW
    assert record.ranked == NOW
    assert record.updated == later


def test_a_delayed_older_event_changes_nothing_in_the_record(tmp_path: Path) -> None:
    _event(tmp_path, "s1", "UserPromptSubmit", prompt="see newer_term")
    record_path = tmp_path / "terms" / "sessions" / "s1.json"
    before = record_path.read_text()
    older = {"session_id": "s1", "hook_event_name": "SessionStart", "cwd": "/x"}
    late_prompt = older | {"hook_event_name": "UserPromptSubmit", "prompt": "see older_term"}

    earlier = NOW - timedelta(seconds=2)
    for payload in (older, late_prompt):
        run_hook(
            json.dumps(payload).encode(),
            state=tmp_path,
            now=earlier,
            clock=lambda: earlier,
            host="box",
        )

    assert record_path.read_text() == before
    logged = [
        json.loads(line)["prompt"] for line in (tmp_path / "prompts.jsonl").read_text().splitlines()
    ]
    assert logged == ["see newer_term", "see older_term"]


def test_a_delayed_hook_from_another_session_does_not_restore_expired_terms(
    tmp_path: Path,
) -> None:
    transcript = _cli_transcript(tmp_path)

    def hook(session: str, event_at: datetime, processed_at: datetime, prompt: str) -> None:
        payload = {
            "session_id": session,
            "hook_event_name": "UserPromptSubmit",
            "cwd": "/x",
            "transcript_path": str(transcript),
            "prompt": prompt,
        }
        run_hook(
            json.dumps(payload).encode(),
            state=tmp_path,
            now=event_at,
            clock=lambda: processed_at,
            host="box",
        )

    hook("a", NOW, NOW, "see alpha_term")
    expired = NOW + timedelta(hours=24, seconds=1)
    hook("c", expired, expired, "see charlie_term")
    # Taken before c's hook, processed after it: a's session is out of reach by then.
    hook("b", expired - timedelta(seconds=2), expired + timedelta(seconds=1), "see bravo_term")

    assert "alpha_term" not in _current(tmp_path / "terms")
    assert "bravo_term" in _current(tmp_path / "terms")
    record = SessionRecord.model_validate_json(
        (tmp_path / "terms" / "sessions" / "b.json").read_text()
    )
    assert record.updated == expired - timedelta(seconds=2)


def test_terms_carry_over_when_a_big_tool_result_fills_the_tail(tmp_path: Path) -> None:
    records: list[dict[str, object]] = [
        {
            "type": "user",
            "timestamp": "2026-10-05T11:50:00Z",
            "origin": _HUMAN,
            "message": {"content": "fix kept_term"},
        },
    ]
    transcript = _transcript(tmp_path, records)
    _event(tmp_path, "s1", "SessionStart", transcript=transcript)
    flood = {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "content": "x" * TAIL_BYTES}]},
    }
    done = {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}}
    with transcript.open("a") as handle:
        handle.write(json.dumps(flood) + "\n" + json.dumps(done) + "\n")

    later = NOW + timedelta(minutes=1)
    payload = {"session_id": "s1", "hook_event_name": "Stop", "cwd": "/x"}
    run_hook(
        json.dumps(payload | {"transcript_path": str(transcript)}).encode(),
        state=tmp_path,
        now=later,
        clock=lambda: later,
        host="box",
    )

    assert _terms(tmp_path / "terms" / "sessions" / "s1.json") == ["kept_term"]


def test_carried_terms_keep_the_higher_count_and_later_sighting(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    earlier = NOW - timedelta(minutes=5)
    stored = SessionRecord(
        session_id="s1",
        cwd="/x",
        updated=earlier,
        ranked=earlier,
        terms=[
            TermCount(term="many_seen", count=5, last_seen=earlier),
            TermCount(term="seen_again", count=1, last_seen=earlier),
            TermCount(term="FULLY", count=9, last_seen=earlier),
            TermCount(term="--force", count=9, last_seen=earlier),
            *(
                TermCount(term=f"old_{n}", count=1, last_seen=earlier - timedelta(seconds=n))
                for n in range(100)
            ),
        ],
    )
    _store(sessions, stored)

    _event(tmp_path, "s1", "UserPromptSubmit", prompt="many_seen and seen_again")

    record = SessionRecord.model_validate_json((sessions / "s1.json").read_text())
    terms = {t.term: (t.count, t.last_seen) for t in record.terms}
    assert terms["many_seen"] == (5, NOW)
    assert terms["seen_again"] == (1, NOW)
    assert "FULLY" not in terms
    assert "--force" not in terms
    assert len(record.terms) == 100
    assert "old_97" in terms
    assert "old_98" not in terms


def test_a_session_seen_only_at_stop_is_not_merged(tmp_path: Path) -> None:
    _event(tmp_path, "s1", "Stop", transcript=_transcript(tmp_path))

    assert _current(tmp_path / "terms") == []


def test_a_headless_session_is_never_merged(tmp_path: Path) -> None:
    headless = [{**record, "entrypoint": "sdk-cli"} for record in _RECORDS]
    worker = tmp_path / "worker"
    worker.mkdir()
    transcript = _transcript(worker, headless)
    _event(tmp_path, "w1", "UserPromptSubmit", transcript=transcript, prompt="worker_only_term")
    interactive = [{**record, "entrypoint": "cli"} for record in _RECORDS]
    _event(tmp_path, "c1", "SessionStart", transcript=_transcript(tmp_path, interactive))
    _event(tmp_path, "n1", "UserPromptSubmit", prompt="unknown_entry_term")

    lines = _current(tmp_path / "terms")
    assert "GadgetPanel" in lines
    assert "unknown_entry_term" not in lines
    assert "worker_only_term" not in lines
    worker_record = SessionRecord.model_validate_json(
        (tmp_path / "terms" / "sessions" / "w1.json").read_text()
    )
    assert worker_record.entrypoint == "sdk-cli"


def test_a_session_is_merged_only_once_its_transcript_shows_cli(tmp_path: Path) -> None:
    missing = tmp_path / "not-yet.jsonl"
    _event(tmp_path, "w1", "SessionStart", transcript=missing, cwd="/worker-repo")

    assert _current(tmp_path / "terms") == []

    missing.write_text(json.dumps({"type": "system", "entrypoint": "cli"}) + "\n")
    _event(tmp_path, "w1", "UserPromptSubmit", transcript=missing, prompt="see cli_term")
    _event(tmp_path, "w1", "UserPromptSubmit", prompt="see blind_term")

    assert "blind_term" in _current(tmp_path / "terms")


def test_code_fragments_flags_caps_and_digit_led_tokens_are_dropped() -> None:
    text = (
        "then call `run_hook()` and print(len(s)) with D1=$(grep x) 2>/dev/null "
        'EXIT=$? average|per $i.v2.log \\"seventeen\\ go(now '
        "git push --force --help -rn -m3 VERDICT: CONFIRMED, RESULT is FULLY MET "
        "8-bit ~410K 5-minute 13-fix keep_me ~/notes.md"
    )

    assert extract_terms(text) == ["run_hook", "keep_me", "notes.md", "~/notes.md"]


def test_a_capitalized_basename_needs_no_shape_but_still_the_drop_rules() -> None:
    assert extract_terms("edit src/Makefile and src/README then a/b/x1 /dev/null") == [
        "Makefile",
        "src/Makefile",
        "src/README",
        "a/b/x1",
        "/dev/null",
    ]


def test_plain_repo_and_branch_words_are_not_terms(tmp_path: Path) -> None:
    cwd = tmp_path / "My Drive"
    cwd.mkdir()
    record: dict[str, object] = {
        "type": "system",
        "gitBranch": "main",
        "timestamp": "2026-10-05T11:59:00Z",
    }
    _event(
        tmp_path / "state",
        "s1",
        "UserPromptSubmit",
        transcript=_transcript(tmp_path, [record]),
        cwd=str(cwd),
        prompt="ok",
    )

    assert _current(tmp_path / "state" / "terms") == []


def test_credential_shaped_tokens_are_not_terms() -> None:
    random36 = (string.ascii_letters + string.digits)[:36]
    text = (
        f"export GH_TOKEN=ghp_{random36} ghp_{random36} github_pat_{random36} "
        f"sk-{random36} xai-{random36} AKIAIOSFODNN7EXAMPLE xoxb-{random36} "
        "aB3dE5gH7jK9mN1pQ3sT5 but xai-stt and task-notification-producer-x stay"
    )

    assert extract_terms(text) == ["xai-stt", "task-notification-producer-x"]


def test_a_path_holding_a_credential_contributes_nothing() -> None:
    # AWS's documented example secret access key: its slashes made it read as a path.
    text = (
        "use wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY and src/AKIAIOSFODNN7EXAMPLE/run.py "
        "or ~/keys/aB3dE5gH7jK9mN1pQ3sT5/run.py"
    )

    assert extract_terms(text + " but src/pkg/sprocket_io.py stays") == [
        "sprocket_io.py",
        "src/pkg/sprocket_io.py",
    ]


def test_log_failure_never_raises(tmp_path: Path) -> None:
    log_failure(tmp_path, NOW, "file \udcff.json unreadable")

    assert "\\udcff" in (tmp_path / "hook.log").read_text()


def test_a_slower_concurrent_merge_does_not_drop_a_newer_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = session_terms._replace_atomically  # pyright: ignore[reportPrivateUsage]  # test seam
    racer: list[threading.Thread] = []

    def interleave(path: Path, text: str) -> None:
        # A's merge has read the sessions; B's hook gets every chance to finish before A writes.
        if path.name == "current.txt" and not racer:
            racer.append(
                threading.Thread(
                    target=_event,
                    args=(tmp_path, "b", "UserPromptSubmit"),
                    kwargs={"transcript": cli, "prompt": "see b_term"},
                )
            )
            racer[0].start()
            time.sleep(0.3)
        real(path, text)

    cli = _cli_transcript(tmp_path)
    monkeypatch.setattr(session_terms, "_replace_atomically", interleave)
    _event(tmp_path, "a", "UserPromptSubmit", transcript=cli, prompt="see a_term")
    racer[0].join()

    assert sorted(_current(tmp_path / "terms")) == ["a_term", "b_term"]


def test_a_stale_session_file_is_not_parsed(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    fresh_claim = _store(sessions, _record("stale", NOW, ["claims_fresh"]))
    broken = sessions / "broken.json"
    broken.write_text("{")
    old = (NOW - timedelta(hours=24, minutes=1)).timestamp()
    for path in (fresh_claim, broken):
        os.utime(path, (old, old))

    write_merged(tmp_path / "terms", now=NOW)

    assert _current(tmp_path / "terms") == []
    assert not (tmp_path / "terms" / "hook.log").exists()


def test_a_prompt_log_failure_still_updates_the_terms(tmp_path: Path) -> None:
    tmp_path.joinpath("prompts.jsonl").write_text("")
    tmp_path.joinpath("prompts.jsonl").chmod(0o400)

    _event(
        tmp_path,
        "s1",
        "UserPromptSubmit",
        transcript=_cli_transcript(tmp_path),
        prompt="see kept_term",
    )

    assert _current(tmp_path / "terms") == ["kept_term"]
    assert "prompt log" in (tmp_path / "terms" / "hook.log").read_text()


def test_a_merge_filters_stored_terms_that_are_not_keyterms(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("s", NOW, ["x_" + "y" * 49, "two\nlines", "fine_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    assert _current(tmp_path / "terms") == ["fine_term"]


def test_two_character_tokens_are_dropped() -> None:
    assert extract_terms("x1 a_ ab_ X1y") == ["ab_", "X1y"]


def test_last_seen_is_the_latest_sighting(tmp_path: Path) -> None:
    records: list[dict[str, object]] = [
        {
            "type": "user",
            "timestamp": "2026-10-05T11:00:00Z",
            "origin": _HUMAN,
            "message": {"content": "twice_term"},
        },
        {
            "type": "user",
            "timestamp": "2026-10-05T11:30:00Z",
            "origin": _HUMAN,
            "message": {"content": "twice_term"},
        },
    ]
    _event(tmp_path, "s1", "SessionStart", transcript=_transcript(tmp_path, records))

    record = SessionRecord.model_validate_json(
        (tmp_path / "terms" / "sessions" / "s1.json").read_text()
    )
    assert [(t.term, t.count, t.last_seen) for t in record.terms] == [
        ("twice_term", 2, datetime(2026, 10, 5, 11, 30, tzinfo=UTC))
    ]


def test_hook_command_survives_a_state_directory_it_cannot_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    read_only = tmp_path / "xdg"
    (read_only / "scribe").mkdir(parents=True)
    (read_only / "scribe").chmod(0o500)
    payload = _payload(_transcript(tmp_path), tmp_path)

    outcomes: list[tuple[int, str]] = []
    for state_home in (blocker, read_only):
        monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
        result = runner.invoke(app, ["terms", "hook"], input=payload)
        outcomes.append((result.exit_code, result.stdout))
    (read_only / "scribe").chmod(0o700)

    assert outcomes == [(0, ""), (0, "")]


def test_tail_reads_no_more_than_the_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "big.jsonl"
    path.write_bytes(b"x" * (4 * TAIL_BYTES) + b"\n")
    reads: list[int] = []
    real_open = Path.open

    class Counting:
        def __init__(self, handle: IO[bytes]) -> None:
            self.handle = handle

        def __enter__(self) -> Counting:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.handle.close()

        def seek(self, offset: int, whence: int = 0) -> int:
            return self.handle.seek(offset, whence)

        def read(self, size: int = -1) -> bytes:
            data = self.handle.read(size)
            reads.append(len(data))
            return data

    def counted(self: Path, mode: str = "r") -> Counting:
        assert mode == "rb"
        return Counting(real_open(self, "rb"))

    monkeypatch.setattr(Path, "open", counted)
    read_tail(path)

    assert sum(reads) <= TAIL_BYTES + 1


def test_hook_log_never_holds_the_raw_input(tmp_path: Path) -> None:
    marker = "PRIVATE_PROMPT_MARKER"
    bad_id = json.dumps(
        {
            "session_id": "../x",
            "hook_event_name": "UserPromptSubmit",
            "cwd": marker,
            "prompt": marker,
        }
    )
    for raw in (bad_id.encode(), f"{marker} {{".encode(), json.dumps({"prompt": marker}).encode()):
        run_hook(raw, state=tmp_path, now=NOW, clock=lambda: NOW, host="box")

    log = (tmp_path / "terms" / "hook.log").read_text()
    assert len(log.splitlines()) == 3
    assert marker not in log


def test_a_missing_transcript_at_session_start_is_not_a_failure(tmp_path: Path) -> None:
    _event(tmp_path, "s1", "SessionStart", transcript=tmp_path / "not-yet.jsonl")

    assert not (tmp_path / "terms" / "hook.log").exists()


def test_hook_log_starts_afresh_past_its_limit(tmp_path: Path) -> None:
    log = tmp_path / "hook.log"
    log.write_text("x" * (LOG_LIMIT + 1))

    log_failure(tmp_path, NOW, "one more")

    assert log.read_text() == "2026-10-05T12:00:00+00:00 one more\n"


_SAFE = st.characters(exclude_categories=["Cs"])


@given(
    session=st.from_regex(r"[A-Za-z0-9_-]{1,128}", fullmatch=True),
    minutes=st.integers(min_value=-10_000, max_value=10_000),
    cwd=st.text(alphabet=_SAFE),
    titles=st.lists(st.text(alphabet=_SAFE), max_size=5),
    terms=st.lists(st.from_regex(r"[a-z_]{3,12}", fullmatch=True), max_size=5),
)
def test_a_block_header_round_trips_any_title_and_cwd(
    session: str, minutes: int, cwd: str, titles: list[str], terms: list[str]
) -> None:
    ranked = NOW + timedelta(minutes=minutes)
    block = SessionBlock(
        session, ranked, ranked + timedelta(minutes=30), tuple(terms), cwd=cwd, titles=tuple(titles)
    )

    text = format_block(block)

    assert text.count("\n") == 1 + len(terms)
    assert text.startswith("# session ")
    assert parse_blocks(text) == [block]
    assert parse_terms(text, where="current.txt").terms == tuple(dict.fromkeys(terms))


def _titled(tmp_path: Path, *titles: tuple[str, str]) -> Path:
    records: list[dict[str, object]] = [{"type": "system", "entrypoint": "cli"}]
    for kind, title in titles:
        key = "customTitle" if kind == "custom-title" else "aiTitle"
        records.append({"type": kind, key: title, "sessionId": "s1"})
    return _transcript(tmp_path, records)


def _titles(tmp_path: Path) -> tuple[str, ...]:
    (block,) = _blocks(tmp_path / "terms")
    return block.titles


def test_a_custom_title_wins_over_a_later_ai_title(tmp_path: Path) -> None:
    transcript = _titled(
        tmp_path,
        ("ai-title", "First topic"),
        ("custom-title", 'My # "named" tab \u00e9'),
        ("ai-title", "Later topic"),
    )

    _event(tmp_path, "s1", "UserPromptSubmit", transcript=transcript, prompt="see a_term")

    assert _titles(tmp_path) == ('My # "named" tab \u00e9', "Later topic", "First topic")
    (block,) = _blocks(tmp_path / "terms")
    assert block.cwd == "/x"


def test_titles_carry_over_when_the_tail_holds_none(tmp_path: Path) -> None:
    titled = _titled(tmp_path, ("ai-title", "Old topic"), ("custom-title", "Renamed"))
    _event(tmp_path, "s1", "UserPromptSubmit", transcript=titled, prompt="see a_term")
    later = tmp_path / "later"
    later.mkdir()

    _event(tmp_path, "s1", "UserPromptSubmit", transcript=_cli_transcript(later), prompt="b_term")

    assert _titles(tmp_path) == ("Renamed", "Old topic")


def test_only_the_five_newest_distinct_titles_are_kept(tmp_path: Path) -> None:
    names = ["t1", "t2", "t3", "t2", "t4", "t5", "t6", "t6"]
    transcript = _titled(tmp_path, *(("ai-title", name) for name in names))

    _event(tmp_path, "s1", "UserPromptSubmit", transcript=transcript, prompt="see a_term")

    assert _titles(tmp_path) == ("t6", "t5", "t4", "t2", "t3")
