"""The terms hook: a session's vocabulary and the prompt log, from hook JSON on stdin."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from scribe.cli import app
from scribe.session_terms import (
    MERGED_CAP,
    TAIL_BYTES,
    SessionRecord,
    TermCount,
    extract_terms,
    read_tail,
    repo_name,
    run_hook,
    state_dir,
    write_merged,
)
from scribe.xai_stt import check_keyterms

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)

_RECORDS: list[dict[str, object]] = [
    {
        "type": "user",
        "timestamp": "2026-10-05T11:00:00Z",
        "gitBranch": "feature/old-branch",
        "message": {"role": "user", "content": "rename widget_factory in GadgetPanel"},
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
    {"type": "attachment", "timestamp": "2026-10-05T11:00:04Z", "attachment": {"x": 1}},
]


def _transcript(tmp_path: Path, records: list[dict[str, object]] = _RECORDS) -> Path:
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


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

    run_hook(_payload(_transcript(tmp_path), cwd).encode(), state=state, now=NOW, host="box")

    terms = _terms(state / "terms" / "sessions" / "11111111-aaaa-4bbb-8ccc-222222222222.json")
    for wanted in (
        "widget_factory",
        "GadgetPanel",
        "sprocket_io.py",
        "src/pkg/sprocket_io.py",
        "lint-all",
        "--fix-only",
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
    ):
        assert unwanted not in terms
    # The prompt is the newest source, so its term leads the session.
    assert terms[0] == "DictationBox"


def test_hook_logs_the_prompt_as_one_private_line(tmp_path: Path) -> None:
    state = tmp_path / "state"
    payload = _payload(_transcript(tmp_path), tmp_path)

    run_hook(payload.encode(), state=state, now=NOW, host="box")
    run_hook(payload.encode(), state=state, now=NOW, host="box")

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

    run_hook(json.dumps(payload).encode(), state=state, now=NOW, host="box")

    assert not (state / "prompts.jsonl").exists()
    assert "GadgetPanel" in (state / "terms" / "current.txt").read_text().split("\n")


def test_session_id_that_is_not_a_plain_name_is_refused(tmp_path: Path) -> None:
    state = tmp_path / "state"
    payload = _payload(_transcript(tmp_path), tmp_path, session_id="../escape")

    run_hook(payload.encode(), state=state, now=NOW, host="box")

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

    run_hook(payload.encode(), state=state, now=NOW, host="box")

    assert "DictationBox" in (state / "terms" / "current.txt").read_text().split("\n")
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
        "Commit 1454fb6 at https://example.com/a_b and 3.14, also akms25; "
        "id 0b8a3c2e-1f4d-4e5a-9b6c-7d8e9f0a1b2c, ok_x, ctx-guard-100! "
        "words like Hello and don't and **bold** stay out, and XaiStt's loses its 's. "
        "/a/very/long/directory/path/that/runs/past/fifty/chars/leaf_file.txt"
    )

    assert extract_terms(text) == [
        "base.nix",
        "modules/darwin/base.nix",
        "--keyterm",
        "XaiStt",
        "akms25",
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
        terms=[TermCount(term=term, count=1, last_seen=updated) for term in terms],
    )


def _store(sessions: Path, record: SessionRecord) -> Path:
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{record.session_id}.json"
    path.write_text(record.model_dump_json())
    return path


def test_merged_file_keeps_the_last_half_hour_newest_first_capped(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("old", NOW - timedelta(minutes=31), ["stale_term"]))
    _store(sessions, _record("mid", NOW - timedelta(minutes=10), ["mid_a", "shared_x"]))
    _store(
        sessions,
        _record("new", NOW - timedelta(minutes=1), ["shared_x", *[f"t_{n}" for n in range(70)]]),
    )

    write_merged(tmp_path / "terms", now=NOW)

    lines = (tmp_path / "terms" / "current.txt").read_text().splitlines()
    assert len(lines) == MERGED_CAP
    assert lines[:2] == ["shared_x", "t_0"]
    assert "stale_term" not in lines
    assert "mid_a" not in lines  # the newest session alone fills the cap
    check_keyterms(lines)


def test_merged_file_ranks_within_a_session_by_last_seen_then_count(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    earlier = NOW - timedelta(minutes=5)
    record = SessionRecord(
        session_id="s",
        cwd="/w",
        updated=NOW,
        terms=[
            TermCount(term="early_many", count=9, last_seen=earlier),
            TermCount(term="late_few", count=1, last_seen=NOW),
            TermCount(term="late_many", count=4, last_seen=NOW),
        ],
    )
    _store(sessions, record)
    _store(sessions, _record("other", NOW - timedelta(minutes=20), ["other_t"]))

    write_merged(tmp_path / "terms", now=NOW)

    assert (tmp_path / "terms" / "current.txt").read_text().splitlines() == [
        "late_many",
        "late_few",
        "early_many",
        "other_t",
    ]


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
        {"type": "user", "message": {"content": 7}},
        {"type": "user", "message": {"content": "keep shaped_term"}},
    ]
    transcript = _transcript(tmp_path, records)
    with transcript.open("a") as handle:
        handle.write('{"type": "user", "mess')
    (state / "terms" / "sessions").mkdir(parents=True)
    (state / "terms" / "sessions" / "broken.json").write_text("{}")

    run_hook(_payload(transcript, tmp_path).encode(), state=state, now=NOW, host="box")

    log = (state / "terms" / "hook.log").read_text().splitlines()
    assert len(log) == 2
    assert "1 transcript records" in log[0]
    assert "broken.json" in log[1]
    assert "shaped_term" in (state / "terms" / "current.txt").read_text().split("\n")
