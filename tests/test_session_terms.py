"""The terms hook: a session's vocabulary and the prompt log, from hook JSON on stdin."""

from __future__ import annotations

import json
import os
import string
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING

from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from scribe import session_terms
from scribe.cli import app
from scribe.session_terms import (
    LOG_LIMIT,
    MERGED_CAP,
    TAIL_BYTES,
    SessionRecord,
    TermCount,
    extract_terms,
    log_failure,
    read_tail,
    repo_name,
    run_hook,
    state_dir,
    write_merged,
)
from scribe.xai_stt import check_keyterms

if TYPE_CHECKING:
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
        ranked=updated,
        terms=[TermCount(term=term, count=1, last_seen=updated) for term in terms],
    )


def _store(sessions: Path, record: SessionRecord) -> Path:
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{record.session_id}.json"
    path.write_text(record.model_dump_json())
    return path


def test_merged_file_fills_round_robin_newest_first_capped(tmp_path: Path) -> None:
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
    assert lines[:4] == ["shared_x", "mid_a", "t_0", "t_1"]
    assert "stale_term" not in lines
    check_keyterms(lines)


def test_merged_file_ranks_within_a_session_by_last_seen_then_count(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    earlier = NOW - timedelta(minutes=5)
    record = SessionRecord(
        session_id="s",
        cwd="/w",
        updated=NOW,
        ranked=NOW,
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
        "other_t",
        "late_few",
        "early_many",
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


def _event(
    state: Path, session: str, event: str, *, transcript: Path | None = None, **extra: object
) -> None:
    payload: dict[str, object] = {
        "session_id": session,
        "hook_event_name": event,
        "cwd": "/x",
        "transcript_path": str(transcript) if transcript else None,
    }
    run_hook(json.dumps(payload | extra).encode(), state=state, now=NOW, host="box")


def _current(terms_dir: Path) -> list[str]:
    return (terms_dir / "current.txt").read_text().splitlines()


def test_window_alone_drops_a_31_minute_old_session(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("old", NOW - timedelta(minutes=31), ["stale_term"]))
    _store(sessions, _record("new", NOW - timedelta(minutes=1), ["fresh_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    assert _current(tmp_path / "terms") == ["fresh_term"]


def test_no_merged_line_reads_as_an_alias_or_a_comment(tmp_path: Path) -> None:
    prompt = "then rows.map(id=>id.name) and #fix_it, curry f=>g=>h"
    _event(tmp_path, "s1", "UserPromptSubmit", prompt=prompt)
    sessions = tmp_path / "terms" / "sessions"
    _store(sessions, _record("stored", NOW, ["heard=>written", "#comment_x", "kept_term"]))

    write_merged(tmp_path / "terms", now=NOW)

    lines = _current(tmp_path / "terms")
    assert "kept_term" in lines
    assert [line for line in lines if "=>" in line or line.startswith("#")] == []


def test_a_second_active_session_keeps_some_slots(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    mine = [f"mine_{n}" for n in range(100)]
    _store(sessions, _record("dictated", NOW - timedelta(minutes=2), mine))
    _store(sessions, _record("worker", NOW, [f"worker_{n}" for n in range(100)]))

    write_merged(tmp_path / "terms", now=NOW)

    lines = _current(tmp_path / "terms")
    assert lines[:4] == ["worker_0", "mine_0", "worker_1", "mine_1"]
    assert len([line for line in lines if line.startswith("mine_")]) == MERGED_CAP // 2


def test_stop_updates_terms_but_not_the_rank_time(tmp_path: Path) -> None:
    transcript = _transcript(tmp_path)
    _event(tmp_path, "s1", "SessionStart", transcript=transcript)
    first = SessionRecord.model_validate_json(
        (tmp_path / "terms" / "sessions" / "s1.json").read_text()
    )
    later = NOW + timedelta(minutes=20)
    payload = {"session_id": "s1", "hook_event_name": "Stop", "cwd": "/x", "prompt": None}
    run_hook(json.dumps(payload).encode(), state=tmp_path, now=later, host="box")

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

    for payload in (older, late_prompt):
        run_hook(
            json.dumps(payload).encode(), state=tmp_path, now=NOW - timedelta(seconds=2), host="box"
        )

    assert record_path.read_text() == before
    logged = [
        json.loads(line)["prompt"] for line in (tmp_path / "prompts.jsonl").read_text().splitlines()
    ]
    assert logged == ["see newer_term", "see older_term"]


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
    assert "unknown_entry_term" in lines
    assert "worker_only_term" not in lines
    worker_record = SessionRecord.model_validate_json(
        (tmp_path / "terms" / "sessions" / "w1.json").read_text()
    )
    assert worker_record.entrypoint == "sdk-cli"


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
                    kwargs={"prompt": "see b_term"},
                )
            )
            racer[0].start()
            time.sleep(0.3)
        real(path, text)

    monkeypatch.setattr(session_terms, "_replace_atomically", interleave)
    _event(tmp_path, "a", "UserPromptSubmit", prompt="see a_term")
    racer[0].join()

    assert sorted(_current(tmp_path / "terms")) == ["a_term", "b_term"]


def test_a_stale_session_file_is_not_parsed(tmp_path: Path) -> None:
    sessions = tmp_path / "terms" / "sessions"
    fresh_claim = _store(sessions, _record("stale", NOW, ["claims_fresh"]))
    broken = sessions / "broken.json"
    broken.write_text("{")
    old = (NOW - timedelta(minutes=31)).timestamp()
    for path in (fresh_claim, broken):
        os.utime(path, (old, old))

    write_merged(tmp_path / "terms", now=NOW)

    assert _current(tmp_path / "terms") == []
    assert not (tmp_path / "terms" / "hook.log").exists()


def test_a_prompt_log_failure_still_updates_the_terms(tmp_path: Path) -> None:
    tmp_path.joinpath("prompts.jsonl").write_text("")
    tmp_path.joinpath("prompts.jsonl").chmod(0o400)

    _event(tmp_path, "s1", "UserPromptSubmit", prompt="see kept_term")

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
        {"type": "user", "timestamp": "2026-10-05T11:00:00Z", "message": {"content": "twice_term"}},
        {"type": "user", "timestamp": "2026-10-05T11:30:00Z", "message": {"content": "twice_term"}},
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
        run_hook(raw, state=tmp_path, now=NOW, host="box")

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
