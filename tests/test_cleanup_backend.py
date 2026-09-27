from __future__ import annotations

import json
import locale
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import stamina
from structlog.testing import capture_logs

from scribe.claude_cli import ClaudeCliBackend, Completion
from scribe.errors import ExternalServiceError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from scribe.cleanup import CleanupBackend

FIXTURES = Path(__file__).parent / "fixtures"
CLAUDE = "/opt/nowhere/bin/claude"
SYSTEM = "you are a transcript editor"
USER = "Speaker 1: we paid $1,234.56"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _found(_name: str) -> str | None:
    return CLAUDE


def _not_found(_name: str) -> str | None:
    return None


@pytest.fixture(autouse=True)
def claude_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # Hermetic on purpose: the suite must not depend on a `claude` install, and
    # resolving the name here is what keeps a bare "claude" out of the argv.
    monkeypatch.setattr(shutil, "which", _found)
    # Session-stub cleanup reads this; the real one is the user's own ~/.claude.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "user-claude-config"))
    # Testing mode drops stamina's backoff waits; capping rather than setting
    # attempts leaves the backend's own limit in force.
    with stamina.set_testing(True, attempts=10, cap=True):
        yield


class Recorder:
    """A stand-in for `subprocess.run` that records the call it was handed."""

    def __init__(self, stdout: str, *, returncode: int = 0, stderr: str = "") -> None:
        self._stdout = stdout
        self._returncode = returncode
        self._stderr = stderr
        self.argv: list[str] = []
        self.cwd = ""
        self.env: dict[str, str] = {}
        self.kwargs: dict[str, object] = {}
        self.system_prompt = ""

    def run(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.argv = argv
        self.kwargs = kwargs
        cwd = kwargs["cwd"]
        assert isinstance(cwd, str)
        self.cwd = cwd
        env = kwargs["env"]
        assert isinstance(env, dict)
        self.env = cast("dict[str, str]", env)
        # Read during the call: the temp directory is gone once it returns.
        self.system_prompt = self.prompt_file(argv).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, self._returncode, stdout=self._stdout, stderr=self._stderr
        )

    @staticmethod
    def prompt_file(argv: list[str]) -> Path:
        return Path(argv[argv.index("--system-prompt-file") + 1])


def test_the_argv_carries_every_mandated_flag() -> None:
    recorder = Recorder(_fixture("claude_p_success.json"))
    backend = ClaudeCliBackend(run=recorder.run)

    backend.complete(SYSTEM, USER)

    argv = recorder.argv
    assert argv[0] == CLAUDE
    assert "-p" in argv
    assert "--no-session-persistence" in argv
    assert "--disable-slash-commands" in argv
    for flag, value in [
        ("--model", "opus"),
        ("--setting-sources", ""),
        ("--max-budget-usd", "6.0"),
        ("--output-format", "json"),
    ]:
        assert argv[argv.index(flag) + 1] == value


def test_tools_are_disabled_only_when_asked() -> None:
    kept = Recorder(_fixture("claude_p_success.json"))
    disabled = Recorder(_fixture("claude_p_success.json"))

    ClaudeCliBackend(run=kept.run).complete(SYSTEM, USER)
    ClaudeCliBackend(run=disabled.run, disable_tools=True).complete(SYSTEM, USER)

    assert "--tools" not in kept.argv
    assert disabled.argv[disabled.argv.index("--tools") + 1] == ""


def test_the_prompt_goes_on_stdin_and_never_into_the_argv() -> None:
    # A long monologue in one argv element dies at exec on Linux (128 KiB).
    user = "Speaker 1: " + "word " * 200_000
    recorder = Recorder(_fixture("claude_p_success.json"))

    ClaudeCliBackend(run=recorder.run).complete(SYSTEM, user)

    assert recorder.kwargs["input"] == user
    assert all(user not in arg for arg in recorder.argv)
    assert sum(len(arg) for arg in recorder.argv) < 1000


# Echoes the prompt back as the reply, reading and writing UTF-8 bytes whatever
# the locale says.
_ECHO_CLAUDE = """\
#!{python}
import json, sys
reply = json.loads(open({fixture!r}, encoding="utf-8").read())
reply["result"] = sys.stdin.buffer.read().decode("utf-8")
sys.stdout.buffer.write(json.dumps(reply, ensure_ascii=False).encode("utf-8"))
"""


@pytest.mark.skipif(sys.flags.utf8_mode == 1, reason="UTF-8 mode ignores the locale")
def test_a_non_ascii_prompt_and_reply_survive_an_ascii_locale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude = tmp_path / "claude"
    fixture = str(FIXTURES / "claude_p_success.json")
    claude.write_text(_ECHO_CLAUDE.format(python=sys.executable, fixture=fixture), encoding="utf-8")
    claude.chmod(0o755)

    def found(_name: str) -> str | None:
        return str(claude)

    monkeypatch.setattr(shutil, "which", found)
    user = "Speaker 1: down \N{MINUS SIGN}5% to \N{EURO SIGN}12"
    previous = locale.setlocale(locale.LC_CTYPE)
    locale.setlocale(locale.LC_CTYPE, "C")
    try:
        completion = ClaudeCliBackend().complete(SYSTEM, user)
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)

    assert completion.text == user


def test_a_claude_that_cannot_be_run_is_a_domain_error() -> None:
    def refuse(_argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise PermissionError(13, "Permission denied", CLAUDE)

    with pytest.raises(ExternalServiceError, match="cannot run claude") as caught:
        ClaudeCliBackend(run=refuse).complete(SYSTEM, USER)

    assert USER not in str(caught.value)


def test_the_system_prompt_file_is_absolute_and_readable_during_the_call() -> None:
    recorder = Recorder(_fixture("claude_p_success.json"))

    ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)

    prompt_file = Recorder.prompt_file(recorder.argv)
    assert prompt_file.is_absolute()
    assert recorder.system_prompt == SYSTEM
    # `claude -p` resolves the path against its own cwd, which is the temp
    # directory the file was written into.
    assert prompt_file.parent == Path(recorder.cwd)
    assert not prompt_file.exists()


def test_the_three_environment_variables_reach_the_cli() -> None:
    recorder = Recorder(_fixture("claude_p_success.json"))

    ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)

    assert recorder.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert recorder.env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert recorder.env["CLAUDE_CODE_DISABLE_CRON"] == "1"
    # The rest of the environment still goes through: the CLI authenticates on
    # the user's subscription, not on a key this process holds.
    assert "PATH" in recorder.env
    assert recorder.kwargs["text"] is True
    assert recorder.kwargs["capture_output"] is True
    assert recorder.kwargs["check"] is False


def test_a_recorded_success_parses_into_a_completion() -> None:
    recorder = Recorder(_fixture("claude_p_success.json"))

    completion = ClaudeCliBackend(model="sonnet", run=recorder.run).complete(SYSTEM, USER)

    assert completion == Completion(
        text="PONG",
        model="sonnet",
        output_tokens=5,
        stop_reason="end_turn",
        is_error=False,
    )


def test_a_recorded_budget_error_becomes_a_domain_error() -> None:
    recorder = Recorder(_fixture("claude_p_budget_error.json"), returncode=1, stderr="  budget  ")
    backend = ClaudeCliBackend(max_budget_usd=0.05, run=recorder.run)

    with pytest.raises(ExternalServiceError, match="error_max_budget_usd") as caught:
        backend.complete(SYSTEM, USER)

    message = str(caught.value)
    assert "exited 1" in message
    assert "budget" in message
    # An error line is a log line: the recording never belongs in one.
    assert USER not in message


def test_a_non_zero_exit_alone_is_a_failure() -> None:
    # The one failure signal with nothing else beside it: a complete, successful
    # reply on stdout and a process that then exited non-zero.
    recorder = Recorder(_fixture("claude_p_success.json"), returncode=1, stderr="killed")

    with pytest.raises(ExternalServiceError, match="exited 1") as caught:
        ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)

    assert "'success'" in str(caught.value)


def test_an_error_flag_on_a_zero_exit_is_still_a_failure() -> None:
    recorder = Recorder('{"is_error": true, "subtype": "error_during_execution", "result": "no"}')

    with pytest.raises(ExternalServiceError, match="error_during_execution"):
        ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)


def test_a_result_that_is_not_a_string_is_a_failure() -> None:
    recorder = Recorder('{"is_error": false, "subtype": "success", "result": null}')

    with pytest.raises(ExternalServiceError, match="'success'"):
        ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)


def test_unparseable_stdout_is_a_failure() -> None:
    recorder = Recorder("not json at all", returncode=2, stderr="claude: command failed")

    with pytest.raises(ExternalServiceError, match="without parseable JSON") as caught:
        ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)

    assert "exited 2" in str(caught.value)


def test_stdout_that_is_not_a_json_object_is_a_failure() -> None:
    recorder = Recorder('["a list of results"]')

    with pytest.raises(ExternalServiceError, match="a JSON list, not an object"):
        ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)


def test_the_stderr_excerpt_stops_at_three_hundred_characters() -> None:
    recorder = Recorder("{}", returncode=1, stderr="E" * 400)

    with pytest.raises(ExternalServiceError) as caught:
        ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)

    assert "E" * 300 in str(caught.value)
    assert "E" * 301 not in str(caught.value)


def test_a_reply_without_usage_reports_no_token_count() -> None:
    recorder = Recorder('{"is_error": false, "subtype": "success", "result": "hi"}')

    completion = ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)

    assert completion.output_tokens is None
    assert completion.stop_reason is None


def test_a_non_integer_token_count_reports_no_token_count() -> None:
    recorder = Recorder('{"result": "hi", "usage": {"output_tokens": "five"}, "stop_reason": 7}')

    completion = ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)

    assert completion.output_tokens is None
    assert completion.stop_reason is None


def test_no_claude_on_path_is_a_domain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", _not_found)
    recorder = Recorder(_fixture("claude_p_success.json"))

    with pytest.raises(ExternalServiceError, match="claude is not on PATH"):
        ClaudeCliBackend(run=recorder.run).complete(SYSTEM, USER)

    assert recorder.argv == []


def test_the_real_backend_satisfies_the_protocol() -> None:
    backend: CleanupBackend = ClaudeCliBackend()

    assert isinstance(backend, ClaudeCliBackend)
    assert backend.model == "opus"
    assert backend.max_budget_usd == 6.0
    assert ClaudeCliBackend.name == "claude-cli"


_SUCCESS = _fixture("claude_p_success.json")
_OVERLOADED = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": 529,
        "result": "API Error: 529 Overloaded",
        "session_id": "1b241630-9e1a-4b45-a828-0c565e72486f",
    }
)
_NOT_LOGGED_IN = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": None,
        "result": "Not logged in · Please run /login",
    }
)


class Script:
    """A stand-in for `subprocess.run` that plays one outcome per call, in order."""

    def __init__(self, *outcomes: str | tuple[str, int, str] | BaseException) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def run(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = cast("dict[str, str]", kwargs["env"])
        self.calls.append((argv, env))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        stdout, returncode, stderr = (outcome, 0, "") if isinstance(outcome, str) else outcome
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)


def _backend(
    script: Script,
    *,
    which: Callable[[str], str | None] = _found,
    stub_root: Path | None = None,
) -> ClaudeCliBackend:
    return ClaudeCliBackend(run=script.run, which=which, stub_root=stub_root)


def test_the_api_key_and_the_nesting_marker_never_reach_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-must-not-bill")
    monkeypatch.setenv("CLAUDECODE", "1")
    script = Script(_SUCCESS)

    _backend(script).complete(SYSTEM, USER)

    env = script.calls[0][1]
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDECODE" not in env
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


def test_the_user_environment_reaches_the_cli_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The user's own credentials and endpoint, left as they set them.
    for name in ("ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"):
        monkeypatch.setenv(name, f"users-own-{name}")
    script = Script(_SUCCESS)

    _backend(script).complete(SYSTEM, USER)

    env = script.calls[0][1]
    for name in ("ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"):
        assert env[name] == f"users-own-{name}"


def test_resolve_finds_claude_before_any_call() -> None:
    lookups: list[str] = []

    def which(name: str) -> str | None:
        lookups.append(name)
        return CLAUDE

    script = Script()
    backend = _backend(script, which=which)

    with capture_logs() as logs:
        assert backend.resolve() == CLAUDE
        assert backend.resolve() == CLAUDE

    assert lookups == ["claude"]
    assert [entry["executable"] for entry in logs if entry["event"] == "claude_cli.executable"] == [
        CLAUDE
    ]
    assert script.calls == []


def test_resolve_without_claude_is_a_domain_error() -> None:
    with pytest.raises(ExternalServiceError, match="claude is not on PATH"):
        _backend(Script(), which=_not_found).resolve()


_THREADS = 6


def _all_at_once(backend: ClaudeCliBackend) -> None:
    start = threading.Barrier(_THREADS)

    def call() -> None:
        start.wait()
        backend.complete(SYSTEM, USER)

    with ThreadPoolExecutor(_THREADS) as pool:
        for future in [pool.submit(call) for _ in range(_THREADS)]:
            future.result()


def test_concurrent_calls_look_claude_up_once() -> None:
    lookups: list[str] = []

    def slow_which(name: str) -> str | None:
        lookups.append(name)
        # Holds every other thread at the unresolved lookup long enough to race.
        time.sleep(0.05)
        return CLAUDE

    script = Script(*[_SUCCESS] * _THREADS)
    backend = ClaudeCliBackend(run=script.run, which=slow_which)

    with capture_logs() as logs:
        _all_at_once(backend)

    assert lookups == ["claude"]
    assert len([entry for entry in logs if entry["event"] == "claude_cli.executable"]) == 1


def test_a_timeout_is_retried_until_a_reply_arrives() -> None:
    script = Script(subprocess.TimeoutExpired([CLAUDE], 300.0), _SUCCESS)

    completion = _backend(script).complete(SYSTEM, USER)

    assert completion.text == "PONG"
    assert len(script.calls) == 2


def test_the_timeout_reaches_the_runner() -> None:
    recorder = Recorder(_SUCCESS)

    ClaudeCliBackend(run=recorder.run, timeout_s=12.5).complete(SYSTEM, USER)

    assert recorder.kwargs["timeout"] == 12.5


def test_timeouts_on_every_attempt_end_in_a_domain_error() -> None:
    script = Script(*[subprocess.TimeoutExpired([CLAUDE], 300.0) for _ in range(3)])

    with pytest.raises(ExternalServiceError, match="timed out after 300"):
        _backend(script).complete(SYSTEM, USER)

    assert len(script.calls) == 3


def test_an_overloaded_reply_is_retried() -> None:
    script = Script((_OVERLOADED, 1, ""), _SUCCESS)

    completion = _backend(script).complete(SYSTEM, USER)

    assert completion.text == "PONG"
    assert len(script.calls) == 2


def test_an_overloaded_reply_on_every_attempt_names_the_status() -> None:
    script = Script(*[(_OVERLOADED, 1, "") for _ in range(3)])

    with pytest.raises(ExternalServiceError, match="api_error_status 529"):
        _backend(script).complete(SYSTEM, USER)

    assert len(script.calls) == 3


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param((_fixture("claude_p_budget_error.json"), 1, ""), id="budget"),
        pytest.param(("not json", 1, ""), id="unparseable"),
        pytest.param(('["a list"]', 0, ""), id="not-an-object"),
        pytest.param((_NOT_LOGGED_IN, 1, ""), id="not-logged-in"),
        pytest.param((_SUCCESS, 1, "killed"), id="non-zero-exit"),
        pytest.param(
            (json.dumps({"is_error": True, "api_error_status": 400, "result": "bad"}), 1, ""),
            id="client-error",
        ),
        pytest.param(
            (json.dumps({"is_error": True, "api_error_status": True, "result": "bad"}), 1, ""),
            id="boolean-status",
        ),
        pytest.param(FileNotFoundError(2, "No such file", CLAUDE), id="gone"),
    ],
)
def test_a_lasting_failure_is_not_retried(
    outcome: tuple[str, int, str] | BaseException,
) -> None:
    script = Script(outcome, _SUCCESS)

    with pytest.raises(ExternalServiceError):
        _backend(script).complete(SYSTEM, USER)

    assert len(script.calls) == 1


def _stub(root: Path, session_id: str, size: int, *, project: str = "-tmp-abc") -> Path:
    path = root / project / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


_SESSION = "0148fa0f-232f-4eef-a51e-d378d8e378ee"


def test_a_call_removes_its_leaked_session_stub(tmp_path: Path) -> None:
    stub = _stub(tmp_path, _SESSION, 110)

    _backend(Script(_SUCCESS), stub_root=tmp_path).complete(SYSTEM, USER)

    assert not stub.exists()
    assert not stub.parent.exists()


def test_the_stub_root_follows_the_config_dir(tmp_path: Path) -> None:
    stub = _stub(Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects", _SESSION, 110)

    _backend(Script(_SUCCESS)).complete(SYSTEM, USER)

    assert not stub.exists()


def test_a_stub_is_removed_after_a_failed_call_too(tmp_path: Path) -> None:
    stub = _stub(tmp_path, "0ac7e0e6-d4ca-4803-bb23-b75066f02174", 110)
    script = Script((_fixture("claude_p_budget_error.json"), 1, ""))

    with pytest.raises(ExternalServiceError):
        _backend(script, stub_root=tmp_path).complete(SYSTEM, USER)

    assert not stub.exists()


def test_a_large_session_file_is_never_removed(tmp_path: Path) -> None:
    transcript = _stub(tmp_path, _SESSION, 4096)

    _backend(Script(_SUCCESS), stub_root=tmp_path).complete(SYSTEM, USER)

    assert transcript.exists()


def test_a_project_dir_holding_other_files_stays(tmp_path: Path) -> None:
    stub = _stub(tmp_path, _SESSION, 110)
    other = _stub(tmp_path, "5d1f7a0e-0000-4000-8000-000000000000", 110)

    _backend(Script(_SUCCESS), stub_root=tmp_path).complete(SYSTEM, USER)

    assert not stub.exists()
    assert other.exists()


@pytest.mark.parametrize("session_id", ["*", "../escape", "", 7])
def test_a_session_id_that_is_not_a_uuid_removes_nothing(
    tmp_path: Path, session_id: object
) -> None:
    stub = _stub(tmp_path, _SESSION, 110)
    reply = json.dumps({"result": "hi", "session_id": session_id})

    _backend(Script(reply), stub_root=tmp_path).complete(SYSTEM, USER)

    assert stub.exists()


def test_a_symlinked_project_dir_is_never_followed(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    outside = _stub(tmp_path, _SESSION, 110, project="outside")
    (projects / "linked").symlink_to(outside.parent, target_is_directory=True)

    _backend(Script(_SUCCESS), stub_root=projects).complete(SYSTEM, USER)

    assert outside.exists()


def test_a_symlinked_stub_is_never_followed(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    outside = _stub(tmp_path, _SESSION, 110, project="outside")
    (projects / "-tmp-abc").mkdir(parents=True)
    link = projects / "-tmp-abc" / outside.name
    link.symlink_to(outside)

    _backend(Script(_SUCCESS), stub_root=projects).complete(SYSTEM, USER)

    assert outside.exists()
    assert link.is_symlink()


def test_a_symlinked_projects_root_still_has_its_stubs_removed(tmp_path: Path) -> None:
    real = tmp_path / "real-projects"
    stub = _stub(real, _SESSION, 110)
    (tmp_path / "projects").symlink_to(real, target_is_directory=True)

    _backend(Script(_SUCCESS), stub_root=tmp_path / "projects").complete(SYSTEM, USER)

    assert not stub.exists()


def test_a_retry_hands_no_prompt_text_to_retry_hooks() -> None:
    # Set here rather than relied on: another module turning stamina's hooks
    # off at import must not be what keeps the prompt out of a log.
    seen: list[stamina.instrumentation.RetryDetails] = []
    previous = stamina.instrumentation.get_on_retry_hooks()

    def record(details: stamina.instrumentation.RetryDetails) -> None:
        seen.append(details)

    stamina.instrumentation.set_on_retry_hooks([record])
    user = "Speaker 1: " + "confidential " * 100
    script = Script(subprocess.TimeoutExpired([CLAUDE], 300.0), _SUCCESS)
    try:
        _backend(script).complete("secret system prompt", user)
    finally:
        stamina.instrumentation.set_on_retry_hooks(previous)

    assert len(seen) == 1
    assert "confidential" not in repr(seen)
    assert "secret system prompt" not in repr(seen)


def test_a_stub_that_cannot_be_removed_does_not_fail_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub(tmp_path, _SESSION, 110)

    def refuse(_self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "unlink", refuse)

    completion = _backend(Script(_SUCCESS), stub_root=tmp_path).complete(SYSTEM, USER)

    assert completion.text == "PONG"
