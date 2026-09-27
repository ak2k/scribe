"""Guard: no starter placeholder token may survive a package rename.

The template ships with a placeholder package name that a rename substitutes
across the tree. A manual or half-finished rename can leave the token behind in
docs or config — the exact failure this template was found shipping downstream.
This test fails any spawned repo where a stray token remains, regardless of
*how* the rename was done.

Self-disabling in the un-renamed template: while the placeholder source
directory still exists, the token is expected everywhere and the check is
skipped.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

# Assembled in two halves so this file never matches itself.
PLACEHOLDER = "my" + "project"

ROOT = Path(__file__).resolve().parents[1]

# Module-local alias so tests can stub git without patching the process-wide
# `subprocess.run` (a plugin or thread calling it mid-test would get the stub).
_run = subprocess.run


def _tracked_files() -> list[str]:
    # The pytest-in-closure flake check runs against a gitless store copy, and
    # a bare environment may lack git entirely — both skip (the guard still
    # runs everywhere `make check` does). A repo ANCESTOR counts: a template
    # vendored into a monorepo subdirectory is still governed by git. Any
    # OTHER git failure — corrupt index, dubious ownership — fails loudly
    # with git's stderr attached: those are the CI-like environments where
    # silently disabling the guard would hurt most.
    if not any((p / ".git").exists() for p in (ROOT, *ROOT.parents)):
        pytest.skip("rename guard needs a git checkout (gitless store copy)")
    try:
        result = _run(
            ["git", "ls-files"],  # git resolved from PATH by design
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        pytest.skip("rename guard needs `git` on PATH")
    except subprocess.CalledProcessError as exc:
        # typeshed types CalledProcessError.stderr as Any (it depends on the
        # run()'s text mode); this call passes text=True, so it is str | None.
        stderr: str = (exc.stderr or "").strip()  # pyright: ignore[reportAny]
        msg = f"`git ls-files` failed in {ROOT} (exit {exc.returncode}): {stderr}"
        raise RuntimeError(msg) from exc
    return result.stdout.splitlines()


def test_no_placeholder_token_remains() -> None:
    """Once renamed, the placeholder token must not survive in any tracked file."""
    if (ROOT / "src" / PLACEHOLDER).is_dir():
        return  # un-renamed template: the placeholder is expected everywhere

    offenders: list[str] = []
    for rel in _tracked_files():
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binaries / staged-then-removed paths
        if PLACEHOLDER in text.lower():
            offenders.append(rel)

    assert not offenders, (
        f"the starter placeholder {PLACEHOLDER!r} survived the rename in {offenders}. "
        "Finish the substitution by hand so no spawned repo ships an un-renamed reference."
    )


def test_guard_skips_in_gitless_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A tree with no .git in it or any ancestor (the nix store copy) skips."""
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        _tracked_files()


def test_guard_skips_when_git_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No git on PATH skips — the guard still runs wherever `make check` does."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)

    def _raise(*args: object, **kwargs: object) -> NoReturn:
        raise FileNotFoundError("git")

    monkeypatch.setattr(sys.modules[__name__], "_run", _raise)
    with pytest.raises(pytest.skip.Exception):
        _tracked_files()


def test_guard_runs_in_nested_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A template vendored into a monorepo subdir (repo in an ANCESTOR) still runs."""
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "services" / "svc"
    sub.mkdir(parents=True)
    monkeypatch.setattr(sys.modules[__name__], "ROOT", sub)

    def _fake(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["git", "ls-files"], 0, stdout="a.py\n", stderr="")

    monkeypatch.setattr(sys.modules[__name__], "_run", _fake)
    assert _tracked_files() == ["a.py"]


def test_guard_fails_loudly_on_broken_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A git-level failure (corrupt index, dubious ownership) must NOT skip.

    Deliberately not written as `pytest.raises` around the call: the round-1
    defect converted git failures into `pytest.skip`, and a Skipped exception
    escaping a non-matching `pytest.raises` block makes the TEST report
    SKIPPED (exit 0) — the regression would pass CI unseen. Asserting on an
    explicit outcome makes a reintroduced catch-and-skip fail red.
    """
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)

    def _raise(*args: object, **kwargs: object) -> NoReturn:
        raise subprocess.CalledProcessError(
            128, ["git", "ls-files"], stderr="fatal: detected dubious ownership"
        )

    monkeypatch.setattr(sys.modules[__name__], "_run", _raise)
    outcome = "returned"
    try:
        _tracked_files()
    except RuntimeError as exc:
        assert "dubious ownership" in str(exc), "git's stderr must survive into the error"
        outcome = "raised"
    except pytest.skip.Exception:  # the pinned bug shape: skipping instead of failing
        outcome = "skipped"
    assert outcome == "raised", f"a git failure must fail loudly, not '{outcome}'"
