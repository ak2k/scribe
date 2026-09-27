#!/usr/bin/env bash
# Hermetic regression matrix for the pre-push gate (.githooks/pre-push) and
# its guarded installer (.githooks/install). Every scenario here pins a
# behavior a review round proved breakable; a failure means a closed hole
# reopened. Needs only git, bash, sh, and coreutils — the two heavy gates
# (`make check`, `nix flake check`) are PATH-shimmed recorders, so the matrix
# runs in seconds anywhere: `./scripts/check-gate.sh`, CI, or the flake's
# `checks.gate` sandbox.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export HOME="$WORK/home" # isolate global git config
mkdir -p "$HOME"
export GIT_CONFIG_NOSYSTEM=1
git config --global user.email t@t && git config --global user.name t
git config --global init.defaultBranch main

Z40=$(printf '0%.0s' $(seq 40))
Z64=$(printf '0%.0s' $(seq 64))
pass=0
fail=0
ok() {
  pass=$((pass + 1))
  printf 'ok   %s\n' "$1"
}
bad() {
  fail=$((fail + 1))
  printf 'FAIL %s\n' "$1" >&2
}
check() { # check <name> <expected-exit> <expect-substring> <actual-exit> <output>
  if [ "$4" = "$2" ] && { [ -z "$3" ] || printf '%s' "$5" | grep -qF "$3"; }; then
    ok "$1"
  else
    bad "$1 (exit=$4 want=$2; output: $(printf '%s' "$5" | head -3 | tr '\n' ' '))"
  fi
}

# Shim make/nix so gated paths terminate instantly and record that they ran.
mkdir -p "$WORK/bin"
cat >"$WORK/bin/make" <<'EOF'
#!/bin/sh
echo "SHIM-MAKE $*"
exit "${SHIM_MAKE_EXIT:-0}"
EOF
cat >"$WORK/bin/nix" <<'EOF'
#!/bin/sh
echo "SHIM-NIX $*"
exit "${SHIM_NIX_EXIT:-0}"
EOF
cat >"$WORK/bin/uv" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$WORK/bin/make" "$WORK/bin/nix" "$WORK/bin/uv"
export PATH="$WORK/bin:$PATH"

new_repo() { # new_repo <dir> — fresh repo carrying the gate files
  rm -rf "$1"
  mkdir -p "$1"
  git -C "$1" init -q
  mkdir -p "$1/.githooks"
  cp "$REPO_ROOT/.githooks/pre-push" "$1/.githooks/pre-push"
  cp "$REPO_ROOT/.githooks/install" "$1/.githooks/install"
  chmod +x "$1/.githooks/pre-push" "$1/.githooks/install"
  touch "$1/flake.nix"
  git -C "$1" add -A
  git -C "$1" commit -qm init
}

# ---------- hook scenarios ----------
new_repo "$WORK/hook"
cd "$WORK/hook"
HEAD_SHA=$(git rev-parse HEAD)
git commit -qm second --allow-empty
HEAD2=$(git rev-parse HEAD)

out=$(SCRIBE_SKIP_PREPUSH=1 ./.githooks/pre-push </dev/null 2>&1) && rc=0 || rc=$?
check "hook: skip var" 0 "skipped" "$rc" "$out"
out=$(SCRIBE_SKIP_PREPUSH=0 ./.githooks/pre-push </dev/null 2>&1) && rc=0 || rc=$?
check "hook: skip var =0 does NOT skip" 0 "running the full gate" "$rc" "$out"
out=$(printf 'refs/heads/x %s refs/heads/x %s\n' "$Z40" "$HEAD_SHA" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: sha1 deletion skips" 0 "only ref deletions" "$rc" "$out"
out=$(printf 'refs/heads/x %s refs/heads/x %s\n' "$Z64" "$HEAD_SHA" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: sha256 deletion skips" 0 "only ref deletions" "$rc" "$out"
out=$(printf 'refs/heads/x %s refs/heads/x %s' "$Z40" "$HEAD_SHA" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: no-trailing-newline deletion skips" 0 "only ref deletions" "$rc" "$out"
out=$(printf 'refs/tags/v1 %s refs/tags/v1 %s\n' "$HEAD2" "$Z40" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: tag push runs both gates" 0 "SHIM-NIX flake check" "$rc" "$out"
out=$(printf 'refs/heads/b %s refs/heads/b %s\n' "$HEAD_SHA" "$Z40" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: non-HEAD push warns" 0 "not HEAD" "$rc" "$out"
out=$(./.githooks/pre-push </dev/null 2>&1) && rc=0 || rc=$?
check "hook: EOF manual invocation runs gate" 0 "SHIM-MAKE check" "$rc" "$out"
out=$(printf 'refs/heads/x %s refs/heads/x %s\nrefs/heads/y %s refs/heads/y %s\n' "$Z40" "$HEAD_SHA" "$HEAD2" "$Z40" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: mixed delete+code runs gate" 0 "SHIM-MAKE check" "$rc" "$out"
echo dirty >>flake.nix
out=$(printf 'refs/heads/m %s refs/heads/m %s\n' "$HEAD2" "$Z40" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: dirty tree NOTE" 0 "working tree is dirty" "$rc" "$out"
git checkout -q flake.nix
touch untracked.txt
out=$(printf 'refs/heads/m %s refs/heads/m %s\n' "$HEAD2" "$Z40" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: untracked NOTE" 0 "untracked files exist" "$rc" "$out"
rm untracked.txt
out=$(printf 'refs/heads/m %s refs/heads/m %s\n' "$HEAD2" "$Z40" | SHIM_MAKE_EXIT=130 ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: make failure preserves status + bypass hint" 130 "make check FAILED" "$rc" "$out"
out=$(printf 'refs/heads/m %s refs/heads/m %s\n' "$HEAD2" "$Z40" | SHIM_NIX_EXIT=143 ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
check "hook: nix failure preserves status + bypass hint" 143 "nix flake check FAILED" "$rc" "$out"
rm flake.nix && git add -A && git commit -qm noflake
H3=$(git rev-parse HEAD)
out=$(printf 'refs/heads/m %s refs/heads/m %s\n' "$H3" "$Z40" | ./.githooks/pre-push 2>&1) && rc=0 || rc=$?
if printf '%s' "$out" | grep -q "SHIM-NIX"; then
  bad "hook: no flake.nix skips nix gate"
else
  check "hook: no flake.nix skips nix gate" 0 "SHIM-MAKE check" "$rc" "$out"
fi

# ---------- installer scenarios ----------
ins() { sh ./.githooks/install "$@" 2>&1; }
hp() { git config --local --get core.hooksPath || true; }

new_repo "$WORK/i1" && cd "$WORK/i1"
out=$(ins) && rc=0 || rc=$?
check "install: fresh spawn arms" 0 "armed" "$rc" "$out"
if [ "$(hp)" = ".githooks" ]; then ok "install: hooksPath written"; else bad "install: hooksPath written"; fi
out=$(ins)
check "install: idempotent re-entry" 0 "armed" 0 "$out"

new_repo "$WORK/i2" && cd "$WORK/i2/.githooks" # SUBDIRECTORY entry — the round-3 bug
out=$(sh ./install 2>&1) && rc=0 || rc=$?
check "install: arms from a subdirectory" 0 "armed" "$rc" "$out"

new_repo "$WORK/i3" && cd "$WORK/i3"
printf '#!/bin/sh\n' >.git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
out=$(ins) && rc=0 || rc=$?
check "install: existing hook refuses" 0 "would be disabled" "$rc" "$out"
if [ -z "$(hp)" ]; then ok "install: refusal wrote nothing"; else bad "install: refusal wrote nothing"; fi

new_repo "$WORK/i4" && cd "$WORK/i4"
git config --local core.hooksPath .husky
out=$(ins) && rc=0 || rc=$?
check "install: local hooksPath refuses" 0 "already '.husky'" "$rc" "$out"

new_repo "$WORK/i5" && cd "$WORK/i5"
git config --global core.hooksPath "$WORK/ghooks"
out=$(ins) && rc=0 || rc=$?
git config --global --unset core.hooksPath
check "install: GLOBAL hooksPath refuses" 0 "already" "$rc" "$out"

new_repo "$WORK/i6" && cd "$WORK/i6"
mkdir .git/hooks/adir # a directory is not a hook
out=$(ins) && rc=0 || rc=$?
check "install: directory in hooks dir still arms" 0 "armed" "$rc" "$out"

new_repo "$WORK/i7" && cd "$WORK/i7"
rm -rf .git/hooks # missing hooks dir = no hooks = safe
out=$(ins) && rc=0 || rc=$?
check "install: missing hooks dir arms" 0 "armed" "$rc" "$out"

new_repo "$WORK/i8" && cd "$WORK/i8"
chmod 555 .git
out=$(ins) && rc=0 || rc=$?
chmod 755 .git
check "install: unwritable config refuses loudly" 0 "write failed" "$rc" "$out"

rm -rf "$WORK/i9" && mkdir -p "$WORK/i9" && cd "$WORK/i9" # foreign repo: no marker
git init -q && git commit -qm init --allow-empty
# invoke the flake's own copy, as `nix develop <url>` from a foreign repo would
out=$(sh "$REPO_ROOT/.githooks/install" 2>&1) && rc=0 || rc=$?
check "install: foreign repo (no marker) silent" 0 "" "$rc" "$out"
if [ -z "$out" ]; then ok "install: foreign repo no output"; else bad "install: foreign repo no output ($out)"; fi

new_repo "$WORK/i10" && cd "$WORK/i10" # flake-identity mismatch
printf '#!/bin/sh\nexit 0\n' >"$WORK/other-hook"
out=$(ins "$WORK/other-hook") && rc=0 || rc=$?
check "install: flake-identity mismatch refuses" 0 "not the one this dev shell ships" "$rc" "$out"
if [ -z "$(hp)" ]; then ok "install: mismatch wrote nothing"; else bad "install: mismatch wrote nothing"; fi

new_repo "$WORK/i11" && cd "$WORK/i11" # flake-identity match arms
out=$(ins "$WORK/i11/.githooks/pre-push") && rc=0 || rc=$?
check "install: flake-identity match arms" 0 "armed" "$rc" "$out"

new_repo "$WORK/i12" # untracked marker: overwrite with untracked content
cd "$WORK/i12"
git rm -q --cached .githooks/pre-push && git commit -qm untrack
out=$(ins) && rc=0 || rc=$?
check "install: untracked marker silent" 0 "" "$rc" "$out"

new_repo "$WORK/i13" && cd "$WORK/i13" # worktree note
git worktree add -q ../i13-wt -b other
out=$(ins) && rc=0 || rc=$?
check "install: multi-worktree caution" 0 "worktrees share this hook config" "$rc" "$out"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
