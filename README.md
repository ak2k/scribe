# scribe

A command line tool that turns an audio file into a diarized, cleaned
transcript. Each stage is a subcommand reading and writing one JSON schema, so
a run can be resumed, re-run, or inspected at any point. Planned, not yet
built: Krisp meetings as an input, and a quality evaluation of the output.

## Transcribe an audio file

```bash
export XAI_API_KEY=...             # required; never logged or written into an artifact
scribe transcribe talk.wav         # writes talk.transcript.json beside the input
scribe turns talk.transcript.json  # renders the speaker turns as markdown
```

Batch transcription costs $0.10 per audio hour. `--out` picks the path,
`--language ""` drops number formatting, `--no-diarize` drops speaker ids.
`--vad-threshold` sets xAI's voice-activity gate from 0 to 1; the default, 0,
turns it off, because the API's own default of 0.5 skipped long stretches of
clear speech.

`--vote` adds two engines, Parakeet TDT v3 run locally and Gemini, and changes
or adds a word of xAI's only where both agree. Three or more words added in a
row inside a pause of 2 s or more in xAI's words get no speaker (`turns` labels
them `Speaker ?`), since xAI may have dropped another speaker's turn there.
`--out` holds the voted transcript, and each engine's own goes beside it
(`talk.parakeet.json`, `talk.xai.json`, `talk.gemini.json`). It needs Apple
silicon, `uvx`, `ffmpeg` and `GEMINI_API_KEY` as well, sends the audio to
Google too, and adds about $0.55 per audio hour, capped by `--max-usd`
(default $3).

xAI can drop a passage of speech and return no words for it. So `transcribe`
then cross-checks its words against Parakeet TDT v3, run locally on Apple
silicon through `uvx`: nothing is uploaded, it takes about 45 s per audio
hour, and a first run downloads a ~1.2 GB model. Parakeet's transcript goes
beside `--out` (`talk.parakeet.json`). Where xAI's words leave a hole of 2 s or
more in which Parakeet heard at least 3 words xAI has nowhere near, those
words are inserted without a speaker (`turns` labels them `Speaker ?`), and
each filled span is printed; a stretch where Parakeet hears many words that no
hole can take is printed as possible dropped speech, to listen to. Where
Parakeet cannot run, or hears no words where xAI heard some, the holes are
checked by loudness, as `scribe gaps` does.
The cross-check never changes the exit code, and a failure in it keeps xAI's
transcript. With `--vote` the voted words are filled the same way.
`--no-cross-check` skips it. `scribe fill TRANSCRIPT REFERENCE` runs the fill
on transcripts already written. `scribe pick TRANSCRIPT REFERENCE` has a model
pick, at each spot where the two disagree, which of their two readings was
said, from the conversation around it (`--context` adds background such as who
was there); the words, not the audio, go to Anthropic through `claude`.

Unless `--no-pick` is given, `transcribe` then runs that pick itself, on its
filled words against Parakeet's (`--pick-model`, default opus, and
`--pick-context` are `pick`'s `--model` and `--context`), so, with the pick on,
the transcript's words, not the audio, go to Anthropic through `claude`. Where
the reading picked has 5 or more fewer words than the transcript's, the pick
is guarded and the transcript's words stay. Where Parakeet's reading is 10 or
more words longer than the transcript's, fillers and repeats aside, its words
go in whatever the model picks: the spot is restored. The pick never changes
the exit code; a failure in it, such as no `claude` on PATH, ends in one line
(a retried `claude` call is logged as a warning before it) and keeps the
filled words. It does not run with `--vote` or `--no-cross-check`, or where
Parakeet cannot run, and `--pick-context` with `--no-pick`, `--vote` or
`--no-cross-check` exits 2.

`scribe disputes talk.transcript.json` writes `talk.disputes.md`: every spot
the pick recorded, and every span the fill filled or could not repair, each
with where it is, the reading the transcript holds and the engine that heard
it, and the reading set aside and its engine. No recorded spot is left out;
entries are ranked so the likeliest errors come first:

- A, words missing on one side: a reading 5 or more words shorter than the other as the pick counts them, fillers and repeats aside; a guarded or restored pick; a filled or unresolved span.
- B, the pick was unsure or gave no answer.
- C, 4 or more words differ.
- D, 1 to 3 words differ.
- E, same words once folded; the difference is with the words around the spot (for example a number split differently).

`--clips` also cuts each entry's audio, 3 s on each side, into
`talk.disputes.clips/`, and each entry names its clip; it needs `ffmpeg`, and
reads `--audio` or the source audio, which must match its recorded sha256.
Not listed: short stretches only one engine heard, unless the fill filled or
flagged them, and words both got wrong the same way. Unlisted text is
unverified. The command sends nothing over the network.

`transcribe` writes this list itself, without clips, beside `--out`
(`talk.disputes.md`) wherever its pick rewrites the transcript, and prints its
counts. Where the run rewrites `--out` but the pick does not, as with
`--no-pick`, `--vote` or `--no-cross-check`, or where the pick finds no spot
even though the fill filled spans, no list is written, and a list an earlier
run left there is removed. A link there to an input or to a directory stays,
named in one line, and a run that exits 2 before it changes `--out` leaves the
list as it was. The list never changes the exit code: a failure in it is one
line, and a list that cannot be written leaves none from an earlier run.

`turns` groups the words into speaker turns. A transcript that carries turns
but no words keeps its turns as they are.

By default `turns` then has a model correct who said what, one `claude -p`
call per ~700 words, six at a time (`--speaker-model`, default opus). The
words themselves never change; only their speaker labels can. It needs the
`claude` CLI on PATH, on its own subscription auth, and records model, prompt
version and per-chunk status in `talk.transcript.speakers.json`.
`--no-llm-speakers` skips it. A failed chunk keeps the diarizer's speakers and
prints one warning line; exit 4 means every chunk failed (artifacts are still
written), and exit 2 with no usable CLI names `--no-llm-speakers`.

## Keep the vocabulary of Claude Code sessions

`scribe terms hook` is a Claude Code hook. On each event it reads the tail of
the session's transcript and keeps the identifiers in it (file names, paths,
functions, commands, the repo and branch), so dictation into that session can
pass them to xAI as keyterms. It writes under `$XDG_STATE_HOME/scribe`, else
`~/.local/state/scribe`:

- `terms/sessions/<session_id>.json`: one session's terms, with counts;
  deleted after 7 days.
- `terms/current.txt`: one block per interactive session you prompted or
  started in the last 24 hours, most recent first. A block opens with the
  comment `# session <id> ranked <UTC time> expires <UTC time> {"cwd": ...,
  "titles": [...]}` (when you last prompted or started it, 30 minutes after
  that, its directory, and its 5 latest titles, `/rename` titles before
  automatic ones), then up to 60 of its terms, one per line, best first. The
  cap is per block, so the file as a whole can hold more terms than xAI
  accepts; `scribe serve` reads it block by block. Nothing rewrites the file
  until the next hook event, so its reader drops a block once it expires;
  only the block of the session you are dictating into still counts after
  its expiry (see `scribe serve` below). A
  session contributes only once its transcript shows it is interactive, so a
  new session's terms join from its first prompt rather than from its start,
  and headless (`claude -p`) sessions are left out. A Stop event
  refreshes a session's terms without making it more recent.
- `prompts.jsonl`: every prompt you submit, with time, host, session and
  directory. It holds your prompts verbatim; delete it whenever you like.
- `terms/hook.log`: one line per failure, started afresh past 1 MB. The hook
  prints nothing and always exits 0.

Install it in `~/.claude/settings.json`; `async` keeps it from delaying a prompt:

```json
{
  "hooks": {
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "scribe terms hook", "async": true}]}],
    "SessionStart": [{"hooks": [{"type": "command", "command": "scribe terms hook", "async": true}]}],
    "Stop": [{"hooks": [{"type": "command", "command": "scribe terms hook", "async": true}]}]
  }
}
```

## Inspect the schema

```bash
scribe schema > transcript.schema.json   # the JSON Schema every stage reads and writes
```

## Clean up a transcript

```bash
scribe cleanup talk.transcript.turns.json \
  --speaker "Speaker 1=Ann Lee" --glossary "ackme=Acme"
```

Writes `talk.transcript.clean.md` (provenance header plus cleaned text) and
`talk.transcript.cleanup.json` beside it. Needs the `claude` CLI on PATH, on
its own subscription auth. Exit 0 is clean. Exit 3 means a number went missing
or changed, a word ended up under another speaker, or a chunk's reply was
malformed (the chunk then keeps its input text uncleaned, and the sidecar's
`malformed_chunks` names the cause), and both files are written anyway;
`--allow-number-drift` and `--allow-speaker-moves` downgrade the number and
speaker cases to 0. A number that survives but appears fewer times, as when
cleanup merges a restated phrase, is listed under `numbers_reduced` with a
warning; alone it does not change the exit code, but beside a number cleanup
added it reads as a changed number and exits 3.
Exit 2 is a bad input or a backend failure. `--max-budget-usd` is the ceiling for ONE
call, and there is one call per chunk, with a floor near $0.10 a call.

`--curated PATH` also writes a reading copy of the same words, each turn as
`**Speaker | HH:MM:SS**` over its cleaned text; a turn holding words a fill
pass inserted (the cross-check or `scribe fill`) opens with a bracketed note of
when. `--front FILE` puts
that file's text, exactly as written, at the top of the reading copy, above a
`---` rule.

## Dictate with scribe serve

```bash
export XAI_API_KEY=...   # required; never logged or written into an artifact
scribe serve             # listens on http://127.0.0.1:8765
```

A local endpoint for dictation apps that speak OpenAI's transcription API. In
VoiceInk, Settings -> AI models -> add a custom cloud model with endpoint
`http://127.0.0.1:8765/v1/audio/transcriptions`, any API key and any model
name. The audio goes to xAI only, and the text comes back in about a second
for a 10 s utterance. It listens on loopback only, and refuses any request a
web page could send.

The terms file (`--terms`, default `~/.config/scribe/terms.txt`) holds one
term per line, sent to xAI to bias recognition; a `#` at the start of a line or
after a space starts a comment. A line `heard => written` is an alias that
rewrites what xAI heard:

```text
# terms
VoiceInk
modules/darwin/base.nix
herder => herdr
```

A run of up to 4 words whose letters and digits spell an identifier-shaped
term comes back as the term ("voice ink" becomes "VoiceInk"). Identifier-shaped
means a digit, a capital after a lowercase letter, or punctuation inside it, and
at least 3 letters and digits; so `TODO`, `c++` and `.bashrc` are sent to xAI but
never snapped to. Use an alias for those. The file is re-read when it changes.

The terms of your live Claude Code sessions follow the file's, snapped the same
way: from this machine's `terms/current.txt` (see above) and from each
`--session-terms-host HOST`, whose file is read over
`ssh -o BatchMode=yes HOST` every 3 s in the background, so a dictation never
waits on it. A host that stops answering keeps its last list until each
session's block expires. Sessions take turns, most recently prompted first, up
to xAI's 100 keyterms; the file's terms always go first. `--no-session-terms`
sends none. `GET /health` shows each source's live sessions, terms and the age
of its last read.

On macOS, serve also asks Ghostty every 3 s, in the background, which tab is
focused. When it shows a Claude Code session on this machine, found by the
tab's title (an untitled session is never picked), that session's terms come
right after the file's, up to 24 hours after its last prompt. No query runs while no session here was prompted in the
last 24 hours. The first query brings up macOS's one-time Automation prompt
asking to let the program running serve control Ghostty; until it is allowed,
the log shows `serve.focus_failed error=-1743`. `--no-focus` turns this off.
Each request's log line and `result.json` say whether focus found a session
and how many terms it gave, never which; `GET /health` shows the poller's
state.

Each dictation (the audio, xAI's reply and `result.json`) is kept under
`--keep`, default `~/.local/state/scribe/serve`, newest 1000; `--no-keep`
keeps nothing.

## Quick start

```bash
uv sync
make check
```

### Nix users

The flake exposes two dev shells, two venv packages, plus `nix fmt` and `nix flake check`:

```bash
nix develop              # default: python + uv + make. `uv sync` populates .venv.
nix develop .#pure       # uv2nix editable venv. No .venv. Worktree edits live.
nix build .#default      # runtime venv: project + [project.dependencies]
nix build .#dev          # dev venv: adds [dependency-groups].dev (pytest, ruff, ...)
nix fmt                  # treefmt: nixfmt + shfmt + yamlfmt (python: `make fix` owns ruff)
nix flake check          # statix + treefmt/shellcheck + gate matrix + pytest-in-closure
```

Entering a dev shell also **arms the pre-push gate** when that is provably
safe: `.githooks/install` sets `core.hooksPath` to the tracked `.githooks/`
dir — but only in a repo spawned from this template, with no existing hooks
or `hooksPath` convention that the setting would silently disable (it prints
the manual command instead of arming when in doubt). Once armed, `git push`
runs `make check` + `nix flake check` first — the latter builds the uv2nix
closure, which takes minutes when cold. Bypass one push with
`git push --no-verify` (or `SCRIBE_SKIP_PREPUSH=1 git push`); unarm with
`git config --local --unset core.hooksPath`.

Pick `default` for daily work — `uv add` / `uv lock` / `uv run` all mutate state
naturally and `make check` is the same command as the non-Nix path. Pick `.#pure`
when you want fully Nix-resolved deps with one-step onboarding; the trade-off is
that `uv.lock` changes require exiting and re-entering the shell so Nix can
re-resolve. The `packages` outputs are for nixpkgs PRs, NixOS modules, or
downstream Nix consumers — not needed for daily work.

direnv auto-activation: edit `.envrc` to uncomment `use flake`, then `direnv allow`.

uv remains the source of truth — `pyproject.toml` + `uv.lock` drive everything.
The Nix layer is a lens, built via [uv2nix](https://pyproject-nix.github.io/uv2nix/).

## What's in here

| File | Purpose |
|---|---|
| `pyproject.toml` | uv deps + ruff strict + basedpyright strict + pytest config — single source of truth |
| `AGENTS.md` | Contract for AI agents. Stack banlist + inner loop + divergence guide. |
| `CLAUDE.md` | Symlink → `AGENTS.md` (compat shim for tools that read CLAUDE.md only) |
| `Makefile` | `make check` = full inner loop. CI runs the same command. |
| `src/scribe/turns.py` | Canonical stage shape — copy for new stages |
| `tests/test_turns.py` | Canonical test shape — copy for new tests |
| `src/scribe/errors.py` | Domain error hierarchy |
| `.github/workflows/ci.yml` | CI runs `make check` + the gate probes; nix path (build + flake check) when `flake.nix` exists |
| `.githooks/` | Pre-push gate (`pre-push`) + its guarded installer (`install`); armed by the dev shell, regression-tested by `scripts/check-gate.sh` |
| `scripts/check-gate.sh` | Hermetic probe matrix for the gate (runs in CI and as `checks.gate`) |
| `flake.nix` + `.envrc` | Optional Nix layer: `default` (uv-managed) + `.#pure` (uv2nix editable) dev shells; `packages.{default,dev}` venvs |

## Philosophy

One tool per concern, banned substitutes, strict types at the boundary,
canonical examples to pattern-match against. The contract lives in
[AGENTS.md](AGENTS.md) — including the "Appropriate divergence" section for
libraries / CLIs / scrapers / data pipelines / one-off scripts.

## Related work

- [Ranteck/PyStrict](https://github.com/Ranteck/PyStrict-strict-python) —
  strict-mode template that inspired the type-checker config here.
- [osprey-oss/cookiecutter-uv](https://github.com/osprey-oss/cookiecutter-uv) —
  heavier mainstream alternative (cookiecutter + extensive Jinja templating).
- [astral-sh/uv](https://github.com/astral-sh/uv),
  [astral-sh/ruff](https://github.com/astral-sh/ruff),
  [DetachHead/basedpyright](https://github.com/DetachHead/basedpyright) —
  upstream tools.

## License

MIT.
