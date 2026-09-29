"""`scribe transcribe` picks, where its filled words and Parakeet's disagree, which was said."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest
from typer.testing import CliRunner

from scribe import ensemble
from scribe.claude_cli import ClaudeCliBackend
from scribe.cli import app
from scribe.coverage import fill_holes
from scribe.parakeet import ParakeetMlx
from scribe.pick import DEFAULT_PICK_MODEL, PICK_PROMPT_VERSION, system_prompt
from scribe.schema import Transcript, Word
from tests.pick_fakes import answering, choosing, numbered
from tests.speakers_fakes import FakeSpeakerBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typer.testing import Result

Timed = tuple[str, float, float]

SAID: list[Timed] = [("Hello", 0.0, 0.4), ("there.", 0.5, 0.9), ("Bye.", 8.0, 8.4)]
# Parakeet hears "there." as "bear.", and three words in the hole xAI left.
HEARD: list[Timed] = [
    *(("Hello", 0.0, 0.4), ("bear.", 0.5, 0.9)),
    *(("we", 3.0, 3.3), ("lost", 4.0, 4.3), ("this", 5.0, 5.3), ("Bye.", 8.0, 8.4)),
]
FILLED = ["Hello", "there.", "we", "lost", "this", "Bye."]
FILL_LINES = [
    "scribe: cross-checking against Parakeet, run locally "
    "(about 45 s per audio hour; a first run downloads a ~1.2 GB model)",
    "scribe: filled 00:00:03.0-00:00:05.3 (3 words from Parakeet, no speaker)",
    "scribe: cross-check: 1 span filled with 3 words from Parakeet, 0 unresolved",
]
KEPT = "; --out keeps the filled words"
runner = CliRunner()


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    said: Sequence[Timed] = SAID,
    heard: Sequence[Timed] = HEARD,
    duration: float = 9.0,
) -> None:
    words = [{"text": text, "start": start, "end": end, "speaker": 0} for text, start, end in said]
    payload: dict[str, object] = {
        "text": " ".join(text for text, _, _ in said),
        "duration": duration,
        "words": words,
    }

    class Xai:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def transcribe(self, _path: Path, **_options: object) -> dict[str, object]:
            return payload

    def run_parakeet(
        argv: list[str], *, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        tokens = [
            {
                "text": f" {text}",
                "start": start,
                "end": end,
                "duration": end - start,
                "confidence": 0.9,
            }
            for text, start, end in heard
        ]
        timing = {"start": 0.0, "end": 0.0, "duration": 0.0, "confidence": 0.9}
        output = {"text": "", "sentences": [{"text": "", **timing, "tokens": tokens}]}
        workdir = Path(argv[argv.index("--output-dir") + 1])
        (workdir / "transcript.json").write_text(json.dumps(output), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    backend = ParakeetMlx(
        run=run_parakeet, which=lambda name: f"/opt/bin/{name}", host=lambda: ("Darwin", "arm64")
    )
    monkeypatch.setattr("scribe.cli.XaiStt", Xai)
    monkeypatch.setattr("scribe.cli.ParakeetMlx", lambda: backend)
    monkeypatch.setenv("XAI_API_KEY", "xai-key")


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: Callable[[str], str] | None = None,
    fail_when: Callable[[str], bool] | None = None,
) -> list[FakeSpeakerBackend]:
    made: list[FakeSpeakerBackend] = []

    def factory(*, model: str, disable_tools: bool) -> FakeSpeakerBackend:
        assert disable_tools
        backend = FakeSpeakerBackend(
            model=model,
            reply=answering(choosing({"bear.", "x100", "x4000"})) if reply is None else reply,
            fail_when=fail_when,
        )
        made.append(backend)
        return backend

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)
    return made


def _transcribe(tmp_path: Path, *args: str) -> Result:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"pretend this is audio")
    return runner.invoke(app, ["transcribe", str(clip), *args])


def _texts(path: Path) -> list[str]:
    return [word.text for word in Transcript.load(path).words]


def _pick_params(path: Path) -> list[str]:
    return [key for key in Transcript.load(path).engine.params if key.startswith("pick_")]


def test_the_default_run_takes_parakeets_reading_where_the_model_picks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    made = _patch(monkeypatch)

    result = _transcribe(tmp_path)

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    picked = Transcript.load(out)
    assert [(word.text, word.speaker) for word in picked.words] == [
        *(("Hello", 0), ("bear.", 0), ("we", None), ("lost", None), ("this", None), ("Bye.", 0))
    ]
    params = picked.engine.params
    assert params["fill_ranges"] == "[[3.0, 5.3]]"
    assert (params["pick_model"], params["pick_spots"], params["pick_to_reference"]) == (
        DEFAULT_PICK_MODEL,
        1,
        1,
    )
    assert params["pick_reference"] == "parakeet-mlx mlx-community/parakeet-tdt-0.6b-v3"
    assert result.stderr.splitlines()[1:] == [
        *FILL_LINES,
        "scribe: picked the reference's reading at 1 of 1 disputed spots (0 unsure) "
        f"with {DEFAULT_PICK_MODEL}, prompt {PICK_PROMPT_VERSION}",
    ]
    assert [system for system, _ in made[0].calls] == [system_prompt()]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        *("clip.parakeet.json", "clip.transcript.json", "clip.wav")
    ]


def _filled(tmp_path: Path) -> bytes:
    """The bytes the fill alone makes of xAI's words and the Parakeet transcript kept."""
    xai = tmp_path / "xai.json"
    assert _transcribe(tmp_path, "--no-cross-check", "--out", str(xai)).exit_code == 0
    filled, _ = fill_holes(Transcript.load(xai), Transcript.load(tmp_path / "clip.parakeet.json"))
    expected = tmp_path / "expected.json"
    filled.dump(expected)
    return expected.read_bytes()


def test_no_pick_builds_no_backend_and_writes_the_fill_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    made = _patch(monkeypatch)

    result = _transcribe(tmp_path, "--no-pick")

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    assert result.stderr.splitlines()[1:] == FILL_LINES
    assert out.read_bytes() == _filled(tmp_path)
    assert made == []


def _nowhere(_name: str) -> str | None:
    return None


class _NoClaude(FakeSpeakerBackend):
    """Fails to resolve as the real backend does with no claude on PATH."""

    def resolve(self) -> str:
        return ClaudeCliBackend(self.model, disable_tools=True, which=_nowhere).resolve()


def _no_claude(monkeypatch: pytest.MonkeyPatch) -> list[FakeSpeakerBackend]:
    made: list[FakeSpeakerBackend] = []

    def factory(*, model: str, disable_tools: bool) -> FakeSpeakerBackend:
        assert disable_tools
        made.append(_NoClaude(model=model))
        return made[-1]

    monkeypatch.setattr("scribe.cli.ClaudeCliBackend", factory)
    return made


def _every_chunk_fails(monkeypatch: pytest.MonkeyPatch) -> list[FakeSpeakerBackend]:
    return _patch(monkeypatch, fail_when=lambda _target: True)


def _rewrite_fails(monkeypatch: pytest.MonkeyPatch) -> list[FakeSpeakerBackend]:
    dump = Transcript.dump

    def refuse(self: Transcript, path: Path) -> None:
        if "pick_record" in self.engine.params:
            raise OSError("disk full")
        dump(self, path)

    monkeypatch.setattr(Transcript, "dump", refuse)
    return _patch(monkeypatch)


def _out_of_order(monkeypatch: pytest.MonkeyPatch) -> list[FakeSpeakerBackend]:
    _setup(monkeypatch, said=[SAID[1], SAID[0], SAID[2]])
    return _patch(monkeypatch)


@pytest.mark.parametrize(
    ("arrange", "message", "built", "called"),
    [
        (_no_claude, "not picking: claude is not on PATH", True, False),
        (_every_chunk_fails, "the pick failed on 1 of 1 chunks", True, True),
        (_rewrite_fails, "cannot write the picked transcript to ", True, True),
        (
            _out_of_order,
            "not picking: the filled transcript's word 1 starts before word 0",
            False,
            False,
        ),
    ],
    ids=["no-claude", "every-chunk-failed", "rewrite-fails", "out-of-order"],
)
def test_a_pick_that_fails_is_one_line_and_keeps_the_filled_words(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[pytest.MonkeyPatch], list[FakeSpeakerBackend]],
    message: str,
    built: bool,
    called: bool,
) -> None:
    _setup(monkeypatch)
    made = arrange(monkeypatch)

    result = _transcribe(tmp_path)

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    lines = result.stderr.splitlines()
    assert lines[-1].startswith(f"scribe: {message}"), lines
    assert lines[-1].endswith(KEPT)
    assert not [line for line in lines[1:-1] if "pick" in line], lines
    assert (len(made), bool(made and made[0].calls)) == (int(built), called)
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]
    assert out.read_bytes() == _filled(tmp_path)


def test_some_chunks_failing_prints_both_lines_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timed(words: list[Word]) -> list[Timed]:
        return [(word.text, word.start, word.end) for word in words]

    changed = {100: "x100", 4000: "x4000"}
    _setup(
        monkeypatch,
        said=timed(numbered(4500)),
        heard=timed(numbered(4500, changed)),
        duration=4500.0,
    )
    _patch(monkeypatch, fail_when=lambda target: "[#1 " in target)

    result = _transcribe(tmp_path)

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    assert result.stderr.splitlines()[-2:] == [
        "scribe: picked the reference's reading at 1 of 2 disputed spots (0 unsure) "
        f"with {DEFAULT_PICK_MODEL}, prompt {PICK_PROMPT_VERSION}",
        "scribe: the pick failed on 1 of 2 chunks (0); their 1 spots keep the transcript's words",
    ]
    words = _texts(out)
    assert (words[100], words[4000]) == ("w100", "x4000")


def test_no_spot_builds_no_backend_and_prints_nothing_of_the_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, heard=[SAID[0], SAID[1], *HEARD[2:]])
    made = _patch(monkeypatch)

    result = _transcribe(tmp_path, "--pick")

    assert result.exit_code == 0, result.output
    out = tmp_path / "clip.transcript.json"
    assert result.stdout == f"{out}\n"
    assert result.stderr.splitlines()[1:] == FILL_LINES
    assert _texts(out) == FILLED
    assert _pick_params(out) == []
    assert made == []


def _voted(_audio: Path, _source: object, destination: Path, *_args: object, **_kw: object) -> Path:
    return destination


def _refuse(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("reached the pick")


@pytest.mark.parametrize("flag", ["--vote", "--no-cross-check"])
def test_no_pick_runs_with_vote_or_without_the_cross_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    _setup(monkeypatch)
    made = _patch(monkeypatch)
    monkeypatch.setattr(ensemble, "transcribe_voted", _voted)
    monkeypatch.setattr("scribe.cli.find_spots", _refuse)
    monkeypatch.setattr("scribe.cli.pick_readings", _refuse)

    result = _transcribe(tmp_path, flag, "--pick")

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{tmp_path / 'clip.transcript.json'}\n"
    assert made == []


def test_pick_model_and_context_reach_the_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    made = _patch(monkeypatch)

    result = _transcribe(
        tmp_path, "--pick-model", "sonnet", "--pick-context", " People at this meeting: Ann Lee "
    )

    assert result.exit_code == 0, result.output
    assert "with sonnet, prompt" in result.stderr
    assert [system for system, _ in made[0].calls] == [
        system_prompt("People at this meeting: Ann Lee")
    ]
    params = Transcript.load(tmp_path / "clip.transcript.json").engine.params
    assert (params["pick_model"], params["pick_context_chars"]) == ("sonnet", 31)


def test_a_fill_that_cannot_be_written_is_not_picked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    made = _patch(monkeypatch)
    dump = Transcript.dump

    def refuse(self: Transcript, path: Path) -> None:
        if path.name.startswith(".clip.transcript.json"):
            raise OSError("disk full")
        dump(self, path)

    monkeypatch.setattr(Transcript, "dump", refuse)

    result = _transcribe(tmp_path, "--pick")

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines()[-1].endswith("disk full; it keeps xAI's words")
    assert _texts(tmp_path / "clip.transcript.json") == [text for text, _, _ in SAID]
    assert made == []


def test_the_help_says_what_the_pick_sends_and_how_to_turn_it_off() -> None:
    result = runner.invoke(app, ["transcribe", "--help"], terminal_width=200)

    assert result.exit_code == 0
    shown = " ".join(result.stdout.split())
    for needed in ["not the audio, go to Anthropic", "--no-pick", "--pick-model", "--pick-context"]:
        assert needed in shown, needed


class _NoUpload:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def transcribe(self, *_args: object, **_options: object) -> NoReturn:
        raise AssertionError("uploaded")


@pytest.mark.parametrize("flag", ["--no-pick", "--vote", "--no-cross-check"])
def test_pick_context_with_no_pick_to_take_it_exits_two_before_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    _setup(monkeypatch)
    monkeypatch.setattr("scribe.cli.XaiStt", _NoUpload)
    monkeypatch.setattr(ensemble, "transcribe_voted", _NoUpload("").transcribe)

    result = _transcribe(tmp_path, flag, "--pick-context", "People at this meeting: Ann Lee")

    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert result.stderr == (
        "scribe: --pick-context is background for the pick, which --no-pick, --vote "
        "and --no-cross-check each turn off\n"
    )
    assert [path.name for path in tmp_path.iterdir()] == ["clip.wav"]
