"""`scribe recheck`: a vote per engine on each fill and unresolved span, written as JSON."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import stamina
from typer.testing import CliRunner

from scribe import recheck
from scribe.cli import app
from scribe.ear import LocalEars
from scribe.schema import Engine, Source, Transcript
from scribe.xai_stt import DEFAULT_MODEL, XaiStt

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from typer.testing import Result

runner = CliRunner()


@pytest.fixture(autouse=True)
def instant_retries() -> Iterator[None]:
    with stamina.set_testing(True, attempts=5, cap=True):
        yield


@pytest.fixture(autouse=True)
def refused_ears(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ears as a host they cannot run on has them, so no test spawns a recognizer."""
    monkeypatch.setattr("scribe.cli.LocalEars", lambda: LocalEars(host=lambda: ("Linux", "x86_64")))


class Calls:
    """What the fakes were asked to do."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.cuts: list[list[str]] = []


def _fakes(
    monkeypatch: pytest.MonkeyPatch, *, status: int = 200, ffmpeg: bool = True
) -> Calls:
    calls = Calls()

    def answer(request: httpx.Request) -> httpx.Response:
        request.read()
        calls.requests.append(request)
        if status != 200:
            return httpx.Response(status, text="no")
        words = [{"text": "one", "start": 1.0, "end": 1.5}]
        return httpx.Response(200, json={"text": "one", "duration": 9.0, "words": words})

    def cut(argv: list[str], **_settings: object) -> subprocess.CompletedProcess[str]:
        calls.cuts.append(argv)
        Path(argv[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    located = "/usr/bin/ffmpeg" if ffmpeg else None
    monkeypatch.setattr(
        "scribe.cli.XaiStt", partial(XaiStt, transport=httpx.MockTransport(answer))
    )
    monkeypatch.setattr(
        "scribe.cli.recheck_spans",
        partial(recheck.recheck, run=cut, which=lambda _name: located),
    )
    monkeypatch.setenv("XAI_API_KEY", "key-from-the-environment")
    return calls


def _meeting(
    tmp_path: Path,
    *,
    fills: Sequence[tuple[float, float]] = ((10.0, 12.0),),
    unresolved: Sequence[tuple[float, float]] = ((60.0, 90.0),),
    model: str = DEFAULT_MODEL,
) -> tuple[Path, Path]:
    audio = tmp_path / "recording.mp3"
    audio.write_bytes(b"pretend this is audio")
    params: dict[str, float | str] = {
        "fill_reference": "parakeet-cpp model.bin",
        "fill_ranges": json.dumps([list(span) for span in fills]),
        "fill_unresolved_ranges": json.dumps([list(span) for span in unresolved]),
    }
    transcript = Transcript(
        source=Source(
            kind="audio", ref=str(audio), sha256=hashlib.sha256(audio.read_bytes()).hexdigest()
        ),
        engine=Engine(name="xai-stt", model=model, params=params),
        duration=120.0,
        text="",
        words=[],
    )
    path = tmp_path / "meeting.transcript.json"
    transcript.dump(path)
    return path, audio


def _invoke(transcript: Path, audio: Path, out: Path) -> Result:
    argv = ["recheck", str(transcript), "--audio", str(audio), "--out", str(out)]
    return runner.invoke(app, argv)


def test_it_writes_one_vote_per_engine_per_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fakes(monkeypatch)
    transcript, audio = _meeting(tmp_path)
    out = tmp_path / "meeting.recheck.json"

    result = _invoke(transcript, audio, out)

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["transcript_sha256"] == hashlib.sha256(transcript.read_bytes()).hexdigest()
    assert written["source_sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert written["model"] == DEFAULT_MODEL
    assert [span["kind"] for span in written["spans"]] == ["fill", "unresolved"]
    assert [span["proposed_by"] for span in written["spans"]] == ["parakeet-cpp model.bin"] * 2
    statuses = [[vote["status"] for vote in span["votes"]] for span in written["spans"]]
    assert statuses == [["heard", "not run", "not run"]] * 2
    assert written["spans"][0]["votes"][0]["words"] == [
        {"text": "one", "start": 6.0, "end": 6.5}
    ]
    assert len(calls.requests) == 2
    assert "2 spans" in result.stderr


def test_span_failure_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fakes(monkeypatch, status=500)
    transcript, audio = _meeting(tmp_path)
    out = tmp_path / "meeting.recheck.json"

    result = _invoke(transcript, audio, out)

    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [span["votes"][0]["status"] for span in written["spans"]] == ["failed", "failed"]
    assert "500" in written["spans"][0]["votes"][0]["reason"]


def test_no_spans_writes_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fakes(monkeypatch)
    transcript, audio = _meeting(tmp_path, fills=(), unresolved=())
    out = tmp_path / "meeting.recheck.json"

    result = _invoke(transcript, audio, out)

    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))["spans"] == []
    assert calls.requests == []
    assert calls.cuts == []


def test_refuses_bad_inputs_no_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fakes(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY")
    transcript, audio = _meeting(tmp_path)
    out = tmp_path / "meeting.recheck.json"

    result = _invoke(transcript, audio, out)

    assert result.exit_code == 2
    assert "XAI_API_KEY" in result.stderr
    assert calls.requests == []
    assert not out.exists()


def test_refuses_bad_inputs_no_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fakes(monkeypatch, ffmpeg=False)
    transcript, audio = _meeting(tmp_path)
    out = tmp_path / "meeting.recheck.json"

    result = _invoke(transcript, audio, out)

    assert result.exit_code == 2
    assert "ffmpeg" in result.stderr
    assert calls.requests == []
    assert not out.exists()


@pytest.mark.parametrize("which", ["transcript", "audio"])
def test_refuses_bad_inputs_out_is_an_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    calls = _fakes(monkeypatch)
    transcript, audio = _meeting(tmp_path)
    before = (transcript.read_bytes(), audio.read_bytes())

    result = _invoke(transcript, audio, transcript if which == "transcript" else audio)

    assert result.exit_code == 2
    assert calls.requests == []
    assert (transcript.read_bytes(), audio.read_bytes()) == before


def test_refuses_a_transcript_of_another_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fakes(monkeypatch)
    transcript, audio = _meeting(tmp_path, model="some-other-model")
    out = tmp_path / "meeting.recheck.json"

    result = _invoke(transcript, audio, out)

    assert result.exit_code == 2
    assert "some-other-model" in result.stderr
    assert calls.requests == []
    assert not out.exists()


def test_refuses_audio_that_is_not_the_transcripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fakes(monkeypatch)
    transcript, _ = _meeting(tmp_path)
    other = tmp_path / "other.mp3"
    other.write_bytes(b"some other audio")
    out = tmp_path / "meeting.recheck.json"

    result = _invoke(transcript, other, out)

    assert result.exit_code == 2
    assert "is not the audio" in result.stderr
    assert calls.requests == []
    assert not out.exists()
