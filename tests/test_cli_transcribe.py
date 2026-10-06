from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from scribe.cli import app
from scribe.schema import Transcript
from scribe.xai_stt import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_VAD_THRESHOLD,
    MAX_UPLOAD_BYTES,
    RETRY_ATTEMPTS,
    XaiStt,
)
from tests.xai_fixtures import xai_payload

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import httpx

runner = CliRunner()


def _clip(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"pretend this is audio")
    return clip


def _stub_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, object] | None = None,
    during: Callable[[Path], None] | None = None,
) -> list[dict[str, object]]:
    """Replace the real client; return the list its calls are recorded into."""
    calls: list[dict[str, object]] = []
    body = xai_payload() if payload is None else payload

    class Stub:
        def __init__(
            self,
            api_key: str,
            *,
            base_url: str = DEFAULT_BASE_URL,
            transport: httpx.BaseTransport | None = None,
            timeout_seconds: float = 600,
            max_bytes: int = MAX_UPLOAD_BYTES,
            attempts: int = RETRY_ATTEMPTS,
            deadline_seconds: float | None = None,
            keep_alive: bool = False,
        ) -> None:
            # Only settings off their defaults are recorded, so a test expecting the
            # key alone also pins the CLI to the client's 5 attempts and 600 s.
            settings: dict[str, object] = {
                "timeout_seconds": timeout_seconds,
                "attempts": attempts,
                "deadline_seconds": deadline_seconds,
                "keep_alive": keep_alive,
            }
            defaults: dict[str, object] = {
                "timeout_seconds": 600,
                "attempts": RETRY_ATTEMPTS,
                "deadline_seconds": None,
                "keep_alive": False,
            }
            changed = {name: value for name, value in settings.items() if value != defaults[name]}
            calls.append({"api_key": api_key, **changed})

        def transcribe(
            self,
            path: Path,
            *,
            model: str = DEFAULT_MODEL,
            language: str | None = "en",
            format_text: bool = True,
            diarize: bool = True,
            keyterms: Sequence[str] = (),
            vad_threshold: float = DEFAULT_VAD_THRESHOLD,
        ) -> dict[str, object]:
            calls.append(
                {
                    "path": path,
                    "model": model,
                    "language": language,
                    "format_text": format_text,
                    "diarize": diarize,
                    "keyterms": list(keyterms),
                    "vad_threshold": vad_threshold,
                }
            )
            if during is not None:
                during(path)
            return body

    # Signatures, defaults included, so the stub cannot keep accepting a call
    # the real client has stopped accepting, or keep a default it has changed.
    assert inspect.signature(Stub.__init__) == inspect.signature(XaiStt.__init__)
    assert inspect.signature(Stub.transcribe) == inspect.signature(XaiStt.transcribe)
    monkeypatch.setattr("scribe.cli.XaiStt", Stub)
    monkeypatch.setenv("XAI_API_KEY", "key-from-the-environment")
    return calls


def test_it_writes_a_transcript_beside_the_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)

    result = runner.invoke(app, ["transcribe", str(clip), "--no-cross-check"])

    assert result.exit_code == 0, result.output
    written = tmp_path / "clip.transcript.json"
    assert result.output.strip().endswith(str(written))
    assert "11.8 s of audio" in result.output
    transcript = Transcript.load(written)
    assert transcript.text.startswith("In the beginning")
    assert len(transcript.words) == 29
    assert transcript.source.sha256 == hashlib.sha256(clip.read_bytes()).hexdigest()
    assert transcript.source.ref == str(clip)
    assert transcript.engine.model == "grok-voice-transcribe-2.0"
    assert transcript.engine.params == {
        "language": "en",
        "format": True,
        "diarize": True,
        "vad_threshold": 0.0,
    }
    assert calls[0] == {"api_key": "key-from-the-environment"}
    assert calls[1]["language"] == "en"
    assert calls[1]["format_text"] is True
    assert calls[1]["diarize"] is True
    assert calls[1]["keyterms"] == []
    assert calls[1]["vad_threshold"] == 0.0


def test_keyterms_from_flags_and_a_file_reach_the_client_and_the_engine_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)
    terms = tmp_path / "terms.txt"
    terms.write_text("# people\nAnn Lee\n\n  Acme Widget  \nClaude\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(clip),
            "--keyterm-file",
            str(terms),
            "--keyterm",
            "Claude",
            "--keyterm",
            "Fable",
            "--no-cross-check",
        ],
    )

    assert result.exit_code == 0, result.output
    # File first, then flags; a repeat is sent once.
    assert calls[1]["keyterms"] == ["Ann Lee", "Acme Widget", "Claude", "Fable"]
    params = Transcript.load(tmp_path / "clip.transcript.json").engine.params
    assert params["keyterms"] == '["Ann Lee", "Acme Widget", "Claude", "Fable"]'


def test_too_many_keyterms_exit_two_before_the_key_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY")
    clip = _clip(tmp_path)
    flags = [arg for index in range(101) for arg in ("--keyterm", f"term{index}")]

    result = runner.invoke(app, ["transcribe", str(clip), *flags])

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert "at most 100" in result.output
    assert calls == []
    assert list(tmp_path.glob("*.transcript.json")) == []


@pytest.mark.parametrize("threshold", ["-0.1", "1.5", "nan"])
def test_a_vad_threshold_outside_zero_to_one_exits_two_before_the_key_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: str
) -> None:
    calls = _stub_client(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY")

    result = runner.invoke(app, ["transcribe", str(_clip(tmp_path)), "--vad-threshold", threshold])

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert "from 0 to 1" in result.output
    assert calls == []


def test_the_help_says_which_keyterms_to_list() -> None:
    result = runner.invoke(app, ["transcribe", "--help"], terminal_width=200)

    assert result.exit_code == 0
    shown = " ".join(result.stdout.split())
    assert "list distinctive names and terms" in shown
    assert "can be forced in where a similar word was said" in shown


def test_a_keyterm_with_a_newline_exits_two_before_the_key_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    monkeypatch.delenv("XAI_API_KEY")
    clip = _clip(tmp_path)

    result = runner.invoke(app, ["transcribe", str(clip), "--keyterm", "Ann\nLee"])

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert "control character" in result.output
    assert calls == []


def test_a_byte_order_mark_is_not_part_of_the_first_keyterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)
    terms = tmp_path / "terms.txt"
    terms.write_text("\N{BYTE ORDER MARK}Ann Lee\nAcme\n", encoding="utf-8")

    result = runner.invoke(
        app, ["transcribe", str(clip), "--keyterm-file", str(terms), "--no-cross-check"]
    )

    assert result.exit_code == 0, result.output
    assert calls[1]["keyterms"] == ["Ann Lee", "Acme"]


def test_an_unreadable_keyterm_file_exits_two_before_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)

    result = runner.invoke(
        app, ["transcribe", str(clip), "--keyterm-file", str(tmp_path / "absent.txt")]
    )

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert "--keyterm-file" in result.output
    assert calls == []


def test_an_out_that_is_the_keyterm_file_is_refused_before_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)
    terms = tmp_path / "terms.txt"
    terms.write_text("Claude\n", encoding="utf-8")

    result = runner.invoke(
        app, ["transcribe", str(clip), "--keyterm-file", str(terms), "--out", str(terms)]
    )

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert calls == [{"api_key": "key-from-the-environment"}]
    assert terms.read_text(encoding="utf-8") == "Claude\n"


def test_the_flags_reach_the_client_and_the_engine_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)
    out = tmp_path / "named.json"

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(clip),
            "--out",
            str(out),
            "--language",
            "",
            "--no-format",
            "--no-diarize",
            "--model",
            "grok-voice-transcribe-1.0",
            "--vad-threshold",
            "0.35",
            "--no-cross-check",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[1]["language"] is None
    assert calls[1]["model"] == "grok-voice-transcribe-1.0"
    assert calls[1]["vad_threshold"] == 0.35
    transcript = Transcript.load(out)
    # No language was sent, so `format` could not have been either.
    assert transcript.engine.params == {
        "language": "",
        "format": False,
        "diarize": False,
        "vad_threshold": 0.35,
    }
    assert not (tmp_path / "clip.transcript.json").exists()


def test_turns_renders_what_transcribe_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_client(monkeypatch)
    clip = _clip(tmp_path)

    written = runner.invoke(app, ["transcribe", str(clip), "--no-cross-check"])
    assert written.exit_code == 0, written.output

    rendered = runner.invoke(app, ["turns", str(tmp_path / "clip.transcript.json"), "--stdout"])

    assert rendered.exit_code == 0, rendered.output
    assert "In the beginning" in rendered.stdout
    assert rendered.stdout.startswith("**Speaker 1** [00:00:00]")


def _one_line_failure(result_output: str) -> bool:
    return "Traceback" not in result_output and len(result_output.strip().splitlines()) == 1


def test_a_missing_key_is_one_stderr_line_and_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    clip = _clip(tmp_path)

    result = runner.invoke(app, ["transcribe", str(clip)])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert _one_line_failure(result.output)
    assert "XAI_API_KEY" in result.output
    assert list(tmp_path.glob("*.transcript.json")) == []


def test_a_missing_input_is_one_stderr_line_and_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real client, not the stub: it must reject the path before it would
    # need a network.
    monkeypatch.setenv("XAI_API_KEY", "unused-by-a-failing-path")

    result = runner.invoke(app, ["transcribe", str(tmp_path / "absent.wav")])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert _one_line_failure(result.output)
    assert "absent.wav" in result.output
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_out_exits_two_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.mkdir()

    result = runner.invoke(app, ["transcribe", str(clip), "--out", str(blocker)])

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert calls == [{"api_key": "key-from-the-environment"}]


def test_an_out_in_a_missing_directory_is_refused_before_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)

    result = runner.invoke(
        app, ["transcribe", str(clip), "--out", str(tmp_path / "typo" / "out.json")]
    )

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert calls == [{"api_key": "key-from-the-environment"}]
    assert [p.name for p in tmp_path.iterdir()] == ["clip.wav"]


def test_a_non_finite_number_in_the_response_exits_two_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An Infinity that reached the artifact would serialize as null and make
    # `scribe turns` reject the file this command reported as written.
    payload: dict[str, object] = {
        "text": "hi",
        "language": "en",
        "duration": 1.0,
        "words": [{"text": "hi", "start": float("inf"), "end": 2.0, "speaker": 0}],
    }
    _stub_client(monkeypatch, payload=payload)
    clip = _clip(tmp_path)

    result = runner.invoke(app, ["transcribe", str(clip)])

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert list(tmp_path.glob("*.transcript.json")) == []


def test_a_lone_surrogate_in_the_response_exits_two_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A lone surrogate is valid JSON on the way in and cannot be encoded as
    # UTF-8 on the way out, so it fails at the write rather than at the parse.
    payload: dict[str, object] = {"text": "a\ud800b", "language": "en", "duration": 1.0}
    _stub_client(monkeypatch, payload=payload)
    clip = _clip(tmp_path)

    result = runner.invoke(app, ["transcribe", str(clip)])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert _one_line_failure(result.output)
    assert list(tmp_path.glob("*.transcript.json")) == []


def test_a_clip_that_vanishes_during_the_upload_still_records_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(tmp_path)
    digest = hashlib.sha256(clip.read_bytes()).hexdigest()
    # Hashing before the request is what ties the recorded digest to the bytes
    # that were sent, however the file changes once the upload is under way.
    _stub_client(monkeypatch, during=Path.unlink)

    result = runner.invoke(app, ["transcribe", str(clip), "--no-cross-check"])

    assert result.exit_code == 0, result.output
    assert not clip.exists()
    assert Transcript.load(tmp_path / "clip.transcript.json").source.sha256 == digest


def test_an_out_that_is_the_audio_is_refused_before_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_client(monkeypatch)
    clip = _clip(tmp_path)

    result = runner.invoke(app, ["transcribe", str(clip), "--out", str(clip)])

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert calls == [{"api_key": "key-from-the-environment"}]
    assert clip.read_bytes() == b"pretend this is audio"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="the platform has no FIFOs")
def test_a_fifo_input_exits_two_before_the_client_is_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Opening a FIFO blocks until a writer arrives, so anything that reads the
    # input before the regular-file check hangs the command instead of failing.
    calls = _stub_client(monkeypatch)
    pipe = tmp_path / "pipe.wav"
    os.mkfifo(pipe)

    result = runner.invoke(app, ["transcribe", str(pipe)])

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert "not an audio file" in result.output
    assert calls == [{"api_key": "key-from-the-environment"}]


def test_a_clip_that_disappears_after_the_check_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every path the CLI can name is rejected before the hash, by the input
    # check or by the framework's own readable-path test, so the hash's guard
    # catches only a file that changes between the two. A check that waves the
    # path through stands in for losing that race.
    def passes(_path: Path, _max_bytes: int) -> None:
        return

    _stub_client(monkeypatch)
    monkeypatch.setattr("scribe.cli.check_input", passes)

    result = runner.invoke(app, ["transcribe", str(tmp_path / "absent.wav")])

    assert result.exit_code == 2
    assert _one_line_failure(result.output)
    assert "cannot read audio file" in result.output
