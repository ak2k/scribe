"""Opt-in live check against the real API. Costs a few seconds of audio."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from scribe.schema import Engine, Source, from_xai_response
from scribe.xai_stt import XaiStt, resolve_api_key

if TYPE_CHECKING:
    from pathlib import Path

SENTENCE = (
    "A pelican flew over the harbor at sunrise while the morning ferry waited. "
    "This clip exists only to check that transcription works end to end."
)
DISTINCTIVE = "pelican"

pytestmark = pytest.mark.skipif(
    os.environ.get("SCRIBE_LIVE") != "1",
    reason="set SCRIBE_LIVE=1 to spend a real API call",
)


def test_a_generated_clip_comes_back_transcribed(tmp_path: Path) -> None:
    say = shutil.which("say")
    if say is None:
        pytest.skip("macOS `say` is needed to generate the clip")
    clip = tmp_path / "clip.wav"
    subprocess.run(  # noqa: S603  # fixed argv, no shell, interpreter from shutil.which
        [say, "-o", str(clip), "--data-format=LEI16@16000", SENTENCE],
        check=True,
    )

    payload = XaiStt(resolve_api_key()).transcribe(clip)
    transcript = from_xai_response(
        payload,
        source=Source(kind="audio", ref=str(clip)),
        engine=Engine(name="xai-stt"),
    )

    assert transcript.duration is not None
    assert transcript.duration > 3
    assert DISTINCTIVE in transcript.text.lower()
    assert transcript.words
