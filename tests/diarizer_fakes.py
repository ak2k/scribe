"""ffmpeg and the pyannote worker, faked in-process, shared by the diarizer tests."""

from __future__ import annotations

import errno
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

from scribe.diarizer import DIMENSION

if TYPE_CHECKING:
    from collections.abc import Callable

UVX = "/opt/nowhere/bin/uvx"
FFMPEG = "/opt/nowhere/bin/ffmpeg"
TOKEN = "hf_" + "x" * 34


def answer(intervals: list[list[float]], **changes: object) -> dict[str, object]:
    """A well-formed OUT for `intervals`, with any top-level key replaced."""
    body: dict[str, object] = {
        "versions": {"python": "3.13.15", "pyannote-audio": "4.0.7"},
        "device": "mps",
        "runtime_s": 1.5,
        "exclusive": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        "embeddings": [[0.5] * DIMENSION for _ in intervals],
    }
    return {**body, **changes}


class FakeWorker:
    """Stands in for ffmpeg and the uvx worker: writes what each argv names as its output."""

    def __init__(
        self,
        reply: Callable[[list[list[float]]], dict[str, object] | str | None] = answer,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        error: Exception | None = None,
        decode_returncode: int = 0,
    ) -> None:
        self.reply, self.returncode, self.stdout, self.stderr = reply, returncode, stdout, stderr
        self.error, self.decode_returncode = error, decode_returncode
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.timeouts: list[float] = []
        self.requests: list[bytes] = []
        self.workers: list[str] = []

    def run(
        self, argv: list[str], *, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        self.envs.append(env)
        self.timeouts.append(timeout)
        if argv[0] == FFMPEG:
            Path(argv[-1]).write_bytes(b"RIFF")
            return subprocess.CompletedProcess(argv, self.decode_returncode, "", "bad mp3")
        if self.error is not None:
            raise self.error
        self.workers.append(Path(argv[-3]).read_text(encoding="utf-8"))
        self.requests.append(Path(argv[-2]).read_bytes())
        request = cast("dict[str, list[list[float]]]", json.loads(self.requests[-1]))
        body = self.reply(request["intervals"])
        if body is not None:
            text = body if isinstance(body, str) else json.dumps(body)
            Path(argv[-1]).write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)


def found(name: str) -> str | None:
    return {"uvx": UVX, "ffmpeg": FFMPEG}.get(name)


def full_disk(*_args: object, **_kwargs: object) -> NoReturn:
    raise OSError(errno.ENOSPC, "No space left on device")
