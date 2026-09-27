"""Diarize one recording and embed stretches of it with pyannote Community-1.

scribe runs this file through `uvx` in an interpreter of its own, where the
pinned pyannote.audio, torch, numpy and soundfile are installed; so it imports
nothing from scribe, whose dependencies are not there.

    python -P diarize_worker.py REQUEST.json OUT.json

REQUEST: {"audio": 16 kHz mono wav, "model", "revision", "intervals": [[start, end], ...]}.
OUT: {"versions", "device", "runtime_s", "exclusive": [{"start", "end", "speaker"}, ...],
"embeddings": [256 floats, or null where an interval cannot be embedded, per interval]}.
A failure it can name is one line on stderr starting with FAILED, and exit 1.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

# Before pyannote is imported, which otherwise reports each run to pyannoteAI.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

import numpy as np
import soundfile
import torch
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError
from pyannote.audio import Pipeline

FAILED = "scribe-diarize: "
_RATE = 16000
# The probes that measured the rule rounded the timeline to the millisecond.
_DIGITS = 3
# The Hub's answer to no token, a rejected one, or one whose account has not accepted the terms.
_REFUSED = (401, 403)


def _fail(cause: str) -> int:
    print(f"{FAILED}{' '.join(cause.split())}", file=sys.stderr, flush=True)
    return 1


def _load_failure(exc: HfHubHTTPError | LocalEntryNotFoundError, model: str, revision: str) -> str:
    where = f"cannot load {model} at {revision}"
    # With the file not cached, the Hub's refusal other than a 401 arrives as
    # LocalEntryNotFoundError raised from the Hub's answer.
    answer = exc if isinstance(exc, HfHubHTTPError) else exc.__cause__
    status = answer.response.status_code if isinstance(answer, HfHubHTTPError) else None
    if isinstance(exc, GatedRepoError) or status in _REFUSED:
        return (
            f"{where} from Hugging Face ({type(exc).__name__}); it is gated: HF_TOKEN, or a saved "
            f"`hf auth login`, must hold a token whose account accepted its terms at "
            f"https://hf.co/{model}"
        )
    if isinstance(exc, LocalEntryNotFoundError):
        return (
            f"{where}: it is not in the Hugging Face cache, and the Hub is unreachable "
            f"({type(exc).__name__})"
        )
    return f"{where} from Hugging Face: HTTP {status} ({type(exc).__name__})"


def _load(model: str, revision: str, device: torch.device) -> Pipeline:
    pipeline = Pipeline.from_pretrained(model, revision=revision)
    if pipeline is None:
        raise LookupError(f"{model}@{revision} has no pipeline config")
    return pipeline.to(device)


def _embed(pipeline: Pipeline, audio: np.ndarray, start: float, end: float) -> list[float] | None:
    embedding = pipeline._embedding
    first, stop = max(0, round(start * _RATE)), min(len(audio), round(end * _RATE))
    # Shorter input makes the model raise, or pyannote's own pipeline returns NaN for it.
    if stop - first < embedding.min_num_samples:
        return None
    waveform = torch.from_numpy(np.ascontiguousarray(audio[first:stop]))[None, None]
    vector = [float(value) for value in embedding(waveform)[0]]
    return vector if all(math.isfinite(value) for value in vector) else None


def main(argv: list[str]) -> int:
    """Run the request named by `argv[0]` and write its answer to `argv[1]`."""
    started = time.monotonic()
    request_path, out_path = argv
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    model, revision = request["model"], request["revision"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        pipeline = _load(model, revision, device)
    # pyannote prints its own multi-line advice to stdout; scribe shows this line instead.
    except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
        return _fail(_load_failure(exc, model, revision))
    audio, rate = soundfile.read(request["audio"], dtype="float32")
    if rate != _RATE or audio.ndim != 1:
        return _fail(f"{request['audio']} is not {_RATE} Hz mono")
    diarized = pipeline({"waveform": torch.from_numpy(audio)[None], "sample_rate": rate})
    exclusive = [
        {"start": round(turn.start, _DIGITS), "end": round(turn.end, _DIGITS), "speaker": label}
        for turn, _, label in diarized.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]
    embeddings = [_embed(pipeline, audio, start, end) for start, end in request["intervals"]]
    versions = {
        str(dist.metadata["Name"]): dist.version for dist in importlib.metadata.distributions()
    }
    answer = {
        "versions": {"python": platform.python_version(), **dict(sorted(versions.items()))},
        "device": device.type,
        "runtime_s": round(time.monotonic() - started, 1),
        "exclusive": exclusive,
        "embeddings": embeddings,
    }
    Path(out_path).write_text(json.dumps(answer), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        code = main(sys.argv[1:])
    # Whatever the cause, scribe reads it from this one line, not from a traceback's end.
    except Exception as exc:  # noqa: BLE001  # reported, then the exit is nonzero
        traceback.print_exc()
        code = _fail(f"{type(exc).__name__}: {exc}")
    sys.exit(code)
