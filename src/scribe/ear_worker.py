"""Transcribe clips of one recording with one pinned local recognizer.

scribe runs this file through `uvx` in an interpreter of its own, where the
pinned transformers, torch, numpy and librosa are installed; so it imports
nothing from scribe, whose dependencies are not there.

    python -P ear_worker.py REQUEST.json OUT.json

REQUEST: {"audio": raw 16 kHz mono float32 samples, "name", "model", "revision",
"intervals": [[start, end], ...] in seconds, "max_new_tokens"}.
OUT: {"versions", "device", "dtype", "runtime_s", "texts": [one per interval]}.
A failure is one line on stderr starting with FAILED, and exit 1.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    Qwen3ASRForConditionalGeneration,
)

FAILED = "scribe-ear: "
_RATE = 16000
# The vote was measured in bfloat16; float32 changed 45 of 164 of Qwen's texts.
_DTYPE = torch.bfloat16


def _fail(cause: str) -> int:
    print(f"{FAILED}{' '.join(cause.split())}", file=sys.stderr, flush=True)
    return 1


def _cohere(processor: Any, model: Any, clip: np.ndarray, max_new_tokens: int) -> str:
    inputs = processor(clip, sampling_rate=_RATE, return_tensors="pt", language="en")
    # A clip longer than the model's window is split; the decode joins the pieces.
    chunks = inputs.pop("audio_chunk_index", None)
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    if chunks is not None and any(chunk is not None for _, chunk in chunks):
        text = processor.decode(
            out, skip_special_tokens=True, audio_chunk_index=chunks, language="en"
        )[0]
    else:
        text = processor.decode(out, skip_special_tokens=True)[0]
    return text.strip()


def _qwen(processor: Any, model: Any, clip: np.ndarray, max_new_tokens: int) -> str:
    inputs = processor.apply_transcription_request(audio=clip, language="English")
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # The answer follows the prompt in the output; only the answer is decoded.
    answer = out[:, inputs["input_ids"].shape[1] :]
    return processor.decode(answer, return_format="transcription_only")[0]


_MODELS = {
    "cohere-transcribe": (CohereAsrForConditionalGeneration, _cohere),
    "qwen3-asr": (Qwen3ASRForConditionalGeneration, _qwen),
}


def main(argv: list[str]) -> int:
    """Run the request named by `argv[0]` and write its answer to `argv[1]`."""
    started = time.monotonic()
    request_path, out_path = argv
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    # On the CPU torch would still answer, with words the vote was not measured on.
    if not torch.backends.mps.is_available():
        return _fail("torch cannot use the Metal GPU (mps) here")
    model_class, transcribe = _MODELS[request["name"]]
    pinned = {"revision": request["revision"]}
    processor = AutoProcessor.from_pretrained(request["model"], **pinned)
    model = model_class.from_pretrained(request["model"], **pinned, dtype=_DTYPE)
    model = model.to("mps").eval()
    audio = np.fromfile(request["audio"], dtype="<f4")
    # One call per clip, never merged: the audio around a spot changes its words.
    clips = [audio[max(0, round(a * _RATE)) : round(b * _RATE)] for a, b in request["intervals"]]
    texts = [transcribe(processor, model, clip, request["max_new_tokens"]) for clip in clips]
    versions = {
        str(dist.metadata["Name"]): dist.version for dist in importlib.metadata.distributions()
    }
    answer = {
        "versions": {"python": platform.python_version(), **dict(sorted(versions.items()))},
        "device": model.device.type,
        "dtype": str(model.dtype).removeprefix("torch."),
        "runtime_s": round(time.monotonic() - started, 1),
        "texts": texts,
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
