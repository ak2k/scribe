"""The recorded xAI reply, shared by the client and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

FIXTURES = Path(__file__).parent / "fixtures"


def xai_payload() -> dict[str, object]:
    """Return the recorded `/stt` reply for the say clip, decoded."""
    decoded: object = json.loads(  # pyright: ignore[reportAny]  # json.loads is Any
        (FIXTURES / "xai_say_clip.json").read_text(encoding="utf-8")
    )
    assert isinstance(decoded, dict)
    return cast("dict[str, object]", decoded)
