"""A fake herdr socket, answering `pane.current` as herdr does."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import anyio
import anyio.abc
from anyio.streams.buffered import BufferedByteReceiveStream

if TYPE_CHECKING:
    from pathlib import Path

SENTINEL = "zq-sentinel-7f3a"


def pane_reply(
    session: str | None = "mine",
    *,
    focused: object = True,
    agent: str | None = "claude",
    source: str = "herdr:claude",
    kind: str = "id",
    kind_of_result: str = "pane_current",
) -> bytes:
    """The herdr answer for a pane running `agent` in `session`; other fields hold SENTINEL."""
    pane: dict[str, object] = {
        "pane_id": f"{SENTINEL}-pane",
        "focused": focused,
        "agent": agent,
        "agent_status": "idle",
        "cwd": f"/{SENTINEL}-cwd",
        "foreground_cwd": f"/{SENTINEL}-cwd",
        "terminal_title": f"{SENTINEL} title",
        "revision": 7,
        "scroll": {"offset": 0},
    }
    if session is not None:
        pane["agent_session"] = {"agent": agent, "kind": kind, "source": source, "value": session}
    reply = {"id": "scribe", "result": {"type": kind_of_result, "pane": pane}}
    return json.dumps(reply).encode() + b"\n"


class Herdr:
    """A fake herdr socket: answers each connection with `reply`, or never when it is None."""

    def __init__(self, path: Path, reply: bytes | None, *, hold: bool = False) -> None:
        """Listen at `path` once served; with `hold`, keep each connection open after `reply`."""
        self.path = path
        self.reply = reply
        self.hold = hold
        self.asked: list[object] = []

    async def serve(self, *, task_status: anyio.abc.TaskStatus[None]) -> None:
        """Answer connections until cancelled."""
        async with await anyio.create_unix_listener(self.path) as listener:
            task_status.started()
            await listener.serve(self._answer)

    async def _answer(self, stream: anyio.abc.SocketStream) -> None:
        async with stream:
            line = await BufferedByteReceiveStream(stream).receive_until(b"\n", 1 << 16)
            self.asked.append(cast("object", json.loads(line)))
            reply = self.reply
            if reply is None:
                await anyio.sleep_forever()
                return
            # A client past its cap or its size bound hangs up mid-reply, as it may.
            with suppress(anyio.BrokenResourceError):
                await stream.send(reply)
            if self.hold:
                await anyio.sleep_forever()
