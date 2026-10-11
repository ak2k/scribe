"""A fake herdr socket, answering `pane.current` as herdr does."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

if TYPE_CHECKING:
    from pathlib import Path

    import anyio.abc

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
    """herdr's answer for a pane running `agent` in `session`, every other field holding SENTINEL."""
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

    def __init__(self, path: Path, reply: bytes | None) -> None:
        """Listen at `path` once served."""
        self.path = path
        self.reply = reply
        self.asked: list[object] = []

    async def serve(self, *, task_status: anyio.abc.TaskStatus[None]) -> None:
        """Answer connections until cancelled."""
        async with await anyio.create_unix_listener(self.path) as listener:
            task_status.started()
            await listener.serve(self._answer)

    async def _answer(self, stream: anyio.abc.SocketStream) -> None:
        async with stream:
            line = await BufferedByteReceiveStream(stream).receive_until(b"\n", 1 << 16)
            self.asked.append(json.loads(line))
            if self.reply is None:
                await anyio.sleep_forever()
            await stream.send(self.reply)
