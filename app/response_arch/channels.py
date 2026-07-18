"""Multi-channel parallel streaming (roadmap Phase 6 #8).

Chat streams one multiplexed SSE stream, but every frame was implicitly the same
"channel" — there was no way to run an answer, its progress, a sandbox run, and
artifact deliveries as *independent* logical channels merged onto the one socket.
This module adds that: a :class:`ChannelMultiplexer` tags each frame with a
channel + a per-channel sequence number (so a client can render four lanes), and
:func:`merge` fans several independent async producers into one ordered SSE
stream, applying frame prioritization (#14) so the answer channel never waits
behind telemetry.

Channels:
    chat      the answer tokens / blocks
    progress  stage / tool / model / ack
    sandbox   code execution output
    artifact  progressive artifact deliveries

Fail-open: a producer that raises is dropped from the merge; the others continue.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from .priority import frame_priority

CHAT = "chat"
PROGRESS = "progress"
SANDBOX = "sandbox"
ARTIFACT = "artifact"

_VALID = frozenset({CHAT, PROGRESS, SANDBOX, ARTIFACT})


def _sse(event: str, data: dict) -> str:
    import json
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@dataclass
class ChannelMultiplexer:
    """Builds channel-tagged SSE frames with per-channel sequencing."""

    _seq: dict = field(default_factory=dict)

    def frame(self, channel: str, event: str, data: dict) -> str:
        """An SSE frame tagged with ``_ch`` (channel) + ``_seq`` (per-channel)."""
        ch = channel if channel in _VALID else CHAT
        n = self._seq.get(ch, 0) + 1
        self._seq[ch] = n
        payload = dict(data or {})
        payload["_ch"] = ch
        payload["_seq"] = n
        return _sse(event, payload)

    def channels_seen(self) -> dict:
        return dict(self._seq)


async def merge(
    sources: dict[str, AsyncIterator[tuple[str, dict]]],
    *,
    prioritize: bool = True,
) -> AsyncIterator[str]:
    """Fan several independent producers into one prioritized SSE stream.

    ``sources`` maps a channel name → an async iterator yielding
    ``(event, data)`` pairs. Frames are emitted as they arrive; when several are
    ready in the same tick they are released answer-first (frame prioritization,
    #14). A producer that raises is dropped; the rest keep streaming.
    """
    mux = ChannelMultiplexer()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    async def _pump(channel: str, ait: AsyncIterator[tuple[str, dict]]):
        try:
            async for event, data in ait:
                await queue.put((channel, event, data))
        except Exception:  # noqa: BLE001 — one bad producer never kills the merge
            pass
        finally:
            await queue.put((_SENTINEL, channel, None))

    tasks = [asyncio.create_task(_pump(ch, ait)) for ch, ait in sources.items()]
    remaining = len(tasks)
    try:
        while remaining > 0:
            item = await queue.get()
            if item[0] is _SENTINEL:
                remaining -= 1
                continue
            # Opportunistically batch whatever else is already queued, then
            # release the batch answer-first.
            batch = [item]
            while not queue.empty():
                nxt = queue.get_nowait()
                if nxt[0] is _SENTINEL:
                    remaining -= 1
                else:
                    batch.append(nxt)
            if prioritize and len(batch) > 1:
                batch.sort(key=lambda t: frame_priority(t[1]))
            for channel, event, data in batch:
                yield mux.frame(channel, event, data)
    finally:
        for t in tasks:
            t.cancel()


__all__ = ["ChannelMultiplexer", "merge", "CHAT", "PROGRESS", "SANDBOX",
           "ARTIFACT"]
