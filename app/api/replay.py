"""In-memory SSE replay buffer for reconnect/resume (Architecture §15).

Once bytes are on the wire a dropped socket loses whatever was mid-flight —
tokens the user never saw, or even the terminal `done`. The fix: every streamed
turn gets a `stream_id`; each SSE frame it emits is tagged with a monotonic event
id and teed into a bounded, TTL'd ring buffer keyed by that id. If the socket
drops, the client reconnects to the replay endpoint with the last id it saw and
receives every frame after it — nothing already produced is lost. The client
de-dupes by id.

Bounded three ways so a long session can't leak memory: frames-per-stream (a
ring), a stream TTL, and a max number of live streams (LRU eviction). Everything
runs on the single asyncio event loop, so no locking is needed; ops are plain
dict mutations between awaits.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict


def new_stream_id() -> str:
    return uuid.uuid4().hex


class ReplayBuffer:
    def __init__(
        self,
        *,
        max_frames_per_stream: int = 512,
        ttl_seconds: float = 900.0,
        max_streams: int = 256,
    ):
        self._max_frames = max_frames_per_stream
        self._ttl = ttl_seconds
        self._max_streams = max_streams
        # stream_id -> {"ts": float, "frames": list[(event_id, frame)]}
        self._streams: "OrderedDict[str, dict]" = OrderedDict()

    def _now(self) -> float:
        return time.monotonic()

    def _evict(self) -> None:
        now = self._now()
        # drop expired
        for sid in [s for s, e in self._streams.items()
                    if now - e["ts"] > self._ttl]:
            self._streams.pop(sid, None)
        # cap live streams (oldest first)
        while len(self._streams) > self._max_streams:
            self._streams.popitem(last=False)

    def append(self, stream_id: str, event_id: int, frame: str) -> None:
        """Record one emitted SSE frame under its stream."""
        entry = self._streams.get(stream_id)
        if entry is None:
            entry = {"ts": self._now(), "frames": []}
            self._streams[stream_id] = entry
        entry["ts"] = self._now()
        frames = entry["frames"]
        frames.append((event_id, frame))
        # ring: keep only the most recent N frames
        if len(frames) > self._max_frames:
            del frames[: len(frames) - self._max_frames]
        self._streams.move_to_end(stream_id)
        self._evict()

    def since(self, stream_id: str, after_id: int) -> list[tuple[int, str]] | None:
        """Frames with event_id > after_id for this stream, in order.

        Returns None when the stream is unknown/expired (client should fall back
        to reloading the conversation); an empty list when it's known but has
        nothing newer than `after_id`.
        """
        entry = self._streams.get(stream_id)
        if entry is None:
            return None
        return [(eid, f) for (eid, f) in entry["frames"] if eid > after_id]

    def drop(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)

    def stats(self) -> dict:
        return {
            "streams": len(self._streams),
            "frames": sum(len(e["frames"]) for e in self._streams.values()),
        }


# Process-wide singleton used by the stream route + replay endpoint.
buffer = ReplayBuffer()


# ── Explicit stream cancellation ────────────────────────────────────────────
# The FE's `http` package cannot abort an in-flight request (cancelling a Dart
# StreamSubscription does NOT close the socket), so a client-side Stop can't rely
# on the server noticing a disconnect. Instead Stop POSTs a cancel for the
# conversation; the streaming generators poll `is_cancelled(...)` between steps
# and tear down immediately (cancel the sandbox verify, close the LLM stream,
# save the partial). Keyed by conversation id; a small TTL'd set, single event
# loop so no locking.
_CANCEL_TTL = 120.0
_cancels: "OrderedDict[str, float]" = OrderedDict()


def request_cancel(key: str) -> None:
    """Flag a conversation's active stream to stop at the next poll."""
    if not key:
        return
    now = time.monotonic()
    _cancels[key] = now + _CANCEL_TTL
    # opportunistic sweep of expired flags
    for k in [k for k, exp in _cancels.items() if exp < now]:
        _cancels.pop(k, None)


def is_cancelled(key: str) -> bool:
    """True if a Stop was requested for this conversation (and not yet cleared)."""
    if not key:
        return False
    exp = _cancels.get(key)
    if exp is None:
        return False
    if exp < time.monotonic():
        _cancels.pop(key, None)
        return False
    return True


def clear_cancel(key: str) -> None:
    """Drop a conversation's cancel flag (call when a fresh turn starts/ends)."""
    if key:
        _cancels.pop(key, None)


__all__ = [
    "ReplayBuffer", "buffer", "new_stream_id",
    "request_cancel", "is_cancelled", "clear_cancel",
]
