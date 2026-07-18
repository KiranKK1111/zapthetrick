"""The async typed shared state — the "blackboard" agents read and write.

Designed for a single session (one user turn). Each slot has an
[asyncio.Event] that fires when first written, so consumers can `await
board.wait_for(KEY_PLAN)` without polling. Subsequent writes don't
re-fire the event but do publish a [BlackboardEvent] to the events
stream that the UI consumes via SSE.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class BlackboardEvent:
    """Every write produces one of these; the UI surfaces them as tool-chips."""
    key: str
    value: Any
    agent: str
    ts_ms: int


class Blackboard:
    """Async typed shared state for one session.

    Not thread-safe; not meant to be. Single asyncio event loop only.
    """

    def __init__(self) -> None:
        self._slots: dict[str, Any] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._stream: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
        self._closed = False

    # ---- read / write -------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self._slots.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._slots

    def write(self, key: str, value: Any, *, agent: str = "system") -> None:
        """Set a slot value and publish a [BlackboardEvent].

        Multiple writes to the same key overwrite the previous value.
        The first write fires the [asyncio.Event] for any waiters.
        """
        self._slots[key] = value
        evt = self._events.setdefault(key, asyncio.Event())
        evt.set()
        if not self._closed:
            self._stream.put_nowait(
                BlackboardEvent(
                    key=key,
                    value=value,
                    agent=agent,
                    ts_ms=int(time.time() * 1000),
                )
            )

    async def wait_for(self, key: str, *, timeout_s: float | None = None) -> Any:
        """Await the first write to `key`, then return its current value."""
        evt = self._events.setdefault(key, asyncio.Event())
        if timeout_s is None:
            await evt.wait()
        else:
            try:
                await asyncio.wait_for(evt.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return None
        return self._slots.get(key)

    # ---- event stream ------------------------------------------------
    async def events(self) -> AsyncIterator[BlackboardEvent]:
        """Yield every blackboard write as a [BlackboardEvent].

        Consumed by the SSE/WS layer to drive the UI's tool-chips.
        """
        while not self._closed:
            try:
                yield await asyncio.wait_for(self._stream.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

    def drain_pending(self) -> list[BlackboardEvent]:
        """Return every queued event without blocking. Used by the
        supervisor to interleave tool-chip events with persona tokens
        without spinning up a second consumer task."""
        out: list[BlackboardEvent] = []
        while True:
            try:
                out.append(self._stream.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    def close(self) -> None:
        """Stop the events stream. Call once the turn is done."""
        self._closed = True

    # ---- introspection ----------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Best-effort dump of every slot — used by debug endpoints."""
        return dict(self._slots)
