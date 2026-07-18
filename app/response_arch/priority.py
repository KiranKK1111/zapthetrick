"""SSE frame prioritization — answer > progress > telemetry (Phase 6 #14).

When frames contend for the wire (a slow client, a burst of telemetry while the
answer is streaming), the *answer* must win. This module assigns every SSE event
a priority class and provides a small, order-stable buffer that releases
higher-priority frames first while never starving lower ones (FIFO within a
class). It is intentionally tiny and synchronous — a drop-in the streaming loop
(or the channel multiplexer) can wrap around a batch of ready frames.

Priority classes (lower = more urgent):
    0  answer      token / block / artifact / clarify / error / done
    1  structure   plan / meta / route / intent / stage / interpretation
    2  progress    tool / model / doc_pending / ack
    3  telemetry   aggregate_confidence / trace / analytics / keepalive
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

P_ANSWER = 0
P_STRUCTURE = 1
P_PROGRESS = 2
P_TELEMETRY = 3

_PRIORITY: dict[str, int] = {
    # answer channel — never delayed
    "token": P_ANSWER, "block": P_ANSWER, "artifact": P_ANSWER,
    "artifacts": P_ANSWER, "clarify": P_ANSWER, "error": P_ANSWER,
    "done": P_ANSWER,
    # structure — the pre/post scaffolding
    "plan": P_STRUCTURE, "meta": P_STRUCTURE, "route": P_STRUCTURE,
    "intent": P_STRUCTURE, "stage": P_STRUCTURE,
    "interpretation": P_STRUCTURE,
    # progress
    "tool": P_PROGRESS, "model": P_PROGRESS, "doc_pending": P_PROGRESS,
    "ack": P_PROGRESS,
    # telemetry
    "aggregate_confidence": P_TELEMETRY, "trace": P_TELEMETRY,
    "analytics": P_TELEMETRY, "keepalive": P_TELEMETRY,
}

# The terminal frame is urgent but must never precede queued answer content.
_DEFAULT = P_PROGRESS


def frame_priority(event: str) -> int:
    """Priority class for an SSE ``event`` name (unknown → progress)."""
    return _PRIORITY.get((event or "").strip(), _DEFAULT)


def sort_frames(frames: list[tuple[str, object]]) -> list[tuple[str, object]]:
    """Stable-sort ``(event, payload)`` pairs by priority (urgent first).

    Order within a class is preserved (FIFO), so nothing starves.
    """
    idx = itertools.count()
    return [f for _, _, f in sorted(
        ((frame_priority(ev), next(idx), (ev, pl)) for ev, pl in frames),
        key=lambda t: (t[0], t[1]))]


@dataclass
class PriorityBuffer:
    """A tiny contention buffer: enqueue frames, drain them urgent-first.

    ``done`` is special-cased to always drain last so it can never overtake
    still-queued answer content.
    """

    _seq: "itertools.count" = field(default_factory=lambda: itertools.count())
    _items: list = field(default_factory=list)

    def push(self, event: str, payload: object) -> None:
        self._items.append((frame_priority(event), next(self._seq),
                            event, payload))

    def __len__(self) -> int:
        return len(self._items)

    def drain(self) -> list[tuple[str, object]]:
        """Return all queued frames urgent-first, FIFO within a class, then clear.

        Any ``done`` frame is forced to the very end regardless of class.
        """
        def _key(it):
            prio, seq, ev, _ = it
            is_done = 1 if ev == "done" else 0
            return (is_done, prio, seq)

        ordered = sorted(self._items, key=_key)
        self._items = []
        return [(ev, pl) for _, _, ev, pl in ordered]


__all__ = [
    "frame_priority", "sort_frames", "PriorityBuffer",
    "P_ANSWER", "P_STRUCTURE", "P_PROGRESS", "P_TELEMETRY",
]
