"""
Live-pipeline event bus.

The live path used to hand results between stages with direct function
calls buried in routes_ws.py, which made the pipeline impossible to
observe, extend, or cancel as a unit. This bus gives every session a tiny
in-process async pub/sub spine:

    UTTERANCE_FINALIZED   raw STT text arrived (post-repair)
    PARTIAL_TRANSCRIPT    interim streaming-STT text (speaker still talking)
    QUESTION_DETECTED     the detector committed a question (qid assigned)
    QUESTION_SKIPPED      decision engine declined to answer (with reason)
    ANSWER_STARTED        generation began for a qid
    ANSWER_DONE           generation finished for a qid
    ANSWER_VERIFIED       post-answer verifier scored a qid
    ANSWER_CANCELLED      an in-flight answer was cancelled (superseded)
    TOPIC_CHANGED         topic tracker observed drift

Design notes:
- Per-session instance (one `LiveEventBus` per WebSocket connection), so
  events never leak across interviews and teardown is trivial.
- `publish` is fire-and-forget: subscriber exceptions are logged, never
  propagated — a broken observer must not break the interview.
- Subscribers are async callables `(event: LiveBusEvent) -> None`.
- Every event is also appended to the session's replayable event log
  (app/live/eventlog.py) when one is attached, giving the debugging /
  replay trail the architecture calls for.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import time
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

# Canonical event kinds (string constants, not an Enum, so ad-hoc kinds can
# be published during experiments without schema churn).
UTTERANCE_FINALIZED = "UTTERANCE_FINALIZED"
PARTIAL_TRANSCRIPT = "PARTIAL_TRANSCRIPT"
QUESTION_DETECTED = "QUESTION_DETECTED"
QUESTION_SKIPPED = "QUESTION_SKIPPED"
ANSWER_STARTED = "ANSWER_STARTED"
ANSWER_DONE = "ANSWER_DONE"
ANSWER_VERIFIED = "ANSWER_VERIFIED"
ANSWER_CANCELLED = "ANSWER_CANCELLED"
TOPIC_CHANGED = "TOPIC_CHANGED"


@dataclass
class LiveBusEvent:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time)


Subscriber = Callable[[LiveBusEvent], Awaitable[None]]


class LiveEventBus:
    """Per-session async pub/sub with answer-task cancellation support."""

    def __init__(self, event_log=None) -> None:
        # kind -> subscribers; "*" subscribers receive every event.
        self._subs: dict[str, list[Subscriber]] = {}
        self._event_log = event_log
        # qid -> in-flight answer task, so a supersede/interruption can
        # cancel generation that no longer matters.
        self._answer_tasks: dict[str, asyncio.Task] = {}

    # ── pub/sub ────────────────────────────────────────────────────────────
    def subscribe(self, kind: str, fn: Subscriber) -> None:
        self._subs.setdefault(kind, []).append(fn)

    def publish(self, kind: str, **data: Any) -> LiveBusEvent:
        """Publish fire-and-forget. Returns the event (for tests/log)."""
        event = LiveBusEvent(kind=kind, data=data)
        if self._event_log is not None:
            try:
                self._event_log.append(kind, data)
            except Exception:  # noqa: BLE001 — logging must never break flow
                pass
        for fn in self._subs.get(kind, []) + self._subs.get("*", []):
            task = asyncio.create_task(self._safe_call(fn, event))
            # Keep a reference until done so tasks aren't GC'd mid-flight.
            task.add_done_callback(lambda _t: None)
        return event

    @staticmethod
    async def _safe_call(fn: Subscriber, event: LiveBusEvent) -> None:
        try:
            await fn(event)
        except Exception:  # noqa: BLE001
            log.exception("live bus subscriber failed for %s", event.kind)

    # ── answer-task registry (cancellation) ────────────────────────────────
    def register_answer_task(self, qid: str, task: asyncio.Task) -> None:
        self._answer_tasks[qid] = task
        task.add_done_callback(lambda _t: self._answer_tasks.pop(qid, None))

    def cancel_answer(self, qid: str, reason: str = "superseded") -> bool:
        """Cancel one in-flight answer. Publishes ANSWER_CANCELLED."""
        task = self._answer_tasks.get(qid)
        if task is None or task.done():
            return False
        task.cancel()
        self.publish(ANSWER_CANCELLED, qid=qid, reason=reason)
        return True

    def cancel_all_answers(self, reason: str = "session teardown") -> int:
        """Cancel every in-flight answer (topic hard-switch / disconnect)."""
        n = 0
        for qid in list(self._answer_tasks):
            if self.cancel_answer(qid, reason=reason):
                n += 1
        return n
