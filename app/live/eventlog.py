"""
Replayable event log (live-conversational-intelligence R14).

An append-only, bounded per-session record of the typed events and state
transitions, used for debugging and the offline evaluation harness (Phase 5).
In-process; no DB. Fail-open — logging never breaks the live turn.
"""
from __future__ import annotations

from collections import deque
from time import time


class EventLog:
    """Bounded append-only log of {ts, type, data} entries for one session."""

    def __init__(self, maxlen: int = 500) -> None:
        self._events: deque = deque(maxlen=maxlen)

    def append(self, etype: str, data: dict | None = None) -> None:
        try:
            self._events.append({"ts": time(), "type": str(etype), "data": dict(data or {})})
        except Exception:  # noqa: BLE001
            pass

    def events(self) -> list[dict]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)


# ---- per-session registry (in-process; no DB) -------------------------
_logs: dict[str, EventLog] = {}


def get_log(session_id: str) -> EventLog:
    log = _logs.get(session_id)
    if log is None:
        log = EventLog()
        _logs[session_id] = log
    return log


def forget_session(session_id: str) -> None:
    _logs.pop(session_id, None)
