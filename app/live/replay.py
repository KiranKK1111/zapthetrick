"""
Session_Replay from the Event_Log (live-conversational-intelligence R45).

Builds a read-only, chronologically ordered replay of a session from the
existing append-only `Event_Log` (app/live/eventlog.py) — no new schema, no new
persistence. Used by a dev/read-only route to step through what the live module
saw and decided. Fail-open → empty replay when nothing was logged.
"""
from __future__ import annotations

from app.live import eventlog as _eventlog


def build_replay(session_id: str) -> dict:
    """Return a read-only replay {session_id, count, steps:[...]} from the event
    log. Steps are ordered by timestamp. Never raises."""
    try:
        log = _eventlog.get_log(session_id)
        events = sorted(log.events(), key=lambda e: e.get("ts", 0.0))
        t0 = events[0]["ts"] if events else 0.0
        steps = []
        for i, e in enumerate(events):
            steps.append({
                "index": i,
                "offset": round(e.get("ts", 0.0) - t0, 3),
                "type": e.get("type", ""),
                "data": e.get("data", {}),
            })
        return {"session_id": session_id, "count": len(steps), "steps": steps}
    except Exception:  # noqa: BLE001
        return {"session_id": session_id, "count": 0, "steps": []}


def summary(session_id: str) -> dict:
    """Coarse counts-by-type summary of a session replay. Never raises."""
    try:
        replay = build_replay(session_id)
        by_type: dict[str, int] = {}
        for step in replay["steps"]:
            by_type[step["type"]] = by_type.get(step["type"], 0) + 1
        return {"session_id": session_id, "count": replay["count"], "by_type": by_type}
    except Exception:  # noqa: BLE001
        return {"session_id": session_id, "count": 0, "by_type": {}}
