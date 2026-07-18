"""Live session-state persistence (crash/restart resilience).

The live module's conversational context — recent Q+A turns, the role-tagged
conversation log, and the interview world model — lives in-process. A backend
restart mid-interview therefore used to lose the whole conversation context
even though the client reconnects with the same session id.

This module snapshots that state into `Session.metadata['live_state']` after
each answered question (background, best-effort) and restores it on reconnect
when the in-process tracker is empty. Bounded: only the turn window the
tracker itself keeps (no embeddings — a restored turn simply can't vote on
follow-up similarity until a fresh embedding is computed, which is fail-open).

Gated by `cfg.live.session_resume`; deterministic; never raises.
"""
from __future__ import annotations

import logging
import uuid as _uuid

log = logging.getLogger("zapthetrick.live")

_KEY = "live_state"
_MAX_TURNS = 20
_MAX_LOG_ENTRIES = 40
_MAX_TEXT = 2000


def _build_snapshot(sid: str) -> dict | None:
    """Serialize the session's in-process live state to a compact dict."""
    from app.question_detection.context_tracker import get_tracker
    tracker = get_tracker(sid)
    turns = [
        {
            "question": (t.question or "")[:_MAX_TEXT],
            "answer": (t.answer or "")[:_MAX_TEXT],
            "topic": t.topic or "",
            "qtype": t.qtype or "unknown",
            "timestamp": float(t.timestamp),
        }
        for t in list(getattr(tracker, "_turns", []))[-_MAX_TURNS:]
    ]
    snap: dict = {"turns": turns}
    try:
        from app.live import conversation as _conv
        snap["log"] = [e.to_dict()
                       for e in _conv.for_tracker(tracker).entries()][-_MAX_LOG_ENTRIES:]
    except Exception:  # noqa: BLE001 — each block is independent, best-effort
        pass
    try:
        from app.live import world_model as _wm
        model = _wm.for_tracker(tracker)
        snap["world"] = {
            **model.snapshot(),
            "assumptions": list(model.assumptions)[-12:],
            "constraints": list(model.constraints)[-12:],
        }
    except Exception:  # noqa: BLE001
        pass
    # Interview memory graph: the topic TREE survives restarts too, so
    # follow-ups and "let's go back to X" still resolve after a reconnect.
    try:
        from app.live import topic_graph as _tg
        g = _tg.for_tracker(tracker)
        snap["topic_graph"] = {
            "nodes": [
                {"name": n.name, "parent": n.parent,
                 "turns": list(n.turns)[-8:], "last_seen": float(n.last_seen)}
                for n in list(g._nodes.values())[-40:]
            ],
            "current": g.current(),
            "previous": g.previous(),
        }
    except Exception:  # noqa: BLE001
        pass
    if not turns and not snap.get("log"):
        return None
    return snap


async def save_state(sid: str) -> bool:
    """Persist the session's live state snapshot. Best-effort; never raises."""
    try:
        snap = _build_snapshot(sid)
        if snap is None:
            return False
        from storage.db import get_session_factory
        from storage.models import Session as _SessionRow
        f = get_session_factory()
        if f is None:
            return False
        async with f() as ws:
            row = await ws.get(_SessionRow, _uuid.UUID(str(sid)))
            if row is None:
                return False
            meta = dict(getattr(row, "session_metadata", None) or {})
            meta[_KEY] = snap
            row.session_metadata = meta   # reassign so SQLAlchemy flags it
            await ws.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        log.info("live state save failed for %s: %s", sid, exc)
        return False


async def restore_state(sid: str) -> int:
    """Rebuild the in-process live state from the persisted snapshot when the
    tracker is EMPTY (fresh process). Returns the number of restored turns
    (0 = nothing restored). Never raises."""
    try:
        from app.question_detection.context_tracker import Turn, get_tracker
        tracker = get_tracker(sid)
        if getattr(tracker, "_turns", None):
            return 0    # live in-process state wins; never overwrite it
        from storage.db import get_session_factory
        from storage.models import Session as _SessionRow
        f = get_session_factory()
        if f is None:
            return 0
        async with f() as ws:
            row = await ws.get(_SessionRow, _uuid.UUID(str(sid)))
            meta = getattr(row, "session_metadata", None) if row else None
            snap = (meta or {}).get(_KEY) if isinstance(meta, dict) else None
        if not isinstance(snap, dict):
            return 0
        restored = 0
        for t in (snap.get("turns") or [])[-_MAX_TURNS:]:
            if not isinstance(t, dict) or not str(t.get("question", "")).strip():
                continue
            # No embedding — follow-up similarity simply can't match this turn
            # (fail-open) until new turns arrive with fresh embeddings.
            tracker._turns.append(Turn(
                question=str(t["question"]),
                answer=str(t.get("answer") or ""),
                topic=str(t.get("topic") or ""),
                qtype=str(t.get("qtype") or "unknown"),
                timestamp=float(t.get("timestamp") or 0.0),
            ))
            restored += 1
        try:
            from app.live import conversation as _conv
            clog = _conv.for_tracker(tracker)
            for e in (snap.get("log") or [])[-_MAX_LOG_ENTRIES:]:
                if isinstance(e, dict) and str(e.get("text", "")).strip():
                    clog.add(str(e.get("role") or "interviewer"),
                             str(e["text"]), str(e.get("topic") or ""))
        except Exception:  # noqa: BLE001
            pass
        try:
            from app.live import world_model as _wm
            world = snap.get("world")
            if isinstance(world, dict):
                model = _wm.for_tracker(tracker)
                model.topic = str(world.get("topic") or "")
                model.subtopic = str(world.get("subtopic") or "")
                for a in (world.get("assumptions") or []):
                    model.add_assumption(str(a))
                for c in (world.get("constraints") or []):
                    model.add_constraint(str(c))
        except Exception:  # noqa: BLE001
            pass
        try:
            from app.live import topic_graph as _tg
            from app.live.topic_graph import TopicNode
            tgs = snap.get("topic_graph")
            if isinstance(tgs, dict):
                g = _tg.for_tracker(tracker)
                for nd in (tgs.get("nodes") or []):
                    name = str((nd or {}).get("name") or "").strip()
                    if not name:
                        continue
                    node = TopicNode(
                        name=name,
                        parent=(str(nd["parent"]) if nd.get("parent") else None),
                    )
                    node.turns = [int(x) for x in (nd.get("turns") or [])]
                    node.last_seen = float(nd.get("last_seen") or 0.0)
                    g._nodes[name] = node
                g._current = (str(tgs["current"]) if tgs.get("current") else None)
                g._previous = (str(tgs["previous"]) if tgs.get("previous") else None)
        except Exception:  # noqa: BLE001
            pass
        if restored:
            log.info("live state restored for %s (%d turns)", sid, restored)
        return restored
    except Exception as exc:  # noqa: BLE001
        log.info("live state restore failed for %s: %s", sid, exc)
        return 0


__all__ = ["save_state", "restore_state"]
