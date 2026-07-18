"""
In-session state validation + recovery (live-conversational-intelligence R36).

Periodically checks the Interview_World_Model for a context gap (e.g. dropped
audio left the active question stale / inconsistent with the topic) and, when one
is detected, rebuilds the active state from the rolling Session_Summary rather
than from a corrupt partial transcript. Distinct from transport reconnect (R23)
— this operates within a connected session. Deterministic + fail-open.
"""
from __future__ import annotations

import re


def detect_gap(world_model, recent_questions: list[str] | None = None) -> bool:
    """True when the live state looks inconsistent (a context gap). Never
    raises."""
    try:
        if world_model is None:
            return False
        topic = (getattr(world_model, "topic", "") or "").strip()
        active = (getattr(world_model, "active_question", "") or "").strip()
        # A topic is set but there's no active question, or we have recent
        # questions yet the active question is empty → a gap.
        if topic and not active:
            return True
        if recent_questions and not active:
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


def recover(world_model, summary: str) -> bool:
    """Rebuild the active topic from the Session_Summary when a gap is detected.
    Returns True if anything was recovered. Never raises."""
    try:
        if world_model is None or not (summary or "").strip():
            return False
        # "Topics covered: a, b, c" → adopt the most recent as the active topic.
        m = re.search(r"topics covered:\s*([^.]+)", summary, re.IGNORECASE)
        if m:
            topics = [t.strip() for t in m.group(1).split(",") if t.strip()]
            if topics and not (getattr(world_model, "topic", "") or "").strip():
                world_model.topic = topics[-1]
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def validate_and_recover(world_model, summary: str = "",
                         recent_questions: list[str] | None = None) -> tuple[bool, bool]:
    """Returns (gap_detected, recovered). Never raises."""
    gap = detect_gap(world_model, recent_questions)
    if not gap:
        return False, False
    return True, recover(world_model, summary)


def should_validate(turn_count: int, interval: int = 5) -> bool:
    """Periodic trigger: validate every `interval` turns."""
    try:
        return interval > 0 and turn_count > 0 and turn_count % interval == 0
    except Exception:  # noqa: BLE001
        return False
