"""
Contradiction + temporal reasoning (live-conversational-intelligence R33).

`is_challenge` recognizes when the interviewer challenges an earlier statement
or a recorded assumption ("but you said…", "does Kafka ALWAYS guarantee
ordering?") so it is handled as a challenge rather than a new topic.
`resolve_temporal` resolves a reference to an earlier point in time ("earlier
you said…", "back to partitions") against the topic graph / event log.
Deterministic + fail-open.
"""
from __future__ import annotations

from app.core import lexicons

_CHALLENGE_CUES = lexicons.LIVE_CONTRADICTION_CHALLENGE_CUES
_TEMPORAL_CUES = lexicons.LIVE_CONTRADICTION_TEMPORAL_CUES


def is_challenge(turn: str, world_model=None, event_log=None) -> bool:
    """True when the turn challenges an earlier statement/assumption. Never
    raises."""
    try:
        t = (turn or "").lower()
        if not t.strip():
            return False
        if any(c in t for c in _CHALLENGE_CUES):
            return True
        # A re-questioned absolute over a recorded assumption is a challenge
        # ("does it ALWAYS …?" after we assumed it does).
        if world_model is not None:
            assumptions = getattr(world_model, "assumptions", []) or []
            if assumptions and ("always" in t or "never" in t) and "?" in t:
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def resolve_temporal(turn: str, topic_graph=None, event_log=None) -> str | None:
    """Resolve a temporal/earlier-topic reference to a topic name (via the topic
    graph) when present; else None. Never raises."""
    try:
        t = (turn or "").lower()
        if not t.strip() or not any(c in t for c in _TEMPORAL_CUES):
            return None
        if topic_graph is not None:
            ref = topic_graph.resolve_reference(turn)
            if ref:
                return ref
        return None
    except Exception:  # noqa: BLE001
        return None
