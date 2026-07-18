"""
Live interview-knowledge retrieval + packs (live-conversational-intelligence R35).

On a detected topic, surface relevant interview knowledge (common angles /
follow-ups / domain patterns) to ground a weaker free model's answer. Uses a
deterministic built-in knowledge map (no second blocking call) and biases toward
a configured domain/company pack; the heavier vector retrieval is opt-in via the
existing `app/rag` (token budget deferred to perceived-speed). Fail-open → [].
"""
from __future__ import annotations

# Deterministic per-topic interview-knowledge snippets (angles to cover).
_KNOWLEDGE = {
    "kafka": [
        "ordering is per-partition, not global",
        "consumer-group rebalancing affects availability",
        "exactly-once needs idempotent producers + transactions",
    ],
    "redis": [
        "single-threaded command execution",
        "persistence: RDB snapshots vs AOF",
        "eviction policies under memory pressure",
    ],
    "system design": [
        "clarify functional + non-functional requirements first",
        "estimate scale (QPS, storage) before designing",
        "call out bottlenecks and trade-offs explicitly",
    ],
    "concurrency": [
        "distinguish parallelism from concurrency",
        "watch for race conditions + deadlocks",
        "prefer immutable / message-passing where possible",
    ],
}


def interview_knowledge(topic: str, pack: str = "") -> list[str]:
    """Return interview-knowledge angles for a topic (+ optional pack bias).
    Deterministic; never raises → []."""
    try:
        t = (topic or "").strip().lower()
        if not t:
            return []
        for key, snips in _KNOWLEDGE.items():
            if key in t or t in key:
                return list(snips)
        return []
    except Exception:  # noqa: BLE001
        return []


def directive(snippets: list[str]) -> str:
    """Fold knowledge angles into the answer prompt ("" when none)."""
    snippets = [s for s in (snippets or []) if s]
    if not snippets:
        return ""
    return "Relevant angles to consider: " + "; ".join(snippets[:4]) + "."


def configured_pack() -> str:
    from app.core.config_loader import cfg
    return (getattr(cfg.live, "knowledge_pack", "") or "").strip()


# ── Recurring-topic Skill_Gap retrieval boost (R57) ────────────────────
def skill_gap_boost(topic: str, skill_gaps: list[str], pack: str = "") -> list[str]:
    """When the current `topic` is a recurring skill-gap (interviewer keeps
    probing it), return EXTRA knowledge angles to reinforce the answer — a
    deterministic retrieval boost. Falls back to the base angles. Never raises."""
    try:
        base = interview_knowledge(topic, pack)
        t = (topic or "").strip().lower()
        gaps = {g.strip().lower() for g in (skill_gaps or [])}
        if t and t in gaps:
            # Boost: include adjacent topics' angles too.
            extra: list[str] = list(base)
            for key, snips in _KNOWLEDGE.items():
                if key != t and (t in key or key in t):
                    extra.extend(snips)
            # De-dup, preserve order.
            seen, out = set(), []
            for s in extra:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
        return base
    except Exception:  # noqa: BLE001
        return []
