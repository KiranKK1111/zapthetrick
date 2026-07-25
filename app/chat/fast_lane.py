"""Trivial-turn fast lane (vNext §3.10).

A greeting / ack / bit of chitchat doesn't need the full agent mesh (understanding
pass, enrichment, tool selection, grounder). The fast lane routes such turns
around that machinery for a snappy fast-tier reply — the model already gets there
via the `trivial` difficulty (which weights SPEED); this just lets the turn skip
the *enrichment* it doesn't need.

**Semantic-first (standing directive):** the trivial decision is the
`trivial_turn` exemplar gate (AUTHORITY); the short hardcoded phrase list is only
the cold-start fallback when the embedder is unavailable. Lives in `chat`
(`chat → semantics` exists), so `api`/`routes_agents` (`api → chat`) can ask it.

Misclassification is cheap by design: a trivial turn still gets a correct answer
from the fast tier, and the difficulty classifier remains the backstop — so the
fast lane only ever *skips enrichment*, never the answer.
"""
from __future__ import annotations


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.chat, "fast_lane", False))
    except Exception:  # noqa: BLE001
        return False


def is_trivial(text: str) -> bool:
    """Whether this turn is trivial (greeting/ack/chitchat). SEMANTIC-first: the
    `trivial_turn` gate is the authority; a tiny phrase list is the cold-start
    fallback. Never raises."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower().strip(" \t\n!.?,;:")
    # Ultra-short (≤2 chars) is trivial regardless of the embedder (a structural
    # fact, not intent) — but a 3-4 char acronym like "DFS"/"OOP?" is NOT.
    if len(low) <= 2:
        return True
    try:
        from app.semantics import gates
        verdict = gates.matches("trivial_turn", low)
        if verdict is not None:
            return bool(verdict)          # embedder answered → trust it
    except Exception:  # noqa: BLE001
        pass
    # Cold-start fallback: the deterministic phrase list (shared with difficulty).
    try:
        from app.core.lexicons import DIFFICULTY_TRIVIAL_PHRASES as _phrases
        return low in _phrases
    except Exception:  # noqa: BLE001
        return False


__all__ = ["enabled", "is_trivial"]
