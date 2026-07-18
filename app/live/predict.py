"""
Question prediction + speculative precompute (live-conversational-intelligence R29).

From the active topic (Topic_Graph) + world model, `predict_next` produces a
ranked set of likely next questions / follow-ups (the "likely follow-ups"
copilot surface). `should_precompute` gates speculative pre-warming on the
perceived-speed speculation budget; the actual pre-warm + cache-serve reuse the
perceived-speed answer-cache (not a new mechanism) and NEVER delay a real
answer. Deterministic + fail-open.
"""
from __future__ import annotations

# Common follow-up sub-topics per well-known topic (deterministic seed).
_FOLLOWUPS = {
    "kafka": ["partitions", "consumer groups", "offsets", "rebalancing", "exactly-once delivery"],
    "redis": ["persistence", "eviction policies", "clustering", "pub/sub", "pipelining"],
    "kubernetes": ["pods", "services", "ingress", "autoscaling", "rolling updates"],
    "postgres": ["indexing", "transactions", "replication", "connection pooling", "vacuum"],
    "react": ["hooks", "context", "reconciliation", "memoization", "suspense"],
}
# Generic templates applied to any topic.
_TEMPLATES = [
    "How does {t} scale?",
    "What are the trade-offs of {t}?",
    "How would you debug a problem with {t}?",
    "What happens when {t} fails?",
]


def predict_next(topic_graph=None, world_model=None, max_n: int = 5) -> list[str]:
    """Ranked likely next questions for the current topic. Never raises → []."""
    try:
        topic = ""
        if world_model is not None and getattr(world_model, "topic", ""):
            topic = world_model.topic
        elif topic_graph is not None and topic_graph.current():
            topic = topic_graph.current()
        topic = (topic or "").strip().lower()
        if not topic:
            return []
        out: list[str] = []
        # Known sub-topics first (most likely follow-ups).
        for key, subs in _FOLLOWUPS.items():
            if key in topic or topic in key:
                out.extend(f"Tell me about {topic} {s}." for s in subs)
                break
        # Then generic templates.
        out.extend(t.format(t=topic) for t in _TEMPLATES)
        # De-dup, cap.
        seen, ranked = set(), []
        for q in out:
            if q not in seen:
                seen.add(q)
                ranked.append(q)
            if len(ranked) >= max_n:
                break
        return ranked
    except Exception:  # noqa: BLE001
        return []


# ── Answer pre-drafting for predicted follow-ups (Phase 2 #10 / 2B-10) ──────
# Speculative PRE-DRAFTING: for the top predicted follow-ups, precompute a
# deterministic answer SCAFFOLD (structure + the angles to hit) and stash it per
# session. When one of those predicted questions is actually asked next, the
# stashed scaffold is injected as a directive so the real answer starts already
# structured — no extra LLM call, no first-token latency (the drafting runs
# POST-answer, off the hot path). Deterministic + fail-open.
import threading as _threading

_LOCK = _threading.RLock()
_PREDRAFTS: dict[str, dict[str, str]] = {}   # sid -> {normalized_q: outline}
_MAX_PER_SESSION = 12


def _norm(q: str) -> str:
    import re as _re
    return " ".join(_re.findall(r"[a-z0-9]+", (q or "").lower()))


def _tokens(q: str) -> set[str]:
    return set(_norm(q).split())


def predraft_outline(question: str) -> str:
    """A deterministic answer scaffold for a (predicted) question. Never raises."""
    try:
        low = (question or "").lower()
        if any(w in low for w in ("tell me about a time", "describe a", "conflict",
                                  "disagree", "challenge you faced")):
            return ("Structure: Situation → Task → Action → Result; lead with the "
                    "outcome and quantify the impact.")
        if any(w in low for w in ("design", "scale", "architecture", "build a system")):
            return ("Structure: clarify requirements → high-level components → data "
                    "flow → bottleneck & trade-off → how it scales.")
        if any(w in low for w in ("trade-off", "tradeoff", "vs", "versus",
                                  "difference between")):
            return ("Structure: define both sides → axes of comparison → when to "
                    "pick each → your default and why.")
        if low.strip().startswith(("what is", "what are", "define", "explain")):
            return ("Structure: one-line definition → why it matters → a concrete "
                    "example → one caveat/edge case.")
        return ("Structure: direct answer first → one supporting detail or example "
                "→ a brief caveat.")
    except Exception:  # noqa: BLE001
        return ""


def predraft(session_id: str, questions: list[str], *, limit: int = 3) -> list[dict]:
    """Pre-draft scaffolds for the top predicted questions and stash them for the
    session. Returns [{question, outline}]. Never raises → []."""
    out: list[dict] = []
    try:
        sid = session_id or ""
        with _LOCK:
            store = _PREDRAFTS.setdefault(sid, {})
            for q in (questions or [])[:limit]:
                outline = predraft_outline(q)
                if not outline:
                    continue
                store[_norm(q)] = outline
                out.append({"question": q, "outline": outline})
            # Bound memory.
            while len(store) > _MAX_PER_SESSION:
                store.pop(next(iter(store)), None)
        return out
    except Exception:  # noqa: BLE001
        return out


def match_predraft(session_id: str, question: str, *, threshold: float = 0.6) -> str | None:
    """If the incoming question matches a stashed pre-draft (token Jaccard ≥
    threshold), return its outline. Never raises → None."""
    try:
        store = _PREDRAFTS.get(session_id or "")
        if not store:
            return None
        qt = _tokens(question)
        if not qt:
            return None
        best, best_sim = None, 0.0
        for norm_q, outline in store.items():
            other = set(norm_q.split())
            if not other:
                continue
            sim = len(qt & other) / len(qt | other)
            if sim > best_sim:
                best, best_sim = outline, sim
        return best if best_sim >= threshold else None
    except Exception:  # noqa: BLE001
        return None


def consume_directive(session_id: str, question: str) -> str:
    """A directive seeding the answer from a matching pre-draft. '' when there's
    no match. Never raises."""
    try:
        outline = match_predraft(session_id, question)
        if not outline:
            return ""
        return "A likely follow-up was anticipated. " + outline
    except Exception:  # noqa: BLE001
        return ""


def forget_session(session_id: str) -> None:
    with _LOCK:
        _PREDRAFTS.pop(session_id or "", None)


def should_precompute(budget=None) -> bool:
    """Whether to speculatively pre-warm predicted answers: requires the
    perceived-speed speculation kill-switch on AND budget headroom. Best-effort;
    never raises → False."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.perceived, "speculation_enabled", False)):
            return False
        if budget is not None and not budget.can_start():
            return False
        return True
    except Exception:  # noqa: BLE001
        return False
