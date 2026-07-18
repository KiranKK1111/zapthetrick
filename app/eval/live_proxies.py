"""
Label-free metric proxies (live-conversational-intelligence R27).

Estimates answer relevance and hallucination risk WITHOUT human labels, so the
live evaluation can be scored even on synthetic / unlabeled transcripts. Reuses
the `quality.critic` review where available and falls back to a deterministic
token-overlap heuristic. Dev/CI-only; no runtime effect; no provider keys.
"""
from __future__ import annotations

import re

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "it", "this", "that", "with", "as", "by", "be", "do", "does",
    "how", "what", "why", "when", "where", "who", "you", "your", "i", "we",
    "would", "can", "could", "should", "will", "about",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in _STOP}


def relevance_proxy(question: str, answer: str) -> float:
    """0..1 estimate that the answer addresses the question (token overlap of
    the question's content words covered by the answer). Never raises."""
    try:
        q = _tokens(question)
        if not q:
            return 0.0
        a = _tokens(answer)
        return round(len(q & a) / len(q), 3)
    except Exception:  # noqa: BLE001
        return 0.0


def hallucination_proxy(answer: str, context: str | None = None) -> float:
    """0..1 estimate of hallucination risk: the fraction of the answer's content
    words NOT grounded in the supplied context. No context → 0.5 (unknown).
    Lower is better. Reuses `quality.critic` when available. Never raises."""
    try:
        if not context or not context.strip():
            return 0.5
        a = _tokens(answer)
        if not a:
            return 0.0
        c = _tokens(context)
        ungrounded = len(a - c) / len(a)
        # Best-effort: let the critic refine if it exposes a contradiction check.
        try:
            from app.quality import critic as _critic  # type: ignore
            checker = getattr(_critic, "hallucination_risk", None)
            if callable(checker):
                return round(float(checker(answer, context)), 3)
        except Exception:  # noqa: BLE001
            pass
        return round(ungrounded, 3)
    except Exception:  # noqa: BLE001
        return 0.5


def proxies_over(samples: list[dict]) -> dict:
    """Average the proxies over [{question, answer, context?}] samples."""
    if not samples:
        return {"count": 0, "avg_relevance": 0.0, "avg_hallucination_risk": 0.0}
    rels, halls = [], []
    for s in samples:
        rels.append(relevance_proxy(s.get("question", ""), s.get("answer", "")))
        halls.append(hallucination_proxy(s.get("answer", ""), s.get("context")))
    n = len(samples)
    return {
        "count": n,
        "avg_relevance": round(sum(rels) / n, 3),
        "avg_hallucination_risk": round(sum(halls) / n, 3),
    }


__all__ = ["relevance_proxy", "hallucination_proxy", "proxies_over"]
