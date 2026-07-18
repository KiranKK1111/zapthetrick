"""Relevance-ranked memory retrieval (memory-graph R3).

`relevant(query, store, workspace_id, k, threshold)` ranks scope-filtered
Memory_Objects by a Relevance_Score that blends semantic similarity (the existing
embedder), recency, and importance, returns the top-k above threshold, and hands
that ranked set to the caller (which forwards it to the perceived-speed
Context_Budget — this module never allocates tokens, R3.3).

Embedder/scorer unavailable → similarity falls back to token overlap so recall
still works (R3.4, Property 3). One-hop graph traversal optionally augments the
result for entity queries (R7). No second blocking LLM call (R8.3).
"""
from __future__ import annotations

import math
import re
import time

# Relevance blend weights (sum ~1.0).
_W_SIM = 0.6
_W_REC = 0.2
_W_IMP = 0.2
# Recency half-life (seconds) — independent of the lifecycle aging half-life.
_REC_HALFLIFE_S = 14 * 24 * 3600.0

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 1}


def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _overlap(q_tokens: set[str], text: str) -> float:
    t = _tokens(text)
    if not q_tokens or not t:
        return 0.0
    return len(q_tokens & t) / float(len(q_tokens | t))


def _recency(updated_at: float, now: float) -> float:
    age = max(0.0, now - float(updated_at or now))
    return math.pow(0.5, age / _REC_HALFLIFE_S)


def _embedder():
    try:
        from app.rag.embedder import embed_one
        return embed_one
    except Exception:  # noqa: BLE001
        return None


def relevant(query: str, store, workspace_id: str | None = None, *,
             k: int | None = None, threshold: float | None = None,
             embed_fn=None, traverse: bool = True) -> list:
    """Return up to `k` Memory_Objects above `threshold`, ranked by relevance.
    Never raises; on any error returns []."""
    try:
        return _relevant(query, store, workspace_id, k, threshold, embed_fn,
                         traverse)
    except Exception:  # noqa: BLE001
        return []


def _relevant(query, store, workspace_id, k, threshold, embed_fn, traverse):
    from app.memory.mstore import retrieval_scopes

    if k is None or threshold is None:
        try:
            from app.core.config_loader import cfg
            k = k or int(getattr(cfg.memory, "retrieval_k", 6))
            threshold = (threshold if threshold is not None
                         else float(getattr(cfg.memory, "relevance_threshold", 0.35)))
        except Exception:  # noqa: BLE001
            k, threshold = k or 6, threshold if threshold is not None else 0.35

    candidates = store.by_scope(retrieval_scopes(workspace_id))
    if not candidates:
        return []

    embed_fn = embed_fn or _embedder()
    q_vec = None
    if embed_fn is not None:
        try:
            q_vec = embed_fn(query)
        except Exception:  # noqa: BLE001 — similarity-only fallback (R3.4)
            q_vec = None
    q_tokens = _tokens(query)
    now = time.time()

    scored = []
    for o in candidates:
        if q_vec is not None and getattr(o, "embedding", None):
            sim = _cosine(q_vec, o.embedding)
        else:
            sim = _overlap(q_tokens, o.content)
        score = (_W_SIM * sim
                 + _W_REC * _recency(o.updated_at, now)
                 + _W_IMP * float(o.importance))
        if score >= threshold:
            scored.append((score, o))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [o for _s, o in scored[:k]]

    # One-hop graph traversal: include objects directly related to the hits
    # (R7.2). No edges → no change (R7.3).
    if traverse and top:
        seen = {o.id for o in top}
        extra = []
        for o in top:
            for rel_obj in store.related(o.id):
                if rel_obj.id not in seen:
                    seen.add(rel_obj.id)
                    extra.append(rel_obj)
        top.extend(extra[: max(0, k)])

    return top


__all__ = ["relevant"]
