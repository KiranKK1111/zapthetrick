"""Semantic conversation/memory search (memory-graph R6).

`search(query, store, workspace_id, k)` finds the most semantically-similar
Memory_Objects within a Workspace's scope (+ global), so "continue the Flutter
work" pulls the right prior context by meaning rather than recency (R6.2). Scope
isolation is enforced (R6.3, Property 6). Reuses the existing embedder; falls
back to token overlap when it's unavailable. No second blocking LLM call.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 1}


def _embedder():
    try:
        from app.rag.embedder import embed_one
        return embed_one
    except Exception:  # noqa: BLE001
        return None


def search(query: str, store, workspace_id: str | None = None, *,
           k: int = 8, embed_fn=None) -> list:
    """Pure semantic similarity ranking (no recency/importance bias), scoped to
    workspace + global. Never raises; returns [] on error."""
    try:
        from app.memory.retriever import _cosine, _overlap
        from app.memory.mstore import retrieval_scopes

        candidates = store.by_scope(retrieval_scopes(workspace_id))
        if not candidates:
            return []
        embed_fn = embed_fn or _embedder()
        q_vec = None
        if embed_fn is not None:
            try:
                q_vec = embed_fn(query)
            except Exception:  # noqa: BLE001
                q_vec = None
        q_tokens = _tokens(query)

        scored = []
        for o in candidates:
            if q_vec is not None and getattr(o, "embedding", None):
                sim = _cosine(q_vec, o.embedding)
            else:
                sim = _overlap(q_tokens, o.content)
            if sim > 0.0:
                scored.append((sim, o))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [o for _s, o in scored[:k]]
    except Exception:  # noqa: BLE001
        return []


__all__ = ["search"]
