"""Relevance-ranked retrieval (memory-graph R3, task 3.3).

Pins Property 3: similarity+recency+importance blend, threshold/max cap,
scope filtering, and a similarity-only (token-overlap) fallback when the
embedder is unavailable. Uses a fake embedder for determinism.
"""
from __future__ import annotations

from app.memory.objects import MemoryObject, SCOPE_GLOBAL, workspace_scope
from app.memory.mstore import MemoryStore
from app.memory.retriever import relevant


def _embed(text: str):
    # Toy 3-bucket embedding, deterministic + comparable.
    v = [0.0, 0.0, 0.0]
    for ch in text.lower():
        v[ord(ch) % 3] += 1.0
    return v


def _store_with_embeddings():
    s = MemoryStore()
    for content, scope in [
        ("the project uses PostgreSQL for storage", workspace_scope("p1")),
        ("user prefers concise answers", SCOPE_GLOBAL),
        ("unrelated pancake recipe notes", workspace_scope("p1")),
        ("other project uses MongoDB", workspace_scope("p2")),
    ]:
        o = MemoryObject(content=content, scope=scope, importance=0.5)
        o.embedding = _embed(content)
        s.add(o)
    return s


def test_relevance_returns_scoped_above_threshold():
    s = _store_with_embeddings()
    hits = relevant("PostgreSQL storage for the project", s, "p1",
                    k=5, threshold=0.2, embed_fn=_embed, traverse=False)
    contents = [o.content for o in hits]
    # The Postgres memory ranks; the OTHER workspace's Mongo memory is excluded.
    assert any("PostgreSQL" in c for c in contents)
    assert not any("MongoDB" in c for c in contents)


def test_threshold_filters_and_k_caps():
    s = _store_with_embeddings()
    hits = relevant("PostgreSQL", s, "p1", k=1, threshold=0.0,
                    embed_fn=_embed, traverse=False)
    assert len(hits) == 1                       # k cap


def test_similarity_only_fallback_when_no_embedder():
    s = MemoryStore()
    o = MemoryObject(content="deploy with docker compose", scope=SCOPE_GLOBAL)
    s.add(o)                                     # no embedding stored
    # embed_fn raises → falls back to token overlap, still recalls.
    def _boom(_):
        raise RuntimeError("embedder offline")
    hits = relevant("docker compose deploy", s, None, k=5, threshold=0.1,
                    embed_fn=_boom, traverse=False)
    assert any("docker" in o.content for o in hits)


def test_importance_and_recency_boost():
    s = MemoryStore()
    import time
    a = MemoryObject(content="topic alpha", scope=SCOPE_GLOBAL, importance=0.1)
    b = MemoryObject(content="topic alpha", scope=SCOPE_GLOBAL, importance=0.95)
    a.embedding = _embed(a.content)
    b.embedding = _embed(b.content)
    a.updated_at = time.time() - 60 * 24 * 3600   # old
    s.add(a)
    s.add(b)
    hits = relevant("topic alpha", s, None, k=2, threshold=0.0,
                    embed_fn=_embed, traverse=False)
    # The important + recent one ranks first.
    assert hits[0].importance == 0.95


def test_empty_store_returns_empty():
    assert relevant("anything", MemoryStore(), "p1", embed_fn=_embed) == []
