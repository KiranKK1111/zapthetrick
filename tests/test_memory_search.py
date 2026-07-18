"""Semantic search + graph traversal (memory-graph R6/R7, tasks 8.2/9.2).

Pins Properties 6 & 7: scope-isolated semantic search; one-hop graph traversal
includes directly-related objects; with no edges retrieval is plain relevance.
"""
from __future__ import annotations

from app.memory.objects import MemoryObject, SCOPE_GLOBAL, workspace_scope
from app.memory.mstore import MemoryStore
from app.memory.search import search
from app.memory.retriever import relevant


def _embed(text: str):
    v = [0.0, 0.0, 0.0]
    for ch in text.lower():
        v[ord(ch) % 3] += 1.0
    return v


def _store():
    s = MemoryStore()
    for content, scope in [
        ("Flutter streaming architecture decisions", workspace_scope("p1")),
        ("global writing-style preference", SCOPE_GLOBAL),
        ("p2 kubernetes deployment notes", workspace_scope("p2")),
    ]:
        o = MemoryObject(content=content, scope=scope)
        o.embedding = _embed(content)
        s.add(o)
    return s


def test_scoped_semantic_search_isolates_workspace():
    s = _store()
    hits = search("Flutter streaming work", s, "p1", embed_fn=_embed)
    contents = [o.content for o in hits]
    assert any("Flutter" in c for c in contents)
    assert not any("kubernetes" in c for c in contents)   # other workspace (R6.3)


def test_search_empty_store():
    assert search("x", MemoryStore(), "p1", embed_fn=_embed) == []


# ── graph traversal (Property 7) ─────────────────────────────────────────────
def test_traversal_includes_related_objects():
    s = MemoryStore()
    entity = MemoryObject(content="entity: PostgreSQL", kind="entity",
                          scope=workspace_scope("p1"), importance=0.6)
    entity.embedding = _embed(entity.content)
    s.add(entity)
    decision = MemoryObject(content="decision: use connection pooling",
                            kind="decision", scope=workspace_scope("p1"),
                            importance=0.4)
    decision.embedding = _embed("totally different vector text zzz")
    s.add(decision)
    # entity → decision edge.
    s.relate(entity.id, "has_decision", decision.id)

    hits = relevant("PostgreSQL", s, "p1", k=3, threshold=0.0,
                    embed_fn=_embed, traverse=True)
    ids = {o.id for o in hits}
    assert entity.id in ids and decision.id in ids   # related pulled in


def test_no_edges_is_plain_relevance():
    s = MemoryStore()
    o = MemoryObject(content="standalone fact", scope=SCOPE_GLOBAL)
    o.embedding = _embed(o.content)
    s.add(o)
    hits = relevant("standalone fact", s, None, k=3, threshold=0.0,
                    embed_fn=_embed, traverse=True)
    assert [h.id for h in hits] == [o.id]        # no traversal additions
