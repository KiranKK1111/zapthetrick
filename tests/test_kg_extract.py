"""Content-KG extraction + multi-hop neighbours (Architecture §3.1 / §6)."""
from __future__ import annotations

import asyncio

from app.rag.kg_extract import build_graph, extract_graph, related_concepts
from app.rag.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph


def _fake_llm(reply: str):
    async def _c(_messages, **_kw):
        return reply
    return _c


def test_extract_parses_entities_and_relations():
    reply = ('{"entities":[{"id":"JWT","kind":"Tech"}],'
             '"relations":[{"src":"JWT","dst":"Refresh Token","kind":"pairs_with"}]}')
    nodes, edges = asyncio.run(extract_graph("...", llm_complete=_fake_llm(reply)))
    ids = {n.id for n in nodes}
    assert "jwt" in ids and "refresh token" in ids   # slugged + endpoint auto-added
    assert edges[0].src == "jwt" and edges[0].dst == "refresh token"


def test_extract_fail_open_on_bad_json():
    nodes, edges = asyncio.run(
        extract_graph("x", llm_complete=_fake_llm("not json at all")))
    assert nodes == [] and edges == []


def test_extract_empty_text_returns_empty():
    called = False
    async def _c(_m, **_k):
        nonlocal called
        called = True
        return "{}"
    nodes, edges = asyncio.run(extract_graph("   ", llm_complete=_c))
    assert (nodes, edges) == ([], []) and called is False


def test_graph_dedup_writeback():
    kg = KnowledgeGraph()
    kg.add_node(GraphNode(id="a", kind="Concept"))
    kg.add_node(GraphNode(id="a", kind="Concept", props={"x": 1}))  # supersede/merge
    kg.add_edge(GraphEdge(src="a", dst="b", kind="rel"))
    kg.add_edge(GraphEdge(src="a", dst="b", kind="rel"))            # dup ignored
    assert len(kg.nodes) == 1
    assert kg.nodes["a"].props == {"x": 1}
    assert len(kg.edges) == 1


def test_multi_hop_neighbors_undirected():
    kg = build_graph(
        [GraphNode(id=x, kind="C") for x in ("a", "b", "c")],
        [GraphEdge(src="a", dst="b", kind="r"), GraphEdge(src="b", dst="c", kind="r")],
    )
    one = {n.id for n in kg.neighbors("a", hops=1)}
    two = {n.id for n in kg.neighbors("a", hops=2)}
    assert one == {"b"}            # 1 hop
    assert two == {"b", "c"}       # 2 hops, undirected reach


def test_related_concepts_excludes_seeds_and_caps():
    kg = build_graph(
        [GraphNode(id=x, kind="C") for x in ("jwt", "refresh token", "expiry", "oauth")],
        [GraphEdge(src="jwt", dst="refresh token", kind="pairs_with"),
         GraphEdge(src="jwt", dst="expiry", kind="has"),
         GraphEdge(src="jwt", dst="oauth", kind="used_in")],
    )
    rel = related_concepts(kg, ["JWT"], limit=2)
    assert "jwt" not in rel                 # seed excluded
    assert len(rel) == 2                     # capped
    assert set(rel).issubset({"refresh token", "expiry", "oauth"})


def test_related_concepts_empty_when_no_seed_match():
    kg = build_graph([GraphNode(id="a", kind="C")], [])
    assert related_concepts(kg, ["nonexistent"]) == []


# --- persistence + cheap per-turn query (§3.1 doc KG) -----------------------

def test_json_roundtrip_and_merge():
    from app.rag.kg_extract import graph_from_json, merge_json, to_json
    j = to_json([GraphNode(id="jwt", kind="Tech")],
                [GraphEdge(src="jwt", dst="oauth", kind="used_in")])
    kg = graph_from_json(j)
    assert "jwt" in kg.nodes and "oauth" in kg.nodes
    # merge new + dup → deduped
    merged = merge_json(j, [GraphNode(id="jwt", kind="Tech")],
                        [GraphEdge(src="jwt", dst="refresh token", kind="pairs_with")])
    kg2 = graph_from_json(merged)
    assert {"jwt", "oauth", "refresh token"}.issubset(set(kg2.nodes))
    assert len(kg2.edges) == 2  # jwt→oauth kept, jwt→refresh token added, no dup


def test_neighbors_in_text_seeds_by_mention():
    from app.rag.kg_extract import graph_from_json, neighbors_in_text
    kg = graph_from_json({
        "nodes": [{"id": "jwt"}, {"id": "refresh token"}, {"id": "oauth"}],
        "edges": [{"src": "jwt", "dst": "refresh token", "kind": "pairs_with"},
                  {"src": "jwt", "dst": "oauth", "kind": "used_in"}],
    })
    # 'jwt' is mentioned in the answer → its neighbours are surfaced.
    rel = neighbors_in_text(kg, "Here is how JWT signing works in practice.",
                            limit=3)
    assert set(rel) == {"refresh token", "oauth"}
    # nothing mentioned → nothing
    assert neighbors_in_text(kg, "an unrelated answer about cooking") == []
