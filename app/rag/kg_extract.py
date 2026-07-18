"""LLM entity/relation extraction for the content Knowledge Graph (Arch §3.1).

Turns free text (an answer, a document, retrieved evidence) into typed entities
+ relations that populate a `KnowledgeGraph`. Multi-hop neighbours of the turn's
topic then become graph-grounded "related concept" follow-up suggestions
(Architecture §6, `knowledge_graph` source).

Design:
  • `extract_graph(text, llm_complete=...)` — one JSON-mode LLM call → (nodes,
    edges). Injectable `llm_complete` so it is unit-testable without a provider.
  • Pure parsing (`_parse`) + fail-open: any error / bad JSON → ([], []). A
    knowledge feature must never break a turn.
  • `related_concepts(kg, seeds, ...)` — multi-hop neighbour NAMES of the seed
    entities (the graph-grounded suggestions), excluding the seeds themselves.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Awaitable, Callable

from app.rag.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph

log = logging.getLogger(__name__)

LLMComplete = Callable[..., Awaitable[str]]

_PROMPT = (
    "Extract a small knowledge graph from the text: the key ENTITIES (concepts, "
    "technologies, components, people, projects) and the RELATIONS between them. "
    "Prefer 4-10 entities and only clear relations. Reply with ONLY compact JSON:\n"
    '{"entities":[{"id":"jwt","kind":"Tech"}],'
    '"relations":[{"src":"jwt","dst":"refresh token","kind":"pairs_with"}]}\n'
    "Use lowercase ids. No prose, no code fences.\n\nText:\n"
)


def _slug(v: str) -> str:
    return " ".join(str(v or "").split()).strip().lower()


def _parse(raw: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        return [], []
    if not isinstance(obj, dict):
        return [], []
    nodes: list[GraphNode] = []
    seen: set[str] = set()
    for e in (obj.get("entities") or []):
        if not isinstance(e, dict):
            continue
        nid = _slug(e.get("id") or e.get("name") or "")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        nodes.append(GraphNode(id=nid, kind=str(e.get("kind") or "Concept")))
    edges: list[GraphEdge] = []
    for r in (obj.get("relations") or []):
        if not isinstance(r, dict):
            continue
        src, dst = _slug(r.get("src")), _slug(r.get("dst"))
        if not src or not dst or src == dst:
            continue
        # Auto-add endpoints that weren't listed as entities.
        for nid in (src, dst):
            if nid not in seen:
                seen.add(nid)
                nodes.append(GraphNode(id=nid, kind="Concept"))
        edges.append(GraphEdge(src=src, dst=dst, kind=str(r.get("kind") or "related_to")))
    return nodes, edges


async def extract_graph(
    text: str, *, llm_complete: LLMComplete | None = None, model: str | None = None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Extract (nodes, edges) from `text` via one JSON LLM call. Fail-open → ([], [])."""
    t = (text or "").strip()
    if not t:
        return [], []
    try:
        if llm_complete is None:
            from app.core.config_loader import cfg, temperature_for
            from app.core.llm_client import llm
            llm_complete = llm.complete
            model = model or (cfg.llm.classifier_model or cfg.llm.model)
            raw = await llm_complete(
                [{"role": "user", "content": _PROMPT + t[:4000]}],
                model=model,
                options={"temperature": temperature_for("classifier"),
                         "num_predict": 300},
            )
        else:
            raw = await llm_complete(
                [{"role": "user", "content": _PROMPT + t[:4000]}])
    except Exception as exc:  # noqa: BLE001 — never break a turn on the KG path
        log.info("kg extraction skipped: %s", exc)
        return [], []
    return _parse(raw)


def build_graph(nodes, edges, *, into: KnowledgeGraph | None = None) -> KnowledgeGraph:
    """Populate a KnowledgeGraph (dedup/supersede write-back handled by the graph)."""
    kg = into or KnowledgeGraph()
    for n in nodes or []:
        kg.add_node(n)
    for e in edges or []:
        kg.add_edge(e)
    return kg


def related_concepts(kg: KnowledgeGraph, seeds, *, hops: int = 1, limit: int = 3):
    """Neighbour entity NAMES of the seed entities (graph-grounded related
    concepts), excluding the seeds. Seeds are matched by slug against node ids."""
    seed_ids = {_slug(s) for s in (seeds or []) if _slug(s)}
    if not seed_ids or not kg.nodes:
        return []
    out: list[str] = []
    seen: set[str] = set(seed_ids)
    for sid in seed_ids:
        if sid not in kg.nodes:
            continue
        for n in kg.neighbors(sid, hops=hops):
            if n.id in seen:
                continue
            seen.add(n.id)
            out.append(n.id)
            if len(out) >= limit:
                return out
    return out


def to_json(nodes, edges) -> dict:
    """Serialize nodes/edges to a compact persistable dict."""
    return {
        "nodes": [{"id": n.id, "kind": n.kind} for n in (nodes or [])],
        "edges": [{"src": e.src, "dst": e.dst, "kind": e.kind}
                  for e in (edges or [])],
    }


def graph_from_json(data) -> KnowledgeGraph:
    """Rebuild a KnowledgeGraph from a `to_json` dict (dedup handled by add_*)."""
    kg = KnowledgeGraph()
    if not isinstance(data, dict):
        return kg
    for n in (data.get("nodes") or []):
        if isinstance(n, dict) and n.get("id"):
            kg.add_node(GraphNode(id=_slug(n["id"]), kind=str(n.get("kind") or "Concept")))
    for e in (data.get("edges") or []):
        if isinstance(e, dict) and e.get("src") and e.get("dst"):
            src, dst = _slug(e["src"]), _slug(e["dst"])
            # Ensure edge endpoints exist as nodes (extraction may add them).
            for nid in (src, dst):
                if nid and nid not in kg.nodes:
                    kg.add_node(GraphNode(id=nid, kind="Concept"))
            kg.add_edge(GraphEdge(src=src, dst=dst,
                                  kind=str(e.get("kind") or "related_to")))
    return kg


def merge_json(existing, nodes, edges) -> dict:
    """Merge new nodes/edges into an existing `to_json` dict (deduped, bounded)."""
    kg = graph_from_json(existing if isinstance(existing, dict) else {})
    for n in nodes or []:
        kg.add_node(n)
    for e in edges or []:
        kg.add_edge(e)
    # Bound growth so a conversation graph can't blow up unbounded.
    if len(kg.nodes) > 400:
        return to_json(list(kg.nodes.values())[:400], kg.edges[:800])
    return to_json(list(kg.nodes.values()), kg.edges)


def neighbors_in_text(kg: KnowledgeGraph, text: str, *, hops: int = 1,
                      limit: int = 3):
    """CHEAP (no LLM) related-concept lookup for a persistent KG: seed with the
    graph nodes whose name appears in `text`, return their neighbours' names."""
    t = " ".join((text or "").lower().split())
    if not t or not kg.nodes:
        return []
    seeds = [nid for nid in kg.nodes if nid and nid in t]
    if not seeds:
        return []
    return related_concepts(kg, seeds, hops=hops, limit=limit)


def relations_in_text(kg: KnowledgeGraph, text: str, *, limit: int = 6):
    """CHEAP (no LLM) grounding lookup (§3.1): the graph RELATIONS whose
    endpoints include an entity mentioned in `text`, as readable triples
    ("jwt —pairs_with→ refresh token"). These feed the Retriever as evidence
    so answers are grounded in what the graph already knows."""
    t = " ".join((text or "").lower().split())
    if not t or not kg.edges:
        return []
    seeds = {nid for nid in kg.nodes if nid and nid in t}
    if not seeds:
        return []
    out: list[str] = []
    for e in kg.edges:
        if e.src in seeds or e.dst in seeds:
            out.append(f"{e.src} —{e.kind}→ {e.dst}")
            if len(out) >= limit:
                break
    return out


__all__ = [
    "extract_graph", "build_graph", "related_concepts",
    "to_json", "graph_from_json", "merge_json", "neighbors_in_text",
    "relations_in_text",
]
