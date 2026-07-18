"""Code-graph retrieval — turn a question into graph-derived evidence.

Used by the RetrieverAgent: load the conversation's code graph, match the
identifiers mentioned in the question against the graph's own symbol names
(data-driven, no keyword lists), and describe each match with its
signature + immediate callers/callees/routes. The Persona agent then cites this
alongside the RAG text chunks.
"""
from __future__ import annotations

import re

from . import query
from .model import CodeGraph
from .store import load_latest
from .summary import summarize_graph

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# Cap on the per-symbol document embedded into RAG (chunked downstream).
_DOC_CHARS = 200_000


def _describe(graph: CodeGraph, node) -> str:
    sig = node.signature if node.kind in ("function", "method", "route") else ""
    head = f"{node.kind} `{node.qualified_name}{sig}` in {node.path}:{node.start_line}"
    callees = [c["qualified_name"] for c in query.callees(graph, node.id, depth=1)][:8]
    callers = [c["qualified_name"] for c in query.callers(graph, node.id, depth=1)][:8]
    routes = [graph.nodes[e.src].qualified_name
              for e in graph.in_edges(node.id, "references")
              if e.src in graph.nodes]
    parts = [head]
    if callees:
        parts.append("calls: " + ", ".join(callees))
    if callers:
        parts.append("called by: " + ", ".join(callers))
    if routes:
        parts.append("serves route(s): " + ", ".join(routes[:8]))
    return ". ".join(parts) + "."


def graph_document(graph: CodeGraph, summary: str = "") -> str:
    """A text document of the graph (overview + per-symbol descriptions with
    callers/callees/routes) to EMBED into the conversation's RAG store, so the
    existing retriever surfaces the project's STRUCTURE semantically on
    follow-ups — on both the upload and the agent-mesh paths."""
    parts = [summary or summarize_graph(graph), "\n## Symbols and relationships"]
    for n in graph.nodes.values():
        if n.kind == "file":
            continue
        parts.append(_describe(graph, n))
    return "\n".join(parts)[:_DOC_CHARS]


def evidence_from_graph(graph: CodeGraph, question: str, *, limit: int = 6) -> list[dict]:
    if not graph or not graph.nodes:
        return []
    out: list[dict] = []
    # Always surface a bounded project overview so the model keeps the codebase's
    # shape even when the question doesn't name a specific symbol (so the overview
    # isn't lost once the upload turn is windowed out of history).
    out.append({"text": summarize_graph(graph, max_chars=1500),
                "source": "code-graph:overview", "score": 0.6})

    names = {n.name.lower() for n in graph.nodes.values() if n.kind != "file"}
    tokens = {t.lower() for t in _TOKEN.findall(question or "")}
    hits = tokens & names
    if not hits:
        return out
    seen: set[str] = set()
    for n in graph.nodes.values():
        if n.kind == "file" or n.name.lower() not in hits or n.id in seen:
            continue
        seen.add(n.id)
        out.append({"text": _describe(graph, n),
                    "source": f"code-graph:{n.path}", "score": 0.9})
        if len(out) >= limit:
            break
    return out


async def retrieve_code_evidence(conversation_id: str, question: str,
                                 *, limit: int = 6) -> list[dict]:
    """Load the conversation's latest code graph and return evidence dicts
    ({text, source, score}) for the symbols the question mentions. [] if there's
    no graph or no match."""
    graph = await load_latest(conversation_id)
    if graph is None:
        return []
    return evidence_from_graph(graph, question, limit=limit)
