"""Top-level: build + persist a code knowledge graph from an uploaded archive.

    summary, stats, graph_id = await ingest_archive_code_graph(cid, data, name)

The heavy parse/build runs in a worker thread; persistence is best-effort. A
graph is only built when the archive actually contains source files.
"""
from __future__ import annotations

import asyncio
import logging

from .archive import iter_source_members
from .builder import build_code_graph
from .model import CodeGraph
from .store import save_code_graph
from .summary import summarize_graph

log = logging.getLogger(__name__)

# Below this many source files we don't bother (a stray .py in a zip of docs).
_MIN_SOURCE_FILES = 2


def looks_like_code_archive(data: bytes, filename: str, *, probe: int = 50) -> bool:
    """Quick check: does the archive contain at least a couple of source files?
    Stops after `probe` matches so it's cheap on huge archives."""
    n = 0
    for _ in iter_source_members(data, filename):
        n += 1
        if n >= max(_MIN_SOURCE_FILES, probe):
            break
    return n >= _MIN_SOURCE_FILES


def _build_sync(data: bytes, filename: str) -> tuple[CodeGraph, str]:
    members = list(iter_source_members(data, filename))
    graph = build_code_graph(members)
    return graph, summarize_graph(graph)


async def ingest_archive_code_graph(
    conversation_id: str, data: bytes, filename: str
) -> tuple[str, dict, str | None] | None:
    """Build + store the graph. Returns (summary, stats, graph_id) or None when
    the archive has no meaningful source to graph."""
    graph, summary = await asyncio.to_thread(_build_sync, data, filename)
    if graph.files_parsed < _MIN_SOURCE_FILES or not graph.nodes:
        return None
    gid = await save_code_graph(conversation_id, filename, graph, summary)

    # Embed the graph's STRUCTURE (overview + per-symbol descriptions with
    # callers/callees/routes) into the conversation's RAG store, so the existing
    # retriever surfaces it semantically on follow-ups — making the code graph
    # available on the agent-mesh path too, not just this upload turn.
    try:
        from app.codegraph.retrieval import graph_document
        from app.rag.documents import ingest_chat_document

        await ingest_chat_document(
            str(conversation_id), f"{filename} · code graph",
            graph_document(graph, summary),
        )
    except Exception as exc:  # noqa: BLE001 — RAG embed is best-effort
        log.info("code graph RAG embed failed: %s", exc)

    log.info("code graph for %s: %s", filename, graph.stats())
    return summary, graph.stats(), gid
