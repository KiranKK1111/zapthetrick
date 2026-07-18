"""Persist + reload a CodeGraph in Postgres (the `code_graphs` table).

Serialised as one JSONB row per built graph, scoped to a conversation. A
follow-up turn reloads it to run the query tools or re-inject the summary.
"""
from __future__ import annotations

import json
import logging

from .model import CodeGraph, Edge, Node

log = logging.getLogger(__name__)


def serialize(graph: CodeGraph) -> dict:
    return {
        "nodes": [n.to_row() for n in graph.nodes.values()],
        "edges": [e.to_row() for e in graph.edges],
    }


def deserialize(data: dict) -> CodeGraph:
    g = CodeGraph()
    for r in (data or {}).get("nodes", []):
        g.add_node(Node(
            id=r["id"], kind=r["kind"], name=r["name"],
            qualified_name=r["qualified_name"], path=r["path"],
            language=r.get("language", ""), start_line=r.get("start_line", 0),
            end_line=r.get("end_line", 0), signature=r.get("signature", ""),
        ))
    for r in (data or {}).get("edges", []):
        g.add_edge(Edge(src=r["src"], dst=r["dst"], kind=r["kind"],
                        line=r.get("line", 0)))
    return g


_DDL = (
    "CREATE TABLE IF NOT EXISTS code_graphs ("
    " id uuid PRIMARY KEY DEFAULT gen_random_uuid(),"
    " conversation_id text NOT NULL, filename text NOT NULL,"
    " files_parsed int NOT NULL DEFAULT 0, nodes_count int NOT NULL DEFAULT 0,"
    " edges_count int NOT NULL DEFAULT 0, languages jsonb, summary text,"
    " graph jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now())"
)
_DDL_IDX = ("CREATE INDEX IF NOT EXISTS ix_code_graphs_conversation "
            "ON code_graphs (conversation_id)")
_ensured = False


async def _ensure_table(s) -> None:
    """Idempotent safety net: create the table if migration 0011 hasn't run.
    No-op after the first success."""
    global _ensured
    if _ensured:
        return
    from sqlalchemy import text
    await s.execute(text(_DDL))
    await s.execute(text(_DDL_IDX))
    _ensured = True


async def save_code_graph(conversation_id: str, filename: str, graph: CodeGraph,
                          summary: str) -> str | None:
    """Insert a built graph. Best-effort: returns the row id, or None on any
    failure (the chat must never break because graph persistence failed)."""
    from sqlalchemy import text

    from storage.db import get_session_factory

    sf = get_session_factory()
    if sf is None:
        return None
    st = graph.stats()
    try:
        async with sf() as s:
            await _ensure_table(s)
            row = (await s.execute(
                text(
                    "INSERT INTO code_graphs "
                    "(conversation_id, filename, files_parsed, nodes_count, "
                    " edges_count, languages, summary, graph) "
                    "VALUES (:cid, :fn, :fp, :nc, :ec, "
                    " cast(:langs AS jsonb), :summ, cast(:graph AS jsonb)) "
                    "RETURNING id"
                ),
                {
                    "cid": str(conversation_id), "fn": filename,
                    "fp": st["files_parsed"], "nc": st["nodes"], "ec": st["edges"],
                    "langs": json.dumps(st["languages"]),
                    "summ": summary,
                    "graph": json.dumps(serialize(graph)),
                },
            )).first()
            await s.commit()
            return str(row[0]) if row else None
    except Exception as exc:  # noqa: BLE001
        log.info("save_code_graph failed: %s", exc)
        return None


def _as_dict(val) -> dict:
    if isinstance(val, dict):
        return val
    if isinstance(val, (str, bytes)):
        try:
            return json.loads(val)
        except Exception:  # noqa: BLE001
            return {}
    return {}


async def load_latest(conversation_id: str) -> CodeGraph | None:
    """Reconstruct the most recent code graph for a conversation, or None."""
    from sqlalchemy import text

    from storage.db import get_session_factory

    sf = get_session_factory()
    if sf is None:
        return None
    try:
        async with sf() as s:
            row = (await s.execute(
                text("SELECT graph FROM code_graphs WHERE conversation_id = :cid "
                     "ORDER BY created_at DESC LIMIT 1"),
                {"cid": str(conversation_id)},
            )).first()
    except Exception as exc:  # noqa: BLE001
        log.info("load_latest code graph failed: %s", exc)
        return None
    if row is None:
        return None
    return deserialize(_as_dict(row[0]))


async def latest_summary(conversation_id: str) -> str | None:
    """The stored project summary for the most recent graph, or None."""
    from sqlalchemy import text

    from storage.db import get_session_factory

    sf = get_session_factory()
    if sf is None:
        return None
    try:
        async with sf() as s:
            row = (await s.execute(
                text("SELECT summary FROM code_graphs WHERE conversation_id = :cid "
                     "ORDER BY created_at DESC LIMIT 1"),
                {"cid": str(conversation_id)},
            )).first()
    except Exception as exc:  # noqa: BLE001
        return None
    return row[0] if row and row[0] else None
