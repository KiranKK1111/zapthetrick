"""Register the code-graph query tools in the orchestrator tool registry, so the
agent can call them by name (find a symbol, trace callers/callees/impact, list a
file's structure) against a conversation's persisted code graph.

Each handler takes the `conversation_id` (the orchestrator binds it) plus the
tool's own arguments, loads the latest graph for that conversation, and returns a
compact JSON-able result.
"""
from __future__ import annotations

from typing import Any

from app.tools.registry import Tool, register

from . import query
from .store import load_latest

_CID = {"conversation_id": {"type": "string",
                            "description": "Conversation whose code graph to query."}}


def _resolve_id(graph, symbol: str) -> str | None:
    hits = query.find_symbol(graph, symbol, limit=1)
    return hits[0]["id"] if hits else None


async def _search(*, conversation_id: str, symbol: str, **_: Any):
    g = await load_latest(conversation_id)
    return {"matches": query.find_symbol(g, symbol)} if g else {"matches": []}


async def _callers(*, conversation_id: str, symbol: str, depth: int = 2, **_: Any):
    g = await load_latest(conversation_id)
    if not g:
        return {"callers": []}
    nid = _resolve_id(g, symbol)
    return {"callers": query.callers(g, nid, depth=depth) if nid else []}


async def _callees(*, conversation_id: str, symbol: str, depth: int = 2, **_: Any):
    g = await load_latest(conversation_id)
    if not g:
        return {"callees": []}
    nid = _resolve_id(g, symbol)
    return {"callees": query.callees(g, nid, depth=depth) if nid else []}


async def _impact(*, conversation_id: str, symbol: str, depth: int = 3, **_: Any):
    g = await load_latest(conversation_id)
    if not g:
        return {"impact": []}
    nid = _resolve_id(g, symbol)
    return {"impact": query.impact(g, nid, depth=depth) if nid else []}


async def _file_structure(*, conversation_id: str, path: str, **_: Any):
    g = await load_latest(conversation_id)
    return query.file_structure(g, path) if g else {"path": path, "exists": False}


def _schema(extra: dict) -> dict:
    props = {**_CID, **extra}
    return {"type": "object", "properties": props,
            "required": [k for k in props if k != "depth"]}


_SYMBOL = {"symbol": {"type": "string", "description": "Symbol name (function/class/method)."}}
_DEPTH = {"depth": {"type": "integer", "description": "Traversal depth (default 2-3)."}}

register(Tool(
    name="code_search",
    description="Find symbols (functions/classes/methods/routes) by name in the "
                "uploaded project's code graph.",
    input_schema=_schema(_SYMBOL), handler=_search,
))
register(Tool(
    name="code_callers",
    description="Who calls this symbol (transitively up to `depth`).",
    input_schema=_schema({**_SYMBOL, **_DEPTH}), handler=_callers,
))
register(Tool(
    name="code_callees",
    description="What this symbol calls (transitively up to `depth`).",
    input_schema=_schema({**_SYMBOL, **_DEPTH}), handler=_callees,
))
register(Tool(
    name="code_impact",
    description="Blast radius — all transitive callers that could break if this "
                "symbol changes.",
    input_schema=_schema({**_SYMBOL, **_DEPTH}), handler=_impact,
))
register(Tool(
    name="code_file_structure",
    description="The symbol outline (classes/functions/imports) of one file in "
                "the project.",
    input_schema=_schema({"path": {"type": "string", "description": "File path within the project."}}),
    handler=_file_structure,
))
