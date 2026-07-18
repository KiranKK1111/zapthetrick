"""Symbol search (code-intelligence R4).

`find(index, symbol) -> {definitions, usages}` returns a symbol's definition
site(s) (from the Symbol_Index) and usage site(s) (from the Call_Graph), scoped
to the workspace (R4.1/R4.2). A symbol that isn't found returns empty so the
caller falls back to the existing text/vector search (R4.3, Property 4).
"""
from __future__ import annotations

from app.codeintel.index import CodeIndex
from app.codeintel.graph import usages_of


def find(index: CodeIndex, symbol: str) -> dict:
    """Return ``{"definitions": [Node], "usages": [Node], "found": bool}``.
    Never raises."""
    try:
        name = (symbol or "").strip()
        if not name or index is None:
            return {"definitions": [], "usages": [], "found": False}
        defs = [n for n in index.graph.by_name(name) if n.kind != "import"]
        uses = usages_of(index, name)
        return {
            "definitions": defs,
            "usages": uses,
            "found": bool(defs or uses),
        }
    except Exception:  # noqa: BLE001
        return {"definitions": [], "usages": [], "found": False}


__all__ = ["find"]
