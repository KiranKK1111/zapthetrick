"""Apache AGE adapter — Cypher inside Postgres.

The init script `migrations/init/00-extensions.sql` creates the `kg`
graph and loads the AGE extension. Every query is wrapped in
`SELECT * FROM cypher('kg', $$ ... $$) AS (result agtype);` because
that's how AGE works.

We borrow the existing Postgres pool via `app.storage.db.SessionFactory`
— no second connection layer. AGE returns `agtype` strings which we
parse loosely (it's JSON with a `::vertex` / `::edge` suffix).
"""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import text

from ..db import SessionFactory
from .base import Node


_AGTYPE_SUFFIX = re.compile(r"::(vertex|edge|path)\s*$")


class AgeStore:
    def __init__(self, *, graph: str = "kg") -> None:
        self.graph = graph

    async def _cypher(self, cypher: str) -> list[Any]:
        """Run a raw Cypher snippet and return loosely-parsed agtype rows.

        We avoid the parameterised-cypher path because AGE's
        `cypher(query, params, graph)` overload only accepts JSON-able
        params and the syntax churn isn't worth it for our workload.
        """
        if SessionFactory is None:
            raise RuntimeError("Database not bootstrapped; call create_engine() first.")
        stmt = text(
            f"SELECT * FROM cypher('{self.graph}', $$ {cypher} $$) AS (result agtype);"
        )
        async with SessionFactory() as session:
            res = await session.execute(stmt)
            rows = res.scalars().all()
        return [self._parse_agtype(r) for r in rows]

    @staticmethod
    def _parse_agtype(raw: Any) -> Any:
        if raw is None:
            return None
        s = str(raw)
        # Strip a trailing `::vertex|edge|path` tag if any.
        s = _AGTYPE_SUFFIX.sub("", s).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s

    # ---- public API --------------------------------------------------
    async def upsert_node(self, node: Node) -> None:
        props_inline = _props(node.props | {"id": node.id})
        await self._cypher(
            f"MERGE (n:{node.label} {{id: '{node.id}'}}) "
            f"SET n += {props_inline}"
        )

    async def upsert_edge(self, edge: Any) -> None:
        props_inline = _props(edge.props)
        await self._cypher(
            f"MATCH (a {{id: '{edge.src}'}}), (b {{id: '{edge.dst}'}}) "
            f"MERGE (a)-[r:{edge.label}]->(b) "
            f"SET r += {props_inline}"
        )

    async def neighbours(
        self,
        node_id: str,
        *,
        edge_label: str | None = None,
        node_label: str | None = None,
    ) -> list[Node]:
        rel = f":{edge_label}" if edge_label else ""
        nlabel = f":{node_label}" if node_label else ""
        rows = await self._cypher(
            f"MATCH (a {{id: '{node_id}'}})-[r{rel}]->(b{nlabel}) RETURN b"
        )
        out: list[Node] = []
        for r in rows:
            if isinstance(r, dict):
                props = dict(r.get("properties", {}))
                label = (r.get("label") or "").strip() or "Node"
                out.append(Node(id=str(props.get("id", "")), label=label, props=props))
        return out

    async def cypher(self, query: str, **params) -> list[dict]:
        if params:
            # Crude param expansion: replace `:name` with the JSON form.
            for k, v in params.items():
                query = query.replace(f":{k}", json.dumps(v))
        rows = await self._cypher(query)
        return [r if isinstance(r, dict) else {"value": r} for r in rows]

    async def close(self) -> None:
        return None


def _props(props: dict[str, Any]) -> str:
    """Render a Python dict as Cypher property syntax: `{k: 'v', ...}`."""
    parts: list[str] = []
    for k, v in props.items():
        if isinstance(v, str):
            esc = v.replace("'", "\\'")
            parts.append(f"{k}: '{esc}'")
        elif v is None:
            continue
        else:
            parts.append(f"{k}: {json.dumps(v)}")
    return "{" + ", ".join(parts) + "}"
