"""Backend-agnostic property-graph interface.

Generic enough to back Apache AGE today (Cypher inside Postgres) and
Kùzu / Neo4j later. The unit of work is `(node, edge)` upserts plus a
free-form Cypher query escape hatch for traversals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Node:
    id: str
    label: str                          # "Person" | "Company" | "Project" | "Technology" | ...
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    label: str                          # "WORKED_AT" | "LED" | "USED" | ...
    props: dict[str, Any] = field(default_factory=dict)


class GraphStore(Protocol):
    async def upsert_node(self, node: Node) -> None: ...
    async def upsert_edge(self, edge: Edge) -> None: ...
    async def neighbours(
        self,
        node_id: str,
        *,
        edge_label: str | None = None,
        node_label: str | None = None,
    ) -> list[Node]: ...
    async def cypher(self, query: str, **params) -> list[dict]:
        """Escape hatch for non-trivial traversals."""
        ...
    async def close(self) -> None: ...
