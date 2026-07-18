"""Optional knowledge-graph layer.

Architecture.md §3:
  (Person: Candidate) ──worked_at──► (Company: Stripe)
                     ──used──► (Tech: Kafka)
                     ──led──► (Project: Payments-V2)

Stored as a property graph — see [storage.graph.AgeStore] for the
prod Apache-AGE-backed implementation. This module's in-memory class
is the unit-test scaffold. Joins as the fourth source feeding the
RRF fusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphNode:
    id: str
    kind: str                       # "Person" | "Company" | "Tech" | "Project" | ...
    props: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    dst: str
    kind: str                       # "worked_at" | "used" | "led" | ...
    props: dict = field(default_factory=dict)


class KnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_node(self, node: GraphNode) -> None:
        # Write-back rule (§16): supersede by id — merge props, don't duplicate.
        existing = self.nodes.get(node.id)
        if existing is not None:
            existing.props.update(node.props or {})
            return
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        # Write-back rule (§16): de-dup edges by (src, dst, kind).
        for e in self.edges:
            if e.src == edge.src and e.dst == edge.dst and e.kind == edge.kind:
                e.props.update(edge.props or {})
                return
        self.edges.append(edge)

    def traverse(self, node_id: str, edge_kind: str | None = None) -> list[GraphNode]:
        out: list[GraphNode] = []
        for e in self.edges:
            if e.src != node_id:
                continue
            if edge_kind is not None and e.kind != edge_kind:
                continue
            n = self.nodes.get(e.dst)
            if n is not None:
                out.append(n)
        return out

    def neighbors(self, node_id: str, *, hops: int = 1) -> list[GraphNode]:
        """Multi-hop UNDIRECTED neighbors of `node_id` (breadth-first), excluding
        the seed itself. `hops` bounds the traversal depth (default 1)."""
        seen: set[str] = {node_id}
        frontier: set[str] = {node_id}
        out: list[GraphNode] = []
        for _ in range(max(1, hops)):
            nxt: set[str] = set()
            for e in self.edges:
                if e.src in frontier and e.dst not in seen:
                    nxt.add(e.dst)
                if e.dst in frontier and e.src not in seen:
                    nxt.add(e.src)
            for nid in nxt:
                seen.add(nid)
                n = self.nodes.get(nid)
                if n is not None:
                    out.append(n)
            if not nxt:
                break
            frontier = nxt
        return out
