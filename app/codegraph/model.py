"""Graph model — nodes, edges, and the in-memory CodeGraph container.

Mirrors codegraph's node/edge vocabulary (a pragmatic subset). A Node is a
symbol (file, class, function, …); an Edge is a relationship between two node
ids (contains, calls, imports, extends, …).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Symbol kinds we extract. 'file' is the container for a source file;
# 'route' is a synthesized HTTP/URL route from a framework resolver.
NODE_KINDS = (
    "file", "module", "class", "interface", "struct", "enum", "trait",
    "function", "method", "constant", "variable", "import", "route",
)

# Relationship kinds.
EDGE_KINDS = (
    "contains",     # file→class, class→method (lexical nesting)
    "calls",        # function→function
    "imports",      # file→file / file→module
    "extends",      # class→base class
    "implements",   # class→interface
    "references",   # generic
)


@dataclass
class Node:
    id: str               # stable id: "<path>::<qualified_name>" (file: the path)
    kind: str             # one of NODE_KINDS
    name: str             # short name ("get_user")
    qualified_name: str   # "UserService.get_user"
    path: str             # source file path within the project
    language: str = ""
    start_line: int = 0   # 1-based
    end_line: int = 0
    signature: str = ""   # e.g. "(self, id: int)"

    def to_row(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "name": self.name,
            "qualified_name": self.qualified_name, "path": self.path,
            "language": self.language, "start_line": self.start_line,
            "end_line": self.end_line, "signature": self.signature,
        }


@dataclass
class Edge:
    src: str              # source node id
    dst: str              # target node id
    kind: str             # one of EDGE_KINDS
    line: int = 0         # source line of the relationship (1-based), 0 if n/a

    def key(self) -> tuple:
        return (self.src, self.dst, self.kind)

    def to_row(self) -> dict:
        return {"src": self.src, "dst": self.dst, "kind": self.kind, "line": self.line}


@dataclass
class CodeGraph:
    """An in-memory graph plus light indexes for fast lookups."""
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    # Parse/skip bookkeeping for the summary.
    files_parsed: int = 0
    files_skipped: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    _edge_keys: set = field(default_factory=set, repr=False)

    def add_node(self, node: Node) -> None:
        # First writer wins for a given id (keeps the defining occurrence).
        self.nodes.setdefault(node.id, node)

    def add_edge(self, edge: Edge) -> None:
        k = edge.key()
        if edge.src == edge.dst or k in self._edge_keys:
            return
        self._edge_keys.add(k)
        self.edges.append(edge)

    # --- indexes (built lazily / on demand) ---
    def by_name(self, name: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.name == name]

    def nodes_in_file(self, path: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.path == path and n.kind != "file"]

    def out_edges(self, node_id: str, kind: str | None = None) -> list[Edge]:
        return [e for e in self.edges
                if e.src == node_id and (kind is None or e.kind == kind)]

    def in_edges(self, node_id: str, kind: str | None = None) -> list[Edge]:
        return [e for e in self.edges
                if e.dst == node_id and (kind is None or e.kind == kind)]

    @property
    def files(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "file"]

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for n in self.nodes.values():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        ekinds: dict[str, int] = {}
        for e in self.edges:
            ekinds[e.kind] = ekinds.get(e.kind, 0) + 1
        return {
            "nodes": len(self.nodes), "edges": len(self.edges),
            "node_kinds": kinds, "edge_kinds": ekinds,
            "files_parsed": self.files_parsed, "files_skipped": self.files_skipped,
            "languages": dict(self.languages),
        }
