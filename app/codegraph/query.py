"""Graph query tools (codegraph-MCP-like), operating on an in-memory CodeGraph.

    find_symbol(graph, name)            -> matching nodes
    callers(graph, id, depth)           -> who (transitively) calls id
    callees(graph, id, depth)           -> what id (transitively) calls
    file_structure(graph, path)         -> the file's symbol outline
    impact(graph, id, depth)            -> transitive callers (blast radius)

The agent can pull these on demand to answer code questions; each returns plain
dicts (JSON/Markdown-friendly).
"""
from __future__ import annotations

from .model import CodeGraph, Node


def _node_dict(n: Node) -> dict:
    return {"id": n.id, "kind": n.kind, "name": n.name,
            "qualified_name": n.qualified_name, "path": n.path,
            "language": n.language, "start_line": n.start_line,
            "signature": n.signature}


def find_symbol(graph: CodeGraph, name: str, *, limit: int = 20) -> list[dict]:
    """Symbols whose name or qualified_name contains `name` (case-insensitive)."""
    q = (name or "").strip().lower()
    if not q:
        return []
    exact, partial = [], []
    for n in graph.nodes.values():
        if n.kind == "file":
            continue
        nm = n.name.lower()
        if nm == q:
            exact.append(n)
        elif q in nm or q in n.qualified_name.lower():
            partial.append(n)
    return [_node_dict(n) for n in (exact + partial)[:limit]]


def _traverse(graph: CodeGraph, node_id: str, *, direction: str, depth: int,
              kind: str = "calls") -> list[dict]:
    """BFS over `kind` edges. direction='in' (callers) or 'out' (callees)."""
    seen = {node_id}
    frontier = [node_id]
    out: list[dict] = []
    for d in range(1, max(1, depth) + 1):
        nxt = []
        for nid in frontier:
            edges = (graph.in_edges(nid, kind) if direction == "in"
                     else graph.out_edges(nid, kind))
            for e in edges:
                other = e.src if direction == "in" else e.dst
                if other in seen:
                    continue
                seen.add(other)
                node = graph.nodes.get(other)
                if node:
                    out.append({**_node_dict(node), "depth": d})
                    nxt.append(other)
        frontier = nxt
        if not frontier:
            break
    return out


def callers(graph: CodeGraph, node_id: str, *, depth: int = 2) -> list[dict]:
    return _traverse(graph, node_id, direction="in", depth=depth, kind="calls")


def callees(graph: CodeGraph, node_id: str, *, depth: int = 2) -> list[dict]:
    return _traverse(graph, node_id, direction="out", depth=depth, kind="calls")


def impact(graph: CodeGraph, node_id: str, *, depth: int = 3) -> list[dict]:
    """Transitive callers — what could break if this symbol changes."""
    return _traverse(graph, node_id, direction="in", depth=depth, kind="calls")


def subclasses(graph: CodeGraph, node_id: str, *, depth: int = 3) -> list[dict]:
    return _traverse(graph, node_id, direction="in", depth=depth, kind="extends")


def file_structure(graph: CodeGraph, path: str) -> dict:
    """The symbol outline of one file (contains tree, one level)."""
    path = path.replace("\\", "/").lstrip("./")
    file_node = graph.nodes.get(path)
    syms = graph.nodes_in_file(path)
    syms.sort(key=lambda n: n.start_line)
    return {
        "path": path,
        "exists": file_node is not None,
        "symbols": [_node_dict(n) for n in syms],
        "imports": [graph.nodes.get(e.dst).path  # type: ignore[union-attr]
                    for e in graph.out_edges(path, "imports")
                    if graph.nodes.get(e.dst)],
    }
