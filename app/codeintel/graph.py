"""Dependency + call graphs (code-intelligence R2/R3).

Derived from the `CodeIndex`'s resolved `CodeGraph`:
  • `dependency_graph` — file → internal files (from `imports` edges); raw
    imports that didn't resolve to a workspace file are reported as `external`
    and excluded from internal edges (R2.3, Property 2).
  • `call_graph` — symbol → referenced symbols (from `calls` edges, best-effort;
    imprecise refs were already bounded by the builder — R3.3, Property 3).
  • `dependents_of(file)` / `usages_of(symbol)` for blast-radius + "where used".

Deterministic; never raises (fail-open to empty results).
"""
from __future__ import annotations

import re

from app.codeintel.index import CodeIndex, _norm

# Minimal raw-import detection for the "external" report (Python + JS/TS).
_PY_IMPORT = re.compile(r"^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+))", re.M)
_JS_IMPORT = re.compile(r"""(?:import\s+[^'"]*from\s+|require\(\s*)['"]([^'"]+)['"]""")


def dependency_graph(index: CodeIndex) -> dict:
    """Return ``{"internal": {file: [files]}, "external": {file: [modules]}}``.
    Internal edges come from the resolved graph; externals are raw imports with
    no internal target. Never raises."""
    try:
        g = index.graph
        internal: dict[str, list[str]] = {}
        for e in g.edges:
            if e.kind == "imports":
                internal.setdefault(e.src, [])
                if e.dst not in internal[e.src]:
                    internal[e.src].append(e.dst)

        external: dict[str, list[str]] = {}
        internal_targets = {f for deps in internal.values() for f in deps}
        file_paths = set(index.all_files())
        for path, source in index.files.items():
            raw = _raw_imports(source)
            ext = []
            for mod in raw:
                if not _resolves_internally(mod, file_paths, internal_targets):
                    ext.append(mod)
            if ext:
                external[_norm(path)] = sorted(set(ext))
        return {"internal": internal, "external": external}
    except Exception:  # noqa: BLE001
        return {"internal": {}, "external": {}}


def _raw_imports(source: str) -> list[str]:
    mods: list[str] = []
    for m in _PY_IMPORT.finditer(source or ""):
        mods.append((m.group(1) or m.group(2) or "").strip())
    for m in _JS_IMPORT.finditer(source or ""):
        mods.append(m.group(1).strip())
    return [m for m in mods if m]


def _resolves_internally(mod: str, file_paths: set[str],
                         internal_targets: set[str]) -> bool:
    m = mod.replace(".", "/").lstrip("/")
    if not m:
        return False
    for p in file_paths:
        stem = p.rsplit(".", 1)[0]
        if stem.endswith(m) or p.endswith(m) or m in stem:
            return True
    return False


def call_graph(index: CodeIndex) -> dict:
    """symbol_id → [referenced symbol_ids] from `calls` edges (R3.1)."""
    try:
        out: dict[str, list[str]] = {}
        for e in index.graph.edges:
            if e.kind in ("calls", "references"):
                out.setdefault(e.src, [])
                if e.dst not in out[e.src]:
                    out[e.src].append(e.dst)
        return out
    except Exception:  # noqa: BLE001
        return {}


def dependents_of(index: CodeIndex, file: str) -> list[str]:
    """Files that import `file` (reverse dependency / blast radius, R2.2)."""
    try:
        target = _norm(file)
        return sorted({e.src for e in index.graph.edges
                       if e.kind == "imports" and e.dst == target})
    except Exception:  # noqa: BLE001
        return []


def usages_of(index: CodeIndex, symbol: str) -> list:
    """Symbols that reference any definition of `symbol` (R3.2). Best-effort:
    resolves the name to its definition node(s), then returns referencing nodes."""
    try:
        defs = index.graph.by_name(symbol)
        def_ids = {n.id for n in defs}
        if not def_ids:
            return []
        callers: list = []
        seen = set()
        for e in index.graph.edges:
            if e.kind in ("calls", "references") and e.dst in def_ids:
                if e.src not in seen:
                    seen.add(e.src)
                    node = index.graph.nodes.get(e.src)
                    if node is not None:
                        callers.append(node)
        return callers
    except Exception:  # noqa: BLE001
        return []


__all__ = ["dependency_graph", "call_graph", "dependents_of", "usages_of"]
