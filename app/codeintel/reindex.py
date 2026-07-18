"""Incremental re-indexing (code-intelligence R6).

On a workspace file change (upload / agent edit / re-materialize) we update only
the changed files' sources and rebuild the affected graph, rather than re-walking
the whole tree (R6.1). Runs in the background and never blocks a turn (R6.2);
falls back to a full `build_index` when the prior index is missing or an
incremental update isn't possible (R6.3, Property 6). Deterministic; never raises.
"""
from __future__ import annotations

from app.codeintel.index import (
    CodeIndex, build_index, build_index_from_files, get_index, cache_put, _norm,
)


def reindex_files(index: CodeIndex, updates: dict) -> CodeIndex:
    """Apply `updates` (path -> new source, or None to delete) to `index` and
    rebuild from the patched source map. Re-reads only the changed files'
    content (the rest is already in memory). Never raises."""
    try:
        files = dict(index.files) if index is not None else {}
        for path, source in (updates or {}).items():
            p = _norm(path)
            if source is None:
                files.pop(p, None)
            else:
                files[p] = source
        new = build_index_from_files(list(files.items()),
                                     workspace_id=getattr(index, "workspace_id", None))
        if index is not None and index.workspace_id:
            cache_put(index.workspace_id, new)
        return new
    except Exception:  # noqa: BLE001
        return index if index is not None else CodeIndex(graph=__import__(
            "app.codegraph.model", fromlist=["CodeGraph"]).CodeGraph())


def reindex(workspace_id: str, changed: list[str] | None = None,
            root: str | None = None) -> CodeIndex:
    """Re-index a workspace. When `changed` paths are given and a prior index
    exists, re-read only those from disk and patch; otherwise full re-index
    (R6.3). Never raises."""
    try:
        prior = get_index(workspace_id, build=False)
        if prior is None or not changed:
            return build_index(workspace_id, root=root)         # full (R6.3)

        import os
        if root is None:
            from app.agent_workspace.materialize import workspace_path
            root = workspace_path(workspace_id)
        root_abs = os.path.realpath(root)
        updates: dict = {}
        for rel in changed:
            full = os.path.join(root_abs, rel.replace("/", os.sep))
            real = os.path.realpath(full)
            # Sandbox guard: only accept paths inside the workspace tree.
            if not (real == root_abs or real.startswith(root_abs + os.sep)):
                continue
            if not os.path.exists(full):
                updates[rel] = None                              # deleted
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    updates[rel] = f.read()
            except Exception:  # noqa: BLE001
                updates[rel] = None
        return reindex_files(prior, updates)
    except Exception:  # noqa: BLE001
        return build_index(workspace_id, root=root)


__all__ = ["reindex", "reindex_files"]
