"""Symbol index over the sandboxed workspace tree (code-intelligence R1).

Reuses `app/codegraph` (`build_code_graph` → Python `ast` + tree-sitter
extraction with location/signature) to produce a per-file Symbol index + the
resolved graph. The build runs ONLY over the `app/agent_workspace` on-disk tree
(sandbox preserved — R1.4/R7.3); unsupported/unparseable files are skipped
without failing the build (R1.3). Deterministic, no LLM call.

A process-wide per-workspace cache lets retrieval read the index cheaply;
`reindex` patches it on file change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.codegraph.builder import build_code_graph
from app.codegraph.model import CodeGraph, Node

_MAX_FILE_BYTES = 1_500_000
_MAX_FILES = 4_000
# Read only plausibly-textual source; skip obvious binaries early.
_BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf", ".zip",
    ".7z", ".gz", ".tar", ".rar", ".exe", ".dll", ".so", ".dylib", ".class",
    ".jar", ".woff", ".woff2", ".ttf", ".otf", ".mp3", ".mp4", ".mov", ".wav",
    ".pyc", ".bin", ".lock",
}


@dataclass
class CodeIndex:
    """The symbol index for one workspace: the resolved CodeGraph + the source
    map it was built from (so reindex/externals can re-derive)."""
    graph: CodeGraph
    files: dict[str, str] = field(default_factory=dict)   # path -> source
    workspace_id: str | None = None

    # ---- reads -----------------------------------------------------------
    def file_symbols(self, path: str) -> list[Node]:
        return self.graph.nodes_in_file(_norm(path))

    def symbols_by_name(self, name: str) -> list[Node]:
        return self.graph.by_name(name)

    def all_files(self) -> list[str]:
        return [n.path for n in self.graph.files]

    def stats(self) -> dict:
        return self.graph.stats()


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./")


def _read_tree(root: str) -> list[tuple[str, str]]:
    """Read source files under `root` (relative paths). Sandbox-preserving:
    only paths that resolve INSIDE `root` are read; binaries/oversized skipped."""
    out: list[tuple[str, str]] = []
    try:
        root_abs = os.path.realpath(root)
    except Exception:  # noqa: BLE001
        return out
    if not os.path.isdir(root_abs):
        return out
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_abs):
        # Prune common vendored/build dirs early (builder also filters).
        dirnames[:] = [d for d in dirnames if d not in (
            "node_modules", ".git", "dist", "build", "__pycache__", ".venv",
            "venv", "target", "vendor", ".dart_tool", "Pods")]
        for fn in filenames:
            if count >= _MAX_FILES:
                return out
            ext = os.path.splitext(fn)[1].lower()
            if ext in _BINARY_EXT:
                continue
            full = os.path.join(dirpath, fn)
            # Sandbox guard: never follow a symlink out of the tree.
            try:
                real = os.path.realpath(full)
                if not real.startswith(root_abs + os.sep) and real != root_abs:
                    continue
                if os.path.getsize(full) > _MAX_FILE_BYTES:
                    continue
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except Exception:  # noqa: BLE001 — unreadable file → skip (R1.3)
                continue
            rel = _norm(os.path.relpath(full, root_abs))
            out.append((rel, source))
            count += 1
    return out


def build_index_from_files(files: list[tuple[str, str]],
                           workspace_id: str | None = None) -> CodeIndex:
    """Build an index directly from (path, source) pairs (used by tests + the
    in-memory path). Never raises — returns an empty index on error."""
    try:
        g = build_code_graph(files)
        return CodeIndex(graph=g, files={_norm(p): s for p, s in files},
                         workspace_id=workspace_id)
    except Exception:  # noqa: BLE001
        return CodeIndex(graph=CodeGraph(), files={}, workspace_id=workspace_id)


def build_index(workspace_id: str, root: str | None = None) -> CodeIndex:
    """Build the index over the workspace's sandboxed on-disk tree. `root`
    overrides the resolved workspace path (tests). Never raises."""
    try:
        if root is None:
            from app.agent_workspace.materialize import workspace_path
            root = workspace_path(workspace_id)
        files = _read_tree(root)
        idx = build_index_from_files(files, workspace_id)
        _CACHE[workspace_id] = idx
        return idx
    except Exception:  # noqa: BLE001
        return CodeIndex(graph=CodeGraph(), files={}, workspace_id=workspace_id)


# Process-wide per-workspace cache.
_CACHE: dict[str, CodeIndex] = {}


def get_index(workspace_id: str, *, build: bool = True) -> CodeIndex | None:
    """Return the cached index for a workspace, building it once if absent."""
    idx = _CACHE.get(workspace_id)
    if idx is None and build:
        idx = build_index(workspace_id)
    return idx


def cache_put(workspace_id: str, index: CodeIndex) -> None:
    _CACHE[workspace_id] = index


def cache_clear() -> None:
    _CACHE.clear()


__all__ = [
    "CodeIndex", "build_index", "build_index_from_files", "get_index",
    "cache_put", "cache_clear",
]
