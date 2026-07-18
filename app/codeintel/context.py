"""Code-aware context builder (code-intelligence R5).

`select(query, files, ...)` starts from the existing `rank_files` ranking, then
augments it with symbol matches, the selected files' direct dependencies, and
key referenced symbols from the index/graphs — returning the relevant slice to
hand to the perceived-speed Context_Budget (R5.2). With the index/graphs
unavailable or the feature gated off, it returns exactly today's `rank_files`
ordering (R5.3/Property 5). Deterministic; never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


@dataclass
class CodeContext:
    files: list[str] = field(default_factory=list)     # ordered relevant slice
    augmented: bool = False
    reasons: list[str] = field(default_factory=list)


def _gated_on() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.advanced_rag, "use_code_knowledge_graph", False))
    except Exception:  # noqa: BLE001
        return False


def select(query: str, files: list[tuple[str, str]], *,
           workspace_id: str | None = None, index=None,
           limit: int = 12) -> CodeContext:
    """Return the relevant file slice for a Code-In query. Fail-open to a plain
    `rank_files` ordering."""
    try:
        return _select(query, files, workspace_id, index, limit)
    except Exception:  # noqa: BLE001
        return CodeContext(files=_rank_only(query, files, limit), augmented=False,
                           reasons=["error → rank_files"])


def _rank_only(query, files, limit) -> list[str]:
    try:
        from app.chat.context_builder import rank_files
        return [r.path for r in rank_files(files or [], query, limit=limit)]
    except Exception:  # noqa: BLE001
        return [p for p, _ in (files or [])][:limit]


def _select(query, files, workspace_id, index, limit) -> CodeContext:
    ranked = _rank_only(query, files, limit)

    # Gated off or no index → today's rank_files unchanged (R5.3).
    if not _gated_on():
        return CodeContext(files=ranked, augmented=False, reasons=["gated off"])
    if index is None:
        from app.codeintel.index import get_index
        index = get_index(workspace_id) if workspace_id else None
    if index is None or not index.all_files():
        return CodeContext(files=ranked, augmented=False, reasons=["no index"])

    from app.codeintel.graph import dependency_graph
    from app.codeintel.search import find

    selected: list[str] = list(ranked)
    seen = set(selected)
    reasons: list[str] = []

    # 1) Symbol matches: query identifiers whose definitions live in files not
    #    already ranked in.
    for tok in {t for t in _WORD_RE.findall(query or "") if len(t) > 2}:
        res = find(index, tok)
        for node in res["definitions"]:
            if node.path and node.path not in seen:
                seen.add(node.path)
                selected.append(node.path)
                reasons.append(f"defines {tok}")

    # 2) Direct dependencies of the already-selected files (R5.1).
    dep = dependency_graph(index).get("internal", {})
    for path in list(selected):
        for d in dep.get(path, []):
            if d not in seen:
                seen.add(d)
                selected.append(d)
                reasons.append(f"dependency of {path}")

    return CodeContext(files=selected[:40],
                       augmented=bool(reasons), reasons=reasons[:8])


__all__ = ["CodeContext", "select"]
