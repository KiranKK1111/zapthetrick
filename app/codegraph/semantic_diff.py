"""Semantic (AST-level) diff (Phase 13, report #106).

A line diff says "12 lines changed"; a SEMANTIC diff says "added method
`User.deactivate`, removed function `legacy_login`, changed signature of
`charge(amount)` → `charge(amount, currency)`". It compares the SYMBOLS of two
versions of a file via the same extractor the code graph uses, so it's
language-aware (Python via `ast`, everything else via tree-sitter) and ignores
formatting/comment churn.

    d = semantic_diff(old_src, new_src, path="app/x.py")
    d["added"]    # ["User.deactivate", ...]
    d["removed"]  # ["legacy_login", ...]
    d["changed"]  # [{"symbol": "charge", "old": "(amount)", "new": "(amount, currency)"}]
    summarize_semantic_diff(d)   # one-line human summary

Pure + offline; never raises (returns an empty diff on a parse failure).
"""
from __future__ import annotations

from .extract import extract_file


def _symbols(src: str, path: str) -> dict[str, str]:
    """{qualified_name: signature} for the defined symbols in `src`."""
    try:
        fx = extract_file(path, src or "")
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for n in fx.nodes:
        if n.kind == "file":
            continue
        out[n.qualified_name] = n.signature or ""
    return out


def semantic_diff(old_src: str, new_src: str, *, path: str = "file.py") -> dict:
    """Symbol-level diff between two versions of a file."""
    old = _symbols(old_src, path)
    new = _symbols(new_src, path)
    added = sorted(q for q in new if q not in old)
    removed = sorted(q for q in old if q not in new)
    changed = [
        {"symbol": q, "old": old[q], "new": new[q]}
        for q in sorted(new)
        if q in old and (old[q] or "") != (new[q] or "")
    ]
    return {"added": added, "removed": removed, "changed": changed}


def has_changes(d: dict) -> bool:
    return bool(d.get("added") or d.get("removed") or d.get("changed"))


def summarize_semantic_diff(d: dict, *, path: str | None = None) -> str:
    """A compact human summary, or '' when there are no symbol-level changes."""
    if not has_changes(d):
        return ""
    parts: list[str] = []
    if d.get("added"):
        parts.append("added " + ", ".join(f"`{s}`" for s in d["added"][:8]))
    if d.get("removed"):
        parts.append("removed " + ", ".join(f"`{s}`" for s in d["removed"][:8]))
    if d.get("changed"):
        parts.append("changed signature of "
                     + ", ".join(f"`{c['symbol']}`" for c in d["changed"][:8]))
    head = f"{path}: " if path else ""
    return head + "; ".join(parts)


__all__ = ["semantic_diff", "has_changes", "summarize_semantic_diff"]
