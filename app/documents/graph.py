"""Artifact relationship graph + cross-artifact search — Phase 6.

DocuementGeneration.md #3 (relationship graph — a response and all its versions/
formats linked) and #10 (cross-artifact search — "find where authentication was
discussed" across generated documents). Built over the Phase-5 `generated_
documents` store: `doc_key` gives the version chain (an edge set per document),
`session_id` links sibling documents in a conversation.

All read-only + fail-open at the callers (the endpoints wrap them). The search
uses ILIKE so it works without a full-text index configured; a tsvector upgrade
can slot in later behind the same signature.
"""
from __future__ import annotations

import uuid
from typing import Optional, Union


def _as_uuid(value: Union[str, uuid.UUID, None]) -> Optional[uuid.UUID]:
    if value is None or isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def _snippet(content: str, query: str, width: int = 140) -> str:
    """A short context window around the first match (for search results)."""
    low = (content or "").lower()
    i = low.find((query or "").lower())
    if i < 0:
        return (content or "")[:width].strip()
    start = max(0, i - width // 3)
    end = min(len(content), i + len(query) + width)
    pre = "…" if start > 0 else ""
    post = "…" if end < len(content) else ""
    return pre + " ".join(content[start:end].split()) + post


async def search_documents(session, query: str, *,
                           session_id: Union[str, uuid.UUID, None] = None,
                           limit: int = 20) -> list[dict]:
    """Cross-artifact text search over generated documents (title + content).
    Newest first; each hit carries a snippet. Empty query → []."""
    from sqlalchemy import or_, select
    from storage.models import GeneratedDocument as GD

    q = (query or "").strip()
    if not q:
        return []
    like = f"%{q}%"
    stmt = select(GD).where(or_(GD.title.ilike(like), GD.content_md.ilike(like)))
    sid = _as_uuid(session_id)
    if sid is not None:
        stmt = stmt.where(GD.session_id == sid)
    stmt = stmt.order_by(GD.created_at.desc()).limit(max(1, int(limit)))
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "doc_key": str(r.doc_key), "version": r.version, "title": r.title,
            "format": r.doc_format,
            "snippet": _snippet(r.content_md, q),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def build_artifact_graph(session,
                               session_id: Union[str, uuid.UUID, None]) -> dict:
    """The relationship graph for a conversation: one node per logical document
    (its `doc_key`), each with its version chain; sibling documents share the
    conversation. Nodes newest-first."""
    from app.documents.store import list_for_session

    rows = await list_for_session(session, session_id, limit=200)
    # Group rows by doc_key, preserving first-seen (newest) order.
    docs: dict[str, dict] = {}
    for r in rows:
        key = str(r.doc_key)
        node = docs.setdefault(key, {
            "doc_key": key, "title": r.title, "format": r.doc_format,
            "latest_version": 0, "versions": [],
        })
        node["versions"].append({
            "version": r.version, "title": r.title, "format": r.doc_format,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
        node["latest_version"] = max(node["latest_version"], r.version)
    for node in docs.values():
        node["versions"].sort(key=lambda v: v["version"])
    node_list = list(docs.values())
    # Sibling edges: every pair of documents in the same conversation.
    keys = [n["doc_key"] for n in node_list]
    edges = [{"from": a, "to": b, "kind": "sibling"}
             for i, a in enumerate(keys) for b in keys[i + 1:]]
    return {
        "session_id": str(_as_uuid(session_id)) if session_id else None,
        "documents": node_list,
        "edges": edges,
    }


__all__ = ["search_documents", "build_artifact_graph"]
