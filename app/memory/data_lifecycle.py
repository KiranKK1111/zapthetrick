"""Data lifecycle & privacy (Architecture §18).

Deterministic, user-initiated operations over the durable memory / knowledge-
graph data — never model-decided:

  * **retention purge** — delete episodes/skills past a retention window
    (`memory.retention_days`); 0 = keep indefinitely (nothing purged silently);
  * **export-all** — everything the device user owns, as one JSON bundle;
  * **delete-all** — erase everything: Postgres rows + vectors + blobs + learned
    exemplars;
  * **provenance forget** — evict one episode, or one KG node **and its incident
    edges** from a conversation/project graph.

Postgres-first; vector/blob cleanup is best-effort so a store outage can't leave
the DB half-deleted. All DB helpers take an explicit `user_id` scope.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models import (
    Episode as EpisodeRow,
    Message as MessageRow,
    Project as ProjectRow,
    Session as SessionRow,
    SkillRow,
)

log = logging.getLogger(__name__)


def _as_uuid(v):
    if v is None or isinstance(v, uuid.UUID):
        return v
    try:
        return uuid.UUID(str(v))
    except (TypeError, ValueError):
        return None


def _retention_days() -> int:
    try:
        from app.core.config_loader import cfg
        return int(getattr(cfg.memory, "retention_days", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


# ---- provenance-based forget: KG node + its edges (pure) -----------------

def forget_kg_node(kg: dict | None, node_id: str) -> dict:
    """Return a copy of a KG JSON with `node_id` removed **and every edge that
    touches it** (its downstream links). Case-insensitive on the node slug. This
    is how a user-facing "forget this" evicts a fact and its inferences (§18).
    Pure; never raises."""
    if not isinstance(kg, dict):
        return {"nodes": [], "edges": []}
    nid = str(node_id or "").strip().lower()
    if not nid:
        return {"nodes": list(kg.get("nodes") or []),
                "edges": list(kg.get("edges") or [])}
    nodes = [n for n in (kg.get("nodes") or [])
             if str(n.get("id", "")).strip().lower() != nid]
    edges = [e for e in (kg.get("edges") or [])
             if str(e.get("src", "")).strip().lower() != nid
             and str(e.get("dst", "")).strip().lower() != nid]
    return {"nodes": nodes, "edges": edges}


# ---- retention purge -----------------------------------------------------

async def purge_expired(
    session: AsyncSession,
    *,
    user_id=None,
    retention_days: int | None = None,
) -> dict:
    """Delete episodes + skills older than the retention window. Returns counts.
    A no-op (nothing deleted) when retention is disabled (days <= 0)."""
    days = retention_days if retention_days is not None else _retention_days()
    if days <= 0:
        return {"enabled": False, "episodes": 0, "skills": 0}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    uid = _as_uuid(user_id)

    ep_sel = select(EpisodeRow).where(EpisodeRow.created_at < cutoff)
    sk_sel = select(SkillRow).where(SkillRow.created_at < cutoff)
    if uid is not None:
        ep_sel = ep_sel.where(EpisodeRow.user_id == uid)
        sk_sel = sk_sel.where(SkillRow.user_id == uid)
    ep_rows = list((await session.execute(ep_sel)).scalars().all())
    sk_rows = list((await session.execute(sk_sel)).scalars().all())

    ep_points = [str(r.vector_point_id) for r in ep_rows if r.vector_point_id]
    sk_points = [str(r.vector_point_id) for r in sk_rows if r.vector_point_id]
    for r in ep_rows:
        await session.delete(r)
    for r in sk_rows:
        await session.delete(r)
    await session.commit()

    await _drop_vectors(f"episodic_memory_{user_id or 'default'}", ep_points)
    await _drop_vectors(f"semantic_memory_{user_id or 'default'}", sk_points)
    return {"enabled": True, "episodes": len(ep_rows), "skills": len(sk_rows),
            "cutoff": cutoff.isoformat()}


# ---- export-all ----------------------------------------------------------

async def export_all(session: AsyncSession, *, user_id) -> dict:
    """One JSON bundle of everything the device user owns (§18)."""
    uid = _as_uuid(user_id)
    sessions = list((await session.execute(
        _scope(select(SessionRow), SessionRow, uid))).scalars().all())
    session_ids = [s.id for s in sessions]
    messages = []
    if session_ids:
        messages = list((await session.execute(
            select(MessageRow).where(MessageRow.session_id.in_(session_ids))
        )).scalars().all())
    episodes = list((await session.execute(
        _scope(select(EpisodeRow), EpisodeRow, uid))).scalars().all())
    skills = list((await session.execute(
        _scope(select(SkillRow), SkillRow, uid))).scalars().all())
    projects = list((await session.execute(
        _scope(select(ProjectRow), ProjectRow, uid))).scalars().all())

    return {
        "user_id": str(uid) if uid else None,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "conversations": [{
            "id": str(s.id), "title": s.title, "type": s.type,
            "project_id": str(s.project_id) if s.project_id else None,
            "kg": (s.session_metadata or {}).get("kg"),
        } for s in sessions],
        "messages": [{
            "id": str(m.id), "conversation_id": str(m.session_id),
            "role": m.role, "content": m.content, "intent": m.intent,
        } for m in messages],
        "episodes": [{
            "id": str(e.id), "session_tag": e.session_tag,
            "project_id": str(e.project_id) if e.project_id else None,
            "question": e.question, "final": e.final, "intent": e.intent,
            "feedback": e.feedback,
        } for e in episodes],
        "skills": [{
            "id": str(k.id), "text": k.text, "kind": k.kind,
            "confidence": float(k.confidence) if k.confidence is not None else None,
        } for k in skills],
        "projects": [{
            "id": str(p.id), "name": p.name, "instructions": p.instructions,
            "kg": (p.project_metadata or {}).get("kg"),
        } for p in projects],
        "counts": {
            "conversations": len(sessions), "messages": len(messages),
            "episodes": len(episodes), "skills": len(skills),
            "projects": len(projects),
        },
    }


# ---- delete-all ----------------------------------------------------------

async def delete_all(session: AsyncSession, *, user_id) -> dict:
    """Erase everything the device user owns across Postgres + vectors + blobs +
    learned exemplars. Deterministic; Postgres-first."""
    uid = _as_uuid(user_id)
    sessions = list((await session.execute(
        _scope(select(SessionRow), SessionRow, uid))).scalars().all())
    session_ids = [s.id for s in sessions]

    # Collect blob paths + memory vector points BEFORE deleting rows.
    blob_paths: list[str] = []
    if session_ids:
        msgs = list((await session.execute(
            select(MessageRow).where(MessageRow.session_id.in_(session_ids))
        )).scalars().all())
        for m in msgs:
            src = m.sources if isinstance(m.sources, dict) else {}
            for key in ("images", "files"):
                for ref in (src.get(key) or []):
                    p = ref.get("path") if isinstance(ref, dict) else None
                    if p:
                        blob_paths.append(p)
    ep_rows = list((await session.execute(
        _scope(select(EpisodeRow), EpisodeRow, uid))).scalars().all())
    sk_rows = list((await session.execute(
        _scope(select(SkillRow), SkillRow, uid))).scalars().all())

    counts = {"conversations": len(sessions), "episodes": len(ep_rows),
              "skills": len(sk_rows)}

    # Postgres deletes (messages/agent_steps cascade off sessions).
    await session.execute(_scope(delete(EpisodeRow), EpisodeRow, uid))
    await session.execute(_scope(delete(SkillRow), SkillRow, uid))
    for s in sessions:
        await session.delete(s)
    proj_res = await session.execute(_scope(delete(ProjectRow), ProjectRow, uid))
    counts["projects"] = proj_res.rowcount or 0
    await session.commit()

    # Vectors: per-user memory collections + per-conversation chat docs.
    await _reset_collection(f"episodic_memory_{user_id or 'default'}")
    await _reset_collection(f"semantic_memory_{user_id or 'default'}")
    for sid in session_ids:
        with _suppress():
            from app.rag.documents import drop_chat_collection
            await drop_chat_collection(str(sid))
    await _delete_blobs(blob_paths)
    # Learned intent exemplars.
    with _suppress():
        from app.clarify.learned_exemplars import clear as _clear_ex
        _clear_ex()

    counts["blobs"] = len(blob_paths)
    return {"deleted": True, **counts}


# ---- provenance forget: one episode --------------------------------------

async def forget_episode(session: AsyncSession, episode_id: str) -> bool:
    """Delete one episode row + its vector (§18). Returns False if absent."""
    eid = _as_uuid(episode_id)
    if eid is None:
        return False
    row = await session.get(EpisodeRow, eid)
    if row is None:
        return False
    point = str(row.vector_point_id) if row.vector_point_id else None
    user_tag = row.user_id or "default"
    await session.delete(row)
    await session.commit()
    if point:
        await _drop_vectors(f"episodic_memory_{user_tag}", [point])
    return True


# ---- helpers -------------------------------------------------------------

def _scope(stmt, model, uid):
    """Scope a select/delete to a user when uid is known (device-local → all)."""
    return stmt.where(model.user_id == uid) if uid is not None else stmt


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True   # swallow — cleanup is best-effort


async def _drop_vectors(collection: str, point_ids: list[str]) -> None:
    if not point_ids:
        return
    with _suppress():
        from storage.vectors import get_vector_store
        await get_vector_store().delete(collection, ids=point_ids)


async def _reset_collection(collection: str) -> None:
    with _suppress():
        from storage.vectors import get_vector_store
        await get_vector_store().reset(collection)


async def _delete_blobs(paths: list[str]) -> None:
    if not paths:
        return
    with _suppress():
        from storage.blobs import get_blobs
        store = get_blobs()
        for p in paths:
            with _suppress():
                await store.delete(p)


__all__ = [
    "forget_kg_node", "purge_expired", "export_all", "delete_all",
    "forget_episode",
]
