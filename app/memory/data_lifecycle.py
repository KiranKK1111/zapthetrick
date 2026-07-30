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
            # Timestamps travel so a re-import (and the human reading the
            # exported file) keeps the thread in its original order.
            "started_at": s.started_at.isoformat() if s.started_at else None,
        } for s in sessions],
        "messages": [{
            "id": str(m.id), "conversation_id": str(m.session_id),
            "role": m.role, "content": m.content, "intent": m.intent,
            "created_at": m.created_at.isoformat() if m.created_at else None,
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


# ---- import (the mirror of export-all) -----------------------------------

#: Section names a caller may ask to import. `chat_sessions` / `live_sessions`
#: are the two halves of `conversations`, split on `Session.type`, because that
#: is the choice users actually make ("bring my Live history over, not my
#: chats"). `memories` covers episodes + skills — they are one concept to a user.
IMPORT_SECTIONS = ("chat_sessions", "live_sessions", "projects", "memories")


def sections_in_bundle(bundle: dict) -> dict[str, int]:
    """What an export bundle actually contains, as `{section: row count}`.

    Pure — the API exposes it so the client can offer only the checkboxes that
    would do something, and the importer uses it for its result summary.
    Unknown/garbage input yields all-zero counts rather than raising."""
    b = bundle if isinstance(bundle, dict) else {}
    convos = [c for c in (b.get("conversations") or []) if isinstance(c, dict)]
    live = [c for c in convos if str(c.get("type") or "chat") == "live"]
    chat = [c for c in convos if str(c.get("type") or "chat") != "live"]
    memories = len([e for e in (b.get("episodes") or []) if isinstance(e, dict)]) \
        + len([s for s in (b.get("skills") or []) if isinstance(s, dict)])
    return {
        "chat_sessions": len(chat),
        "live_sessions": len(live),
        "projects": len([p for p in (b.get("projects") or [])
                         if isinstance(p, dict)]),
        "memories": memories,
    }


def _parse_dt(v):
    """Best-effort ISO-8601 → aware datetime. None/garbage → None (the column
    default then supplies `now()`)."""
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def import_bundle(
    session: AsyncSession,
    *,
    user_id,
    bundle: dict,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Add the contents of an export bundle to this user's data.

    ADDITIVE by design: every row is inserted under a FRESH id and owned by
    `user_id`, so importing never overwrites (or silently merges into) something
    the user already has — the failure mode of an id-preserving import is losing
    live data to a stale file, which is unrecoverable. Duplicates are visible and
    deletable; clobbered history is not.

    Ids inside the bundle are used only to re-link the graph: message →
    conversation, and conversation/episode → project.

    Returns `{"imported": {...counts...}, "skipped": {...}}`.
    """
    uid = _as_uuid(user_id)
    b = bundle if isinstance(bundle, dict) else {}
    wanted = set(sections or IMPORT_SECTIONS)
    unknown = sorted(wanted - set(IMPORT_SECTIONS))
    wanted &= set(IMPORT_SECTIONS)

    counts = {k: 0 for k in IMPORT_SECTIONS}
    counts["messages"] = 0

    # Projects first — conversations and episodes reference them.
    project_map: dict[str, uuid.UUID] = {}
    if "projects" in wanted:
        for p in (b.get("projects") or []):
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "Imported project").strip() \
                or "Imported project"
            row = ProjectRow(
                user_id=uid,
                name=name,
                instructions=(p.get("instructions") or None),
                project_metadata={"kg": p.get("kg")} if p.get("kg") else {},
            )
            session.add(row)
            await session.flush()
            if p.get("id"):
                project_map[str(p["id"])] = row.id
            counts["projects"] += 1

    want_chat = "chat_sessions" in wanted
    want_live = "live_sessions" in wanted
    session_map: dict[str, uuid.UUID] = {}
    if want_chat or want_live:
        for c in (b.get("conversations") or []):
            if not isinstance(c, dict):
                continue
            kind = str(c.get("type") or "chat")
            is_live = kind == "live"
            if is_live and not want_live:
                continue
            if not is_live and not want_chat:
                continue
            row = SessionRow(
                user_id=uid,
                type=kind,
                title=str(c.get("title") or "Imported session"),
                project_id=project_map.get(str(c.get("project_id") or "")),
                session_metadata={"kg": c.get("kg")} if c.get("kg") else {},
            )
            started = _parse_dt(c.get("started_at"))
            if started is not None:
                row.started_at = started
            session.add(row)
            await session.flush()
            if c.get("id"):
                session_map[str(c["id"])] = row.id
            counts["live_sessions" if is_live else "chat_sessions"] += 1

        # Messages follow their conversation; ones whose conversation wasn't
        # imported (unselected half, or a bundle missing the parent) are dropped.
        orphans = 0
        for m in (b.get("messages") or []):
            if not isinstance(m, dict):
                continue
            parent = session_map.get(str(m.get("conversation_id") or ""))
            if parent is None:
                orphans += 1
                continue
            content = m.get("content")
            if content is None:
                orphans += 1
                continue
            row = MessageRow(
                session_id=parent,
                role=str(m.get("role") or "user")[:20],
                content=str(content),
                intent=(str(m["intent"])[:50] if m.get("intent") else None),
            )
            created = _parse_dt(m.get("created_at"))
            if created is not None:
                row.created_at = created
            session.add(row)
            counts["messages"] += 1
        counts["orphan_messages"] = orphans

    if "memories" in wanted:
        for e in (b.get("episodes") or []):
            if not isinstance(e, dict):
                continue
            session.add(EpisodeRow(
                user_id=uid,
                session_tag=(e.get("session_tag") or None),
                project_id=project_map.get(str(e.get("project_id") or "")),
                question=str(e.get("question") or ""),
                final=str(e.get("final") or ""),
                intent=(e.get("intent") or None),
                feedback=(e.get("feedback") or None),
            ))
            counts["memories"] += 1
        for k in (b.get("skills") or []):
            if not isinstance(k, dict):
                continue
            text = str(k.get("text") or "").strip()
            if not text:
                continue
            row = SkillRow(user_id=uid, text=text,
                           kind=str(k.get("kind") or "lesson"))
            if k.get("confidence") is not None:
                try:
                    row.confidence = float(k["confidence"])
                except (TypeError, ValueError):
                    pass
            session.add(row)
            counts["memories"] += 1

    await session.commit()
    result = {"imported": counts, "available": sections_in_bundle(b)}
    if unknown:
        result["ignored_sections"] = unknown
    return result


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
    "forget_episode", "import_bundle", "sections_in_bundle", "IMPORT_SECTIONS",
]
