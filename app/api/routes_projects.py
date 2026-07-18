"""Projects API (Architecture §17).

A project groups conversations (sessions) and scopes their graphs. CRUD plus
assignment of a conversation to a project. Everything is scoped to the current
device user so one machine's projects never leak into another's.

    POST   /api/projects                         create {name, instructions?}
    GET    /api/projects                          list (newest first; ?archived=)
    GET    /api/projects/{id}                      one project + its conversation count
    PATCH  /api/projects/{id}                      update {name?, instructions?, archived?}
    DELETE /api/projects/{id}                      delete (sessions → ungrouped)
    GET    /api/projects/{id}/conversations        conversations in the project
    PUT    /api/projects/{id}/conversations/{cid}  assign a conversation to the project
    DELETE /api/projects/{id}/conversations/{cid}  remove a conversation from the project
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from storage.device import ensure_device_user
from storage.models import Project, Session as SessionRow

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["projects"])

_INSTR_CAP = 4000


def _project_dict(p: Project, *, conversation_count: int | None = None) -> dict:
    d = {
        "id": str(p.id),
        "name": p.name,
        "instructions": p.instructions or "",
        "archived": bool(p.archived),
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }
    if conversation_count is not None:
        d["conversation_count"] = conversation_count
    return d


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid {what} id")


async def _owned_project(session: AsyncSession, project_id: str,
                         uid: uuid.UUID | None) -> Project:
    """Fetch a project scoped to the current device user, or 404."""
    pid = _parse_uuid(project_id, "project")
    proj = await session.get(Project, pid)
    if proj is None or (uid is not None and proj.user_id not in (None, uid)):
        raise HTTPException(status_code=404, detail="Project not found")
    return proj


@router.post("")
async def create_project(body: dict,
                         session: AsyncSession = Depends(get_session)) -> dict:
    uid = await ensure_device_user()
    name = (body.get("name") or "").strip()[:200] or "New project"
    instructions = (body.get("instructions") or "").strip()[:_INSTR_CAP] or None
    proj = Project(user_id=uid, name=name, instructions=instructions)
    session.add(proj)
    await session.commit()
    await session.refresh(proj)
    return _project_dict(proj, conversation_count=0)


@router.get("")
async def list_projects(archived: bool = False,
                        session: AsyncSession = Depends(get_session)) -> dict:
    uid = await ensure_device_user()
    # Conversation counts in one grouped query (no per-project round-trip).
    counts = dict(
        (await session.execute(
            select(SessionRow.project_id, func.count(SessionRow.id))
            .where(SessionRow.project_id.isnot(None))
            .group_by(SessionRow.project_id)
        )).all()
    )
    stmt = select(Project).where(Project.archived == archived)
    if uid is not None:
        stmt = stmt.where((Project.user_id == uid) | (Project.user_id.is_(None)))
    stmt = stmt.order_by(Project.updated_at.desc())
    rows = (await session.execute(stmt)).scalars().all()
    return {"projects": [
        _project_dict(p, conversation_count=int(counts.get(p.id, 0)))
        for p in rows
    ]}


@router.get("/{project_id}")
async def get_project(project_id: str,
                      session: AsyncSession = Depends(get_session)) -> dict:
    uid = await ensure_device_user()
    proj = await _owned_project(session, project_id, uid)
    count = int((await session.execute(
        select(func.count(SessionRow.id))
        .where(SessionRow.project_id == proj.id)
    )).scalar() or 0)
    return _project_dict(proj, conversation_count=count)


@router.patch("/{project_id}")
async def update_project(project_id: str, body: dict,
                         session: AsyncSession = Depends(get_session)) -> dict:
    uid = await ensure_device_user()
    proj = await _owned_project(session, project_id, uid)
    if "name" in body:
        name = (body.get("name") or "").strip()[:200]
        if name:
            proj.name = name
    if "instructions" in body:
        instr = (body.get("instructions") or "").strip()[:_INSTR_CAP]
        proj.instructions = instr or None
    if "archived" in body:
        proj.archived = bool(body.get("archived"))
    await session.commit()
    await session.refresh(proj)
    return _project_dict(proj)


@router.delete("/{project_id}")
async def delete_project(project_id: str,
                         delete_conversations: bool = False,
                         session: AsyncSession = Depends(get_session)) -> dict:
    """Delete a project. By default its conversations are DETACHED (moved back
    to ungrouped). With `delete_conversations=true` the conversations are
    deleted too, along with their owned artifacts (blobs, vector collections,
    code graphs) — same cleanup as a normal conversation delete."""
    uid = await ensure_device_user()
    proj = await _owned_project(session, project_id, uid)

    deleted_convos = 0
    convo_ids: list = []
    if delete_conversations:
        convo_ids = list((await session.execute(
            select(SessionRow.id).where(SessionRow.project_id == proj.id)
        )).scalars().all())
    if delete_conversations and convo_ids:
        # Gather owned blob paths BEFORE the cascade removes the message rows.
        from storage.models import Message
        blob_paths: list[str] = []
        try:
            msgs = (await session.execute(
                select(Message).where(Message.session_id.in_(convo_ids))
            )).scalars().all()
            for m in msgs:
                src = getattr(m, "sources", None)
                if not isinstance(src, dict):
                    continue
                for key in ("images", "files"):
                    for ref in (src.get(key) or []):
                        p = ref.get("path") if isinstance(ref, dict) else None
                        if p:
                            blob_paths.append(p)
        except Exception:  # noqa: BLE001 — never block the delete on collection
            blob_paths = []

        # Delete the sessions (messages cascade via FK ondelete=CASCADE).
        from sqlalchemy import delete as _delete
        await session.execute(
            _delete(SessionRow).where(SessionRow.id.in_(convo_ids))
        )
        deleted_convos = len(convo_ids)
        await session.delete(proj)
        await session.commit()

        # Best-effort artifact cleanup (never fail the delete on these).
        if blob_paths:
            try:
                from storage.blobs import get_blobs
                store = get_blobs()
                for p in blob_paths:
                    try:
                        await store.delete(p)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
        for cid in convo_ids:
            try:
                from app.rag.documents import drop_chat_collection
                await drop_chat_collection(str(cid))
            except Exception:  # noqa: BLE001
                pass
        try:
            from sqlalchemy import text as _text
            await session.execute(
                _text("DELETE FROM code_graphs WHERE conversation_id = ANY(:ids)"),
                {"ids": [str(c) for c in convo_ids]},
            )
            await session.commit()
        except Exception:  # noqa: BLE001
            pass
    else:
        # Conversations survive — the FK is ON DELETE SET NULL, but detach them
        # explicitly first so the response reflects the count.
        await session.execute(
            update(SessionRow).where(SessionRow.project_id == proj.id)
            .values(project_id=None)
        )
        await session.delete(proj)
        await session.commit()

    return {"deleted": True, "id": project_id,
            "deleted_conversations": deleted_convos}


@router.get("/{project_id}/conversations")
async def project_conversations(project_id: str,
                                session: AsyncSession = Depends(get_session)) -> dict:
    uid = await ensure_device_user()
    proj = await _owned_project(session, project_id, uid)
    rows = (await session.execute(
        select(SessionRow).where(SessionRow.project_id == proj.id)
        .order_by(SessionRow.updated_at.desc())
    )).scalars().all()
    return {"conversations": [
        {"id": str(s.id), "title": s.title,
         "updated_at": s.updated_at.isoformat() if s.updated_at else None,
         "message_count": s.message_count}
        for s in rows
    ]}


@router.put("/{project_id}/conversations/{conversation_id}")
async def assign_conversation(project_id: str, conversation_id: str,
                              session: AsyncSession = Depends(get_session)) -> dict:
    uid = await ensure_device_user()
    proj = await _owned_project(session, project_id, uid)
    cid = _parse_uuid(conversation_id, "conversation")
    convo = await session.get(SessionRow, cid)
    if convo is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    convo.project_id = proj.id
    await session.commit()
    return {"ok": True, "project_id": str(proj.id),
            "conversation_id": conversation_id}


@router.delete("/{project_id}/conversations/{conversation_id}")
async def unassign_conversation(project_id: str, conversation_id: str,
                                session: AsyncSession = Depends(get_session)) -> dict:
    uid = await ensure_device_user()
    await _owned_project(session, project_id, uid)   # 404 if not owned
    cid = _parse_uuid(conversation_id, "conversation")
    convo = await session.get(SessionRow, cid)
    if convo is not None and convo.project_id is not None:
        convo.project_id = None
        await session.commit()
    return {"ok": True, "conversation_id": conversation_id}
