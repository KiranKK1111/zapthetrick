"""Product-Workspace grouping + artifact endpoints (workspace-and-artifacts).

Distinct from `routes_workspace.py` (DB connection profiles) — these manage the
product-level Workspace that groups a user's conversations / files / artifacts,
and expose artifact versions + restore for the FE artifact panel.

All endpoints are additive + fail-open: with no workspace action the
Default_Workspace is returned transparently (Property 1). Metadata lives in
`User.preferences` (no schema migration); artifact bytes live in the blob store.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.database import get_session
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/wsgroups", tags=["workspace-grouping"])


async def _manager(session: AsyncSession):
    """Load a WorkspaceManager over the device user's preferences. Returns
    (manager, store, user) or (None, None, None) when unavailable."""
    from app.clarify import load_store
    from app.wsgroup import WorkspaceManager
    from storage.device import ensure_device_user

    uid = await ensure_device_user()
    store, user = await load_store(session, uid)
    if store is None:
        return None, None, None
    return WorkspaceManager(store.root), store, user


@router.get("")
async def list_workspaces(session: AsyncSession = Depends(get_session)) -> dict:
    mgr, _, _ = await _manager(session)
    if mgr is None:
        return {"active": "default", "workspaces": [
            {"id": "default", "name": "Personal", "active": True}]}
    return {"active": mgr.active_id(), "workspaces": mgr.list()}


class CreateBody(BaseModel):
    name: str


@router.post("")
async def create_workspace(body: CreateBody,
                           session: AsyncSession = Depends(get_session)) -> dict:
    from app.clarify import save_store
    mgr, store, user = await _manager(session)
    if mgr is None:
        raise HTTPException(503, detail="workspaces unavailable")
    wid = mgr.create(body.name)
    await save_store(session, user, store)
    return {"ok": True, "id": wid, "name": body.name}


class ActiveBody(BaseModel):
    id: str


@router.post("/active")
async def set_active(body: ActiveBody,
                     session: AsyncSession = Depends(get_session)) -> dict:
    from app.clarify import save_store
    mgr, store, user = await _manager(session)
    if mgr is None:
        raise HTTPException(503, detail="workspaces unavailable")
    if not mgr.set_active(body.id):
        raise HTTPException(404, detail="workspace not found")
    await save_store(session, user, store)
    return {"ok": True, "active": body.id}


@router.get("/artifacts/{artifact_id}/versions")
async def artifact_versions(artifact_id: str) -> dict:
    from app.artifacts import artifact_store
    store = artifact_store()
    art = store.get(artifact_id)
    if art is None:
        raise HTTPException(404, detail="artifact not found")
    return {
        "id": art.id, "kind": art.kind, "title": art.title,
        "current": art.current_version,
        "versions": [{"version": v.version, "format": v.fmt,
                      "created": v.created} for v in store.versions(artifact_id)],
    }


@router.get("/artifacts/{artifact_id}/versions/{version}/content")
async def artifact_version_content(artifact_id: str, version: int) -> dict:
    """A single version's content as text — powers the inter-version diff view."""
    from app.artifacts import artifact_store
    store = artifact_store()
    if store.get(artifact_id) is None:
        raise HTTPException(404, detail="artifact not found")
    data = await store.content(artifact_id, version)
    return {"version": version,
            "content": data.decode("utf-8", errors="replace")}


class EditBody(BaseModel):
    content: str
    format: str | None = None


@router.post("/artifacts/{artifact_id}/content")
async def artifact_edit(artifact_id: str, body: EditBody) -> dict:
    """Save edited artifact content as a NEW version (editable artifact pane).
    The version chain is append-only, so every edit is undoable via /restore."""
    from app.artifacts import artifact_store
    store = artifact_store()
    if store.get(artifact_id) is None:
        raise HTTPException(404, detail="artifact not found")
    ver = await store.append_version(artifact_id, body.content, body.format)
    return {"ok": True, "version": ver.version}


class RestoreBody(BaseModel):
    version: int


@router.post("/artifacts/{artifact_id}/restore")
async def artifact_restore(artifact_id: str, body: RestoreBody) -> dict:
    from app.artifacts import artifact_store
    store = artifact_store()
    ver = await store.restore(artifact_id, body.version)
    if ver is None:
        raise HTTPException(404, detail="artifact or version not found")
    return {"ok": True, "restored_from": body.version, "version": ver.version}


__all__ = ["router"]
