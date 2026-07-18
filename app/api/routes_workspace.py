"""Workspace management endpoints.

    GET  /api/workspaces                 list saved workspaces + active
    GET  /api/workspaces/drivers         driver catalog (UI form schema)
    GET  /api/workspaces/{name}          one workspace's full config
    POST /api/workspaces                 create / update a workspace
    POST /api/workspaces/{name}/activate switch the active workspace
    POST /api/workspaces/{name}/probe    run the 10-step connection probe
    DELETE /api/workspaces/{name}        remove
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.workspace import (
    DRIVERS,
    Workspace,
    default_workspace_repo,
    probe_workspace,
)


router = APIRouter(prefix="/api/workspaces")


def _redact(ws: Workspace) -> dict:
    """Strip secret fields before sending to the UI."""
    out = ws.to_dict()
    for slot in ("relational", "vector", "cache", "blob"):
        v = out.get(slot) or {}
        for k in ("password", "secret", "api_key"):
            if k in v and v[k]:
                v[k] = "********"
        out[slot] = v
    return out


@router.get("")
async def list_workspaces() -> dict:
    repo = default_workspace_repo()
    active = repo.active()
    return {
        "active": active.name if active else None,
        "workspaces": [_redact(w) for w in repo.list()],
    }


@router.get("/drivers")
async def driver_catalog() -> dict:
    return {
        drv_id: {
            "kind": drv.kind.value,
            "label": drv.label,
            "requires_pkg": drv.requires_pkg,
            "default_port": drv.default_port,
            "fields": [
                {
                    "name": f.name,
                    "type": f.type,
                    "label": f.label,
                    "default": f.default,
                    "choices": f.choices,
                    "secret": f.secret,
                    "required": f.required,
                    "notes": f.notes,
                }
                for f in drv.fields
            ],
        }
        for drv_id, drv in DRIVERS.items()
    }


@router.get("/{name}")
async def get_workspace(name: str) -> dict:
    ws = default_workspace_repo().get(name)
    if ws is None:
        raise HTTPException(404, detail="workspace not found")
    return _redact(ws)


class WorkspaceBody(BaseModel):
    name: str
    relational: dict = {}
    vector: dict = {}
    cache: dict = {}
    blob: dict = {}


@router.post("")
async def upsert_workspace(body: WorkspaceBody) -> dict:
    repo = default_workspace_repo()
    existing = repo.get(body.name)
    # Preserve secrets the UI re-sent as ********.
    if existing:
        for slot in ("relational", "vector", "cache", "blob"):
            incoming = getattr(body, slot) or {}
            current = getattr(existing, slot) or {}
            for k in ("password", "secret", "api_key"):
                if incoming.get(k) in ("********", "***"):
                    incoming[k] = current.get(k)
            setattr(body, slot, incoming)
    repo.upsert(Workspace(
        name=body.name,
        relational=body.relational,
        vector=body.vector,
        cache=body.cache,
        blob=body.blob,
    ))
    return {"ok": True, "name": body.name}


@router.post("/{name}/activate")
async def activate_workspace(name: str) -> dict:
    """Switch the active workspace AND swap the live storage engine.

    `set_active` updates `~/.zapthetrick/workspaces.json`; `_apply_db_changes`
    then disposes the current Postgres pool + re-runs Alembic against
    the workspace's connection. Mirrors what
    `POST /api/settings (with database.postgres)` does today, so the
    switch is live by the time the route returns.
    """
    ok = default_workspace_repo().set_active(name)
    if not ok:
        raise HTTPException(404, detail="workspace not found")

    # Drop the cached vector-store client so the next request rebuilds
    # against the new workspace's Qdrant URL + API key. The relational
    # reinit happens inside _apply_db_changes.
    try:
        from storage.vectors import factory as _vstore

        _vstore.reset()
    except Exception:  # noqa: BLE001
        pass

    try:
        from app.api.routes_settings import _apply_db_changes

        await _apply_db_changes()
    except Exception as exc:  # noqa: BLE001 — never fail an activate on apply
        # Surface the state but keep the activation — Settings/Database
        # will reflect "error" so the user can retry from the UI.
        return {"ok": True, "active": name, "apply_warning": str(exc)}
    return {"ok": True, "active": name}


@router.post("/{name}/probe")
async def probe(name: str) -> dict:
    ws = default_workspace_repo().get(name)
    if ws is None:
        raise HTTPException(404, detail="workspace not found")
    report = await probe_workspace(ws)
    return {
        "workspace": report.workspace,
        "overall_ok": report.overall_ok,
        "steps": [asdict(s) for s in report.steps],
    }


@router.delete("/{name}")
async def delete_workspace(name: str) -> dict:
    ok = default_workspace_repo().delete(name)
    if not ok:
        raise HTTPException(404, detail="workspace not found")
    return {"ok": True}
