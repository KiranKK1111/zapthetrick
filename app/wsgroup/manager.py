"""Workspace-grouping manager (workspace-and-artifacts R1/R2).

Pure manager over the user's ``preferences`` blob (under the ``ws_group`` key) +
the per-session ``session_metadata.workspace_id`` owner reference. No schema
migration: a Workspace is metadata only, and conversation ownership is an
additive field on the existing JSONB. The Default_Workspace is implicit — a
session with no recorded ``workspace_id`` belongs to it (Property 1/3).

Isolation (R2) is the pure `belongs_to(session_metadata, workspace_id)` filter,
scoped to the current user by the caller.
"""
from __future__ import annotations

import time
import uuid

DEFAULT_WORKSPACE_ID = "default"


def session_workspace_id(session_metadata: dict | None) -> str:
    """The workspace a session belongs to — its recorded id, else the
    Default_Workspace (a flat/pre-existing conversation)."""
    try:
        wid = (session_metadata or {}).get("workspace_id")
        return str(wid) if wid else DEFAULT_WORKSPACE_ID
    except Exception:  # noqa: BLE001
        return DEFAULT_WORKSPACE_ID


def belongs_to(session_metadata: dict | None, workspace_id: str | None) -> bool:
    """Isolation predicate (R2.1): does this session belong to `workspace_id`?
    A None/'default' filter matches flat (unassigned) conversations."""
    target = str(workspace_id) if workspace_id else DEFAULT_WORKSPACE_ID
    return session_workspace_id(session_metadata) == target


class WorkspaceManager:
    """Manage the workspaces dict in a shared preferences root. Mutates in place;
    the caller persists (same pattern as the clarify/learning stores)."""

    def __init__(self, prefs: dict | None):
        self.root: dict = prefs if isinstance(prefs, dict) else {}
        ws = self.root.get("ws_group")
        if not isinstance(ws, dict):
            ws = {}
            self.root["ws_group"] = ws
        self._ws = ws

    def ensure_default(self) -> str:
        """Guarantee the Default_Workspace exists; return its id (R1.2)."""
        if DEFAULT_WORKSPACE_ID not in self._ws:
            self._ws[DEFAULT_WORKSPACE_ID] = {
                "name": "Personal", "created": time.time(), "active": True}
        return DEFAULT_WORKSPACE_ID

    def list(self) -> list[dict]:
        self.ensure_default()
        return [{"id": wid, **meta} for wid, meta in self._ws.items()]

    def create(self, name: str) -> str:
        self.ensure_default()
        wid = uuid.uuid4().hex[:12]
        self._ws[wid] = {"name": (name or "Workspace").strip()[:80],
                         "created": time.time(), "active": False}
        return wid

    def get(self, workspace_id: str) -> dict | None:
        m = self._ws.get(workspace_id)
        return {"id": workspace_id, **m} if isinstance(m, dict) else None

    def active_id(self) -> str:
        self.ensure_default()
        for wid, meta in self._ws.items():
            if isinstance(meta, dict) and meta.get("active"):
                return wid
        return DEFAULT_WORKSPACE_ID

    def set_active(self, workspace_id: str) -> bool:
        self.ensure_default()
        if workspace_id not in self._ws:
            return False
        for wid, meta in self._ws.items():
            if isinstance(meta, dict):
                meta["active"] = (wid == workspace_id)
        return True


__all__ = [
    "DEFAULT_WORKSPACE_ID", "WorkspaceManager", "session_workspace_id",
    "belongs_to",
]
