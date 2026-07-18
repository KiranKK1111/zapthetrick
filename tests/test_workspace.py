"""Workspace manager (workspace-and-artifacts R1/R2, task 1.3).

Pins Properties 1, 2, 3: default ensure/backfill transparency, isolation by
(metadata) workspace, and additive (no-migration) ownership over Session rows.
"""
from __future__ import annotations

from app.wsgroup.manager import (
    DEFAULT_WORKSPACE_ID, WorkspaceManager, session_workspace_id, belongs_to,
)


def test_ensure_default_creates_personal_workspace():
    root: dict = {}
    mgr = WorkspaceManager(root)
    assert mgr.ensure_default() == DEFAULT_WORKSPACE_ID
    assert any(w["id"] == DEFAULT_WORKSPACE_ID for w in mgr.list())


def test_flat_session_belongs_to_default():
    # A pre-existing conversation with no workspace_id → Default_Workspace.
    assert session_workspace_id(None) == DEFAULT_WORKSPACE_ID
    assert session_workspace_id({}) == DEFAULT_WORKSPACE_ID
    assert belongs_to({}, None) is True
    assert belongs_to({}, DEFAULT_WORKSPACE_ID) is True


def test_create_and_isolation():
    root: dict = {}
    mgr = WorkspaceManager(root)
    wid = mgr.create("Project X")
    assert mgr.get(wid)["name"] == "Project X"
    # A session tagged to wid belongs only to wid, not to default.
    meta = {"workspace_id": wid}
    assert belongs_to(meta, wid) is True
    assert belongs_to(meta, DEFAULT_WORKSPACE_ID) is False
    assert belongs_to({}, wid) is False


def test_set_active_is_exclusive():
    root: dict = {}
    mgr = WorkspaceManager(root)
    a = mgr.create("A")
    b = mgr.create("B")
    assert mgr.set_active(a) is True
    assert mgr.active_id() == a
    mgr.set_active(b)
    assert mgr.active_id() == b
    # Only one active at a time.
    actives = [w for w in mgr.list() if w.get("active")]
    assert len(actives) == 1


def test_ownership_is_additive_metadata_only():
    # Ownership rides session_metadata — no new column, content untouched.
    meta = {"some": "existing", "workspace_id": "ws1"}
    assert session_workspace_id(meta) == "ws1"
    assert meta["some"] == "existing"        # existing metadata preserved


def test_set_active_unknown_workspace_fails():
    mgr = WorkspaceManager({})
    assert mgr.set_active("nope") is False
