"""Memory scope isolation (memory-graph R2, task 1.3).

Pins Property 2: a workspace object is scoped to it; retrieval returns only that
workspace + global; another workspace's memory is never returned; no-workspace
uses the Default_Workspace scope.
"""
from __future__ import annotations

from app.memory.objects import MemoryObject, SCOPE_GLOBAL, workspace_scope
from app.memory.mstore import MemoryStore, retrieval_scopes


def _store():
    s = MemoryStore()
    s.add(MemoryObject(content="global pref", scope=SCOPE_GLOBAL))
    s.add(MemoryObject(content="p1 decision", scope=workspace_scope("p1")))
    s.add(MemoryObject(content="p2 decision", scope=workspace_scope("p2")))
    return s


def test_workspace_scope_helper():
    assert workspace_scope("p1") == "workspace:p1"
    assert workspace_scope(None) == "workspace:default"


def test_retrieval_scopes_are_workspace_plus_global():
    assert retrieval_scopes("p1") == ["workspace:p1", SCOPE_GLOBAL]
    assert retrieval_scopes(None) == ["workspace:default", SCOPE_GLOBAL]


def test_by_scope_returns_only_requested():
    s = _store()
    got = s.by_scope(retrieval_scopes("p1"))
    contents = {o.content for o in got}
    assert "p1 decision" in contents and "global pref" in contents
    assert "p2 decision" not in contents        # other workspace isolated (R2.2)


def test_no_workspace_uses_default_scope():
    s = MemoryStore()
    s.add(MemoryObject(content="default work", scope=workspace_scope(None)))
    got = s.by_scope(retrieval_scopes(None))
    assert any(o.content == "default work" for o in got)
