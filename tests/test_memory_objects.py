"""Memory_Object model + store (memory-graph R1, task 1.3).

Pins Property 1: typed objects, persistence round-trip (no migration), and a
clean fallback (empty store → caller uses today's recall).
"""
from __future__ import annotations

from app.memory.objects import MemoryObject, KINDS, SCOPE_GLOBAL
from app.memory.mstore import MemoryStore


def test_typed_object_fields():
    o = MemoryObject(content="user prefers dark mode", kind="preference",
                     scope=SCOPE_GLOBAL, importance=0.8, durable=True)
    assert o.kind in KINDS and o.id
    assert o.created_at > 0 and o.updated_at > 0
    d = o.to_dict()
    assert d["content"] == "user prefers dark mode" and d["durable"] is True


def test_add_get_and_len():
    s = MemoryStore()
    o = s.add(MemoryObject(content="x", kind="fact"))
    assert s.get(o.id) is o
    assert len(s) == 1
    assert s.get("missing") is None


def test_persistence_round_trip_no_migration():
    s = MemoryStore()
    s.add(MemoryObject(content="stack is Flutter", kind="entity",
                       scope="workspace:p1", importance=0.7))
    prefs: dict = {}
    s.export_to(prefs)
    assert "memory_objects" in prefs and len(prefs["memory_objects"]) == 1

    s2 = MemoryStore()
    s2.load_from(prefs)
    assert len(s2) == 1
    assert s2.all()[0].content == "stack is Flutter"


def test_empty_store_is_fallback():
    s = MemoryStore()
    assert s.all() == []
    assert s.by_scope(SCOPE_GLOBAL) == []        # nothing → caller falls back


def test_clear_user_removes_all():
    s = MemoryStore()
    s.add(MemoryObject(content="a"))
    s.add(MemoryObject(content="b"))
    assert s.clear_user() == 2 and len(s) == 0
