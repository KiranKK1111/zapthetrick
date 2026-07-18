"""Memory lifecycle (memory-graph R4/R5, tasks 5.2/6.2).

Pins Properties 4 & 5: aging decay, reinforcement, durable floor; promotion of
durable/important objects, duplicate + low-importance eviction, and bounded
per-scope storage.
"""
from __future__ import annotations

import time

from app.memory.objects import MemoryObject, SCOPE_GLOBAL, workspace_scope
from app.memory.mstore import MemoryStore
from app.memory import lifecycle as L


def test_aging_decays_importance():
    s = MemoryStore()
    o = MemoryObject(content="x", importance=1.0)
    o.updated_at = time.time() - 30 * 86400      # one half-life ago
    s.add(o)
    L.age(s, half_life_s=30 * 86400)
    assert 0.4 < o.importance < 0.6              # ~halved


def test_durable_floor_holds():
    s = MemoryStore()
    o = MemoryObject(content="durable pref", importance=1.0, durable=True)
    o.updated_at = time.time() - 365 * 86400     # very old
    s.add(o)
    L.age(s, half_life_s=30 * 86400)
    assert o.importance >= 0.4                   # never below the durable floor


def test_reinforce_raises_importance():
    s = MemoryStore()
    o = MemoryObject(content="x", importance=0.3)
    s.add(o)
    L.reinforce(s, o.id, 0.2)
    assert abs(o.importance - 0.5) < 1e-9


def test_promotion_to_global():
    s = MemoryStore()
    o = MemoryObject(content="confirmed stack", scope=workspace_scope("p1"),
                     importance=0.9, durable=True)
    s.add(o)
    res = L.maintain(s)
    assert res["promoted"] == 1
    assert s.get(o.id).scope == SCOPE_GLOBAL


def test_eviction_of_low_importance():
    s = MemoryStore()
    s.add(MemoryObject(content="junk", importance=0.01, durable=False))
    keep = s.add(MemoryObject(content="keep", importance=0.9, durable=True))
    res = L.maintain(s)
    assert res["evicted"] >= 1
    assert s.get(keep.id) is not None


def test_dedupe_keeps_most_important():
    s = MemoryStore()
    weak = s.add(MemoryObject(content="same fact", importance=0.3))
    strong = s.add(MemoryObject(content="same fact", importance=0.9))
    L.maintain(s)
    assert s.get(strong.id) is not None
    assert s.get(weak.id) is None                # duplicate evicted


def test_maintain_scheduled_consolidates_global_store(monkeypatch):
    """P7 #11: the nightly-schedulable trigger runs age + maintain over the
    global memory store without a live turn."""
    import app.memory.mstore as M
    store = MemoryStore()
    # A durable promotable + a duplicate + a low-importance evictable.
    store.add(MemoryObject(content="fact", scope=workspace_scope("p1"),
                           importance=0.9, durable=True))
    store.add(MemoryObject(content="dup", importance=0.3))
    store.add(MemoryObject(content="dup", importance=0.2))
    store.add(MemoryObject(content="junk", importance=0.01, durable=False))
    monkeypatch.setattr(M, "memory_store", lambda: store)

    res = L.maintain_scheduled()
    assert res["aged"] is True
    assert res["promoted"] >= 1
    assert res["evicted"] >= 1


def test_maintain_scheduled_is_fail_open(monkeypatch):
    import app.memory.mstore as M

    def boom():
        raise RuntimeError("no store")
    monkeypatch.setattr(M, "memory_store", boom)
    res = L.maintain_scheduled()
    assert res["aged"] is False               # swallowed, no raise


def test_bounded_per_scope_eviction():
    s = MemoryStore()
    s2_cap = 3
    # Force a tiny cap by monkeypatching the helper via many adds + manual bound.
    import app.memory.mstore as M
    orig = M._max_per_scope
    M._max_per_scope = lambda: s2_cap
    try:
        for i in range(10):
            s.add(MemoryObject(content=f"o{i}", scope="workspace:p1",
                               importance=i / 10.0))
        in_scope = [o for o in s.all() if o.scope == "workspace:p1"]
        assert len(in_scope) <= s2_cap
    finally:
        M._max_per_scope = orig
