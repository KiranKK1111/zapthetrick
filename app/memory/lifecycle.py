"""Memory lifecycle (memory-graph R4/R5).

Aging (importance decays over time unless reinforced), reinforcement (on reuse/
reconfirmation), a durable importance floor (R4), and the background maintenance
step (promotion of durable/important objects + eviction of stale/duplicate/
low-importance ones, bounded per scope — R5). All synchronous + fail-open;
``maintain`` is meant to run on the Reflector's existing idle/session-end trigger
so it never blocks a turn (R5.3/R8.3).
"""
from __future__ import annotations

import time

from app.memory.objects import SCOPE_GLOBAL

_DURABLE_FLOOR = 0.4
_PROMOTE_IMPORTANCE = 0.8
_EVICT_FLOOR = 0.08


def _half_life_s() -> float:
    try:
        from app.core.config_loader import cfg
        return max(1.0, float(getattr(cfg.memory, "half_life_days", 30.0))) * 86400.0
    except Exception:  # noqa: BLE001
        return 30 * 86400.0


def age(store, half_life_s: float | None = None, now: float | None = None) -> None:
    """Decay each object's importance toward 0 by elapsed time since last
    update; durable objects never fall below the floor (R4.1/R4.3). No-op safe."""
    try:
        hl = half_life_s if half_life_s is not None else _half_life_s()
        now = now if now is not None else time.time()
        for o in store.all():
            elapsed = max(0.0, now - float(o.updated_at or now))
            factor = 0.5 ** (elapsed / hl)
            decayed = o.importance * factor
            if o.durable:
                decayed = max(decayed, _DURABLE_FLOOR)
            o.importance = max(0.0, min(1.0, decayed))
    except Exception:  # noqa: BLE001
        pass


def reinforce(store, obj_id: str, amount: float = 0.15) -> None:
    """Reinforce on reuse/reconfirmation (R4.2)."""
    try:
        store.reinforce(obj_id, amount)
    except Exception:  # noqa: BLE001
        pass


def _promote_scope(scope: str) -> str | None:
    """Next longer-lived scope: session → workspace's... → global. Sessions and
    workspaces promote to global (the durable, cross-project tier)."""
    if scope == SCOPE_GLOBAL:
        return None
    if scope.startswith("session:") or scope.startswith("workspace:"):
        return SCOPE_GLOBAL
    return None


def maintain(store, *, promote_threshold: float = _PROMOTE_IMPORTANCE,
             evict_floor: float = _EVICT_FLOOR) -> dict:
    """Background maintenance: promote durable/important objects up a scope and
    evict stale/duplicate/low-importance ones. Returns counts. Never raises —
    a failed cycle is simply skipped (R5.3)."""
    try:
        promoted = _promote(store, promote_threshold)
        deduped = _dedupe(store)
        evicted = store.evict(
            lambda o: (not o.durable) and o.importance < evict_floor)
        return {"promoted": promoted, "evicted": evicted + deduped}
    except Exception:  # noqa: BLE001
        return {"promoted": 0, "evicted": 0}


def maintain_scheduled() -> dict:
    """Schedulable consolidation trigger (R5 / Phase 7 #11).

    `maintain` already runs on the Reflector's session-end/idle trigger; this is
    the *nightly-schedulable* entry point the roadmap asks for — it ages then
    consolidates the in-process global memory store on a timer (wired into the
    maintenance loop) without needing a live turn. Fail-open; returns counts."""
    try:
        from app.memory.mstore import memory_store
        store = memory_store()
        age(store)
        result = maintain(store)
        result["aged"] = True
        return result
    except Exception:  # noqa: BLE001 — scheduled maintenance never raises
        return {"promoted": 0, "evicted": 0, "aged": False}


def _promote(store, threshold: float) -> int:
    n = 0
    for o in list(store.all()):
        if o.durable and o.importance >= threshold:
            target = _promote_scope(o.scope)
            if target and target != o.scope:
                store.promote(o.id, target)
                n += 1
    return n


def _dedupe(store) -> int:
    """Evict duplicate objects (same kind + normalized content) within a scope,
    keeping the most important/recent (R5.2)."""
    seen: dict[tuple, object] = {}
    victims: set[str] = set()
    for o in store.all():
        key = (o.scope, o.kind, " ".join((o.content or "").lower().split()))
        keep = seen.get(key)
        if keep is None:
            seen[key] = o
            continue
        loser = o if (o.importance, o.updated_at) <= (keep.importance, keep.updated_at) else keep
        winner = keep if loser is o else o
        seen[key] = winner
        victims.add(loser.id)
    if victims:
        return store.evict(lambda o: o.id in victims)
    return 0


__all__ = ["age", "reinforce", "maintain", "maintain_scheduled"]
