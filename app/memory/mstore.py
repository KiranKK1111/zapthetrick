"""Memory_Object store (memory-graph R1/R2/R7).

In-process, user-scoped, bounded store of typed ``MemoryObject``s with a
relationship graph. Persistence reuses ``User.preferences`` (a JSON sidecar) so
there's no destructive migration (R1.4/R8.5) — the existing episodes/skills +
vector collections remain the raw tiers this layers over.

All operations are pure + synchronous and never raise; scope isolation is
enforced on every read (Property 2). Named ``mstore`` to avoid colliding with
``app/rag/store.py``.
"""
from __future__ import annotations

import time

from app.memory.objects import MemoryObject, SCOPE_GLOBAL


def _max_per_scope() -> int:
    try:
        from app.core.config_loader import cfg
        return max(1, int(getattr(cfg.memory, "max_objects_per_scope", 500)))
    except Exception:  # noqa: BLE001
        return 500


class MemoryStore:
    """Per-user typed memory. Mutates in place; the caller persists via
    ``export_to``/``load_from`` against the user's preferences blob."""

    def __init__(self) -> None:
        self._objs: dict[str, MemoryObject] = {}

    # ---- writes ----------------------------------------------------------
    def add(self, obj: MemoryObject) -> MemoryObject:
        try:
            self._objs[obj.id] = obj
            self._bound_scope(obj.scope)
        except Exception:  # noqa: BLE001
            pass
        return obj

    def reinforce(self, obj_id: str, amount: float = 0.1) -> None:
        o = self._objs.get(obj_id)
        if o is not None:
            o.importance = min(1.0, o.importance + amount)
            o.updated_at = time.time()

    def relate(self, obj_id: str, rel: str, other_id: str) -> None:
        o = self._objs.get(obj_id)
        if o is None or not other_id or other_id == obj_id:
            return
        edge = (rel, other_id)
        if edge not in o.relations:
            o.relations.append(edge)

    def promote(self, obj_id: str, new_scope: str) -> None:
        o = self._objs.get(obj_id)
        if o is not None and new_scope:
            o.scope = new_scope
            o.durable = True
            o.updated_at = time.time()

    def evict(self, predicate) -> int:
        try:
            victims = [oid for oid, o in self._objs.items() if predicate(o)]
            for oid in victims:
                self._objs.pop(oid, None)
            return len(victims)
        except Exception:  # noqa: BLE001
            return 0

    def clear_user(self) -> int:
        """Data-clear (R8.2): drop everything for this user."""
        n = len(self._objs)
        self._objs.clear()
        return n

    # ---- reads -----------------------------------------------------------
    def get(self, obj_id: str) -> MemoryObject | None:
        return self._objs.get(obj_id)

    def all(self) -> list[MemoryObject]:
        return list(self._objs.values())

    def by_scope(self, scopes, kinds=None) -> list[MemoryObject]:
        """Objects in any of `scopes` (str or iterable). Optional kind filter.
        Scope isolation: only the requested scopes are ever returned (R2.2)."""
        want = {scopes} if isinstance(scopes, str) else set(scopes or [])
        kset = set(kinds) if kinds else None
        out = []
        for o in self._objs.values():
            if o.scope in want and (kset is None or o.kind in kset):
                out.append(o)
        return out

    def related(self, obj_id: str) -> list[MemoryObject]:
        """One-hop graph traversal: the objects directly related to `obj_id`
        (R7.2). No edges → empty (caller does plain relevance retrieval)."""
        o = self._objs.get(obj_id)
        if o is None:
            return []
        out = []
        for _rel, other_id in o.relations:
            t = self._objs.get(other_id)
            if t is not None:
                out.append(t)
        return out

    # ---- bounds + persistence -------------------------------------------
    def _bound_scope(self, scope: str) -> None:
        cap = _max_per_scope()
        in_scope = [o for o in self._objs.values() if o.scope == scope]
        if len(in_scope) <= cap:
            return
        # Evict the least-important, oldest non-durable objects first (R5.4).
        evictable = sorted(
            (o for o in in_scope if not o.durable),
            key=lambda o: (o.importance, o.updated_at))
        overflow = len(in_scope) - cap
        for o in evictable[:overflow]:
            self._objs.pop(o.id, None)

    def export_to(self, prefs: dict) -> None:
        try:
            prefs["memory_objects"] = [o.to_dict() for o in self._objs.values()]
        except Exception:  # noqa: BLE001
            pass

    def load_from(self, prefs: dict | None) -> None:
        try:
            if not isinstance(prefs, dict):
                return
            data = prefs.get("memory_objects")
            if not isinstance(data, list):
                return
            self._objs.clear()
            for d in data:
                if isinstance(d, dict):
                    o = MemoryObject.from_dict(d)
                    self._objs[o.id] = o
        except Exception:  # noqa: BLE001
            pass

    def __len__(self) -> int:
        return len(self._objs)


# Process-wide store (user-scoped persistence layered on top by the caller).
_STORE: MemoryStore | None = None


def memory_store() -> MemoryStore:
    global _STORE
    if _STORE is None:
        _STORE = MemoryStore()
    return _STORE


def retrieval_scopes(workspace_id: str | None) -> list[str]:
    """The scopes a turn retrieves from: its workspace + global (R2.2)."""
    from app.memory.objects import workspace_scope
    return [workspace_scope(workspace_id), SCOPE_GLOBAL]


__all__ = ["MemoryStore", "memory_store", "retrieval_scopes"]
