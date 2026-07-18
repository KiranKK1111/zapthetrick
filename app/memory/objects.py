"""Typed Memory_Objects (memory-graph R1/R2).

A structured memory item layered over the existing episodic/semantic stores —
``fact | decision | preference | entity | episode_summary`` with a scope,
importance, timestamps, durable flag, an optional embedding, and relationship
edges. Additive: when no structured object matches a query the caller falls back
to today's episodic/semantic recall (Property 1).

Scope is ``global`` (user), ``workspace:<id>`` (project), or ``session:<id>`` —
consistent with `workspace-and-artifacts` (Property 2).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

KINDS = ("fact", "decision", "preference", "entity", "episode_summary")

SCOPE_GLOBAL = "global"


def workspace_scope(workspace_id: str | None) -> str:
    wid = (workspace_id or "").strip() or "default"
    return f"workspace:{wid}"


def session_scope(session_id: str | None) -> str:
    sid = (session_id or "").strip() or "unknown"
    return f"session:{sid}"


@dataclass
class MemoryObject:
    content: str
    kind: str = "fact"
    scope: str = SCOPE_GLOBAL
    importance: float = 0.5
    durable: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    embedding: list[float] | None = None
    relations: list[tuple[str, str]] = field(default_factory=list)  # (rel, other_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "content": self.content, "kind": self.kind,
            "scope": self.scope, "importance": round(self.importance, 4),
            "durable": self.durable, "created_at": self.created_at,
            "updated_at": self.updated_at,
            "relations": [list(r) for r in self.relations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryObject":
        obj = cls(
            content=d.get("content", ""),
            kind=d.get("kind", "fact"),
            scope=d.get("scope", SCOPE_GLOBAL),
            importance=float(d.get("importance", 0.5)),
            durable=bool(d.get("durable", False)),
            id=d.get("id") or uuid.uuid4().hex,
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
            embedding=d.get("embedding"),
        )
        obj.relations = [tuple(r) for r in (d.get("relations") or [])
                         if isinstance(r, (list, tuple)) and len(r) == 2]
        return obj


__all__ = [
    "MemoryObject", "KINDS", "SCOPE_GLOBAL", "workspace_scope", "session_scope",
]
