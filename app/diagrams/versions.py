"""Diagram version history (MermaidDiagramVisualizations.md #9).

    Version 1 → Version 2 → Version 3 → Restore

Iterative editing without history is a trap: the third "actually, go back to how
it was" is unanswerable, so users stop iterating and start re-prompting from
scratch. Every compose / edit / repair / critic-accept pushes a version here, and
:meth:`DiagramVersionStore.restore` makes any of them current again — which is
itself recorded, so a restore is never destructive.

Storage is **process-local and in-memory**, matching the existing render caches
(`response_arch.mermaid`'s source-hash cache, the FE's PNG cache): a diagram's
history lives as long as the app does, and a restart starts clean. That is a
deliberate scope choice, not an oversight — persisting it means a schema, a
migration and a retention policy, and the value (undo *within* a working session)
is delivered without them. The store is capped per diagram and globally so a long
session cannot grow without bound, and it is guarded by a lock because FastAPI
serves requests from a thread pool.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

MAX_VERSIONS_PER_DIAGRAM = 20     # deepest undo we keep
MAX_DIAGRAMS = 200                # distinct diagrams tracked (LRU by touch)


def diagram_id(source: str) -> str:
    """A stable id for a diagram from its FIRST source, so a client that has no
    id of its own still gets a coherent history."""
    return hashlib.sha256(
        (source or "").encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class DiagramVersion:
    version: int
    source: str
    origin: str = "compose"        # compose | edit | repair | critic | restore | manual
    note: str = ""
    score: float | None = None
    ir: dict | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self, *, include_ir: bool = False) -> dict:
        out = {"version": self.version, "source": self.source,
               "origin": self.origin, "note": self.note,
               "score": self.score, "created_at": self.created_at,
               "chars": len(self.source)}
        if include_ir:
            out["ir"] = self.ir
        return out

    def summary(self) -> dict:
        """History-list shape: no source body, so listing 20 versions is cheap."""
        return {"version": self.version, "origin": self.origin,
                "note": self.note, "score": self.score,
                "created_at": self.created_at, "chars": len(self.source)}


class DiagramVersionStore:
    """Bounded, thread-safe, per-diagram version history."""

    def __init__(self, *, max_versions: int = MAX_VERSIONS_PER_DIAGRAM,
                 max_diagrams: int = MAX_DIAGRAMS) -> None:
        self._lock = threading.Lock()
        # Insertion-ordered: the oldest-touched diagram is evicted first.
        self._history: dict[str, list[DiagramVersion]] = {}
        self._max_versions = max_versions
        self._max_diagrams = max_diagrams

    # -- writes -----------------------------------------------------------
    def push(self, key: str, source: str, *, origin: str = "compose",
             note: str = "", score: float | None = None,
             ir: dict | None = None) -> DiagramVersion:
        """Record a new version. A push whose source is IDENTICAL to the current
        head is a no-op returning that head — repeated renders of an unchanged
        diagram must not fill the history."""
        key = (key or "").strip() or diagram_id(source)
        with self._lock:
            versions = self._history.get(key)
            if versions is None:
                versions = []
                self._history[key] = versions
            else:
                # Refresh LRU position.
                self._history.pop(key)
                self._history[key] = versions
            if versions and versions[-1].source == source:
                return versions[-1]
            entry = DiagramVersion(
                version=(versions[-1].version + 1) if versions else 1,
                source=source, origin=origin, note=note, score=score, ir=ir)
            versions.append(entry)
            if len(versions) > self._max_versions:
                # Drop the OLDEST, keeping version numbers monotonic so a client
                # holding a number never gets a different diagram back.
                del versions[0:len(versions) - self._max_versions]
            while len(self._history) > self._max_diagrams:
                self._history.pop(next(iter(self._history)))
            return entry

    def restore(self, key: str, version: int) -> DiagramVersion | None:
        """Make `version` current by pushing it again (origin `restore`), so the
        history stays append-only and a restore is itself undoable."""
        with self._lock:
            versions = list(self._history.get(key or "", []))
        target = next((v for v in versions if v.version == version), None)
        if target is None:
            return None
        return self.push(key, target.source, origin="restore",
                         note=f"restored v{version}", score=target.score,
                         ir=target.ir)

    def clear(self, key: str = "") -> None:
        with self._lock:
            if key:
                self._history.pop(key, None)
            else:
                self._history.clear()

    # -- reads ------------------------------------------------------------
    def list(self, key: str) -> list[DiagramVersion]:
        with self._lock:
            return list(self._history.get(key or "", []))

    def get(self, key: str, version: int) -> DiagramVersion | None:
        for entry in self.list(key):
            if entry.version == version:
                return entry
        return None

    def head(self, key: str) -> DiagramVersion | None:
        versions = self.list(key)
        return versions[-1] if versions else None

    def stats(self) -> dict:
        with self._lock:
            return {"diagrams": len(self._history),
                    "versions": sum(len(v) for v in self._history.values()),
                    "max_versions_per_diagram": self._max_versions,
                    "max_diagrams": self._max_diagrams}


# The process-wide store the API routes use.
versions = DiagramVersionStore()

__all__ = ["DiagramVersion", "DiagramVersionStore", "versions", "diagram_id",
           "MAX_VERSIONS_PER_DIAGRAM", "MAX_DIAGRAMS"]
