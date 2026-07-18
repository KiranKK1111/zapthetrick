"""Durable background continuation (roadmap Phase 6 #23).

`obs/jobs.py` tracks background jobs in-process only, so a backend restart loses
the Task Center: a long export or research job that was "running" simply vanishes,
and a "done" job the user hasn't reopened yet is gone too. This module adds a
**durable** registry that mirrors job records to a small JSON file and restores
them on construction, so jobs survive a restart.

It is deliberately standalone (the `obs/` module is owned elsewhere): the route
layer / obs.jobs can mirror its writes here, and on boot call
:meth:`DurableJobRegistry.load` to rehydrate. On restart, any job left in a
non-terminal state is marked ``interrupted`` so the UI can offer resume/retry
instead of showing a spinner forever.

Fail-open: a corrupt or unwritable store degrades to in-memory only; a persist
failure never propagates to the caller.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

_TERMINAL = frozenset({"done", "failed", "cancelled", "interrupted"})


def _default_path() -> str:
    base = os.environ.get("DTT_DATA_DIR") or os.path.join(
        tempfile.gettempdir(), "dtt")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return os.path.join(base, "background_jobs.json")


@dataclass
class DurableJobRegistry:
    path: str = field(default_factory=_default_path)
    _jobs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.load()

    # -- persistence --------------------------------------------------------
    def load(self) -> None:
        """Rehydrate from disk; mark orphaned non-terminal jobs interrupted."""
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._jobs = {}
                for jid, rec in data.items():
                    if not isinstance(rec, dict):
                        continue
                    # A job that was still running when we died can't be running
                    # now — surface it as interrupted so the UI can resume it.
                    if rec.get("status") not in _TERMINAL:
                        rec = {**rec, "status": "interrupted",
                               "interrupted": True}
                    self._jobs[jid] = rec
                self._persist()
        except FileNotFoundError:
            self._jobs = {}
        except Exception:  # noqa: BLE001 — corrupt store → start clean
            self._jobs = {}

    def _persist(self) -> None:
        try:
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._jobs, fh, default=str)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 — durability is best-effort
            pass

    # -- job lifecycle ------------------------------------------------------
    def start(self, job_id: str, kind: str, meta: dict | None = None) -> dict:
        rec = {
            "id": job_id, "kind": kind, "status": "running",
            "meta": dict(meta or {}), "created_at": time.time(),
            "updated_at": time.time(), "result": None,
        }
        self._jobs[job_id] = rec
        self._persist()
        return rec

    def update(self, job_id: str, *, status: str | None = None,
               result: Any = None, meta: dict | None = None) -> dict | None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return None
        if status:
            rec["status"] = status
        if result is not None:
            rec["result"] = result
        if meta:
            rec["meta"] = {**rec.get("meta", {}), **meta}
        rec["updated_at"] = time.time()
        self._persist()
        return rec

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def all(self) -> list[dict]:
        return sorted(self._jobs.values(),
                     key=lambda r: r.get("created_at", 0), reverse=True)

    def pending(self) -> list[dict]:
        """Jobs that need attention on boot (running or interrupted)."""
        return [r for r in self._jobs.values()
                if r.get("status") not in _TERMINAL or r.get("interrupted")]

    def drop(self, job_id: str) -> None:
        if self._jobs.pop(job_id, None) is not None:
            self._persist()


# Process-wide singleton (lazily rehydrated). Wire obs.jobs / routes_jobs to
# mirror into this so the Task Center survives a restart.
_registry: DurableJobRegistry | None = None


def registry() -> DurableJobRegistry:
    global _registry
    if _registry is None:
        _registry = DurableJobRegistry()
    return _registry


__all__ = ["DurableJobRegistry", "registry"]
