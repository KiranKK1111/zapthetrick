"""In-process background-job registry — the Task Center's data source.

Tracks user-facing async work (answer generation, document/archive exports) so
the FE can show what's running and what recently finished. Bounded, lock-guarded
for the single-process app, and fail-open: registry errors never break the work
they track.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass
class Job:
    id: str
    label: str
    kind: str  # chat | export | agent | task
    status: str = "running"  # running | done | error | cancelled
    progress: float | None = None
    detail: str = ""
    started: float = 0.0
    finished: float | None = None


class JobRegistry:
    def __init__(self, max_jobs: int = 100) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._max = max_jobs
        self._lock = threading.Lock()

    def start(self, label: str, kind: str = "task", detail: str = "") -> str:
        jid = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[jid] = Job(
                id=jid, label=(label or "Task")[:120], kind=kind,
                detail=detail, started=time.time(),
            )
            self._order.append(jid)
            self._evict_locked()
        return jid

    def update(self, jid: str, *, progress: float | None = None,
               status: str | None = None, detail: str | None = None) -> None:
        with self._lock:
            j = self._jobs.get(jid)
            if j is None:
                return
            if progress is not None:
                j.progress = progress
            if status is not None:
                j.status = status
            if detail is not None:
                j.detail = detail

    def finish(self, jid: str, ok: bool = True, detail: str | None = None) -> None:
        with self._lock:
            j = self._jobs.get(jid)
            if j is None:
                return
            j.status = "done" if ok else "error"
            j.finished = time.time()
            if detail is not None:
                j.detail = detail

    def snapshot(self) -> list[dict]:
        """Newest first."""
        with self._lock:
            return [asdict(self._jobs[i]) for i in reversed(self._order)
                    if i in self._jobs]

    def clear_finished(self) -> None:
        with self._lock:
            keep = [i for i in self._order
                    if self._jobs.get(i) and self._jobs[i].status == "running"]
            self._jobs = {i: self._jobs[i] for i in keep}
            self._order = keep

    def _evict_locked(self) -> None:
        while len(self._order) > self._max:
            old = self._order.pop(0)
            self._jobs.pop(old, None)


_registry = JobRegistry()


def jobs() -> JobRegistry:
    return _registry
