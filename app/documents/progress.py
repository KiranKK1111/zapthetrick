"""In-process staged-progress registry for resume upload / delete pipelines.

The upload pipeline (parse -> chunk -> embed -> index) and the delete
pipeline (vectors -> chunks -> row) report per-stage progress here, keyed
by `resume_id`, so the client can poll
`GET /api/resumes/{resume_id}/progress` and render a stage checklist +
progress bar with live metrics ("Embedding chunks 42/120").

Design constraints:
  - THREAD-SAFE: the embed step runs inside `asyncio.to_thread`, and the
    warmup thread may race it, so every mutation holds a `threading.Lock`.
  - FAIL-OPEN: progress reporting must NEVER break the pipeline. Every
    public function swallows its own exceptions; a reporting bug degrades
    to a stale progress bar, not a failed upload.
  - IN-PROCESS ONLY: this is a per-process dict (the app is a single
    desktop-bundled server). Entries are evicted oldest-first past a cap
    so long sessions don't leak.

Entry shape (what the endpoint returns):
    {
      "op": "upload" | "delete",
      "stage": "embed",              # current stage name
      "stage_index": 3,              # 0-based index into `stages`
      "total_stages": 6,
      "stages": ["upload", ...],     # full ordered list for the checklist
      "percent": 57.5,               # OVERALL percent across all stages
      "detail": "Embedding chunks 42/120",
      "counts": {"chunks": 120, "embedded": 42},
      "done": false,
      "error": null,
      "updated_at": 1719990000.0,
    }
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

# Ordered stage names for each operation. The final stage is the terminal
# "all good" state that `finish()` lands on.
UPLOAD_STAGES: list[str] = ["upload", "parse", "chunk", "embed", "index", "ready"]
DELETE_STAGES: list[str] = ["vectors", "chunks", "row", "gone"]

# Oldest entries are evicted past this cap (dicts preserve insertion order).
_MAX_ENTRIES = 256

_LOCK = threading.Lock()
_ENTRIES: dict[str, dict[str, Any]] = {}


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def _overall_percent(stage_index: int, total: int, fraction: float) -> float:
    """Overall 0-100 percent: completed stages + fraction of the current one."""
    if total <= 0:
        return 0.0
    pct = ((stage_index + _clamp01(fraction)) / total) * 100.0
    return max(0.0, min(100.0, round(pct, 1)))


def _evict_locked() -> None:
    while len(_ENTRIES) > _MAX_ENTRIES:
        try:
            _ENTRIES.pop(next(iter(_ENTRIES)))
        except (StopIteration, KeyError):  # pragma: no cover — racing pop
            break


# ---- Public API (every function is fail-open) ---------------------------


def begin(resume_id: str, op: str, stages: Optional[list[str]] = None) -> None:
    """Start (or restart) tracking `resume_id`. Overwrites any prior entry."""
    try:
        if stages is None:
            stages = DELETE_STAGES if op == "delete" else UPLOAD_STAGES
        stages = list(stages) or ["working"]
        entry = {
            "op": op,
            "stage": stages[0],
            "stage_index": 0,
            "total_stages": len(stages),
            "stages": stages,
            "percent": 0.0,
            "detail": "",
            "counts": {},
            "done": False,
            "error": None,
            "updated_at": time.time(),
        }
        with _LOCK:
            _ENTRIES.pop(str(resume_id), None)  # re-insert at the tail (LRU-ish)
            _ENTRIES[str(resume_id)] = entry
            _evict_locked()
    except Exception:  # noqa: BLE001 — fail-open, never break the pipeline
        pass


def set_stage(
    resume_id: str,
    stage: str,
    detail: str = "",
    fraction: float = 0.0,
    counts: Optional[dict] = None,
) -> None:
    """Move to `stage` (by name). Unknown stage names are appended-safe:
    they keep the current index so percent never jumps backwards."""
    try:
        with _LOCK:
            entry = _ENTRIES.get(str(resume_id))
            if entry is None or entry.get("done"):
                return
            stages = entry["stages"]
            try:
                idx = stages.index(stage)
            except ValueError:
                idx = entry["stage_index"]  # unknown name: hold position
            # Never move backwards — late/racing updates from a slower
            # thread must not rewind the bar.
            idx = max(idx, entry["stage_index"])
            entry["stage"] = stage
            entry["stage_index"] = idx
            entry["percent"] = max(
                entry["percent"],
                _overall_percent(idx, entry["total_stages"], fraction),
            )
            if detail:
                entry["detail"] = detail
            if counts:
                entry["counts"].update(counts)
            entry["updated_at"] = time.time()
    except Exception:  # noqa: BLE001
        pass


def update(
    resume_id: str,
    fraction: Optional[float] = None,
    detail: Optional[str] = None,
    counts: Optional[dict] = None,
) -> None:
    """Progress WITHIN the current stage (e.g. embed batch 3/8)."""
    try:
        with _LOCK:
            entry = _ENTRIES.get(str(resume_id))
            if entry is None or entry.get("done"):
                return
            if fraction is not None:
                entry["percent"] = max(
                    entry["percent"],
                    _overall_percent(
                        entry["stage_index"], entry["total_stages"], fraction
                    ),
                )
            if detail is not None:
                entry["detail"] = detail
            if counts:
                entry["counts"].update(counts)
            entry["updated_at"] = time.time()
    except Exception:  # noqa: BLE001
        pass


def finish(resume_id: str, detail: str = "") -> None:
    """Mark the operation complete: last stage, 100%, done=True."""
    try:
        with _LOCK:
            entry = _ENTRIES.get(str(resume_id))
            if entry is None:
                return
            stages = entry["stages"]
            entry["stage"] = stages[-1]
            entry["stage_index"] = len(stages) - 1
            entry["percent"] = 100.0
            entry["done"] = True
            entry["error"] = None
            if detail:
                entry["detail"] = detail
            entry["updated_at"] = time.time()
    except Exception:  # noqa: BLE001
        pass


def fail(resume_id: str, error: str) -> None:
    """Mark the operation failed. `done` flips True so pollers stop."""
    try:
        with _LOCK:
            entry = _ENTRIES.get(str(resume_id))
            if entry is None:
                return
            entry["done"] = True
            entry["error"] = str(error) or "Unknown error"
            entry["updated_at"] = time.time()
    except Exception:  # noqa: BLE001
        pass


def note_background(
    resume_id: str,
    counts: Optional[dict] = None,
    detail: Optional[str] = None,
) -> None:
    """Update METRICS on an entry even after `finish()` — for background
    work that continues past the blocking flow (e.g. instant-answer
    pre-generation after a resume upload). Never touches done/percent/error,
    so pollers that stop on `done` are unaffected and late watchers see the
    metric ticking."""
    try:
        with _LOCK:
            entry = _ENTRIES.get(str(resume_id))
            if entry is None:
                return
            if counts:
                entry["counts"].update(counts)
            if detail is not None:
                entry["detail"] = detail
            entry["updated_at"] = time.time()
    except Exception:  # noqa: BLE001
        pass


def get(resume_id: str) -> Optional[dict[str, Any]]:
    """Snapshot of the entry (a copy — callers can't mutate the registry)."""
    try:
        with _LOCK:
            entry = _ENTRIES.get(str(resume_id))
            if entry is None:
                return None
            snap = dict(entry)
            snap["counts"] = dict(entry["counts"])
            snap["stages"] = list(entry["stages"])
            return snap
    except Exception:  # noqa: BLE001
        return None


def clear(resume_id: str) -> None:
    """Drop the entry (e.g. after the client saw the terminal state)."""
    try:
        with _LOCK:
            _ENTRIES.pop(str(resume_id), None)
    except Exception:  # noqa: BLE001
        pass
