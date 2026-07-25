"""Progressive stage protocol (vNext §5.3).

Replaces free-form ``stage`` strings with a typed frame every pipeline emits from
its REAL step registry:

    {id, label, state: pending|active|done|skipped|failed,
     parent?, progress?: 0..1, detail?, attempt?, ts}

``id`` is the stable key the FE state machine dedups on — the same id updates the
same row (repeats become impossible), ``attempt`` renders as an iteration badge
on that row (a repair loop is "verify · attempt 2", not a new row), and a stage
that doesn't run emits ``skipped`` (or nothing). Backward compatible: the legacy
``{"name": str}`` shape adapts via ``from_legacy``.
"""
from __future__ import annotations

import time

STATES = ("pending", "active", "done", "skipped", "failed")


def stage_event(stage_id: str, label: str = "", *, state: str = "active",
                parent: str | None = None, progress: float | None = None,
                detail: str = "", attempt: int = 0,
                ts: float | None = None) -> dict:
    """Build a schema-valid stage frame. Unknown ``state`` → ``active`` (never
    invalid). ``progress`` is clamped to 0..1."""
    ev: dict = {
        "id": str(stage_id),
        "label": label or str(stage_id),
        "state": state if state in STATES else "active",
        "ts": ts if ts is not None else time.time(),
    }
    if parent:
        ev["parent"] = str(parent)
    if progress is not None:
        try:
            ev["progress"] = max(0.0, min(1.0, float(progress)))
        except (TypeError, ValueError):
            pass
    if detail:
        ev["detail"] = detail
    if attempt and attempt > 0:
        ev["attempt"] = int(attempt)
    return ev


def from_legacy(name: str) -> dict:
    """Adapt the old free-form ``{'name': str}`` stage to the schema: a slug id
    (so repeats of the same phrase dedup) + the phrase as the label, active."""
    slug = (name or "").strip().lower().replace(" ", "_").replace("-", "_") or "step"
    return stage_event(slug, label=(name or slug).strip(), state="active")


def validate_stage(ev: dict) -> list[str]:
    """Return a list of errors ([] = valid) for a stage frame."""
    if not isinstance(ev, dict):
        return ["not an object"]
    errs: list[str] = []
    if not ev.get("id"):
        errs.append("id: required")
    if ev.get("state") not in STATES:
        errs.append(f"state: not in {STATES}")
    p = ev.get("progress")
    if p is not None and not (isinstance(p, (int, float))
                              and not isinstance(p, bool) and 0.0 <= p <= 1.0):
        errs.append("progress: must be 0..1")
    return errs


__all__ = ["STATES", "stage_event", "from_legacy", "validate_stage"]
