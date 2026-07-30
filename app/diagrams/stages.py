"""Pipeline-stage vocabulary (MermaidDiagramVisualizations.md #14 and #17).

    Prompt → Planning ✔ → Generating ✔ → Validating ✔ → Compiling ✔ → Rendering ✔

The doc's argument for exposing the pipeline is diagnostic, not decorative: "when
something fails, users immediately know where and why". A spinner that dies leaves
the user guessing whether the model, the parser or the renderer let them down.

Before this, the FE surfaced exactly one string — "Fixing diagram syntax…" — which
only appeared in the rare LLM-repair branch. This module is the shared, ordered
vocabulary both ends agree on, plus a tiny state machine (:class:`StageTracker`)
that turns a sequence of stage transitions into the frame a UI can paint.

Pure and dependency-free so the FE can mirror the same constants (see
`lib/widgets/diagram_stages.dart`) and both stay in step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

PENDING = "pending"
ACTIVE = "active"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"

# id, label, the one-line "what is happening" a UI shows while it's active.
STAGES: tuple[tuple[str, str, str], ...] = (
    ("planning", "Planning", "Working out the entities and how they relate"),
    ("generating", "Generating", "Building the diagram structure"),
    ("validating", "Validating", "Checking syntax, logic, style and accessibility"),
    ("compiling", "Compiling", "Parsing the diagram source"),
    ("repairing", "Repairing", "Fixing the syntax the parser rejected"),
    ("rendering", "Rendering", "Drawing the diagram"),
)
STAGE_IDS: tuple[str, ...] = tuple(stage[0] for stage in STAGES)
# `repairing` only happens when compiling failed — a clean run skips it, and a UI
# should not draw a gap for a stage that never needed to run.
CONDITIONAL: frozenset[str] = frozenset({"repairing"})


@dataclass
class Stage:
    id: str
    label: str
    detail: str
    state: str = PENDING
    note: str = ""
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def ms(self) -> int | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return int((self.ended_at - self.started_at) * 1000)

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "detail": self.detail,
                "state": self.state, "note": self.note, "ms": self.ms}


def stage_ladder() -> list[Stage]:
    """A fresh, all-pending ladder."""
    return [Stage(id=sid, label=label, detail=detail)
            for sid, label, detail in STAGES]


class StageTracker:
    """Drive the ladder: `begin` → `complete`/`fail`/`skip`, then :meth:`frame`.

    Monotonic by construction — beginning a stage completes every earlier one that
    is still pending (a conditional stage is marked skipped instead), so a caller
    that jumps from `generating` straight to `rendering` still produces a coherent
    ladder rather than a row of half-lit steps.
    """

    def __init__(self) -> None:
        self._stages = stage_ladder()
        self._index = {stage.id: position
                       for position, stage in enumerate(self._stages)}

    def _at(self, stage_id: str) -> Stage | None:
        position = self._index.get(stage_id)
        return self._stages[position] if position is not None else None

    def begin(self, stage_id: str, note: str = "") -> "StageTracker":
        position = self._index.get(stage_id)
        if position is None:
            return self
        now = time.time()
        for earlier in self._stages[:position]:
            if earlier.state in (PENDING, ACTIVE):
                earlier.state = SKIPPED if (
                    earlier.id in CONDITIONAL and earlier.state == PENDING) else DONE
                earlier.ended_at = earlier.ended_at or now
                earlier.started_at = earlier.started_at or now
        stage = self._stages[position]
        stage.state = ACTIVE
        stage.note = note
        stage.started_at = now
        return self

    def complete(self, stage_id: str, note: str = "") -> "StageTracker":
        stage = self._at(stage_id)
        if stage:
            stage.state = DONE
            stage.note = note or stage.note
            stage.ended_at = time.time()
            stage.started_at = stage.started_at or stage.ended_at
        return self

    def fail(self, stage_id: str, note: str = "") -> "StageTracker":
        stage = self._at(stage_id)
        if stage:
            stage.state = FAILED
            stage.note = note
            stage.ended_at = time.time()
            stage.started_at = stage.started_at or stage.ended_at
        return self

    def skip(self, stage_id: str, note: str = "") -> "StageTracker":
        stage = self._at(stage_id)
        if stage:
            stage.state = SKIPPED
            stage.note = note
        return self

    def finish(self, note: str = "") -> "StageTracker":
        """Everything still open is done; unrun conditional stages are skipped."""
        now = time.time()
        for stage in self._stages:
            if stage.state in (PENDING, ACTIVE):
                stage.state = SKIPPED if (stage.id in CONDITIONAL
                                          and stage.state == PENDING) else DONE
                stage.started_at = stage.started_at or now
                stage.ended_at = stage.ended_at or now
        if note:
            self._stages[-1].note = note
        return self

    # -- reads ------------------------------------------------------------
    @property
    def failed_stage(self) -> Stage | None:
        return next((s for s in self._stages if s.state == FAILED), None)

    @property
    def active_stage(self) -> Stage | None:
        return next((s for s in self._stages if s.state == ACTIVE), None)

    def frame(self) -> dict:
        """The payload a client paints: the ladder plus a one-line summary."""
        failed = self.failed_stage
        active = self.active_stage
        if failed:
            summary = f"Failed at {failed.label.lower()}" + (
                f": {failed.note}" if failed.note else "")
        elif active:
            summary = active.detail
        else:
            summary = "Done"
        return {"stages": [s.to_dict() for s in self._stages],
                "summary": summary,
                "ok": failed is None,
                "failed_at": failed.id if failed else None,
                "total_ms": sum(s.ms or 0 for s in self._stages)}


__all__ = ["STAGES", "STAGE_IDS", "CONDITIONAL", "Stage", "StageTracker",
           "stage_ladder", "PENDING", "ACTIVE", "DONE", "FAILED", "SKIPPED"]
