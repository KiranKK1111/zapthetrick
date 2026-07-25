"""Durable background-task core (vNext §9.3, Stage 9 Component D).

The deterministic heart of the background-task engine — everything that can be
tested with no database and no event loop:

  * a **state machine** (pending → running → {needs_input, paused, completed,
    failed, cancelled}) with legal transitions + terminal/runnable predicates;
  * a **schedule** model + `next_run`/`due_tasks` — the cron ticker's maths
    (once / every-N-seconds / daily-at), computed from an INJECTED `now`;
  * a **checkpoint** (the C.2 todo schema — step/total/todos/artifacts/data) +
    `advance_checkpoint` and `rehydrate` (resume from the last checkpoint after a
    pod restart);
  * the runner's **`pick_runnable`** — choose the next tasks to run under a
    concurrency cap;
  * `step_needs_approval` — a side-effectful step PARKS on `needs_input` (reuses
    the §9.9 side-effect taxonomy), the human-in-the-loop gate.

Persistence (Postgres) and the asyncio runner are the infra seams that call into
this. Pure + fail-open. Flag-gated (`tasks.enabled`, default OFF).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# States.
PENDING = "pending"
RUNNING = "running"
NEEDS_INPUT = "needs_input"     # parked on human approval / clarification
PAUSED = "paused"              # checkpointed (e.g. drain) — resumable
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

_TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED})
_RUNNABLE = frozenset({PENDING, PAUSED})   # can be picked up by the runner

# Legal transitions.
_TRANSITIONS: dict[str, frozenset] = {
    PENDING: frozenset({RUNNING, CANCELLED}),
    RUNNING: frozenset({NEEDS_INPUT, PAUSED, COMPLETED, FAILED, CANCELLED}),
    NEEDS_INPUT: frozenset({RUNNING, CANCELLED}),
    PAUSED: frozenset({RUNNING, CANCELLED}),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.tasks, "enabled", False))
    except Exception:  # noqa: BLE001
        return False


def _max_concurrency() -> int:
    try:
        from app.core.config_loader import cfg
        return max(1, int(getattr(cfg.tasks, "max_concurrency", 2) or 2))
    except Exception:  # noqa: BLE001
        return 2


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #
def is_terminal(state: str) -> bool:
    return state in _TERMINAL


def is_runnable(state: str) -> bool:
    return state in _RUNNABLE


def can_transition(frm: str, to: str) -> bool:
    return to in _TRANSITIONS.get(frm, frozenset())


def transition(record: "TaskRecord", to: str) -> bool:
    """Apply a state transition if legal. Returns True on success (mutates the
    record), False if the transition is illegal. Never raises."""
    try:
        if can_transition(record.state, to):
            record.state = to
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class Schedule:
    kind: str = "once"            # once | interval | daily
    interval_s: float = 0.0       # for kind=interval
    at_second_of_day: int = 0     # for kind=daily (0..86399)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "interval_s": self.interval_s,
                "at_second_of_day": self.at_second_of_day}


@dataclass
class Checkpoint:
    """The C.2 todo-schema checkpoint — durable progress state."""
    step: int = 0
    total: int = 0
    todos: list = field(default_factory=list)      # [{text, done}]
    artifacts: list = field(default_factory=list)  # artifact ids/paths
    data: dict = field(default_factory=dict)       # opaque resume state

    def progress(self) -> float:
        return (self.step / self.total) if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {"step": self.step, "total": self.total, "todos": list(self.todos),
                "artifacts": list(self.artifacts), "data": dict(self.data),
                "progress": round(self.progress(), 3)}


@dataclass
class TaskSpec:
    id: str
    goal: str = ""
    kind: str = "chat"            # chat | deep_research | doc | code
    schedule: Schedule = field(default_factory=Schedule)
    side_effectful: bool = False  # whole-task hint; per-step gate is authoritative

    def to_dict(self) -> dict:
        return {"id": self.id, "goal": self.goal, "kind": self.kind,
                "schedule": self.schedule.to_dict(),
                "side_effectful": self.side_effectful}


@dataclass
class TaskRecord:
    spec: TaskSpec
    state: str = PENDING
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    created_at: float = 0.0
    last_run: float | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {"spec": self.spec.to_dict(), "state": self.state,
                "checkpoint": self.checkpoint.to_dict(),
                "last_run": self.last_run, "error": self.error}


# --------------------------------------------------------------------------- #
# Schedule / cron ticker
# --------------------------------------------------------------------------- #
def next_run(schedule: Schedule, *, now: float, last_run: float | None = None,
             created_at: float = 0.0) -> "float | None":
    """When (epoch seconds) the task should NEXT run, or None if never again.
    `now` is INJECTED (no wall-clock read here → deterministic + resume-safe).
      * once     — at `created_at` if it has never run, else None;
      * interval — `last_run + interval_s` (or `now` if never run);
      * daily    — the next `at_second_of_day` boundary at/after `now`.
    Never raises → None."""
    try:
        k = schedule.kind
        if k == "once":
            return None if last_run is not None else (created_at or now)
        if k == "interval":
            step = max(1.0, float(schedule.interval_s or 0))
            return now if last_run is None else last_run + step
        if k == "daily":
            sod = int(schedule.at_second_of_day) % 86400
            day_start = (int(now) // 86400) * 86400
            candidate = day_start + sod
            return candidate if candidate >= now else candidate + 86400
        return None
    except Exception:  # noqa: BLE001
        return None


def due_tasks(records, *, now: float) -> "list[TaskRecord]":
    """The runnable records whose next_run is due (<= now). Never raises → []."""
    try:
        out = []
        for r in records or ():
            if not is_runnable(r.state):
                continue
            nr = next_run(r.spec.schedule, now=now, last_run=r.last_run,
                          created_at=r.created_at)
            if nr is not None and nr <= now:
                out.append(r)
        return out
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# Runner scheduling
# --------------------------------------------------------------------------- #
def pick_runnable(records, *, running: int = 0,
                  max_concurrency: int | None = None) -> "list[TaskRecord]":
    """Choose the next runnable records to start, filling the free concurrency
    slots (cap − currently running). Runnable = PENDING/PAUSED. Never raises."""
    try:
        cap = max_concurrency if max_concurrency is not None else _max_concurrency()
        free = max(0, cap - max(0, running))
        if free == 0:
            return []
        runnable = [r for r in (records or ()) if is_runnable(r.state)]
        return runnable[:free]
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# Checkpoint + rehydrate
# --------------------------------------------------------------------------- #
def advance_checkpoint(cp: Checkpoint, *, done_index: int | None = None,
                       artifact: str = "", data: dict | None = None) -> Checkpoint:
    """Advance a checkpoint one step: mark a todo done, bump the step, append an
    artifact, merge resume data. Returns the same checkpoint (mutated). Never
    raises."""
    try:
        if done_index is not None and 0 <= done_index < len(cp.todos):
            td = cp.todos[done_index]
            if isinstance(td, dict):
                td["done"] = True
        cp.step = min(cp.total or (cp.step + 1), cp.step + 1)
        if artifact:
            cp.artifacts.append(artifact)
        if data:
            cp.data.update(data)
        return cp
    except Exception:  # noqa: BLE001
        return cp


def rehydrate(record: "TaskRecord") -> "TaskRecord":
    """Resume a task after a pod restart: a RUNNING or PAUSED task rehydrates to
    RUNNING (from its last checkpoint); a NEEDS_INPUT task stays parked; terminal
    tasks are untouched. Never raises → the record unchanged."""
    try:
        if record.state in (RUNNING, PAUSED):
            record.state = RUNNING          # resume from checkpoint
        return record
    except Exception:  # noqa: BLE001
        return record


def step_needs_approval(tool_or_action: str) -> bool:
    """Whether a task STEP must PARK on `needs_input` — a side-effectful action
    (write/push/egress/config/create) needs human approval before it runs.
    Reuses the §9.9 side-effect taxonomy. Never raises → True (fail SAFE)."""
    try:
        from app.security.quarantine import is_side_effectful
        return is_side_effectful(tool_or_action)
    except Exception:  # noqa: BLE001
        return True
