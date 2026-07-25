"""Background tasks & automations (vNext §9.3, Stage 9 Component D).

Durable, checkpointed tasks that survive a pod restart. The persistence (Postgres)
and the asyncio runner are the infra seams; the deterministic core — the state
machine, the schedule/cron ticker, the checkpoint model, and the runner's
pick-next scheduling — lives in `core.py` and is unit-tested with no DB.
"""
from app.tasks.core import (  # noqa: F401
    CANCELLED, COMPLETED, FAILED, NEEDS_INPUT, PAUSED, PENDING, RUNNING,
    Checkpoint, Schedule, TaskRecord, TaskSpec, advance_checkpoint, can_transition,
    due_tasks, enabled, is_runnable, is_terminal, next_run, pick_runnable,
    rehydrate, step_needs_approval, transition,
)

__all__ = [
    "PENDING", "RUNNING", "NEEDS_INPUT", "PAUSED", "COMPLETED", "FAILED",
    "CANCELLED", "enabled", "TaskSpec", "Checkpoint", "TaskRecord", "Schedule",
    "can_transition", "transition", "is_terminal", "is_runnable", "next_run",
    "due_tasks", "pick_runnable", "advance_checkpoint", "rehydrate",
    "step_needs_approval",
]
