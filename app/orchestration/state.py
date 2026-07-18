"""Agent state persistence (agent-orchestration R7).

`AgentState` holds a long-horizon goal's per-task status (completed/pending),
scoped to a workspace + user and bounded in size, so a run can pause and resume
from pending work (Property 7). Persists to the user's preferences blob (no
schema migration), reusing the same sidecar pattern as the other specs. Pure +
synchronous; never raises.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

_MAX_TASKS = 64

PENDING = "pending"
DONE = "done"


@dataclass
class Task:
    id: int
    text: str
    status: str = PENDING
    output: str = ""


@dataclass
class AgentState:
    goal: str
    scope: str = "default"            # workspace:<id> / global, etc.
    tasks: list[Task] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def set_tasks(self, texts: list[str]) -> None:
        self.tasks = [Task(id=i, text=t) for i, t in enumerate(texts)][:_MAX_TASKS]
        self.updated_at = time.time()

    def pending(self) -> list[Task]:
        return [t for t in self.tasks if t.status != DONE]

    def is_done(self, task_id: int) -> bool:
        for t in self.tasks:
            if t.id == task_id:
                return t.status == DONE
        return False

    def mark_done(self, task_id: int, output: str = "") -> None:
        for t in self.tasks:
            if t.id == task_id:
                t.status = DONE
                t.output = (output or "")[:2000]
                self.updated_at = time.time()
                return

    @property
    def complete(self) -> bool:
        return bool(self.tasks) and all(t.status == DONE for t in self.tasks)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "scope": self.scope,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "tasks": [{"id": t.id, "text": t.text, "status": t.status,
                       "output": t.output} for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentState":
        st = cls(goal=d.get("goal", ""), scope=d.get("scope", "default"),
                 created_at=float(d.get("created_at", time.time())),
                 updated_at=float(d.get("updated_at", time.time())))
        st.tasks = [Task(id=int(t.get("id", i)), text=t.get("text", ""),
                         status=t.get("status", PENDING), output=t.get("output", ""))
                    for i, t in enumerate(d.get("tasks") or [])][:_MAX_TASKS]
        return st


def save_state(prefs: dict, state: AgentState) -> None:
    """Persist `state` into the prefs blob under `orch_state[scope]`. Bounded:
    one state per scope. Never raises."""
    try:
        if not isinstance(prefs, dict):
            return
        store = prefs.get("orch_state")
        if not isinstance(store, dict):
            store = {}
            prefs["orch_state"] = store
        store[state.scope] = state.to_dict()
        # Bound total persisted states.
        if len(store) > 32:
            oldest = min(store, key=lambda k: store[k].get("updated_at", 0))
            store.pop(oldest, None)
    except Exception:  # noqa: BLE001
        pass


def load_state(prefs: dict | None, scope: str) -> AgentState | None:
    """Restore the persisted state for `scope`, or None. Never raises."""
    try:
        store = (prefs or {}).get("orch_state")
        if isinstance(store, dict) and scope in store:
            return AgentState.from_dict(store[scope])
    except Exception:  # noqa: BLE001
        pass
    return None


__all__ = ["AgentState", "Task", "save_state", "load_state", "PENDING", "DONE"]
