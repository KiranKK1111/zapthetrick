"""Long-horizon task engine (agent-orchestration R6).

`run_goal(goal, plan_fn, exec_fn, ...)` drives goal→plan→execute→review→complete
with per-task status (R6.1): a completed task advances rather than redoing
(R6.2), and a small goal is answered directly (R6.3, Property 6). State persists
via `AgentState` so a run resumes from pending work (R7). Injectable + async;
never raises — on error it returns whatever completed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.orchestration.state import AgentState


@dataclass
class RunGoalResult:
    direct: bool
    answer: str
    state: AgentState | None = None
    completed: list[str] = field(default_factory=list)
    resumed: bool = False


def _is_small_default(goal: str) -> bool:
    """Heuristic: a short single-sentence goal is small (→ direct answer)."""
    if not (goal or "").strip():
        return True
    from app.orchestration.decompose import decompose
    words = len(" ".join(goal.split()).split())
    return len(decompose(goal)) < 2 and words < 25


async def run_goal(
    goal: str,
    *,
    plan_fn: Callable[[str], Awaitable[list]],
    exec_fn: Callable[[str, object], Awaitable[str]],
    state: AgentState | None = None,
    is_small: Callable[[str], bool] | None = None,
    scope: str = "default",
    combine: Callable[[list], str] | None = None,
) -> RunGoalResult:
    """Run a goal as a tracked plan. `plan_fn(goal) -> [task_text]`;
    `exec_fn(task_text, state) -> output`. Pass a prior `state` to resume.
    Never raises."""
    try:
        small = (is_small or _is_small_default)
        resumed = state is not None and bool(state.tasks)

        # Small goal with no prior plan → answer directly (R6.3).
        if not resumed and small(goal):
            ans = await exec_fn(goal, None)
            return RunGoalResult(direct=True, answer=ans, state=None)

        st = state or AgentState(goal=goal, scope=scope)
        if not st.tasks:
            tasks = await plan_fn(goal)
            st.set_tasks([t for t in (tasks or []) if str(t).strip()])
            if not st.tasks:
                # Planner produced nothing → direct answer fallback.
                ans = await exec_fn(goal, None)
                return RunGoalResult(direct=True, answer=ans, state=st)

        completed: list[str] = []
        for task in st.pending():               # skips already-DONE (no redo)
            out = await exec_fn(task.text, st)
            st.mark_done(task.id, out)
            completed.append(out)

        done_outputs = [t.output for t in st.tasks if t.output]
        answer = (combine(done_outputs) if combine
                  else "\n\n".join(done_outputs))
        return RunGoalResult(direct=False, answer=answer, state=st,
                             completed=completed, resumed=resumed)
    except Exception:  # noqa: BLE001
        return RunGoalResult(direct=True, answer="", state=state)


__all__ = ["run_goal", "RunGoalResult"]
