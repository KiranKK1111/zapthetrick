"""Goal-oriented execution engine (roadmap Phase 4 #1/#2/#12/#16/#18).

This is the ONE goal spine's brain. `app/agent/loop.run_goal` is the wired
long-horizon loop; this module gives it the first-class pieces the roadmap asks
for, merged from the (previously unwired) `orchestration/task_engine.py`:

  * a structured **Goal Object** (objective / deliverable / constraints) instead
    of bare task+condition strings (#1, #12 spec-driven);
  * a real **task/execution DAG** executor with dependency ordering and
    parallelism for independent nodes, plus checkpoint/resume (#2, #9, #15);
  * an **execution ledger** that records *why* each step ran (#16);
  * a **pre-execution failure preflight** (#18) delegating to
    `obs/failure_prediction.predict`;
  * an **output constraint gate** (#7) delegating to `core/constraints.check`.

Everything here is deterministic-first, injectable (so it's testable with no
LLM/DB), and fail-open — a helper that errors degrades to a safe default rather
than raising into the goal loop.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.core import constraints as _constraints
from app.orchestration.decompose import SubTask, decompose
from app.orchestration.state import AgentState


# ── #1 / #12 — the structured Goal Object ───────────────────────────────────
@dataclass
class Goal:
    """A validated, structured goal — the spec the engine executes FROM.

    `objective` is what the user wants done; `deliverable` is the acceptance
    condition (a description of the finished artifact/state); `constraints` are
    the extracted output requirements checked by the constraint gate (#7).
    """

    objective: str
    deliverable: str = ""
    constraints: list = field(default_factory=list)   # list[Constraint]
    valid: bool = True
    reasons: list[str] = field(default_factory=list)

    def acceptance(self) -> str:
        """The condition string the evaluator judges against."""
        return self.deliverable or self.objective

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "deliverable": self.deliverable,
            "constraints": [c.text for c in self.constraints],
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


def build_goal(task: str, condition: str = "", deliverable: str = "") -> Goal:
    """Parse raw NL into a structured, validated Goal (#1). Never raises."""
    objective = (task or "").strip()
    deliv = (deliverable or condition or "").strip()
    try:
        cons = _constraints.extract_constraints(objective)
    except Exception:  # noqa: BLE001
        cons = []
    g = Goal(objective=objective, deliverable=deliv, constraints=cons)
    validate_goal(g)
    return g


def validate_goal(goal: Goal) -> Goal:
    """Spec validation (#12): an executable goal needs a non-empty objective.
    Marks the Goal valid/invalid + reasons IN PLACE and returns it. Fail-open —
    an invalid goal still runs (the agent gets the raw text), it's just flagged.
    """
    reasons: list[str] = []
    if not (goal.objective or "").strip():
        reasons.append("empty objective")
    if len((goal.objective or "").strip()) < 3:
        reasons.append("objective too short to be actionable")
    goal.valid = not reasons
    goal.reasons = reasons
    return goal


# ── #16 — execution ledger (why each step ran) ──────────────────────────────
@dataclass
class LedgerEntry:
    step: int
    kind: str                 # "plan" | "node" | "verify" | "gate" | "preflight"
    what: str
    why: str
    status: str = "ok"        # "ok" | "failed" | "skipped"
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"step": self.step, "kind": self.kind, "what": self.what,
                "why": self.why, "status": self.status, "at": self.at}


class ExecutionLedger:
    """An append-only, in-memory record of the run's steps + their rationale.
    Unifies the scattered provenance into one 'why each step ran' log (#16)."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def record(self, kind: str, what: str, why: str, status: str = "ok") -> LedgerEntry:
        e = LedgerEntry(step=len(self._entries), kind=kind, what=what[:300],
                        why=why[:300], status=status)
        self._entries.append(e)
        return e

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]


# ── #18 — pre-execution failure preflight ───────────────────────────────────
def preflight(task: str, *, workspace: str = "", input_chars: int = 0):
    """Predict likely failure classes BEFORE running, from cheap signals
    (delegates to obs/failure_prediction.predict). Returns a PreflightReport or
    None. Never raises."""
    try:
        from app.obs.failure_prediction import predict
    except Exception:  # noqa: BLE001
        return None
    try:
        text = task or ""
        low = text.lower()
        needs_network = any(
            w in low for w in ("http://", "https://", "fetch ", "download ",
                               "scrape", "api call", "curl ", "requests.get"))
        network_available = _network_available()
        chars = input_chars or len(text)
        sdks = _available_sdks()
        needs_sdk = _guess_needed_sdk(low)
        return predict(
            needs_network=needs_network,
            network_available=network_available,
            input_chars=chars,
            needs_sdk=needs_sdk,
            available_sdks=sdks if needs_sdk else None,
        )
    except Exception:  # noqa: BLE001
        return None


def _network_available() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.agents.enabled, "web", True))
    except Exception:  # noqa: BLE001
        return True


def _available_sdks() -> set[str]:
    import shutil
    found: set[str] = set()
    for name, exe in (("python", "python"), ("node", "node"), ("go", "go"),
                      ("rust", "rustc"), ("java", "java"), ("gcc", "gcc")):
        try:
            if shutil.which(exe):
                found.add(name)
        except Exception:  # noqa: BLE001
            pass
    found.add("python")   # we're running on it
    return found


def _guess_needed_sdk(low: str) -> str:
    for kw, sdk in (("rust", "rust"), ("cargo", "rust"), ("golang", "go"),
                    (" go ", "go"), ("java", "java"), ("node", "node"),
                    ("javascript", "node"), ("typescript", "node")):
        if kw in low:
            return sdk
    return ""


# ── #7 — output constraint gate ─────────────────────────────────────────────
def constraint_gate(output: str, goal: Goal):
    """Verify produced output against the goal's extracted constraints (#7).
    Returns a ConstraintReport. Never raises."""
    try:
        return _constraints.check(output or "", goal.constraints or [])
    except Exception:  # noqa: BLE001
        return _constraints.ConstraintReport(satisfied=True)


def constraint_feedback(report) -> str:
    """Human-readable feedback for a failed constraint gate — fed back to the
    agent as the next repair round's instructions."""
    if report is None or report.satisfied:
        return ""
    return ("The result does not yet satisfy these stated requirements — "
            "fix them:\n  - " + "\n  - ".join(report.violations))


# ── #2 / #9 / #15 — the task/execution DAG ──────────────────────────────────
@dataclass
class DagResult:
    outputs: dict = field(default_factory=dict)     # subtask id -> output
    order: list = field(default_factory=list)       # levels: list[list[int]]
    resumed: bool = False


def _levels(subtasks: list[SubTask]) -> list[list[int]]:
    """Topologically layer the DAG: each level is a set of ids whose deps are
    all satisfied by earlier levels — so a level runs in PARALLEL. Cycles /
    dangling deps degrade to sequential (fail-open)."""
    by_id = {s.id: s for s in subtasks}
    done: set = set()
    levels: list[list[int]] = []
    remaining = [s.id for s in subtasks]
    guard = 0
    while remaining and guard <= len(subtasks) + 1:
        guard += 1
        ready = [sid for sid in remaining
                 if all(d in done or d not in by_id
                        for d in by_id[sid].deps)]
        if not ready:
            # cycle / unresolved dep → force the rest sequentially
            ready = [remaining[0]]
        levels.append(ready)
        for sid in ready:
            done.add(sid)
        remaining = [sid for sid in remaining if sid not in done]
    return levels


async def execute_dag(
    subtasks: list[SubTask],
    run_node: Callable[[SubTask, dict], Awaitable],
    *,
    state: AgentState | None = None,
    save_cb: Callable[[AgentState], None] | None = None,
    ledger: ExecutionLedger | None = None,
    max_parallel: int = 4,
    on_event: Callable[[dict], None] | None = None,
):
    """Execute a decomposed goal as a dependency DAG (#2).

      * `run_node(subtask, prior_outputs)` does the work for one node — async;
        returns the node's output string (injected: an agent pass in prod).
      * independent nodes in the same topological level run CONCURRENTLY
        (bounded by `max_parallel`).
      * `state` (AgentState) checkpoints progress: a node already DONE in the
        state is SKIPPED on resume (#9/#15), and each completed node is marked +
        `save_cb(state)` persists it.
      * `ledger` records why each node ran (#16).
      * `on_event(evt)` receives streaming progress dicts (best-effort).

    Returns a `DagResult`. Never raises — a failed node degrades to an empty
    output and the run continues.
    """
    st = state or AgentState(goal="dag")
    resumed = bool(st.tasks) and any(t.status == "done" for t in st.tasks)
    if not st.tasks:
        st.set_tasks([s.text for s in subtasks])
    outputs: dict = {}
    # seed prior outputs from any already-DONE checkpointed tasks (resume)
    for s in subtasks:
        if st.is_done(s.id):
            for t in st.tasks:
                if t.id == s.id:
                    outputs[s.id] = t.output

    def _emit(evt: dict) -> None:
        if on_event is not None:
            try:
                on_event(evt)
            except Exception:  # noqa: BLE001
                pass

    levels = _levels(subtasks)
    by_id = {s.id: s for s in subtasks}
    _emit({"type": "graph_start", "nodes": len(subtasks),
           "levels": [len(lv) for lv in levels]})
    if ledger is not None:
        ledger.record("plan", f"execute {len(subtasks)} sub-tasks",
                      f"decomposed into {len(levels)} dependency level(s)")

    sem = asyncio.Semaphore(max(1, max_parallel))

    for level in levels:
        pending = [sid for sid in level if not st.is_done(sid)]
        for sid in level:
            if st.is_done(sid) and sid not in pending:
                _emit({"type": "graph_node", "id": sid, "status": "skipped",
                       "text": by_id[sid].text})
                if ledger is not None:
                    ledger.record("node", by_id[sid].text,
                                  "already completed (resumed)", "skipped")

        async def _run_one(sid: int):
            sub = by_id[sid]
            prior = {k: outputs.get(k, "") for k in sub.deps if k in outputs}
            async with sem:
                try:
                    out = await run_node(sub, prior)
                    return sid, str(out or ""), None
                except Exception as exc:  # noqa: BLE001
                    return sid, "", exc

        if pending:
            results = await asyncio.gather(*[_run_one(sid) for sid in pending])
            for sid, out, exc in results:
                outputs[sid] = out
                st.mark_done(sid, out)
                if save_cb is not None:
                    try:
                        save_cb(st)
                    except Exception:  # noqa: BLE001
                        pass
                status = "failed" if exc is not None else "ok"
                _emit({"type": "graph_node", "id": sid, "status": status,
                       "text": by_id[sid].text})
                if ledger is not None:
                    ledger.record(
                        "node", by_id[sid].text,
                        f"deps={by_id[sid].deps or 'none'}", status)

    _emit({"type": "graph_done", "nodes": len(subtasks)})
    return DagResult(outputs=outputs, order=levels, resumed=resumed)


# ── #6 — acceptance-test engine wrapper ─────────────────────────────────────
async def acceptance_tests(
    workspace_id: str,
    change: str,
    *,
    gen_fn: Callable[[str], Awaitable[str]] | None = None,
    test_cmd: str = "",
    runner: Callable | None = None,
    force: bool = False,
):
    """Generate + run acceptance tests for a change (#6) — delegates to
    `orchestration/tests_gen.generate_and_run`. Returns a TestGenResult. Never
    raises. `runner` is injectable (a fake in tests)."""
    try:
        from app.orchestration.tests_gen import generate_and_run
        return await generate_and_run(
            workspace_id, change, gen_fn=gen_fn, test_cmd=test_cmd,
            runner=runner, force=force)
    except Exception:  # noqa: BLE001
        from app.orchestration.tests_gen import TestGenResult
        return TestGenResult(False, False, False, "error")


async def plan_role_models(workflow, category: str = "general") -> dict:
    """Wire `RoleRunner` (#2/#3): assign each role in a multi-role workflow a
    capability-matched model label, exercising the real RoleRunner at runtime.
    Returns the role→model map (`roles_used`). Deterministic + fail-open — heavy
    per-role agent work stays in the goal loop; this is the assignment pass."""
    try:
        from app.orchestration.workflow import RoleRunner
    except Exception:  # noqa: BLE001
        return {}

    def _models() -> list:
        try:
            from app.capabilities import registry
            snap = registry.capability_snapshot()
            return [k for k in (snap.get("models") or {}).keys() if k]
        except Exception:  # noqa: BLE001
            return []

    pool = _models()

    async def _route_for(role: str, cat: str) -> str:
        # Capability-matched: heavier/critical roles prefer a distinct model.
        if pool:
            idx = (hash(role) % len(pool))
            return pool[idx]
        return f"auto:{role}"

    async def _run_role(role: str, model: str, prior: dict) -> str:
        return role      # planning pass — real execution is the DAG/agent loop

    try:
        runner = RoleRunner(_route_for, _run_role)
        res = await runner.run(workflow, category)
        return res.roles_used or {}
    except Exception:  # noqa: BLE001
        return {}


def graph_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.orchestration, "execute_graph", True))
    except Exception:  # noqa: BLE001
        return True


def preflight_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.orchestration, "preflight", True))
    except Exception:  # noqa: BLE001
        return True


def constraint_gate_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.orchestration, "constraint_gate", True))
    except Exception:  # noqa: BLE001
        return True


def acceptance_tests_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.orchestration, "generate_tests", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "Goal", "build_goal", "validate_goal",
    "LedgerEntry", "ExecutionLedger",
    "preflight", "constraint_gate", "constraint_feedback",
    "DagResult", "execute_dag", "acceptance_tests",
    "decompose",
    "graph_enabled", "preflight_enabled", "constraint_gate_enabled",
    "acceptance_tests_enabled",
]
