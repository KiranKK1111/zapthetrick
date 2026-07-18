"""Multi-agent workflows (agent-orchestration R3).

`select_workflow(signals)` picks a role workflow; `RoleRunner` assigns each role
a capability-matched model (via injected `route_for` → `intelligent-model-routing`),
runs it (injected `run_role` → `run_agent`/answer), routes validator/review roles
through the injected `verify` (`verified_answer`), and combines the role outputs
into one result (R3). A single-agent path is used when sufficient (R3.4,
Property 3). Async + injectable so it's testable with fakes; never raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

SINGLE = "single"
PLAN_EXEC_VALIDATE = "plan_exec_validate"
RESEARCH = "research"
CODE_REVIEW = "code_review"

_ROLES = {
    PLAN_EXEC_VALIDATE: ("planner", "executor", "validator"),
    RESEARCH: ("researcher", "fact_checker", "summarizer"),
    CODE_REVIEW: ("coder", "reviewer", "security_reviewer"),
}
# Roles whose pass goes through verified_answer (validation/review).
_VERIFY_ROLES = {"validator", "fact_checker", "reviewer", "security_reviewer"}


@dataclass
class Workflow:
    kind: str
    roles: tuple[str, ...] = ()
    reasons: list[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    answer: str
    kind: str
    roles_used: dict = field(default_factory=dict)
    degraded_to_single: bool = False


def _cfg_max_roles() -> int:
    try:
        from app.core.config_loader import cfg
        return max(1, int(getattr(cfg.orchestration, "max_roles", 4)))
    except Exception:  # noqa: BLE001
        return 4


def select_workflow(signals: dict | None, *, enabled: bool = True) -> Workflow:
    """Pick a role workflow from the request signals. Disabled / simple →
    single (R3.4, Property 3). Never raises."""
    try:
        if not enabled or not signals:
            return Workflow(SINGLE, (), ["disabled or no signals"])
        difficulty = str(signals.get("difficulty", "standard")).lower()
        category = str(signals.get("task_category", "general")).lower()
        multi = bool(signals.get("multi_goal", False))

        if difficulty not in ("hard", "expert") and not multi:
            return Workflow(SINGLE, (), ["simple request → single agent"])

        if category in ("coding", "agentic"):
            return Workflow(CODE_REVIEW, _ROLES[CODE_REVIEW],
                            [f"{category} → coder/reviewer/security"])
        if category in ("research", "writing"):
            return Workflow(RESEARCH, _ROLES[RESEARCH],
                            [f"{category} → research/fact-check/summarize"])
        if difficulty == "expert" or multi:
            return Workflow(PLAN_EXEC_VALIDATE, _ROLES[PLAN_EXEC_VALIDATE],
                            ["complex → plan/execute/validate"])
        return Workflow(SINGLE, (), ["single sufficient"])
    except Exception:  # noqa: BLE001
        return Workflow(SINGLE, (), ["error → single"])


class RoleRunner:
    """Runs a workflow's roles, each on a capability-matched model, combining
    outputs. All external work is injected (testable, no models/DB):
      route_for(role, category) -> model_label  (intelligent-model-routing)
      run_role(role, model, prior) -> str       (run_agent / answer pass)
      verify(role, text) -> str                 (verified_answer for review roles)
    """

    def __init__(self, route_for: Callable[[str, str], Awaitable[str]],
                 run_role: Callable[[str, str, dict], Awaitable[str]],
                 verify: Callable[[str, str], Awaitable[str]] | None = None,
                 combine: Callable[[dict], str] | None = None):
        self._route_for = route_for
        self._run_role = run_role
        self._verify = verify
        self._combine = combine or self._default_combine

    @staticmethod
    def _default_combine(role_outputs: dict) -> str:
        for key in ("summarizer", "security_reviewer", "validator", "reviewer",
                    "executor", "fact_checker", "coder", "researcher", "planner"):
            if role_outputs.get(key):
                return role_outputs[key]
        return next((v for v in role_outputs.values() if v), "")

    async def run(self, workflow: Workflow, category: str = "general") -> WorkflowResult:
        try:
            return await self._run(workflow, category)
        except Exception:  # noqa: BLE001
            return WorkflowResult("", SINGLE, {}, degraded_to_single=True)

    async def _run(self, workflow: Workflow, category: str) -> WorkflowResult:
        if workflow.kind == SINGLE or not workflow.roles:
            model = await self._route_for("single", category)
            ans = await self._run_role("single", model, {})
            return WorkflowResult(ans, SINGLE, {"single": model})

        roles = workflow.roles[: _cfg_max_roles()]
        outputs: dict = {}
        models: dict = {}
        for role in roles:
            try:
                model = await self._route_for(role, category)
                models[role] = model
                text = await self._run_role(role, model, dict(outputs))
                if role in _VERIFY_ROLES and self._verify is not None:
                    verified = await self._verify(role, text)
                    text = verified or text
                outputs[role] = text
            except Exception:  # noqa: BLE001 — a failed role degrades, doesn't abort
                outputs[role] = ""
        if not any(outputs.values()):
            # Everything failed → degrade to a single pass.
            model = await self._route_for("single", category)
            ans = await self._run_role("single", model, {})
            return WorkflowResult(ans, SINGLE, {"single": model},
                                  degraded_to_single=True)
        return WorkflowResult(self._combine(outputs), workflow.kind, models)


__all__ = [
    "select_workflow", "Workflow", "WorkflowResult", "RoleRunner",
    "SINGLE", "PLAN_EXEC_VALIDATE", "RESEARCH", "CODE_REVIEW",
]
