"""Multi-agent orchestration (agent-orchestration spec).

Grows the single agent loop into a coordinated, role-based, sandboxed,
self-testing multi-agent layer — by EXTENDING `app/agent/loop.run_agent`,
`app/agent_workspace.run_in_workspace`, `app/mcp` tools, and reusing
`verified_answer`. Per-role model selection delegates to
`intelligent-model-routing`, degradation/quality to `evaluation-and-reliability`,
and the task list to `workspace-and-artifacts`.

Opt-in + flag-gated (`cfg.orchestration.enabled`): off → today's single
agent/answer path. All code execution stays inside the existing workspace
sandbox under the existing caps. Every entry point is deterministic-first and
fail-open.
"""
from .decompose import decompose, SubTask
from .tool_plan import plan_tools, ToolPlan
from .workflow import select_workflow, Workflow, RoleRunner
from .goal_engine import (
    Goal, build_goal, validate_goal, ExecutionLedger, execute_dag,
    constraint_gate, preflight, acceptance_tests,
)

__all__ = [
    "decompose", "SubTask", "plan_tools", "ToolPlan",
    "select_workflow", "Workflow", "RoleRunner",
    "Goal", "build_goal", "validate_goal", "ExecutionLedger", "execute_dag",
    "constraint_gate", "preflight", "acceptance_tests",
]
