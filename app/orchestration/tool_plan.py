"""Tool planning (agent-orchestration R2).

`plan_tools(request, sub_tasks, tools, is_granted)` selects + orders MCP tools
for a request from the provided registry tools, honoring permission gating
(never plans an ungranted tool — R2.2). A planned tool that fails at runtime is
routed around via `route_around` per the degradation policy (R2.3, Property 2).
Deterministic; never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

_MAX_TOOLS_DEFAULT = 8
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class PlannedTool:
    name: str
    server: str = ""
    reason: str = ""


@dataclass
class ToolPlan:
    tools: list[PlannedTool] = field(default_factory=list)
    skipped_ungranted: list[str] = field(default_factory=list)

    def names(self) -> list[str]:
        return [t.name for t in self.tools]


def _cfg_max() -> int:
    try:
        from app.core.config_loader import cfg
        return max(1, int(getattr(cfg.orchestration, "max_tools", 8)))
    except Exception:  # noqa: BLE001
        return _MAX_TOOLS_DEFAULT


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 2}


def _reliability(name: str) -> float:
    """Measured success rate for `name`, or the neutral 0.5 when unknown.

    A never-seen tool scores exactly 0.5, so it is neither rewarded nor
    punished for having no history — only tools we have actually watched fail
    sink below a fresh one.
    """
    try:
        from app.tools.reliability import reliability
        return float(reliability(name))
    except Exception:  # noqa: BLE001
        return 0.5


def plan_tools(request: str, sub_tasks: list | None, tools: list,
               is_granted: Callable[[object], bool] | None = None) -> ToolPlan:
    """Select + order tools relevant to the request from `tools` (objects with
    `.name`/`.description`/`.server`), filtered by `is_granted`. Never raises."""
    try:
        return _plan(request, sub_tasks, tools, is_granted)
    except Exception:  # noqa: BLE001
        return ToolPlan()


def _plan(request, sub_tasks, tools, is_granted) -> ToolPlan:
    text = request or ""
    for st in (sub_tasks or []):
        text += " " + getattr(st, "text", "")
    q = _tokens(text)

    scored: list[tuple[float, object]] = []
    skipped: list[str] = []
    for t in tools or []:
        name = getattr(t, "name", "") or ""
        if is_granted is not None and not is_granted(t):
            skipped.append(name)            # permission gating (R2.2)
            continue
        hay = _tokens(f"{name} {getattr(t, 'description', '')}")
        overlap = len(q & hay)
        # A tool whose name appears verbatim in the request is a strong match.
        name_hit = 1 if name and name.lower() in text.lower() else 0
        score = overlap + 3 * name_hit
        if score > 0:
            scored.append((score, t))

    # Relevance decides the plan; measured reliability breaks ties, so that
    # between two equally-relevant tools the one that actually works wins the
    # slot. Relevance stays dominant — a reliable-but-irrelevant tool must
    # never displace the tool the request is actually about.
    scored.sort(key=lambda x: (x[0], _reliability(getattr(x[1], "name", ""))),
                reverse=True)
    cap = _cfg_max()
    plan = ToolPlan(skipped_ungranted=skipped)
    for _s, t in scored[:cap]:
        plan.tools.append(PlannedTool(
            name=getattr(t, "name", ""), server=getattr(t, "server", ""),
            reason="relevant to request"))
    return plan


def route_around(plan: ToolPlan, failed_tool: str) -> ToolPlan:
    """Degradation (R2.3): drop a failed tool from the plan and continue with the
    rest rather than aborting. Returns a new ToolPlan."""
    try:
        remaining = [t for t in plan.tools if t.name != failed_tool]
        return ToolPlan(tools=remaining,
                        skipped_ungranted=list(plan.skipped_ungranted))
    except Exception:  # noqa: BLE001
        return plan


__all__ = ["plan_tools", "route_around", "ToolPlan", "PlannedTool"]
