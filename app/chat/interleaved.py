"""Interleaved (mid-generation) tool use (vNext §9.2, Stage 9 Component B).

The existing `run_tool_loop` runs tools BEFORE the answer streams. §9.2 lets the
model call tools DURING generation: it emits a `tool_use` block, the stream
pauses, the call is guarded (§9.9 capability-drop) and executed, the framed
result is spliced onto the CACHED prefix, and generation resumes — so a long
answer can fetch, compute, or search exactly when it needs to.

This module owns the deterministic ORCHESTRATION that ties it together:
  * `ToolBudget` — bounds the loop (≤3 interactive in a chat turn, ≤15 in a
    background agent-task);
  * `parse_tool_use` — extract a tool_use block from a generated fragment;
  * `decide_step` — the per-step decision: parse → budget → §9.9 gate (via the
    turn's `TaintTracker`) → EXECUTE | PARK_FOR_APPROVAL | BUDGET_EXCEEDED | ANSWER;
  * `frame_result` — quarantine-wrap a tool result (untrusted) AND taint the turn.

The streaming pause/splice/resume itself is the engine/WS integration (an injected
seam); this pure core is unit-tested with no model. Fail-open. Flag-gated
(`tool_loop.interleaved`, default OFF → today's pre-answer-only loop).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Step actions.
EXECUTE = "execute"
PARK_FOR_APPROVAL = "park_for_approval"
BUDGET_EXCEEDED = "budget_exceeded"
ANSWER = "answer"                    # no tool_use → keep generating the answer

INTERACTIVE = "interactive"
AGENT_TASK = "agent_task"


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.tool_loop, "interleaved", False))
    except Exception:  # noqa: BLE001
        return False


def _budget_for(mode: str) -> int:
    try:
        from app.core.config_loader import cfg
        if mode == AGENT_TASK:
            return int(getattr(cfg.tool_loop, "interleaved_task_budget", 15) or 15)
        return int(getattr(cfg.tool_loop, "interleaved_interactive_budget", 3) or 3)
    except Exception:  # noqa: BLE001
        return 15 if mode == AGENT_TASK else 3


@dataclass
class ToolBudget:
    mode: str = INTERACTIVE
    limit: int = 0                   # 0 → resolved from config lazily
    used: int = 0

    def _cap(self) -> int:
        return self.limit or _budget_for(self.mode)

    def can_call(self) -> bool:
        return self.used < self._cap()

    def remaining(self) -> int:
        return max(0, self._cap() - self.used)

    def record(self) -> None:
        self.used += 1

    def to_dict(self) -> dict:
        return {"mode": self.mode, "used": self.used, "limit": self._cap(),
                "remaining": self.remaining()}


@dataclass
class ToolUse:
    tool: str
    args: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"tool": self.tool, "args": dict(self.args)}


def parse_tool_use(fragment: str) -> "ToolUse | None":
    """Extract a `tool_use` block from a generated fragment. Reuses the tolerant
    balanced-brace JSON extractor from `chat.tool_loop`. None when there's no
    parseable tool call. Never raises."""
    try:
        from app.chat.tool_loop import _extract_action
        obj = _extract_action(fragment or "")
        if not obj or not isinstance(obj, dict):
            return None
        tool = str(obj.get("tool") or "").strip()
        if not tool:
            return None
        args = obj.get("args") if isinstance(obj.get("args"), dict) else {
            k: v for k, v in obj.items() if k not in ("tool", "args")}
        return ToolUse(tool=tool, args=args or {})
    except Exception:  # noqa: BLE001
        return None


@dataclass
class StepDecision:
    action: str
    tool: str = ""
    args: dict = field(default_factory=dict)
    reason: str = ""
    needs_approval: bool = False

    def to_dict(self) -> dict:
        return {"action": self.action, "tool": self.tool, "args": dict(self.args),
                "reason": self.reason, "needs_approval": self.needs_approval}


def decide_step(fragment: str, *, budget: ToolBudget, taint=None,
                allowed_tools=None) -> StepDecision:
    """The per-step interleaved decision. Parse the fragment for a tool_use; if
    none → ANSWER (keep generating). Else: enforce the allow-list, the BUDGET,
    and the §9.9 capability gate (a side-effectful tool on a tainted turn →
    PARK_FOR_APPROVAL). Otherwise EXECUTE. When disabled → always ANSWER
    (byte-identical: the interleaved path never engages). Never raises."""
    try:
        if not enabled():
            return StepDecision(ANSWER, reason="interleaved disabled")
        use = parse_tool_use(fragment)
        if use is None:
            return StepDecision(ANSWER, reason="no tool_use")
        # Allow-list (when provided).
        if allowed_tools is not None and use.tool not in allowed_tools:
            return StepDecision(ANSWER, tool=use.tool,
                                reason=f"tool '{use.tool}' not in allow-list")
        # Budget.
        if not budget.can_call():
            return StepDecision(BUDGET_EXCEEDED, tool=use.tool, args=use.args,
                                reason=f"tool budget exhausted ({budget.used}/"
                                       f"{budget._cap()})")
        # §9.9 capability gate.
        if taint is not None:
            try:
                dec = taint.gate(use.tool)
                if not dec.allow:
                    return StepDecision(PARK_FOR_APPROVAL, tool=use.tool,
                                        args=use.args, reason=dec.reason,
                                        needs_approval=True)
            except Exception:  # noqa: BLE001 — a gate error must fail SAFE
                return StepDecision(PARK_FOR_APPROVAL, tool=use.tool,
                                    args=use.args, reason="gate error — parked",
                                    needs_approval=True)
        return StepDecision(EXECUTE, tool=use.tool, args=use.args,
                            reason="within budget, capability ok")
    except Exception:  # noqa: BLE001
        return StepDecision(ANSWER, reason="decide error → answer")


def frame_result(tool: str, result, *, source: str = "mcp", taint=None) -> str:
    """Frame a tool result for splicing back into the prompt: it is UNTRUSTED
    (from web/mcp/…), so quarantine-wrap it AND taint the turn (so a later
    side-effectful tool_use on the strength of it is gated). Returns the framed
    block. Never raises."""
    try:
        body = result if isinstance(result, str) else _stringify(result)
        from app.security import quarantine as _q
        if taint is not None:
            try:
                taint.ingest(body, source=source)
            except Exception:  # noqa: BLE001
                pass
        wrapped = _q.quarantine_wrap(body, source=source, provenance=tool)
        return wrapped or body
    except Exception:  # noqa: BLE001
        return result if isinstance(result, str) else str(result)


def _stringify(result) -> str:
    try:
        import json
        return json.dumps(result, ensure_ascii=False, default=str)[:16000]
    except Exception:  # noqa: BLE001
        return str(result)[:16000]


__all__ = ["EXECUTE", "PARK_FOR_APPROVAL", "BUDGET_EXCEEDED", "ANSWER",
           "INTERACTIVE", "AGENT_TASK", "enabled", "ToolBudget", "ToolUse",
           "parse_tool_use", "StepDecision", "decide_step", "frame_result"]
