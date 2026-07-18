"""Unified per-turn world state (ArchitectureVerdict Phase 5).

The audit found three disjoint state models: the agent-mesh `Blackboard`, the
live `InterviewWorldModel`, and the chat `Assessment`/goal-ledger. [TurnState]
is the canonical view the design doc's "shared world state" calls for — one
JSON-friendly object carrying everything Phase 1-4 now produce about a turn:

    goal · intent · requirement matrix · assumptions · risk · policy decision
    · capability snapshot · plan · artifacts

Wrap, don't rewrite: TurnState is BUILT FROM the existing structures (it never
replaces them) and travels on the blackboard under KEY "turn_state", so every
mesh agent, the SSE layer, and the trace read the same picture. The live
pipeline keeps its InterviewWorldModel and can project into the same shape via
`from_live_snapshot`.

Fail-open: builders never raise; missing sub-structures simply leave fields
empty.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

KEY_TURN_STATE = "turn_state"


@dataclass
class TurnState:
    """Canonical decision state for one turn."""
    goal: str = ""                        # the user's request (normalized)
    intent: str = ""                      # detected intent label
    decision: str = ""                    # answer | clarify | defer
    confidence: float = 0.0
    ambiguity: float = 0.0
    risk: float = 0.0
    risk_level: str = "low"
    missing_required: list[str] = field(default_factory=list)
    matrix: dict | None = None            # RequirementMatrix.as_dict()
    assumptions: list[str] = field(default_factory=list)
    policy: dict | None = None            # PolicyDecision record
    capabilities: dict | None = None      # capability snapshot (trimmed)
    plan: list[dict] = field(default_factory=list)      # [{id,text,deps}]
    artifacts: list[dict] = field(default_factory=list)  # produced this turn
    horizon: str = ""                     # temporal planning horizon (Phase 3 #16)
    deadline: bool = False                # request carries a deadline
    constraints: list[dict] = field(default_factory=list)  # output constraints (Phase 4 #7)
    created_at: float = field(default_factory=time.time)

    # ---- builders ----------------------------------------------------------
    @classmethod
    def from_assessment(cls, assessment: Any, *, goal: str = "",
                        capabilities: bool = True) -> "TurnState":
        """Project the Phase 1-3 Assessment into the unified state."""
        ts = cls(goal=goal or "")
        try:
            ts.intent = getattr(assessment, "intent", "") or ""
            ts.decision = getattr(assessment, "decision", "") or ""
            ts.confidence = float(getattr(assessment, "confidence", 0.0) or 0)
            ts.ambiguity = float(getattr(assessment, "ambiguity", 0.0) or 0)
            ts.risk = float(getattr(assessment, "risk", 0.0) or 0)
            ts.risk_level = getattr(assessment, "risk_level", "low") or "low"
            ts.missing_required = list(
                getattr(assessment, "missing_required", []) or [])
            m = getattr(assessment, "matrix", None)
            if m is not None and hasattr(m, "as_dict"):
                ts.matrix = m.as_dict()
            ts.policy = getattr(assessment, "policy", None)
        except Exception:  # noqa: BLE001 — a partial state is still useful
            pass
        try:
            from app.core import temporal
            sig = temporal.temporal_signal(ts.goal)
            ts.horizon = sig["horizon"]
            ts.deadline = sig["deadline"]
        except Exception:  # noqa: BLE001
            pass
        try:
            from app.core import constraints as _constraints
            ts.constraints = [
                {"kind": c.kind, "key": c.key, "text": c.text}
                for c in _constraints.extract_constraints(ts.goal)
            ]
        except Exception:  # noqa: BLE001
            pass
        if capabilities:
            try:
                from app.capabilities import (available_document_formats,
                                              capability_snapshot)
                snap = capability_snapshot()
                ts.capabilities = {
                    "document_formats": sorted(available_document_formats()),
                    "sandbox": bool(snap.get("sandbox")),
                    "gpu": bool((snap.get("gpu") or {}).get("available")),
                    "web_search": bool(snap.get("web_search")),
                }
            except Exception:  # noqa: BLE001
                ts.capabilities = None
        return ts

    @classmethod
    def from_live_snapshot(cls, snapshot: dict | None, *,
                           goal: str = "") -> "TurnState":
        """Project the live InterviewWorldModel snapshot into the same shape,
        so live and chat expose one protocol to consumers."""
        ts = cls(goal=goal)
        try:
            s = snapshot or {}
            ts.intent = "live_interview"
            ts.assumptions = [str(a) for a in (s.get("assumptions") or [])]
            ts.plan = [{"id": 0, "text": s.get("active_question") or "",
                        "deps": []}] if s.get("active_question") else []
            from app.core import temporal
            ts.horizon = temporal.classify_horizon(s.get("active_question") or "")
        except Exception:  # noqa: BLE001
            pass
        return ts

    # ---- mutation helpers ---------------------------------------------------
    def set_plan(self, subtasks: list) -> None:
        try:
            self.plan = [{"id": getattr(t, "id", i),
                          "text": getattr(t, "text", str(t)),
                          "deps": list(getattr(t, "deps", []) or [])}
                         for i, t in enumerate(subtasks or [])]
        except Exception:  # noqa: BLE001
            pass

    def add_artifact(self, name: str, fmt: str, meta: dict | None = None) -> None:
        try:
            self.artifacts.append({"name": name, "format": fmt,
                                   "meta": dict(meta or {})})
        except Exception:  # noqa: BLE001
            pass

    # ---- consumption -------------------------------------------------------
    def answer_directive(self) -> str:
        """A compact directive that lets the answering model HONOR this turn's
        extracted output constraints + planning horizon — the piece that makes
        TurnState a genuine runtime CONSUMER, not just a produced record.

        Empty for a plain question (no constraints, conversation horizon, no
        deadline) so a normal answer is left completely unchanged. Fail-open."""
        try:
            from app.core import temporal
            parts: list[str] = []
            if self.constraints:
                reqs = "; ".join(
                    str(c.get("text") or "") for c in self.constraints
                    if isinstance(c, dict) and c.get("text"))
                if reqs:
                    parts.append(
                        "OUTPUT REQUIREMENTS the answer MUST satisfy: "
                        f"{reqs}.")
            if self.horizon == temporal.IMMEDIATE:
                parts.append("The user wants a quick, direct answer — be concise.")
            elif self.horizon in (temporal.PROJECT, temporal.LONG_TERM):
                parts.append("This is a broad, multi-step goal — be thorough "
                             "and well-structured.")
            if self.deadline:
                parts.append("The request is time-sensitive; lead with the "
                             "essential answer first.")
            return " ".join(parts)
        except Exception:  # noqa: BLE001
            return ""

    def check_output(self, output: str) -> dict | None:
        """Verify a produced answer against this turn's extracted constraints
        (Phase 3 truth-maintenance on the chat side). Returns a compact report
        dict, or None when the turn imposed no checkable constraint. Fail-open."""
        try:
            if not self.constraints:
                return None
            from app.core import constraints as _c
            cons = [_c.Constraint(kind=x.get("kind"), key=str(x.get("key") or ""),
                                  text=str(x.get("text") or ""))
                    for x in self.constraints if isinstance(x, dict) and x.get("kind")]
            if not cons:
                return None
            rep = _c.check(output or "", cons)
            return {
                "satisfied": rep.satisfied,
                "violations": list(rep.violations),
                "checked": rep.checked,
                "unchecked": list(rep.unchecked),
            }
        except Exception:  # noqa: BLE001
            return None

    def as_dict(self) -> dict:
        return {
            "goal": self.goal, "intent": self.intent,
            "decision": self.decision,
            "confidence": round(self.confidence, 3),
            "ambiguity": round(self.ambiguity, 3),
            "risk": round(self.risk, 3), "risk_level": self.risk_level,
            "missing_required": list(self.missing_required),
            "matrix": self.matrix, "assumptions": list(self.assumptions),
            "policy": self.policy, "capabilities": self.capabilities,
            "plan": list(self.plan), "artifacts": list(self.artifacts),
            "horizon": self.horizon, "deadline": self.deadline,
            "constraints": list(self.constraints),
            "created_at": self.created_at,
        }


__all__ = ["TurnState", "KEY_TURN_STATE"]
