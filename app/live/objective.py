"""
Evaluation-objective + expected-depth estimation
(live-conversational-intelligence R30; multi-pass understanding extended in R50).

Estimates what a question is really probing (e.g. "What is CAP theorem?" →
trade-off reasoning) and the depth it expects (definition / architecture /
internals / source-level), reusing the existing `difficulty` (no new LLM call).
Feeds answer-strategy + length. Deterministic + fail-open.
"""
from __future__ import annotations

from app.core import lexicons

# Evaluation objectives.
TRADEOFF = "tradeoff_reasoning"
DESIGN_ABILITY = "system_design_ability"
DEPTH_KNOWLEDGE = "depth_of_knowledge"
PROBLEM_SOLVING = "problem_solving"
COMMUNICATION = "communication"
KNOWLEDGE = "knowledge"

# Expected depth levels.
DEFINITION = "definition"
ARCHITECTURE = "architecture"
INTERNALS = "internals"
SOURCE_LEVEL = "source_level"

_TRADEOFF_CUES = lexicons.LIVE_OBJECTIVE_TRADEOFF_CUES
_DESIGN_CUES = lexicons.LIVE_OBJECTIVE_DESIGN_CUES
_INTERNALS_CUES = lexicons.LIVE_OBJECTIVE_INTERNALS_CUES
_SOURCE_CUES = lexicons.LIVE_OBJECTIVE_SOURCE_CUES
_BEHAVIORAL_CUES = lexicons.LIVE_OBJECTIVE_BEHAVIORAL_CUES


def estimate(question: str, qtype: str = "", phase: str = "",
             difficulty: str = "standard") -> tuple[str, str]:
    """Return (evaluation_objective, expected_depth). Never raises."""
    try:
        t = (question or "").lower()
        qt = (qtype or "").lower()

        # Objective.
        if any(c in t for c in _TRADEOFF_CUES):
            objective = TRADEOFF
        elif any(c in t for c in _DESIGN_CUES):
            objective = DESIGN_ABILITY
        elif qt == "coding":
            objective = PROBLEM_SOLVING
        elif qt == "behavioral" or any(c in t for c in _BEHAVIORAL_CUES):
            objective = COMMUNICATION
        elif t.startswith(("what is", "what are", "define", "explain")):
            objective = KNOWLEDGE
        else:
            objective = DEPTH_KNOWLEDGE

        # Expected depth.
        if any(c in t for c in _SOURCE_CUES):
            depth = SOURCE_LEVEL
        elif any(c in t for c in _INTERNALS_CUES):
            depth = INTERNALS
        elif any(c in t for c in _DESIGN_CUES):
            depth = ARCHITECTURE
        else:
            d = (difficulty or "standard").lower()
            depth = {"expert": INTERNALS, "hard": ARCHITECTURE}.get(d, DEFINITION)
        return objective, depth
    except Exception:  # noqa: BLE001
        return KNOWLEDGE, DEFINITION


def directive(objective: str, depth: str) -> str:
    """A one-line guidance fragment for the answer prompt (folded into the same
    call)."""
    obj = (objective or "").replace("_", " ")
    dep = (depth or "").replace("_", " ")
    if not obj and not dep:
        return ""
    return f"This question probes {obj}; answer at the '{dep}' level of depth."


# ── Multi-pass understanding (R50) ─────────────────────────────────────
# A second deterministic "pass" over the question that refines the objective +
# depth using the recent context — WITHOUT a second LLM call. The first pass is
# `estimate()`; this pass adjusts when the recent turns reveal a deeper probe
# (e.g. a follow-up "but why?" on a definition escalates to internals).
_ESCALATE_CUES = lexicons.LIVE_OBJECTIVE_ESCALATE_CUES


def multi_pass(question: str, qtype: str = "", phase: str = "",
               difficulty: str = "standard",
               recent: list[str] | None = None) -> tuple[str, str]:
    """Two-pass understanding: run `estimate`, then refine with recent context.
    Deterministic, no extra LLM call. Never raises."""
    try:
        objective, depth = estimate(question, qtype, phase, difficulty)
        rec = " ".join((recent or [])[-3:]).lower()
        q = (question or "").lower()
        if any(c in q or c in rec for c in _ESCALATE_CUES):
            # Escalate the depth one notch.
            order = [DEFINITION, ARCHITECTURE, INTERNALS, SOURCE_LEVEL]
            try:
                i = order.index(depth)
                depth = order[min(i + 1, len(order) - 1)]
            except ValueError:
                depth = INTERNALS
        return objective, depth
    except Exception:  # noqa: BLE001
        return KNOWLEDGE, DEFINITION
