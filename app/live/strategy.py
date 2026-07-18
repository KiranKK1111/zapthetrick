"""
Question-type-aware answer strategy (live-conversational-intelligence R8).

Picks an Answer_Strategy from the question type + interview phase + question
text and returns a prompt-shaping scaffold (STAR / design-session / coding-flow
/ definition / comparison / trade-off / debugging). The scaffold is injected
into the SAME answer-generation call (no second blocking LLM call). Fail-open:
unknown → the generic strategy (empty scaffold = today's prompt).
"""
from __future__ import annotations

from app.core import lexicons
from app.live import phase as _phase

STAR = "star"
DESIGN_SESSION = "design_session"
CODING_FLOW = "coding_flow"
DEFINITION = "definition"
COMPARISON = "comparison"
TRADEOFF = "tradeoff"
DEBUGGING = "debugging"
GENERAL = "general"

STRATEGIES = {
    STAR, DESIGN_SESSION, CODING_FLOW, DEFINITION, COMPARISON, TRADEOFF,
    DEBUGGING, GENERAL,
}

_SCAFFOLD = {
    STAR: ("Structure the answer as STAR — Situation, Task, Action, Result — "
           "and end with a one-line takeaway."),
    DESIGN_SESSION: ("Structure the answer as a system-design walkthrough: "
                     "clarify requirements & scale, propose a high-level "
                     "architecture, cover key components and data flow, then "
                     "trade-offs and bottlenecks."),
    CODING_FLOW: ("Structure the answer as a coding walkthrough: restate the "
                  "problem and constraints, outline the approach, give the time "
                  "and space complexity, and note edge cases (include code only "
                  "if asked)."),
    DEFINITION: ("Define the concept clearly, give one concrete example, then "
                 "say where/why it is used."),
    COMPARISON: ("Compare the options across the key dimensions and finish with "
                 "a short verdict on when to pick each."),
    TRADEOFF: ("Lay out the trade-offs explicitly and state the conditions under "
               "which each option wins."),
    DEBUGGING: ("Walk through debugging methodically: reproduce, hypothesize, "
                "isolate, fix, then verify."),
    GENERAL: "",
}

_COMPARE_CUES = lexicons.LIVE_STRATEGY_COMPARE_CUES
_TRADEOFF_CUES = lexicons.LIVE_STRATEGY_TRADEOFF_CUES
_DEBUG_CUES = lexicons.LIVE_STRATEGY_DEBUG_CUES
_CONCEPT_PREFIXES = lexicons.LIVE_STRATEGY_CONCEPT_PREFIXES


def select_strategy(qtype: str, phase: str = "", question: str = "") -> str:
    """Choose an Answer_Strategy. Deterministic; never raises."""
    try:
        qt = (qtype or "").lower()
        ph = (phase or "").lower()
        t = (question or "").lower()

        if ph == _phase.BEHAVIORAL or qt == "behavioral" or ph == _phase.HR:
            return STAR
        if ph == _phase.SYSTEM_DESIGN:
            return DESIGN_SESSION
        if ph == _phase.CODING or qt == "coding":
            return CODING_FLOW
        if any(c in t for c in _DEBUG_CUES):
            return DEBUGGING
        if any(c in t for c in _TRADEOFF_CUES):
            return TRADEOFF
        if any(c in t for c in _COMPARE_CUES):
            return COMPARISON
        if qt == "technical_concept" or t.startswith(_CONCEPT_PREFIXES):
            return DEFINITION
        return GENERAL
    except Exception:  # noqa: BLE001
        return GENERAL


def prompt_shaping(strategy: str) -> str:
    """Return the scaffold instruction for a strategy ("" for general)."""
    return _SCAFFOLD.get((strategy or "").lower(), "")
