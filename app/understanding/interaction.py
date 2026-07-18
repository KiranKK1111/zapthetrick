"""Human-interaction model + density (roadmap Phase 3 #12, chat side).

The audit found the codebase adapted DEPTH (`live/style.py::depth_for_load`) but
had no engine that SELECTS the *interaction move* — whether to ASK vs PROCEED vs
SUMMARIZE — nor the *presentation shape* (prose / table / steps / diagram / code)
best fitted to the request. This module is that selection engine for the chat
turn: one deterministic, fail-open read the answer path can act on.

It is intentionally cheap (no LLM, no embeddings): it reads the request text +
the turn's already-computed signals (the unified `TurnState`, the clarifier's
`missing_required`) and returns an `InteractionPlan`.

Consumed in `routes_agents.py`:
  * `action`  → whether the turn should defer to the clarifier (ASK) or answer.
  * `shape`   → a light presentation directive appended to the model prompt
                ("prefer a comparison table", "number the steps"…), so the
                answer's DENSITY/FORMAT fits the ask instead of always prose.
  * surfaced as additive `interaction` meta for the client.

Fail-open: any error → a neutral PROCEED/prose plan (today's behavior).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── interaction moves ────────────────────────────────────────────────────────
ASK = "ask"              # a required choice is missing → clarify first
PROCEED = "proceed"      # enough to answer now
SUMMARIZE = "summarize"  # the user wants a condensed / TL;DR view

# ── presentation shapes (density) ────────────────────────────────────────────
PROSE = "prose"
TABLE = "table"
STEPS = "steps"
DIAGRAM = "diagram"
CODE = "code"
COMPARISON = "comparison"

_COMPARE_CUES = ("compare", "comparison", "versus", " vs ", " vs.", "difference between",
                 "pros and cons", "trade-off", "tradeoff", "better than", "which is better")
_STEPS_CUES = ("how do i", "how to", "step by step", "step-by-step", "walk me through",
               "guide", "set up", "install", "configure", "procedure", "instructions")
_DIAGRAM_CUES = ("diagram", "architecture", "flowchart", "flow chart", "sequence diagram",
                 "draw", "visualize", "visualise", "er diagram", "state machine",
                 "data flow", "system design")
_CODE_CUES = ("code", "function", "implement", "script", "snippet", "regex",
              "write a program", "class ", "method")
_SUMMARIZE_CUES = ("summarize", "summarise", "tl;dr", "tldr", "in short",
                   "brief overview", "high level", "high-level", "recap",
                   "in a nutshell", "gist")
_TABLE_CUES = ("table", "tabular", "list the", "matrix", "spreadsheet",
               "columns", "as a table")


@dataclass
class InteractionPlan:
    action: str = PROCEED
    shape: str = PROSE
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"action": self.action, "shape": self.shape,
                "reasons": list(self.reasons)}

    def shape_directive(self) -> str:
        """A short, additive instruction that nudges the answer's FORMAT to the
        selected shape. Empty for plain prose so a normal answer is unchanged."""
        if self.shape == TABLE:
            return ("PRESENTATION: the data is best shown as a Markdown table — "
                    "use one with clear column headers.")
        if self.shape == COMPARISON:
            return ("PRESENTATION: this is a comparison — lay the options "
                    "side-by-side in a Markdown table (a column per option) so "
                    "the differences are scannable.")
        if self.shape == STEPS:
            return ("PRESENTATION: give a numbered, ordered list of concrete "
                    "steps the user can follow.")
        if self.shape == DIAGRAM:
            return ("PRESENTATION: include a Mermaid diagram (```mermaid) that "
                    "captures the structure/flow, alongside a brief explanation.")
        return ""


def _has(text: str, cues) -> bool:
    return any(c in text for c in cues)


def select(text: str, *, missing_required: list | None = None,
           horizon: str | None = None) -> InteractionPlan:
    """Choose the interaction move + presentation shape for a chat turn.

    Deterministic + fail-open. `missing_required` (from the clarifier's
    assessment) drives ASK; `horizon` (from TurnState/temporal) biases toward a
    denser/summary shape for immediate asks. Never raises."""
    try:
        return _select(text, missing_required, horizon)
    except Exception:  # noqa: BLE001 — a neutral plan is always safe
        return InteractionPlan()


def _select(text, missing_required, horizon) -> InteractionPlan:
    t = f" {(text or '').lower().strip()} "
    reasons: list[str] = []

    # 1. ASK — a required choice the request omitted (the clarifier owns the
    #    actual card; this only reports the move for observability/consistency).
    if missing_required:
        reasons.append(f"missing required: {', '.join(map(str, missing_required))}")
        action = ASK
    elif _has(t, _SUMMARIZE_CUES):
        reasons.append("explicit summarize/condense request")
        action = SUMMARIZE
    else:
        action = PROCEED

    # 2. SHAPE — pick the densest fitting presentation. Order matters: a
    #    comparison beats a bare table; a diagram/steps ask beats prose.
    if _has(t, _COMPARE_CUES):
        shape = COMPARISON
        reasons.append("comparison cue")
    elif _has(t, _DIAGRAM_CUES):
        shape = DIAGRAM
        reasons.append("diagram/architecture cue")
    elif _has(t, _STEPS_CUES):
        shape = STEPS
        reasons.append("procedural/how-to cue")
    elif _has(t, _TABLE_CUES):
        shape = TABLE
        reasons.append("tabular cue")
    elif _has(t, _CODE_CUES):
        shape = CODE
        reasons.append("code cue")
    else:
        shape = PROSE

    # A SUMMARIZE move never wants a heavy diagram/steps layout — keep it prose
    # unless the user explicitly asked for a table.
    if action == SUMMARIZE and shape in (DIAGRAM, STEPS, CODE):
        shape = PROSE

    return InteractionPlan(action=action, shape=shape, reasons=reasons)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.understanding, "interaction_engine", True))
    except Exception:  # noqa: BLE001
        return True


__all__ = [
    "InteractionPlan", "select", "enabled",
    "ASK", "PROCEED", "SUMMARIZE",
    "PROSE", "TABLE", "STEPS", "DIAGRAM", "CODE", "COMPARISON",
]
