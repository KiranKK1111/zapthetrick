"""Clarification engine vNext — the one decision policy (vNext §3.7, Stage 7 B).

The old gate asked whenever confidence was low; that produces a stream of
questions. §3.7 replaces it with ONE decision policy over the §3.13 interpretation
brief: a missing slot is only turned into a QUESTION when it is **material**
(changing it changes the answer) AND guessing wrong is **costly**. Otherwise the
engine ASSUMES a sensible default, LABELS it in an assumption ledger (the FE shows
"Assumed Python · tap to change"), and proceeds — so a vague prompt still gets a
useful answer immediately instead of a clarifying interrogation.

Rules (§3.7.1–.6):
  * a **contradiction** always asks (you cannot assume through "short but
    exhaustive") — highest priority;
  * **one question per turn**, at most `clarify_budget` (default 2) per task;
  * a **sticky** slot already resolved in the goal ledger is never re-asked;
  * everything non-material, or material-but-low-cost, is **assume-and-label**.

Reads the brief duck-typed (`.missing_slots` / `.contradictions`) so `clarify`
takes no new edge. Pure + fail-open — any error → PROCEED (never block a turn on
the policy). Flag-gated (`decision_core.clarify_v2`, default OFF).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Slot keywords whose value MATERIALLY changes the answer (worth a question when
# also costly). Everything else is assume-and-label — tone, length, formatting.
_MATERIAL = (
    "language", "framework", "format", "platform", "database", "scope",
    "target", "version", "audience", "deliverable", "runtime", "cloud",
)
# Assume-and-label defaults: slot keyword → (assumed value, why). Conservative,
# reversible (the FE chip lets the user change it), never a silent commitment.
_DEFAULTS: dict[str, tuple[str, str]] = {
    "language": ("Python", "most common for this kind of task"),
    "format": ("Markdown", "readable inline; ask if you need a file"),
    "framework": ("the stack in your prompt/history", "inferred from context"),
    "audience": ("a technical reader", "matches the question's register"),
    "scope": ("a focused, self-contained answer", "safest default"),
    "database": ("PostgreSQL", "the common default"),
    "platform": ("Linux", "the common default"),
}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.decision_core, "clarify_v2", False))
    except Exception:  # noqa: BLE001
        return False


def _budget() -> int:
    try:
        from app.core.config_loader import cfg
        return int(getattr(cfg.decision_core, "clarify_budget", 2))
    except Exception:  # noqa: BLE001
        return 2


def is_material(slot: str) -> bool:
    s = (slot or "").strip().lower()
    return any(k in s for k in _MATERIAL)


def _default_for(slot: str) -> tuple[str, str]:
    s = (slot or "").strip().lower()
    for key, (val, why) in _DEFAULTS.items():
        if key in s:
            return val, why
    return ("a sensible default", "not specified; proceeding conservatively")


@dataclass
class Assumption:
    slot: str
    value: str
    why: str

    def as_dict(self) -> dict:
        return {"slot": self.slot, "value": self.value, "why": self.why}


@dataclass
class ClarifyDecision:
    action: str                       # "ask" | "assume" | "proceed"
    question: str = ""                # the ONE question, when action == "ask"
    slot: str = ""                    # the slot being asked about
    assumptions: list[Assumption] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"action": self.action, "question": self.question,
                "slot": self.slot,
                "assumptions": [a.as_dict() for a in self.assumptions]}


def _question_for(slot: str) -> str:
    s = (slot or "the missing detail").strip()
    return f"Quick check — which {s}?"


def decide(brief, *, resolved: "set[str] | None" = None,
           asked: int = 0) -> ClarifyDecision:
    """The one clarification decision for this turn, from the interpretation
    `brief`. Returns ASK (one question) / ASSUME (labeled defaults) / PROCEED.
    `resolved` = sticky slots already settled (never re-asked); `asked` = how many
    questions this task has already spent (budget cap). Never raises → PROCEED."""
    try:
        if not enabled():
            return ClarifyDecision(action="proceed")
        resolved = {r.strip().lower() for r in (resolved or set())}
        contradictions = [str(c) for c in
                          (getattr(brief, "contradictions", None) or [])]
        missing = [str(m) for m in (getattr(brief, "missing_slots", None) or [])
                   if m and m.strip().lower() not in resolved]

        within_budget = asked < _budget()

        # A contradiction can't be assumed through → ask (one per turn, budget).
        if contradictions and within_budget:
            c = contradictions[0]
            return ClarifyDecision(
                action="ask", slot="contradiction",
                question=f"Those two asks conflict — {c}. Which do you want?")

        if not missing:
            return ClarifyDecision(action="proceed")

        # Split unresolved slots into "worth a question" (material) vs assume.
        material = [m for m in missing if is_material(m)]
        assume = [m for m in missing if not is_material(m)]

        # Ask ONE material slot if we still have budget; assume the rest.
        if material and within_budget:
            ask_slot = material[0]
            extra_assumptions = [
                Assumption(m, *_default_for(m)) for m in (material[1:] + assume)]
            return ClarifyDecision(
                action="ask", slot=ask_slot,
                question=_question_for(ask_slot), assumptions=extra_assumptions)

        # No budget left / nothing material → assume-and-label everything.
        assumptions = [Assumption(m, *_default_for(m)) for m in missing]
        return ClarifyDecision(action="assume", assumptions=assumptions)
    except Exception as exc:  # noqa: BLE001
        log.info("clarify.policy.decide failed: %s", exc)
        return ClarifyDecision(action="proceed")


# --------------------------------------------------------------------------- #
# Assumption ledger (per-conversation) — for the envelope + sticky slots
# --------------------------------------------------------------------------- #
@dataclass
class AssumptionLedger:
    """Records the assume-and-label decisions of a conversation so they ride the
    envelope (→ FE 'Assumed X · tap to change' chips) and a slot assumed once is
    sticky (not re-assumed / re-asked)."""
    entries: dict = field(default_factory=dict)   # slot(lower) -> Assumption

    def record(self, assumptions: "list[Assumption]") -> None:
        for a in assumptions or []:
            key = (a.slot or "").strip().lower()
            if key:
                self.entries.setdefault(key, a)

    def slots(self) -> set:
        return set(self.entries.keys())

    def as_list(self) -> list:
        return [a.as_dict() for a in self.entries.values()]


__all__ = ["enabled", "is_material", "Assumption", "ClarifyDecision", "decide",
           "AssumptionLedger"]
