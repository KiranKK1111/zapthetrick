"""Adaptive reasoning effort dial (vNext §8.2, Stage 7 Component E).

"How hard should I think about this?" is normally answered piecemeal — a tier
here, a thinking budget there, best-of-N somewhere else. §8.2 unifies them into
ONE coherent effort profile the whole turn reads, chosen from the turn's
difficulty and shaded by the user's reasoning mode:

    EffortProfile = { tier, thinking_budget (tokens), best_of_n, use_judge,
                      reasoning (route to a reasoning model), escalate }

  * a **trivial** turn spends nothing (fast tier, no thinking budget, N=1);
  * a **hard** turn thinks (bigger budget, best-of-N=2 + a judge);
  * an **expert** / DSA-optimal turn routes to a reasoning model with a large
    budget, N=3 + judge;
  * a **repair loop** (`repair_stage > 0`) escalates to the reasoning tier with
    the error + counterexample, regardless of the base difficulty;
  * the Fast / Thorough **reasoning mode** shifts the band down / up.

Pure mapping over data (reuses `reasoning_mode`'s ladder + mode signal); the
quality machinery (best-of-N + judge in `app/quality/*`) and the router (tier)
CONSUME the profile. Fail-open — any error returns the base profile for the
difficulty. Flag-gated (`llm.effort_dial`, default OFF → today's per-knob path).
"""
from __future__ import annotations

from dataclasses import dataclass

_LADDER = ["trivial", "standard", "hard", "expert"]


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.llm, "effort_dial", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class EffortProfile:
    difficulty: str
    tier: str                # fast | standard | hard | reasoning
    thinking_budget: int     # thinking/scratchpad token budget (0 = none)
    best_of_n: int           # candidates to sample before judging
    use_judge: bool          # run the §quality judge to pick the best
    reasoning: bool          # route to a reasoning-capable model
    escalate: bool = False   # this profile is a repair/escalation bump

    @property
    def escalated(self) -> bool:
        """True when the turn thinks *harder than a normal turn* — it samples
        best-of-N, routes to a reasoning model, or is a repair bump. Drives the
        user-visible "thinking carefully" chip + the persona thinking-summary
        directive; a plain standard turn (N=1, small budget) is NOT escalated,
        so it stays quiet."""
        return self.best_of_n > 1 or self.reasoning or self.escalate

    def as_dict(self) -> dict:
        return {"difficulty": self.difficulty, "tier": self.tier,
                "thinking_budget": self.thinking_budget,
                "best_of_n": self.best_of_n, "use_judge": self.use_judge,
                "reasoning": self.reasoning, "escalate": self.escalate,
                "escalated": self.escalated}


# Base profiles per difficulty (the dial's data table).
_BASE: dict[str, EffortProfile] = {
    "trivial": EffortProfile("trivial", "fast", 0, 1, False, False),
    "standard": EffortProfile("standard", "standard", 512, 1, False, False),
    "hard": EffortProfile("hard", "hard", 2048, 2, True, False),
    "expert": EffortProfile("expert", "reasoning", 8192, 3, True, True),
}
_DEFAULT = _BASE["standard"]

# The reasoning tier a repair/DSA escalation lands on.
_ESCALATED = EffortProfile("expert", "reasoning", 8192, 2, True, True,
                           escalate=True)


def _shift(difficulty: str, steps: int) -> str:
    try:
        i = _LADDER.index((difficulty or "standard").strip().lower())
    except ValueError:
        i = 1
    return _LADDER[max(0, min(len(_LADDER) - 1, i + steps))]


def _mode_steps(mode: str | None) -> int:
    m = (mode or "").strip().lower()
    if m in ("thorough", "deep", "deeper", "exhaustive"):
        return 1
    if m in ("fast", "tldr", "quick"):
        return -1
    return 0


def effort_for(difficulty: str, *, mode: str | None = None,
               repair_stage: int = 0, dsa_optimal: bool = False) -> EffortProfile:
    """The effort profile for a turn. `mode` = the Fast/Balanced/Thorough
    reasoning mode; `repair_stage > 0` → escalate to the reasoning tier;
    `dsa_optimal` → route a DSA-optimal turn to reasoning. Disabled → the base
    profile (no dial). Never raises."""
    try:
        base = _BASE.get((difficulty or "standard").strip().lower(), _DEFAULT)
        if not enabled():
            return base                    # dial off → base profile, no shading
        if repair_stage > 0 or dsa_optimal:
            # A failed attempt (or an optimize-this-algorithm ask) always thinks.
            return _ESCALATED
        # Mode shades the band up (Thorough) or down (Fast).
        return _BASE.get(_shift(difficulty, _mode_steps(mode)), _DEFAULT)
    except Exception:  # noqa: BLE001
        return _DEFAULT


def thinking_summary_directive() -> str:
    """§8.2 — the thinking stream is SUMMARIZED into the stage protocol, never
    shown raw. This directive tells the model to surface only short 'considering
    X vs Y' step summaries."""
    return ("When you reason, surface only brief 'considering X vs Y…' progress "
            "notes as steps — never stream your raw chain-of-thought.")


__all__ = ["enabled", "EffortProfile", "effort_for", "thinking_summary_directive"]
