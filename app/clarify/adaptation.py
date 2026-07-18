"""Clarification fatigue + trust adaptation (advanced-intent-reasoning R3/R4).

Pure functions that lower the bar for *answering* (i.e. make the system ask
fewer questions) as the user accumulates recent clarifications (fatigue) and as
they repeatedly skip/override them (eroded trust) — and let that bar recover as
they engage or go quiet. The clarifier feeds the resulting "answer band" into
the confidence-band clamp; safety cards bypass banding entirely, so they are
never suppressed (R3.4 / R4.4).

Counters come from [OutcomeStore]:
    recent   — clarifications the user has dealt with in the current window
               (decays on quiet turns)
    skips    — skipped + overridden clarifications
    answers  — answered clarifications
"""
from __future__ import annotations

# Each recent clarification lowers the answer band by this much (before trust).
_FATIGUE_STEP = 0.07
# Trust multiplier bounds.
_TRUST_LO, _TRUST_HI = 0.5, 1.5


def _band_floor() -> float:
    """Lowest the answer band may fall (keeps genuinely ambiguous turns asking).
    Central config `cfg.confidence.band_floor` (default 0.5)."""
    from app.core.config_loader import cfg
    return cfg.confidence.band_floor


def trust_factor(skips: int, answers: int) -> float:
    """0.5..1.5 multiplier on the fatigue adjustment (R4).

    More skips/overrides → toward 1.5 (trust eroded → suppress asking harder).
    More answers → toward 0.5 (the user engages → don't over-suppress). With no
    history → 1.0 (neutral). Always recoverable (bounded, answers pull it back).
    """
    s = max(0, int(skips))
    a = max(0, int(answers))
    total = s + a
    if total <= 0:
        return 1.0
    skip_rate = s / total
    return max(_TRUST_LO, min(_TRUST_HI, 0.6 + skip_rate))


def fatigue_threshold(base: float, recent: int, trust: float = 1.0) -> float:
    """Adjusted "answer band" (R3): the confidence at/above which the clarifier
    answers instead of asking. Lowers monotonically as `recent` rises (scaled by
    `trust`), floored at `_BAND_FLOOR`. `recent <= 0` → `base` unchanged, so a
    quiet conversation behaves exactly as the base band (recovery, R3.3).
    """
    try:
        b = float(base)
    except (TypeError, ValueError):
        return 0.9
    r = max(0, int(recent))
    if r <= 0:
        return b
    t = max(_TRUST_LO, min(_TRUST_HI, float(trust)))
    return max(_band_floor(), b - _FATIGUE_STEP * r * t)


def adapted_answer_band(base: float, recent: int, skips: int,
                        answers: int) -> float:
    """Convenience: combine fatigue + trust into the final answer band."""
    return fatigue_threshold(base, recent, trust_factor(skips, answers))


__all__ = ["trust_factor", "fatigue_threshold", "adapted_answer_band"]
