"""Real-Time (mid-stream) Quality Controller (roadmap Phase 5 #13).

Monitors an answer *as it streams* from cheap deterministic signals — refusal
leakage on a legit turn, degenerate repetition, error/apology spikes, emptiness —
and recommends an action (continue / flag / regenerate) before the whole answer
lands. Complements the post-`done` verifier (which runs after) and the
`quality/governor`. Deterministic + fail-open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CONTINUE = "continue"
FLAG = "flag"
REGENERATE = "regenerate"

_REFUSAL = re.compile(
    r"\b(i can'?t help with that|i cannot help with that|i'?m not able to help|"
    r"as an ai(?: language model)?|i'?m unable to assist)\b", re.IGNORECASE)
_ERROR_SPIKE = re.compile(r"\b(error|failed|exception|undefined|null null|traceback)\b",
                          re.IGNORECASE)


@dataclass
class QualityVerdict:
    action: str            # continue | flag | regenerate
    score: float           # 0..1 running quality estimate
    reasons: list[str]


def _repetition_ratio(text: str) -> float:
    """Fraction of repeated tokens — high = degenerate loop."""
    toks = (text or "").split()
    if len(toks) < 8:
        return 0.0
    uniq = len(set(toks))
    return 1.0 - (uniq / len(toks))


def assess_partial(text: str, *, expect_refusal_ok: bool = False) -> QualityVerdict:
    """Assess an in-progress answer. `expect_refusal_ok=True` for turns where a
    refusal is legitimate (so we don't flag it)."""
    reasons: list[str] = []
    score = 1.0
    try:
        t = text or ""
        if not expect_refusal_ok and _REFUSAL.search(t):
            reasons.append("refusal_leak")
            score -= 0.6
        rep = _repetition_ratio(t)
        if rep > 0.8:              # severe loop → strong signal to regenerate
            reasons.append(f"degenerate_repetition({rep:.2f})")
            score -= 0.7
        elif rep > 0.6:
            reasons.append(f"degenerate_repetition({rep:.2f})")
            score -= 0.4
        errs = len(_ERROR_SPIKE.findall(t))
        if errs >= 4:
            reasons.append(f"error_spike({errs})")
            score -= 0.3
        if len(t.strip()) == 0:
            reasons.append("empty")
            score -= 0.2
    except Exception:  # noqa: BLE001
        return QualityVerdict(CONTINUE, 1.0, [])
    score = max(0.0, min(1.0, score))
    if score < 0.4:
        action = REGENERATE
    elif score < 0.75:
        action = FLAG
    else:
        action = CONTINUE
    return QualityVerdict(action, round(score, 3), reasons)


__all__ = ["CONTINUE", "FLAG", "REGENERATE", "QualityVerdict", "assess_partial"]
