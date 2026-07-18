"""Multi-modal fusion — Architecture.md §"Multi-modal question detection".

Combines:
  - text-based classifier        weight 0.55
  - prosody/acoustic features    weight 0.30
  - context (recent Q&A flow)    weight 0.15

The exact weights come from the doc's "0.55 text + 0.30 prosody +
0.15 context" recipe. They're tunable via cfg.

Public surface:
    fuse(text_score, prosody_score, context_score) -> FusedDecision
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedDecision:
    is_question: bool
    score: float
    components: dict
    rationale: str = ""


def fuse(
    text_score: float,
    prosody_score: float,
    context_score: float,
    *,
    threshold: float = 0.55,
    weights: tuple[float, float, float] = (0.55, 0.30, 0.15),
) -> FusedDecision:
    """Linear blend of the three sub-scores. Decision threshold at
    0.55 by default — the doc notes this gives the best precision/
    recall on the eval set."""
    wt, wp, wc = weights
    total = wt + wp + wc or 1.0
    score = (wt * text_score + wp * prosody_score + wc * context_score) / total
    score = max(0.0, min(1.0, score))
    is_q = score >= threshold

    bits = []
    if text_score >= 0.7:
        bits.append("text strong")
    elif text_score >= 0.4:
        bits.append("text moderate")
    if prosody_score >= 0.5:
        bits.append("rising pitch")
    if context_score >= 0.5:
        bits.append("interview Q context")
    rationale = ", ".join(bits) or "low confidence"

    return FusedDecision(
        is_question=is_q,
        score=score,
        components={
            "text": text_score,
            "prosody": prosody_score,
            "context": context_score,
        },
        rationale=rationale,
    )


__all__ = ["fuse", "FusedDecision"]
