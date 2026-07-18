"""Answer Readiness Score (roadmap Phase 2 #13).

A deterministic 0..1 estimate of how *ready* a forming answer is to be shown —
combining answer confidence, whether it's grounded in evidence, length adequacy,
and verification. The live pipeline can stream as soon as readiness crosses a
threshold instead of waiting for the whole answer. Pure + fail-open.
"""
from __future__ import annotations


def readiness_score(
    *,
    confidence: float | None = None,
    has_evidence: bool = False,
    answer_chars: int = 0,
    verified: bool | None = None,
    min_chars: int = 40,
) -> float:
    """Weighted, deterministic readiness in [0, 1].

    Weights: confidence 0.5 · evidence 0.2 · length-adequacy 0.2 · verification
    ±0.1. Unknown confidence is treated as neutral (0.25 of the 0.5 band)."""
    try:
        score = 0.0
        if confidence is not None:
            score += 0.5 * _clamp01(float(confidence))
        else:
            score += 0.25
        if has_evidence:
            score += 0.2
        denom = max(1, int(min_chars))
        score += 0.2 * min(1.0, max(0, int(answer_chars)) / denom)
        if verified is True:
            score += 0.1
        elif verified is False:
            score -= 0.1
        return round(_clamp01(score), 3)
    except Exception:  # noqa: BLE001 — a scoring hiccup must not break a turn
        return 0.0


def ready_to_stream(score: float, threshold: float = 0.5) -> bool:
    return score >= threshold


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


__all__ = ["readiness_score", "ready_to_stream"]
