"""
Advisory Outcome_Estimate (live-conversational-intelligence R44).

Aggregates per-session signals already captured (answered-question ratio,
average answer confidence, satisfaction signals, contradictions, health
warnings) into a coarse, ADVISORY outcome estimate. It is explicitly labeled
"not a hiring decision" and carries no authority — it's a self-coaching readout.
Deterministic + fail-open. No new schema; reads the per-session event log /
trackers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Coarse bands.
STRONG = "strong"
SOLID = "solid"
MIXED = "mixed"
NEEDS_WORK = "needs_work"
UNKNOWN = "unknown"

DISCLAIMER = "Advisory self-coaching estimate only — NOT a hiring decision."


@dataclass
class OutcomeEstimate:
    band: str = UNKNOWN
    score: float = 0.0          # [0,1]
    factors: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER
    advisory: bool = True

    def to_dict(self) -> dict:
        return {"band": self.band, "score": round(self.score, 3),
                "factors": self.factors, "disclaimer": self.disclaimer,
                "advisory": True}


def estimate(
    *,
    answered: int = 0,
    total: int = 0,
    avg_confidence: float | None = None,
    satisfaction: float | None = None,   # [0,1]
    contradictions: int = 0,
    health_warnings: int = 0,
) -> OutcomeEstimate:
    """Aggregate session signals into an advisory outcome. Never raises."""
    o = OutcomeEstimate()
    try:
        factors: list[str] = []
        comps: list[float] = []
        if total > 0:
            cov = answered / total
            comps.append(cov)
            factors.append(f"Answered {answered}/{total} questions.")
        if avg_confidence is not None:
            comps.append(max(0.0, min(1.0, avg_confidence)))
            factors.append(f"Average answer confidence {avg_confidence:.0%}.")
        if satisfaction is not None:
            comps.append(max(0.0, min(1.0, satisfaction)))
            factors.append(f"Interviewer satisfaction signal {satisfaction:.0%}.")
        if not comps:
            return o  # UNKNOWN — nothing to go on
        score = sum(comps) / len(comps)
        # Penalize contradictions / health warnings (advisory).
        score -= min(0.3, 0.05 * contradictions + 0.03 * health_warnings)
        score = max(0.0, min(1.0, score))
        if contradictions:
            factors.append(f"{contradictions} contradiction(s) detected.")
        if health_warnings:
            factors.append(f"{health_warnings} session-health warning(s).")
        o.score = score
        o.factors = factors
        if score >= 0.8:
            o.band = STRONG
        elif score >= 0.6:
            o.band = SOLID
        elif score >= 0.4:
            o.band = MIXED
        else:
            o.band = NEEDS_WORK
        return o
    except Exception:  # noqa: BLE001
        return o
