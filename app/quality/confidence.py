"""Aggregate confidence + gating (evaluation-and-reliability R3/R4).

Each decision subsystem (intent, reference resolution, retrieval relevance,
routing, answer trust) exposes a ``SubsystemConfidence`` — a thin
``{source, value, reasons}`` that reuses the ``app/chat/trust.py`` representation
(R3.3). ``aggregate`` blends the available signals into a single
``trust.ConfidenceResult`` (band high|medium|low); a missing signal is simply
absent → treated as neutral, never a failure (R3.2/R4.1, Property 4).

``gate`` maps the band to the existing control flow: high → proceed, low → defer
to the EXISTING clarifier (no new ask path / no second LLM call), middle → the
existing judgment path (R4.2–R4.4, Property 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.chat.trust import ConfidenceResult

# Band thresholds — identical to `trust.confidence_band` so the representation
# is shared, not parallel (R3.3).
_HIGH = 0.75
_MEDIUM = 0.45
_NEUTRAL = 0.70     # used when no signals are available

# Relative weights per source (answer-trust + intent matter most).
_WEIGHTS = {
    "trust": 1.5,
    "intent": 1.2,
    "reference": 1.0,
    "routing": 0.8,
    "retrieval": 0.8,
}


@dataclass
class SubsystemConfidence:
    """A confidence signal emitted by one decision subsystem."""
    source: str                 # "intent" | "reference" | "retrieval" | "routing" | "trust"
    value: float                # 0..1
    reasons: list[str] = field(default_factory=list)

    def clamped(self) -> float:
        return max(0.0, min(1.0, float(self.value)))


def _band(score: float) -> str:
    return "high" if score >= _HIGH else "medium" if score >= _MEDIUM else "low"


def aggregate(signals: list[SubsystemConfidence | None]) -> ConfidenceResult:
    """Blend available subsystem signals into one ConfidenceResult. Missing /
    None signals are skipped (neutral). No signals → neutral medium band."""
    try:
        present = [s for s in (signals or []) if s is not None]
        if not present:
            return ConfidenceResult(band=_band(_NEUTRAL), score=_NEUTRAL,
                                    reasons=["no confidence signals — neutral"])
        num = 0.0
        den = 0.0
        reasons: list[str] = []
        for s in present:
            w = _WEIGHTS.get(s.source, 1.0)
            num += w * s.clamped()
            den += w
            for r in s.reasons[:2]:
                reasons.append(f"{s.source}: {r}")
        score = round(num / den, 3) if den else _NEUTRAL
        if not reasons:
            reasons.append("aggregated subsystem confidence")
        return ConfidenceResult(band=_band(score), score=score, reasons=reasons)
    except Exception:  # noqa: BLE001 — never break a turn (Property 9)
        return ConfidenceResult(band=_band(_NEUTRAL), score=_NEUTRAL,
                                reasons=["confidence aggregation error — neutral"])


def gate(result: ConfidenceResult) -> str:
    """Map a band to the existing control flow (no new ask path):
    high → 'proceed', low → 'clarify' (defer to the existing clarifier),
    middle → 'judgment' (existing assumption/answer path)."""
    if result.band == "high":
        return "proceed"
    if result.band == "low":
        return "clarify"
    return "judgment"


@dataclass
class Presentation:
    """UX hints derived from the aggregate confidence (R4 — presentation, not
    control flow). Additive: a client that ignores it renders exactly as today.

    * ``hedge``           — prepend an epistemic qualifier ("I'm fairly sure…").
    * ``show_confidence`` — surface the band/score as a badge.
    * ``offer_alternatives`` — invite the user to see other interpretations.
    * ``offer_regenerate``   — invite a "try again / think harder" affordance.
    * ``verbosity``       — "brief" (high conf) | "normal" | "detailed" (low conf,
      show reasoning so the user can sanity-check).
    """
    band: str
    score: float
    hedge: str
    show_confidence: bool
    offer_alternatives: bool
    offer_regenerate: bool
    verbosity: str
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "band": self.band, "score": self.score, "hedge": self.hedge,
            "show_confidence": self.show_confidence,
            "offer_alternatives": self.offer_alternatives,
            "offer_regenerate": self.offer_regenerate,
            "verbosity": self.verbosity, "note": self.note,
        }


_HEDGE = {
    "high": "",
    "medium": "Here's my best read — worth a quick sanity check: ",
    "low": "I'm not fully certain here, so treat this as a starting point: ",
}


def presentation(result: ConfidenceResult) -> Presentation:
    """Map a ConfidenceResult to richer presentation variation (P5 #20).

    High confidence answers stay clean and brief (no badge, no hedging). As
    confidence drops the UX gets progressively more transparent: a hedge, a
    visible confidence badge, an offer to show alternative interpretations, and
    at the low band an explicit regenerate affordance + more detail so the user
    can verify the reasoning. Fail-open to the neutral/high presentation."""
    try:
        band = result.band
        score = float(getattr(result, "score", _NEUTRAL) or _NEUTRAL)
    except Exception:  # noqa: BLE001
        band, score = "high", _NEUTRAL
    if band == "high":
        return Presentation(band, score, "", False, False, False, "brief",
                            "confident — clean presentation")
    if band == "low":
        return Presentation(band, score, _HEDGE["low"], True, True, True,
                            "detailed", "low confidence — hedge + show reasoning "
                            "+ offer regenerate/alternatives")
    return Presentation(band, score, _HEDGE["medium"], True, True, False,
                        "normal", "medium confidence — hedge + badge + alternatives")


# ── per-subsystem adapters (missing data → None = neutral, R3.2) ─────────────
def from_assessment(assessment) -> SubsystemConfidence | None:
    """Intent pre-gate confidence (`intent_pipeline.assess(...)`)."""
    try:
        v = float(getattr(assessment, "confidence", None))
    except (TypeError, ValueError):
        return None
    reasons = list(getattr(assessment, "reasons", []) or [])[:2]
    return SubsystemConfidence("intent", v, reasons or ["intent pre-gate"])


def from_resolution(resolution) -> SubsystemConfidence | None:
    """Reference-resolution confidence (`followup.reference.resolve(...)`)."""
    if resolution is None:
        return None
    refs = getattr(resolution, "refs", None)
    if not refs:
        return None                      # no references → no signal (neutral)
    v = float(getattr(resolution, "confidence", 0.0) or 0.0)
    note = "needs clarification" if getattr(resolution, "needs_clarification",
                                            False) else "resolved"
    return SubsystemConfidence("reference", v, [f"reference {note}"])


def from_trust(result) -> SubsystemConfidence | None:
    """Answer-trust confidence (`trust.confidence_band(...)` result)."""
    if result is None:
        return None
    v = float(getattr(result, "score", 0.0) or 0.0)
    return SubsystemConfidence("trust", v, list(getattr(result, "reasons", []) or [])[:2])


def from_routing(difficulty: str | None) -> SubsystemConfidence | None:
    """Routing confidence proxy from the difficulty label (deterministic).
    Clear/easy levels → high confidence; expert/unknown → lower."""
    if not difficulty:
        return None
    table = {"trivial": 0.9, "standard": 0.85, "hard": 0.7, "expert": 0.6}
    v = table.get(str(difficulty).lower())
    if v is None:
        return None
    return SubsystemConfidence("routing", v, [f"difficulty={difficulty}"])


def from_retrieval(relevance: float | None) -> SubsystemConfidence | None:
    """Retrieval-relevance confidence from a 0..1 top-result relevance."""
    if relevance is None:
        return None
    return SubsystemConfidence("retrieval", float(relevance),
                               [f"top relevance={round(float(relevance), 2)}"])


__all__ = [
    "SubsystemConfidence", "aggregate", "gate", "presentation", "Presentation",
    "from_assessment", "from_resolution", "from_trust", "from_routing",
    "from_retrieval",
]
