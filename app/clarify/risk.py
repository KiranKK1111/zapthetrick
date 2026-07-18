"""Numeric risk assessment for the clarifier gate (SeveralFeatures.md Step 8).

The design doc's risk ladder: read-only/explanatory work is LOW risk (proceed
freely), document/report generation is MEDIUM (sometimes clarify), and
project/architecture/infrastructure/production work is HIGH (clarify when in
doubt) — with destructive/irreversible operations always confirmation-worthy.

Previously risk existed only as the binary destructive-op regex guard in the
clarifier agent. This module scores it 0..1 with attributed reasons and maps
the score to an ANSWER-BAND DELTA: high risk raises the bar to answer without
asking (ask sooner), low risk lowers it (interrupt less). The delta is small
and centrally capped — risk nudges the existing confidence-band machinery, it
never overrides it.

Fail-open: `assess_risk` never raises; on any error it returns the neutral
LOW assessment (delta 0.0 → behavior identical to before this module existed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

LOW = "low"
MEDIUM = "medium"
HIGH = "high"

# ---- intent base risk (doc Step 8's three tiers) ---------------------------
# Read-only / explanatory / conversational → LOW. Deliverable generation →
# MEDIUM. Project/architecture scale → HIGH-leaning.
_INTENT_BASE = {
    "chitchat": 0.05,
    "knowledge": 0.10,
    "comparison": 0.10,
    "debugging": 0.15,       # analysis of existing code — read-mostly
    "test_gen": 0.20,
    "docs": 0.35,            # generated document — medium tier
    "archive": 0.30,
    "code_gen": 0.35,
    "design": 0.45,
    "project_build": 0.55,   # high tier: wrong assumptions are expensive
}
_DEFAULT_BASE = 0.25         # unknown intent → mild

# ---- additive cues ----------------------------------------------------------
# Destructive / irreversible operations (always the strongest signal).
_DESTRUCTIVE_RE = re.compile(
    r"\b(delete|drop|truncate|wipe|erase|destroy|remove\s+all|rm\s+-rf|"
    r"format\s+(?:the\s+)?(?:disk|drive)|overwrite|force[- ]push|"
    r"reset\s+--hard|purge)\b", re.IGNORECASE)
# Production / live-environment blast radius.
_PRODUCTION_RE = re.compile(
    r"\b(prod(?:uction)?|live\s+(?:site|system|server|database|db)|"
    r"customer[- ]facing|in\s+production)\b", re.IGNORECASE)
# Deployment / infrastructure work (doc: High Risk tier).
_INFRA_RE = re.compile(
    r"\b(deploy(?:ment)?|infrastructure|terraform|kubernetes|k8s|helm|"
    r"provision|cloudformation|ci/?cd\s+pipeline)\b", re.IGNORECASE)
# Schema / data-model changes (migrations ripple).
_SCHEMA_RE = re.compile(
    r"\b(schema|migration|alter\s+table|database\s+design|data\s+model)\b",
    re.IGNORECASE)
# Security-sensitive surface.
_SECURITY_RE = re.compile(
    r"\b(auth(?:entication|orization)?|credential|secret|api[- ]key|"
    r"password|encryption|payment|billing)\b", re.IGNORECASE)

_CUES: tuple[tuple[re.Pattern, float, str], ...] = (
    (_DESTRUCTIVE_RE, 0.40, "destructive_operation"),
    (_PRODUCTION_RE, 0.25, "production_environment"),
    (_INFRA_RE, 0.20, "deployment_or_infrastructure"),
    (_SCHEMA_RE, 0.15, "schema_or_data_model"),
    (_SECURITY_RE, 0.10, "security_sensitive"),
)


@dataclass
class RiskAssessment:
    """0..1 risk with attribution and the answer-band nudge it implies."""
    score: float = 0.0
    level: str = LOW                    # low | medium | high
    reasons: list[str] = field(default_factory=list)
    band_delta: float = 0.0             # added to the answer band downstream

    def as_dict(self) -> dict:
        return {"score": round(self.score, 3), "level": self.level,
                "reasons": list(self.reasons),
                "band_delta": round(self.band_delta, 3)}


def _level(score: float) -> str:
    if score >= 0.60:
        return HIGH
    if score >= 0.30:
        return MEDIUM
    return LOW


def assess_risk(text: str, intent: str, slots: dict | None = None,
                *, weight: float | None = None) -> RiskAssessment:
    """Score the risk of proceeding on assumptions for this request.

    band_delta mapping (weight w = cfg.decision_core.risk_band_weight):
        HIGH   → +w      (raise the answer band: clarify sooner)
        MEDIUM → 0.0     (today's behavior — the bands already handle it)
        LOW    → -w/2    (lower the bar: don't interrupt cheap read-only asks)
    """
    try:
        t = (text or "").strip()
        reasons: list[str] = []
        score = _INTENT_BASE.get((intent or "").lower(), _DEFAULT_BASE)
        reasons.append(f"intent_base:{intent or 'unknown'}")
        for pattern, add, why in _CUES:
            if pattern.search(t):
                score += add
                reasons.append(why)
        # A named performance/complexity constraint means correctness of the
        # approach matters more than usual — small bump.
        if (slots or {}).get("constraints"):
            score += 0.05
            reasons.append("explicit_constraints")
        score = max(0.0, min(1.0, score))
        level = _level(score)
        if weight is None:
            try:
                from app.core.config_loader import cfg
                weight = float(cfg.decision_core.risk_band_weight)
            except Exception:  # noqa: BLE001
                weight = 0.05
        delta = weight if level == HIGH else (-weight / 2 if level == LOW else 0.0)
        return RiskAssessment(score=score, level=level, reasons=reasons,
                              band_delta=delta)
    except Exception:  # noqa: BLE001 — neutral fallback keeps behavior unchanged
        return RiskAssessment()


__all__ = ["RiskAssessment", "assess_risk", "LOW", "MEDIUM", "HIGH"]
