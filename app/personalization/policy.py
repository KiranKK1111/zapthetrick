"""Topic-risk policy gate (personalization-and-governance R4).

`classify(text) -> TopicRisk` (general | medical | legal | financial | other)
deterministically (lexicons/cues; no second blocking LLM call — R4.5), and
`strategy_for(risk) -> PolicyStrategy` that ADDS appropriate caveats + a
"consult a professional" recommendation and forbids prohibited professional
advice while still offering general/educational help (R4.1/R4.2).

This gate COMPOSES WITH — never weakens — the existing content-safety and
destructive-action guards, which run first and take precedence (R4.3, Property 4).
General topics behave exactly as today (R4.4). Pure; fail-open to "general".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GENERAL = "general"
MEDICAL = "medical"
LEGAL = "legal"
FINANCIAL = "financial"
OTHER = "other"

# Deterministic cues. Tuned to fire on personal-advice phrasing, not generic
# mentions (e.g. "build a medical app" is general engineering, not medical advice).
_MEDICAL = re.compile(
    r"\b(should i take|is it safe to take|diagnos|my symptoms|i have a (?:rash|"
    r"fever|pain)|prescrib|dosage|medication for|treat my|is this cancer|"
    r"mg of|side effects? of (?:my|the))\b", re.I)
_LEGAL = re.compile(
    r"\b(can i sue|is it legal for me|am i liable|my lawsuit|my contract says|"
    r"will i be charged|my landlord|custody|legal advice|should i plead|"
    r"my rights in court)\b", re.I)
_FINANCIAL = re.compile(
    r"\b(should i invest|should i buy (?:stock|shares|crypto)|my portfolio|"
    r"is .+ a good investment|financial advice|should i sell my|retirement "
    r"savings|which stock should|put my money)\b", re.I)


@dataclass
class PolicyStrategy:
    risk: str
    add_caveat: bool = False
    recommend_professional: str = ""     # "" when general
    prohibited: list[str] = field(default_factory=list)
    directive: str = ""                  # appended to the answer prompt (additive)


def classify(text: str) -> str:
    """Deterministic topic-risk classification. Fail-open to general."""
    try:
        t = text or ""
        if _MEDICAL.search(t):
            return MEDICAL
        if _LEGAL.search(t):
            return LEGAL
        if _FINANCIAL.search(t):
            return FINANCIAL
        return GENERAL
    except Exception:  # noqa: BLE001
        return GENERAL


_PROF = {
    MEDICAL: ("a licensed medical professional",
              ["a diagnosis", "a specific treatment/dosage recommendation"]),
    LEGAL: ("a qualified lawyer",
            ["specific legal counsel", "a definitive legal determination"]),
    FINANCIAL: ("a licensed financial advisor",
                ["specific buy/sell recommendations", "personalized financial advice"]),
}


def strategy_for(risk: str) -> PolicyStrategy:
    """The additive response strategy for a topic risk. General → no-op."""
    try:
        if risk in (GENERAL, "", None):
            return PolicyStrategy(risk=GENERAL)
        prof, prohibited = _PROF.get(risk, ("a qualified professional", []))
        directive = (
            f"SENSITIVE TOPIC ({risk}): provide general, educational information "
            f"only. Do NOT give {', '.join(prohibited) if prohibited else 'professional advice'}. "
            f"Add a brief caveat and recommend consulting {prof}. Be helpful and "
            f"clear within those bounds; do not refuse a legitimate general question."
        )
        return PolicyStrategy(risk=risk, add_caveat=True,
                              recommend_professional=prof,
                              prohibited=list(prohibited), directive=directive)
    except Exception:  # noqa: BLE001
        return PolicyStrategy(risk=GENERAL)


TopicRisk = str

__all__ = [
    "classify", "strategy_for", "PolicyStrategy", "TopicRisk",
    "GENERAL", "MEDICAL", "LEGAL", "FINANCIAL", "OTHER",
]
