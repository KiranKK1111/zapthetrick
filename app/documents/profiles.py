"""Export profiles + document persona engine — Phase 7 of the roadmap.

DocuementGeneration.md first-doc #6 (export profiles) + second-doc #7 (persona):
the SAME content should read differently for different audiences. This maps a
request to an audience, then to a persona that shapes tone + depth + emphasis via
a generation directive — so "write this for my manager" produces a concise,
outcome-focused document while "for the engineering team" stays deep + technical.

Deterministic audience detection (no LLM); the persona is a directive the answer
model consumes, mirroring the Phase-2 blueprint directive. Unknown audience →
GENERAL (no persona imposed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Audience(str, Enum):
    DEVELOPER = "developer"
    MANAGER = "manager"
    EXECUTIVE = "executive"
    CLIENT = "client"
    HR = "hr"
    STUDENT = "student"
    GENERAL = "general"


@dataclass
class Persona:
    audience: Audience
    tone: str
    detail: str          # deep | balanced | high-level
    emphasis: str

    def directive(self) -> str:
        if self.audience == Audience.GENERAL:
            return ""
        return (
            f"\nAUDIENCE ({self.audience.value}): write for this reader — "
            f"{self.tone} tone, {self.detail} detail, emphasize {self.emphasis}. "
            f"Match their vocabulary and only the depth they need."
        )

    def as_dict(self) -> dict:
        return {"audience": self.audience.value, "tone": self.tone,
                "detail": self.detail, "emphasis": self.emphasis}


_PERSONAS: dict[Audience, Persona] = {
    Audience.DEVELOPER: Persona(
        Audience.DEVELOPER, "precise + technical", "deep",
        "architecture, code, trade-offs, and edge cases"),
    Audience.MANAGER: Persona(
        Audience.MANAGER, "clear + pragmatic", "balanced",
        "status, risks, timeline, and what's needed to proceed"),
    Audience.EXECUTIVE: Persona(
        Audience.EXECUTIVE, "concise + outcome-focused", "high-level",
        "business impact, cost, and the decision to make (lead with a summary)"),
    Audience.CLIENT: Persona(
        Audience.CLIENT, "professional + reassuring", "balanced",
        "benefits, deliverables, and next steps (minimal internal jargon)"),
    Audience.HR: Persona(
        Audience.HR, "plain + people-focused", "balanced",
        "roles, process, and clear non-technical language"),
    Audience.STUDENT: Persona(
        Audience.STUDENT, "friendly + explanatory", "deep",
        "fundamentals, worked examples, and building intuition step by step"),
    Audience.GENERAL: Persona(
        Audience.GENERAL, "clear", "balanced", "the key points"),
}

# Ordered specific → general.
_AUDIENCE_PATTERNS: list[tuple[Audience, re.Pattern]] = [
    (Audience.EXECUTIVE, re.compile(
        r"\b(?:for|to) (?:the |my )?(?:cto|ceo|cfo|coo|vp|director|leadership|"
        r"board|exec(?:utive)?s?)\b|\bexecutive (?:summary|audience|report)\b|"
        r"\bc-?suite\b|\bstakeholders?\b", re.I)),
    (Audience.MANAGER, re.compile(
        r"\bfor (?:the |my )?(?:manager|boss|team lead|lead|supervisor)\b|"
        r"\bfor management\b", re.I)),
    (Audience.HR, re.compile(
        r"\bfor (?:the )?hr\b|\bhuman resources\b|\bfor recruiters?\b", re.I)),
    (Audience.CLIENT, re.compile(
        r"\bfor (?:the |our |a )?(?:client|customer|end[- ]?user)s?\b", re.I)),
    (Audience.STUDENT, re.compile(
        r"\bfor (?:students?|beginners?|learners?|a beginner)\b|"
        r"\bexplain like i'?m\b|\beli5\b|\bfor a novice\b", re.I)),
    (Audience.DEVELOPER, re.compile(
        r"\bfor (?:the )?(?:developers?|engineers?|engineering team|dev team|"
        r"technical (?:team|audience|reader))\b|\btechnical audience\b", re.I)),
]


# Semantic exemplars per audience — the AUTHORITY (generalizes to paraphrases).
# The regex patterns above remain only as the cold-start / embedder-down fallback.
_AUDIENCE_EXEMPLARS: dict[str, list[str]] = {
    Audience.EXECUTIVE.value: [
        "write this for the CTO", "an executive summary for leadership",
        "a brief for the board", "send this to the CEO",
        "something concise for senior management to decide on"],
    Audience.MANAGER.value: [
        "write this for my manager", "a status update for my team lead",
        "something my boss can read", "for my supervisor to review"],
    Audience.HR.value: [
        "write this for HR", "for the human resources team",
        "something the recruiters can use"],
    Audience.CLIENT.value: [
        "write this for the client", "something to send the customer",
        "a document for our end users"],
    Audience.STUDENT.value: [
        "explain this for beginners", "write it for students learning this",
        "explain like i'm five", "for someone new to the topic"],
    Audience.DEVELOPER.value: [
        "write this for the engineering team", "for developers",
        "a technical document for engineers", "for the dev team to implement"],
    Audience.GENERAL.value: [
        "just explain this", "give me a normal answer",
        "a general explanation", "tell me about this topic"],
}


def detect_audience(text: str) -> Audience:
    """Detect the target audience SEMANTICALLY (embedding nearest-class); the
    regex patterns are only the fallback when the embedder is unavailable."""
    try:
        from app.semantics.gates import classify
        cls = classify(text, _AUDIENCE_EXEMPLARS,
                       cache_key="doc_audience", threshold=0.45)
        if cls is not None:
            return Audience(cls)
    except Exception:  # noqa: BLE001
        pass
    for audience, pat in _AUDIENCE_PATTERNS:  # fallback: deterministic cues
        if pat.search(text or ""):
            return audience
    return Audience.GENERAL


def persona_for(audience: Audience) -> Persona:
    return _PERSONAS.get(audience, _PERSONAS[Audience.GENERAL])


def persona_directive(text: str) -> str:
    """The persona directive for a request's detected audience (empty for a
    general audience — no persona imposed)."""
    return persona_for(detect_audience(text)).directive()


__all__ = ["Audience", "Persona", "detect_audience", "persona_for",
           "persona_directive"]
