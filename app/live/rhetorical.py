"""
Rhetorical-question disambiguation (live-conversational-intelligence R52).

Not every '?' wants an answer: "Right?", "Make sense?", "You know what I mean?"
and lead-ins like "So what happens next? Well, ..." are rhetorical. This
suppresses an answer for clearly rhetorical questions, but on LOW confidence it
falls through and treats the utterance as a REAL question (never drop a genuine
question). Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core import lexicons

# High-confidence rhetorical tags (the whole utterance, roughly).
_RHETORICAL_TAGS = lexicons.LIVE_RHETORICAL_TAGS

# Self-answered lead-in: a question immediately followed by its own answer.
_SELF_ANSWER_CONNECTORS = lexicons.LIVE_RHETORICAL_SELF_ANSWER_CONNECTORS


@dataclass
class RhetoricalSignal:
    is_rhetorical: bool = False
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {"is_rhetorical": self.is_rhetorical,
                "confidence": round(self.confidence, 3), "reason": self.reason}


def classify(utterance: str) -> RhetoricalSignal:
    """Classify whether a '?'-bearing utterance is rhetorical. Never raises."""
    sig = RhetoricalSignal()
    try:
        t = (utterance or "").strip().lower()
        if not t or "?" not in t:
            return sig
        # Short trailing tag question → high-confidence rhetorical.
        for tag in _RHETORICAL_TAGS:
            if t.endswith(tag) or t == tag.strip():
                sig.is_rhetorical = True
                sig.confidence = 0.85
                sig.reason = "tag_question"
                return sig
        # Self-answered: "...? well, ...".
        qpos = t.find("?")
        tail = t[qpos + 1:]
        if tail.strip() and any(c in (" " + tail + " ") for c in _SELF_ANSWER_CONNECTORS):
            sig.is_rhetorical = True
            sig.confidence = 0.7
            sig.reason = "self_answered"
            return sig
        return sig
    except Exception:  # noqa: BLE001
        return sig


def should_answer(utterance: str, threshold: float = 0.6) -> bool:
    """Whether to ANSWER the utterance. A low-confidence rhetorical signal falls
    through to answering (never drop a genuine question)."""
    try:
        sig = classify(utterance)
        if sig.is_rhetorical and sig.confidence >= threshold:
            return False
        return True
    except Exception:  # noqa: BLE001
        return True
