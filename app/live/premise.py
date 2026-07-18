"""
False-premise / adversarial-input detection (live-conversational-intelligence R32).

Flags a question that embeds a likely-false absolute assertion seeking
confirmation ("Kafka stores data only in memory, right?") so the answer corrects
the premise rather than affirming it. Composes with the knowledge-gap guard +
`quality.critic` (it adds a directive, never weakens them) and hedges when the
premise's truth is itself uncertain. Deterministic + fail-open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core import lexicons

# Absolute claims are over-strong and often the planted false premise.
_ABSOLUTES = lexicons.LIVE_PREMISE_ABSOLUTES
# Confirmation tags inviting a yes.
_CONFIRM_TAGS = lexicons.LIVE_PREMISE_CONFIRM_TAGS


@dataclass
class Premise:
    false_premise: bool
    confidence: float        # 0..1 that a (false) premise is being asserted
    note: str = ""


def check_premise(question: str, world_model=None) -> Premise:
    """Detect a confirmation-seeking absolute claim (likely false premise).
    Never raises."""
    try:
        t = (question or "").lower()
        if not t.strip():
            return Premise(False, 0.0)
        has_absolute = any(re.search(r"\b" + re.escape(a) + r"\b", t) for a in _ABSOLUTES)
        has_confirm = any(tag in t for tag in _CONFIRM_TAGS)
        if has_absolute and has_confirm:
            return Premise(True, 0.7,
                           "Verify the premise before agreeing; correct it if it's wrong.")
        if has_absolute:
            # An absolute statement without a tag — lower-confidence flag.
            return Premise(True, 0.4,
                           "Check the absolute claim; qualify it if it's not strictly true.")
        return Premise(False, 0.0)
    except Exception:  # noqa: BLE001
        return Premise(False, 0.0)


def directive(premise: Premise) -> str:
    """Prompt directive used when a false premise is suspected."""
    if not premise.false_premise:
        return ""
    base = ("The question may contain a false or over-stated premise. Do NOT simply "
            "agree — verify it, and if it is wrong or only partly true, correct it "
            "clearly before answering.")
    if premise.confidence < 0.5:
        base += " If you are unsure whether the premise holds, say so."
    return base
