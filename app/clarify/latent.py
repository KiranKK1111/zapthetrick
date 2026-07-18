"""Latent-requirement prediction (advanced-intent-reasoning R7).

Answer-first, suggest-later: after the system decides to answer directly, it MAY
attach a few non-blocking, intent-appropriate follow-ups the user is likely to
want next (tests, edge cases, complexity, …) — surfaced as suggestions, never as
required clarifications. Pure + deterministic.
"""
from __future__ import annotations

from .intent_pipeline import (
    INTENT_CODE_GEN,
    INTENT_COMPARISON,
    INTENT_DEBUGGING,
    INTENT_DESIGN,
    INTENT_KNOWLEDGE,
    INTENT_PROJECT_BUILD,
)

_MAX_SUGGESTIONS = 3

# Likely next steps per intent. Kept short, concrete, and genuinely useful so a
# suggestion never reads as filler.
_BY_INTENT: dict[str, list[str]] = {
    INTENT_CODE_GEN: [
        "Add unit tests for this",
        "Handle edge cases / invalid input",
        "Show the time & space complexity",
    ],
    INTENT_PROJECT_BUILD: [
        "Add authentication",
        "Set up persistence / a database",
        "Add a test suite",
    ],
    INTENT_DEBUGGING: [
        "Add a regression test for this bug",
        "Explain the root cause",
    ],
    INTENT_DESIGN: [
        "Sketch the data model",
        "List the main trade-offs",
        "Outline a phased rollout",
    ],
    INTENT_KNOWLEDGE: [
        "Show a concrete code example",
        "Compare it with the alternatives",
    ],
    INTENT_COMPARISON: [
        "Recommend one for a specific use case",
    ],
}


def suggest(intent: str, slots: dict | None = None) -> list[str]:
    """Up to [_MAX_SUGGESTIONS] non-blocking follow-ups for the intent (R7).
    Returns [] for intents with no useful proactive next step (chitchat,
    unknown). `slots` is accepted for future tailoring; unused today."""
    base = _BY_INTENT.get(intent or "", [])
    # Don't suggest "add tests" when the user already asked for tests, etc.
    return list(base[:_MAX_SUGGESTIONS])


__all__ = ["suggest"]
