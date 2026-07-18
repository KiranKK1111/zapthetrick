"""Conversational-act classifier (followup-context-engine R2).

`classify(turn, state, model_act=None) -> (act, followup_confidence)` over the
closed act set. Deterministic cues first (reusing small lexicons aligned with the
`goal_ledger` cue lists); an optional `model_act` label parsed from the existing
gate call may break ties — never a second LLM call (Property 11).

Gating (R2.3): when a turn would be a *continuity* act (follow_up / continuation
/ comparison / expansion) but the follow-up confidence is below the configured
threshold, it is treated as a `new_topic`. Explicit acts (approval / rejection /
correction / clarification_answer) are not downgraded. Any error → `new_topic`
with neutral confidence so the route uses today's continuity prompt (R2.4).
"""
from __future__ import annotations

import re

from app.core import lexicons

# Closed act set.
NEW_TOPIC = "new_topic"
FOLLOW_UP = "follow_up"
CORRECTION = "correction"
CONTINUATION = "continuation"
COMPARISON = "comparison"
EXPANSION = "expansion"
APPROVAL = "approval"
REJECTION = "rejection"
CLARIFICATION_ANSWER = "clarification_answer"

ACTS = (NEW_TOPIC, FOLLOW_UP, CORRECTION, CONTINUATION, COMPARISON, EXPANSION,
        APPROVAL, REJECTION, CLARIFICATION_ANSWER)

# Continuity acts are the ones gated by followup_confidence (R2.3).
_CONTINUITY_ACTS = (FOLLOW_UP, CONTINUATION, COMPARISON, EXPANSION)

# Deterministic lexicons (lowercased, matched on word/phrase boundaries).
# Phrase DATA lives in the central registry (app/core/lexicons.py).
_APPROVAL = lexicons.ACT_APPROVAL
_REJECTION = lexicons.ACT_REJECTION
_CONTINUATION = lexicons.ACT_CONTINUATION
_CORRECTION = lexicons.ACT_CORRECTION
_COMPARISON = lexicons.ACT_COMPARISON
_EXPANSION = lexicons.ACT_EXPANSION
_IMPROVE = lexicons.ACT_IMPROVE
# Explicit TOPIC-SHIFT cues — phrases that unambiguously abandon the current
# thread and start a fresh subject. These must win over the correction/continuity
# lexicons (a switch like "let's move on to Rust instead" contains "instead",
# which would otherwise read as a correction of the PRIOR request and drag its
# context along, giving a wrong answer). Matched as substrings on the normalized
# turn; deliberately curated to unambiguous switch phrasing so an ordinary
# follow-up is never mis-flagged.
_TOPIC_SHIFT = lexicons.ACT_TOPIC_SHIFT

# Reference cues — pronouns + selection references (drives follow-up confidence).
_PRONOUNS = lexicons.ACT_PRONOUNS
_SELECTION = re.compile(lexicons.ACT_SELECTION, re.IGNORECASE)


def _has_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def is_topic_shift(turn: str) -> bool:
    """True when the turn starts a NEW subject (reset the topic-scoped state so a
    switch answers fresh instead of inheriting the prior thread's goal/entities).

    SEMANTIC-first (the `topic_shift` gate understands paraphrases); the cue list
    is only the fallback when the embedder is unavailable."""
    t = " ".join((turn or "").lower().split())
    if not t:
        return False
    try:
        from app.semantics.gates import matches
        v = matches("topic_shift", t)
        if v is not None:
            return v
    except Exception:  # noqa: BLE001
        pass
    return _has_any(t, _TOPIC_SHIFT)  # fallback: deterministic cues


def _word_in(text: str, words) -> bool:
    toks = set(re.findall(r"[a-z']+", text))
    return any(w in toks for w in words if " " not in w) or _has_any(
        text, [w for w in words if " " in w])


def _state_has_context(state) -> bool:
    try:
        return bool(state.goal() or state.entities() or state.decisions()
                    or state.enumerations() or state.open_questions())
    except Exception:  # noqa: BLE001
        return False


def _threshold() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.followup, "followup_confidence_threshold", 0.6))
    except Exception:  # noqa: BLE001
        return 0.6


def classify(turn: str, state, model_act: str | None = None):
    """Return ``(act, followup_confidence)``. Deterministic-first; fail-open."""
    try:
        return _classify(turn, state, model_act)
    except Exception:  # noqa: BLE001 — never break a turn (R2.4)
        return NEW_TOPIC, 0.0


def _classify(turn: str, state, model_act: str | None):
    t = " ".join((turn or "").lower().split())
    if not t:
        return NEW_TOPIC, 0.0

    has_ctx = _state_has_context(state)
    words = t.split()
    short = len(words) <= 6
    has_pronoun = _word_in(t, _PRONOUNS)
    has_selection = bool(_SELECTION.search(t))
    ref_boost = 0.25 if (has_pronoun or has_selection) else 0.0
    ctx_boost = 0.15 if has_ctx else 0.0

    # 0) Explicit topic shift wins over everything: a clear "new topic / different
    #    question / let's move on" abandons the current thread. Return NEW_TOPIC
    #    with high confidence so the route does NOT apply a continuation directive
    #    or a reference rewrite (which would answer for the OLD topic).
    if _has_any(t, _TOPIC_SHIFT):
        return NEW_TOPIC, 0.9

    # 1) Clarification answer: the assistant has an open question and this short
    #    turn looks like an answer to it.
    try:
        open_qs = state.open_questions()
    except Exception:  # noqa: BLE001
        open_qs = []
    if open_qs and (short or ":" in t):
        return CLARIFICATION_ANSWER, min(1.0, 0.7 + ctx_boost)

    # 2) Correction / reversal (explicit; not downgraded).
    if _has_any(t, _CORRECTION):
        return CORRECTION, min(1.0, 0.8 + ctx_boost)

    # 3) Approval / rejection (explicit; short confirmations only so a longer
    #    "no, build X instead" routes to correction/new_topic above/below).
    if short and _word_in(t, _APPROVAL):
        return APPROVAL, min(1.0, 0.8 + ctx_boost)
    if short and _word_in(t, _REJECTION):
        return REJECTION, min(1.0, 0.8 + ctx_boost)

    # 4) Continuity acts (gated by followup_confidence). Expansion/comparison
    #    are checked before continuation so multi-word cues ("explain more")
    #    aren't swallowed by a bare continuation match.
    if _has_any(t, _EXPANSION):
        return _gate(EXPANSION, 0.65 + ref_boost + ctx_boost)
    if _has_any(t, _COMPARISON):
        return _gate(COMPARISON, 0.65 + ref_boost + ctx_boost)
    if _has_any(t, _CONTINUATION) or (short and t in ("more", "next", "go on")):
        conf = 0.7 + ref_boost + ctx_boost
        return _gate(CONTINUATION, conf)
    if _has_any(t, _IMPROVE) or (has_pronoun and short):
        return _gate(FOLLOW_UP, 0.6 + ref_boost + ctx_boost)

    # 5) A short turn that references prior content with context → follow-up.
    if (has_pronoun or has_selection) and has_ctx:
        return _gate(FOLLOW_UP, 0.55 + ref_boost)

    # 6) Optional model tie-breaker (from the existing gate call).
    if model_act in ACTS and model_act != NEW_TOPIC and has_ctx:
        return _gate(model_act, 0.6 + ctx_boost)

    # 7) Default: a self-contained / new request.
    return NEW_TOPIC, 0.2 + (0.1 if has_ctx else 0.0)


def _gate(act: str, confidence: float):
    """Downgrade a continuity act to new_topic when under the threshold (R2.3)."""
    confidence = max(0.0, min(1.0, confidence))
    if act in _CONTINUITY_ACTS and confidence < _threshold():
        return NEW_TOPIC, confidence
    return act, confidence


__all__ = [
    "classify", "is_topic_shift", "ACTS",
    "NEW_TOPIC", "FOLLOW_UP", "CORRECTION", "CONTINUATION", "COMPARISON",
    "EXPANSION", "APPROVAL", "REJECTION", "CLARIFICATION_ANSWER",
]
