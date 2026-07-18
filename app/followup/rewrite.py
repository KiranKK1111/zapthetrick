"""Follow-up prompt rewriting (followup-context-engine R5).

`rewrite(turn, act, resolution, state) -> (text, confidence)` turns a vague
follow-up into an explicit, self-contained instruction using the resolved
reference + ``ConversationState`` — with deterministic templates, no second
blocking LLM call (Property 11).

Contract (Property 6):
  • Preserve the user's intent; introduce NO decision or constraint the user did
    not state.
  • Below ``rewrite_confidence_threshold`` (or when nothing resolved) → return
    the original turn so the caller falls back to today's continuity prompt
    (R5.4 / Property 1).
"""
from __future__ import annotations

from app.followup import acts as A


def _threshold() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.followup, "rewrite_confidence_threshold", 0.6))
    except Exception:  # noqa: BLE001
        return 0.6


def _antecedent_phrase(resolution, state) -> tuple[str | None, bool]:
    """The concrete thing the follow-up acts on. Returns ``(phrase, resolved)``
    — `resolved` is True only when it came from actual reference resolution
    (an antecedent in the RECENT exchange). The conversation GOAL is a stale
    fallback: in a long multi-topic thread it is the FIRST turn's subject, so
    anchoring a follow-up to it can attach the wrong topic entirely (observed:
    "…apply this to Explain how a hash map handles collisions" on a follow-up
    about a later Sudoku answer)."""
    if resolution is not None and getattr(resolution, "antecedents", None):
        return resolution.antecedents[0], True
    try:
        g = state.goal()
    except Exception:  # noqa: BLE001
        g = None
    return g, False


def rewrite(turn: str, act: str, resolution, state):
    """Return ``(text, confidence)``. Deterministic; fail-open to the original."""
    try:
        return _rewrite(turn, act, resolution, state)
    except Exception:  # noqa: BLE001 — never break a turn (R5.4)
        return (turn or ""), 0.0


def _rewrite(turn: str, act: str, resolution, state):
    original = (turn or "").strip()
    target, resolved = _antecedent_phrase(resolution, state)

    # No concrete target → can't produce a confident, self-contained rewrite.
    if not target:
        return original, 0.0

    # Resolution confidence (when present) bounds the rewrite confidence so a
    # shaky reference never produces an over-confident rewrite.
    res_conf = float(getattr(resolution, "confidence", 0.0) or 0.0) if resolution else 0.0
    base_conf = max(res_conf, 0.7 if target else 0.0)

    text = None
    if act == A.FOLLOW_UP:
        # "make it better" / "improve" → improve the RESOLVED subject, keeping
        # the user's own words as the directive verb. A GOAL-derived target is
        # NOT bound here: in a multi-topic thread the goal is the FIRST turn's
        # subject, and "…apply this to <first topic>" corrupts a follow-up
        # about a later one (observed live). The model sees the recent turns
        # anyway, so passing the turn through unchanged is strictly safer.
        # (CONTINUATION below keeps the goal anchor — resuming the
        # conversation goal is the right default for a bare "continue".)
        if not resolved:
            return original, base_conf
        text = f"{original} — specifically, apply this to {target}."
    elif act == A.CONTINUATION:
        text = (f"Continue the previous response about {target} from where it "
                f"ended, without repeating earlier content.")
    elif act == A.COMPARISON:
        text = f"{original} — compare with respect to {target}."
    elif act == A.EXPANSION:
        text = f"Expand on {target} in more detail: {original}."
    else:
        # Other acts (correction/approval/rejection/clarification_answer) are
        # handled by the state updater, not the rewriter.
        return original, 0.0

    confidence = max(0.0, min(1.0, base_conf))
    if confidence < _threshold():
        return original, confidence       # low confidence → original (R5.4)
    return text, confidence


__all__ = ["rewrite"]
