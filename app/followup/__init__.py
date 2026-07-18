"""Follow-up / conversation-state engine (followup-context-engine spec).

Makes vague follow-ups ("make it better", "continue", "do the second one",
"actually use X", "undo that") resolve reliably by reasoning over an explicit,
persisted ``ConversationState`` rather than re-reading raw chat history.

Every subsystem is **additive, deterministic-first, and fail-open**: with the
engine flag off (``cfg.followup.enabled``) or on any error, behavior is exactly
today's prompt-driven follow-ups. The single answer/clarifier LLM call is
preserved — classification, reference resolution, and rewriting are
deterministic.

Phase 1 ships the state spine (this package's ``state`` module); later phases
add act classification, reference resolution, rewriting, and surfacing.
"""
from .state import ConversationState

__all__ = ["ConversationState"]
