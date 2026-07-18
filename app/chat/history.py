"""Conversation history windowing + rolling summary.

A long thread must not send its whole history to the model every turn — cost
and latency grow linearly and the context window eventually overflows. So:

  * `window_messages(prior)` keeps only the most RECENT turns that fit a token
    budget (the rest are "dropped").
  * the dropped older turns are represented by a rolling `summary` on the
    session row (Session.summary / summary_count), updated in the background by
    `maybe_update_summary(...)` after a turn.

The route passes the kept window as `prior_messages` and the summary text as
`history_summary`; the Persona agent appends the summary to its system prompt
(provider-safe — a single system message, not a second one mid-conversation).
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from app.core.prompt import fill

log = logging.getLogger(__name__)

# Verbatim recent-window budget (input tokens). Generous enough that ordinary
# conversations are never trimmed; only genuinely long threads window.
HISTORY_TOKEN_BUDGET = 12_000

# How many of the most recent messages the rolling summary deliberately leaves
# OUT (they're recent enough to keep verbatim), and the min batch of newly-aged
# messages before we spend an LLM call to fold them into the summary.
SUMMARY_KEEP_RECENT = 12
SUMMARY_BATCH = 6

def _est(text: str) -> int:
    """Rough input-token estimate (chars/4), matching app/llm/engine.py."""
    return max(1, len(text or "") // 4)

def window_messages(
    prior: list[dict], *, budget: int = HISTORY_TOKEN_BUDGET
) -> tuple[list[dict], int]:
    """Keep the most recent messages that fit `budget` tokens.

    `prior` is chronological [{role, content}] (NOT including the new user
    message). Returns (kept_recent, dropped_count). Always keeps at least the
    last message even if it alone exceeds the budget.

    Each message is first run through `condense_oversized`, so a single huge
    paste in the history is reduced to its important lines BEFORE windowing —
    otherwise one 40 MB message would dominate the budget (or blow the context
    window) on its own.
    """
    from app.chat.condense import condense_oversized

    prior = [
        {**m, "content": condense_oversized(m.get("content"))[0]} for m in prior
    ]
    kept: list[dict] = []
    total = 0
    for m in reversed(prior):
        t = _est(m.get("content") or "")
        if kept and total + t > budget:
            break
        kept.append(m)
        total += t
    kept.reverse()
    return kept, len(prior) - len(kept)

_SUMMARY_PROMPT = """You maintain a running summary of a conversation so it can \
continue past the model's context window.

Update the summary below with the NEW messages, then output ONLY the updated \
summary (no preamble). Keep it under ~250 words. Preserve concrete facts, \
decisions, names, code/file references, numbers, and any open threads or the \
user's stated goals/preferences. Drop pleasantries. Write in compact prose or \
short bullets.

CURRENT SUMMARY (may be empty):
{summary}

NEW MESSAGES TO FOLD IN:
{new_messages}
"""

async def maybe_update_summary(conversation_id: uuid.UUID | str) -> None:
    """Fold newly-aged messages into the session's rolling summary.

    Runs in the background after a turn. No-op unless enough messages have aged
    past the recent window to be worth an LLM call. Failures are swallowed — a
    stale summary just means slightly less long-range context, never a crash.
    """
    from storage.db import get_session_factory
    from storage.models import Message, Session

    factory = get_session_factory()
    if factory is None:
        return
    try:
        async with factory() as s:
            convo = await s.get(Session, conversation_id)
            if convo is None:
                return
            total = (
                await s.execute(
                    select(func.count(Message.id)).where(
                        Message.session_id == conversation_id
                    )
                )
            ).scalar_one()
            target = total - SUMMARY_KEEP_RECENT  # cover everything but recent
            already = convo.summary_count or 0
            if target <= 0 or target - already < SUMMARY_BATCH:
                return  # not enough aged messages to bother yet

            # The newly-aged slice: messages [already, target) in order.
            rows = (
                (
                    await s.execute(
                        select(Message)
                        .where(Message.session_id == conversation_id)
                        .order_by(Message.created_at)
                        .offset(already)
                        .limit(target - already)
                    )
                )
                .scalars()
                .all()
            )
            if not rows:
                return
            new_block = "\n".join(
                f"{m.role}: {(m.content or '').strip()[:1200]}" for m in rows
            )
            prompt = fill(_SUMMARY_PROMPT, 
                summary=(convo.summary or "(none yet)"),
                new_messages=new_block[:8000],
            )
            try:
                raw = await llm.complete(
                    [{"role": "user", "content": prompt}],
                    model=(cfg.llm.classifier_model or cfg.llm.model),
                    options={"temperature": cfg.temperature.planning,
                             "num_predict": cfg.output_tokens.verdict},
                )
            except LLMError as exc:
                log.info("rolling-summary LLM failed (keeping old): %s", exc)
                return
            new_summary = (raw or "").strip()
            if not new_summary:
                return
            convo.summary = new_summary[:4000]
            convo.summary_count = target
            await s.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("rolling-summary update failed: %s", exc)

__all__ = ["window_messages", "maybe_update_summary", "HISTORY_TOKEN_BUDGET"]
