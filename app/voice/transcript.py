"""Turn ledger → chat bubbles (Requirements 8.5, 8.6, 10.3).

A voice conversation must not be a separate universe: when a session ends, the
spoken turns should be sitting in the chat thread the user was already in.

The duplication trap
--------------------
The two engines differ in **who generates the answer**, so they differ in who
has already written to chat:

* ``staged``   — the CLIENT runs its normal chat pipeline for the reply, which
  persists both bubbles exactly as a typed turn would. Persisting them again
  here would double every message.
* ``realtime`` — the SERVER owns generation; nothing has written to chat, so
  this module must.

That is why :class:`~app.voice.engine.VoiceTurn` records the engine that
produced it, and why :func:`persist_ledger` filters on it. Getting this wrong in
either direction violates Requirement 10.3 ("turns neither lost nor duplicated"),
and a handover mid-session means one ledger can legitimately contain both kinds.

Everything here is fail-open: losing a transcript is bad, but breaking session
teardown over it is worse.
"""
from __future__ import annotations

import logging
import uuid

from app.voice.engine import SessionLedger, VoiceTurn

log = logging.getLogger("zapthetrick.voice.transcript")

# Engines whose turns the CLIENT has already persisted through the normal chat
# path. Turns from these are skipped here.
CLIENT_PERSISTED = frozenset({"staged"})


def needs_persisting(turn: VoiceTurn) -> bool:
    """Whether this turn still has to be written to the chat thread."""
    if turn.engine in CLIENT_PERSISTED:
        return False
    return bool((turn.user or "").strip() or (turn.assistant or "").strip())


def pending(ledger: SessionLedger) -> list[VoiceTurn]:
    return [t for t in ledger.turns if needs_persisting(t)]


async def persist_ledger(ledger: SessionLedger,
                         conversation_id: str | None) -> int:
    """Write the server-generated turns of `ledger` into the chat conversation.

    Returns the number of MESSAGES written (two per complete turn). Never
    raises — a storage failure logs and returns 0, and the session still closes
    cleanly.
    """
    turns = pending(ledger)
    if not turns or not conversation_id:
        return 0
    try:
        conv_uuid = uuid.UUID(str(conversation_id))
    except (ValueError, AttributeError, TypeError):
        log.info("voice: conversation id %r is not a uuid — transcript not "
                 "persisted", conversation_id)
        return 0

    try:
        from storage.db import get_session_factory
        from storage.models import Conversation, Message
        factory = get_session_factory()
        if factory is None:
            return 0
        written = 0
        async with factory() as session:
            convo = await session.get(Conversation, conv_uuid)
            if convo is None:
                log.info("voice: conversation %s is gone — transcript not "
                         "persisted", conversation_id)
                return 0
            for turn in turns:
                user = (turn.user or "").strip()
                assistant = (turn.assistant or "").strip()
                if user:
                    session.add(Message(conversation_id=conv_uuid, role="user",
                                        content=user))
                    written += 1
                if assistant:
                    session.add(Message(
                        conversation_id=conv_uuid, role="assistant",
                        content=assistant, intent="Voice",
                        model=turn.engine,
                        agents_used=list(turn.tools_used) or None,
                    ))
                    written += 1
            # No-op assignment so SQLAlchemy's onupdate bumps the thread's
            # timestamp — the conversation should surface as recently active.
            convo.title = convo.title
            await session.commit()
        return written
    except Exception:  # noqa: BLE001 — teardown must never fail on transcripts
        log.warning("voice: failed to persist %d turn(s) to conversation %s",
                    len(turns), conversation_id, exc_info=True)
        return 0


__all__ = ["CLIENT_PERSISTED", "needs_persisting", "pending", "persist_ledger"]
