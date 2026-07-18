"""Message-level actions: feedback (👍/👎) and delete (with cascade).

Kept separate from routes_chat so message operations are modular.

  POST   /api/messages/{id}/feedback   -> store/toggle a like/dislike
  DELETE /api/messages/{id}            -> delete one message (?cascade=after
                                          also drops every later message in the
                                          session — powers retry / edit)

Feedback is persisted in the `feedback` table via FeedbackRepo, keyed by
`message_id`. We keep at most one signal per message (delete-then-insert) so
toggling like→dislike→off is clean; `GET /api/conversations/{id}` surfaces the
current signal per message.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete as _delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Feedback, get_session

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/messages")


# FE may send friendly names; normalize to the repo's valid signals.
_SIGNAL_ALIASES = {
    "like": "thumb_up",
    "dislike": "thumb_down",
    "up": "thumb_up",
    "down": "thumb_down",
    "thumb_up": "thumb_up",
    "thumb_down": "thumb_down",
}


class MessageFeedback(BaseModel):
    # None / "" clears any existing feedback on the message (un-toggle).
    signal: str | None = None


@router.post("/{message_id}/feedback")
async def set_message_feedback(
    message_id: str,
    body: MessageFeedback,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Store (or clear) the user's 👍/👎 on a message. At most one per message."""
    from storage.repos import FeedbackRepo, MessageRepo

    msg = await MessageRepo(session).get(message_id)
    if msg is None:
        raise HTTPException(404, detail="Message not found")

    raw = (body.signal or "").strip().lower()
    signal = _SIGNAL_ALIASES.get(raw) if raw else None
    if raw and signal is None:
        raise HTTPException(400, detail=f"Unknown feedback signal {body.signal!r}")

    # One signal per message: drop any prior rows first, then insert the new one.
    await session.execute(_delete(Feedback).where(Feedback.message_id == msg.id))
    if signal is not None:
        await FeedbackRepo(session).record(signal=signal, message_id=msg.id)
    await session.commit()
    return {"ok": True, "message_id": str(msg.id), "feedback": signal}


@router.post("/{message_id}/resolve")
async def resolve_message(
    message_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Clear the `incomplete` flag on a message (the user resumed/accepted it).

    Used when the user clicks Continue on an interrupted turn so the
    "Response interrupted" bar doesn't reappear after a reload.
    """
    from storage.repos import MessageRepo

    msg = await MessageRepo(session).get(message_id)
    if msg is None:
        raise HTTPException(404, detail="Message not found")
    msg.incomplete = False
    await session.commit()
    return {"ok": True, "message_id": str(msg.id)}


@router.delete("/{message_id}")
async def delete_message(
    message_id: str,
    cascade: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Delete a message. `?cascade=after` also deletes every later message in
    its session (used by retry/edit to regenerate from a clean point)."""
    from storage.repos import MessageRepo

    repo = MessageRepo(session)
    if cascade == "after":
        deleted = await repo.delete_from(message_id)
        if deleted == 0:
            raise HTTPException(404, detail="Message not found")
        await session.commit()
        return {"ok": True, "deleted": deleted}

    ok = await repo.delete(message_id)
    if not ok:
        raise HTTPException(404, detail="Message not found")
    await session.commit()
    return {"ok": True, "deleted": 1}
