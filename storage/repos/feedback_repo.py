"""Feedback rows — user 👍/👎/edit signals for messages and episodes."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from ..models import Feedback
from .base import Repo


_VALID_SIGNALS = frozenset({"up", "down", "edit", "redo", "thumb_up", "thumb_down"})


class FeedbackRepo(Repo):
    async def record(
        self,
        *,
        signal: str,
        message_id: uuid.UUID | str | None = None,
        episode_id: uuid.UUID | str | None = None,
        payload: dict | None = None,
    ) -> Feedback:
        if signal not in _VALID_SIGNALS:
            raise ValueError(
                f"feedback signal must be one of {sorted(_VALID_SIGNALS)}; got {signal!r}"
            )
        if isinstance(message_id, str):
            message_id = uuid.UUID(message_id)
        if isinstance(episode_id, str):
            episode_id = uuid.UUID(episode_id)
        row = Feedback(
            message_id=message_id,
            episode_id=episode_id,
            signal=signal,
            payload=payload,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def for_message(self, message_id: uuid.UUID | str) -> list[Feedback]:
        if isinstance(message_id, str):
            message_id = uuid.UUID(message_id)
        result = await self.session.execute(
            select(Feedback).where(Feedback.message_id == message_id)
        )
        return list(result.scalars().all())

    async def for_episode(self, episode_id: uuid.UUID | str) -> list[Feedback]:
        if isinstance(episode_id, str):
            episode_id = uuid.UUID(episode_id)
        result = await self.session.execute(
            select(Feedback).where(Feedback.episode_id == episode_id)
        )
        return list(result.scalars().all())
