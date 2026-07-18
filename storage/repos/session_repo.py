"""Sessions repo — chat / live / solve session rows.

Replaces direct ORM access to the old `conversations` table. The
external API still calls these "conversations" for backward
compatibility with the Flutter client, but the table is `sessions`
and the discriminator is `type`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, or_, select, update

from ..models import Message, Session
from .base import Repo


class SessionRepo(Repo):
    async def create(
        self,
        *,
        type: str = "chat",
        title: str = "New session",
        user_id: uuid.UUID | None = None,
        resume_id: uuid.UUID | None = None,
        session_metadata: dict | None = None,
    ) -> Session:
        row = Session(
            type=type,
            title=title,
            user_id=user_id,
            resume_id=resume_id,
            session_metadata=session_metadata or {},
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(self, session_id: uuid.UUID | str) -> Session | None:
        if isinstance(session_id, str):
            session_id = _to_uuid(session_id)
        return await self.session.get(Session, session_id)

    async def list(
        self,
        *,
        type: str | None = None,
        user_id: uuid.UUID | None = None,
        archived: bool | None = False,
        pinned_first: bool = True,
        tag: str | None = None,
        limit: int = 50,
    ) -> list[Session]:
        """Drawer list. By default hides archived sessions and surfaces
        pinned ones at the top (pinned DESC, then updated_at DESC).

        `archived=None` returns everything regardless. `tag` filters to
        sessions whose `tags` array contains that label.
        """
        stmt = select(Session)
        if pinned_first:
            stmt = stmt.order_by(Session.pinned.desc(), Session.updated_at.desc())
        else:
            stmt = stmt.order_by(Session.updated_at.desc())
        stmt = stmt.limit(limit)
        if type is not None:
            stmt = stmt.where(Session.type == type)
        if user_id is not None:
            stmt = stmt.where(Session.user_id == user_id)
        if archived is not None:
            stmt = stmt.where(Session.archived.is_(archived))
        if tag:
            # `tags @> ARRAY[<tag>]` — array containment.
            stmt = stmt.where(Session.tags.contains([tag]))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def search(
        self,
        query: str,
        *,
        user_id: uuid.UUID | None = None,
        limit: int = 30,
    ) -> list[Session]:
        """Full-text search across session titles AND message bodies.

        Uses the `content_tsv` GIN index on messages plus a plain ILIKE
        on the session title — gives the user "find that conversation
        where I asked about X" without needing a separate search index.
        """
        if not query.strip():
            return []
        tsq = func.plainto_tsquery("english", query)
        msg_match = (
            select(Message.session_id)
            .where(Message.content_tsv.op("@@")(tsq))
            .distinct()
        )
        stmt = (
            select(Session)
            .where(
                or_(
                    Session.id.in_(msg_match),
                    Session.title.ilike(f"%{query}%"),
                )
            )
            .order_by(Session.pinned.desc(), Session.updated_at.desc())
            .limit(limit)
        )
        if user_id is not None:
            stmt = stmt.where(Session.user_id == user_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def touch(self, session_id: uuid.UUID | str) -> None:
        """Force `updated_at` refresh — used when a new message lands. Sets it
        SERVER-SIDE (`func.now()`); the old `row.title = row.title` no-op didn't
        reliably mark the row dirty (SQLAlchemy sees no net change), so `onupdate`
        often never fired and the chat kept a stale timestamp."""
        if isinstance(session_id, str):
            session_id = _to_uuid(session_id)
        await self.session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(updated_at=func.now())
        )

    async def record_message(
        self,
        session_id: uuid.UUID | str,
        *,
        message_at: datetime | None = None,
    ) -> None:
        """Bump message_count + last_message_at after persisting a turn.

        Cheaper than a sub-select on every list — the drawer reads
        these columns directly. Called from chat / agents / solve
        endpoints once they commit a Message row.
        """
        if isinstance(session_id, str):
            session_id = _to_uuid(session_id)
        await self.session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                message_count=Session.message_count + 1,
                last_message_at=message_at or func.now(),
            )
        )

    async def set_flags(
        self,
        session_id: uuid.UUID | str,
        *,
        pinned: bool | None = None,
        archived: bool | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> Session | None:
        """Partial update — only the supplied fields are touched."""
        row = await self.get(session_id)
        if row is None:
            return None
        if pinned is not None:
            row.pinned = pinned
        if archived is not None:
            row.archived = archived
        if title is not None:
            row.title = title
        if tags is not None:
            row.tags = tags
        # Bump updated_at to a CONCRETE tz-aware value. Relying on the model's
        # server-side `onupdate=func.now()` leaves `row.updated_at` as an
        # unevaluated SQL expression on the returned object → the response
        # serializer (_iso_utc) then blows up with a 500 ("Rename failed").
        row.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return row

    async def set_resume(
        self,
        session_id: uuid.UUID | str,
        resume_id: uuid.UUID | str | None,
    ) -> None:
        """Associate (or clear) the resume that grounds this session's answers.
        Used by the resume upload (link the just-uploaded resume to the live
        session) and by resume delete (clear any dangling reference)."""
        if isinstance(session_id, str):
            session_id = _to_uuid(session_id)
        if isinstance(resume_id, str):
            resume_id = _to_uuid(resume_id)
        await self.session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(resume_id=resume_id)
        )

    async def clear_resume_everywhere(self, resume_id: uuid.UUID | str) -> None:
        """Null out `resume_id` on every session pointing at this resume —
        called when the resume is deleted so no session dangles."""
        if isinstance(resume_id, str):
            resume_id = _to_uuid(resume_id)
        await self.session.execute(
            update(Session)
            .where(Session.resume_id == resume_id)
            .values(resume_id=None)
        )

    async def delete(self, session_id: uuid.UUID | str) -> bool:
        if isinstance(session_id, str):
            session_id = _to_uuid(session_id)
        result = await self.session.execute(
            delete(Session).where(Session.id == session_id)
        )
        return (result.rowcount or 0) > 0

    async def delete_many(self, session_ids: list) -> int:
        """Delete many sessions in ONE statement (children cascade at the DB
        level, same as `delete`). Returns the number of rows removed. Skips ids
        that don't parse rather than failing the whole batch."""
        ids = []
        for s in session_ids:
            try:
                ids.append(_to_uuid(s) if isinstance(s, str) else s)
            except Exception:  # noqa: BLE001
                continue
        if not ids:
            return 0
        result = await self.session.execute(
            delete(Session).where(Session.id.in_(ids))
        )
        return result.rowcount or 0


def _to_uuid(s: str) -> uuid.UUID:
    try:
        return uuid.UUID(s)
    except ValueError:
        # Allow short-hex ids by zero-padding into a UUID.
        return uuid.UUID(s.ljust(32, "0")[:32])
