"""Messages repo — chat messages with provider / model / latency / source metadata."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from ..models import Message
from .base import Repo


class MessageRepo(Repo):
    async def append(
        self,
        *,
        session_id: uuid.UUID | str,
        role: str,
        content: str,
        intent: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        tokens: int | None = None,
        latency_ms: int | None = None,
        agents_used: list[str] | None = None,
        sources: dict | None = None,
        confidence: float | None = None,
    ) -> Message:
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)
        row = Message(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            provider=provider,
            model=model,
            tokens=tokens,
            latency_ms=latency_ms,
            agents_used=agents_used,
            sources=sources,
            confidence=confidence,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(self, message_id: uuid.UUID | str) -> Message | None:
        if isinstance(message_id, str):
            message_id = uuid.UUID(message_id)
        return await self.session.get(Message, message_id)

    async def delete(self, message_id: uuid.UUID | str) -> bool:
        """Delete a single message. Returns False if it didn't exist."""
        row = await self.get(message_id)
        if row is None:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True

    async def delete_from(self, message_id: uuid.UUID | str) -> int:
        """Delete `message_id` and every later message in the same session.

        Powers Claude-style retry / edit: drop the turn (and everything after
        it) so the caller can regenerate from a clean point. Returns the count
        deleted (0 if the message doesn't exist).

        Best-effort: also removes the blobs (uploaded images/files) owned by the
        deleted messages so repeated edit/retry of attachment turns doesn't leak
        orphaned blobs — mirrors the conversation-delete cleanup.
        """
        from sqlalchemy import delete as _delete

        anchor = await self.get(message_id)
        if anchor is None:
            return 0
        # Collect owned blob paths from the to-be-deleted rows BEFORE deleting.
        blob_paths: list[str] = []
        try:
            doomed = (
                await self.session.execute(
                    select(Message).where(
                        Message.session_id == anchor.session_id,
                        Message.created_at >= anchor.created_at,
                    )
                )
            ).scalars().all()
            for m in doomed:
                src = getattr(m, "sources", None)
                if not isinstance(src, dict):
                    continue
                for key in ("images", "files"):
                    for ref in (src.get(key) or []):
                        p = ref.get("path") if isinstance(ref, dict) else None
                        if p:
                            blob_paths.append(p)
        except Exception:  # noqa: BLE001 — never block the delete on collection
            blob_paths = []
        result = await self.session.execute(
            _delete(Message).where(
                Message.session_id == anchor.session_id,
                Message.created_at >= anchor.created_at,
            )
        )
        await self.session.flush()
        if blob_paths:
            try:
                from storage.blobs import get_blobs

                store = get_blobs()
                for p in blob_paths:
                    try:
                        await store.delete(p)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
        return result.rowcount or 0

    async def list_for_session(
        self,
        session_id: uuid.UUID | str,
    ) -> list[Message]:
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)
        result = await self.session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        return list(result.scalars().all())

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Message]:
        """BM25-ish search via Postgres tsvector — used by history search."""
        # plainto_tsquery is more forgiving than to_tsquery (no operator chars).
        stmt = (
            select(Message)
            .where(Message.content_tsv.op("@@")(_plain_query(query)))
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


def _plain_query(q: str):
    """Build a `plainto_tsquery('english', :q)` expression."""
    from sqlalchemy import func

    return func.plainto_tsquery("english", q)
