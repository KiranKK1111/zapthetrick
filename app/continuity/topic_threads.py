"""Topic threads — groups sessions that share a topic.

Schema (added lazily on first call):
    session_topics(
        session_id UUID PRIMARY KEY,
        topic      TEXT NOT NULL,
        keywords   TEXT[] NOT NULL,
        embedding  VECTOR(?)            -- optional, when pgvector is present
    )

A "topic" is a short stable label (e.g. "Kafka migration plan",
"system design for X"). Sessions with the same topic value are part
of the same thread; the UI surfaces them as a stacked card group in
the history drawer.

The detection itself is heuristic: extract proper-noun-shaped terms
from the session's first user turn, hash the top three terms into
the topic label, and store. Good enough as a default; an LLM-based
relabeler can replace this without changing the API.
"""
from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class Topic:
    session_id: str
    topic: str
    keywords: list[str]


_PROPER_NOUN_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)?)\b")
_STOP = frozenset({"I", "A", "The", "And", "Or", "So", "If", "It"})


class TopicThreadRepo:
    # Schema lives in Alembic 0004_continuity. DDL below is a guard.
    _DDL = """
        CREATE TABLE IF NOT EXISTS session_topics (
            session_id UUID PRIMARY KEY,
            topic      TEXT NOT NULL,
            keywords   TEXT[] NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_session_topics_topic ON session_topics(topic);
    """

    _schema_checked: bool = False

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def ensure_schema(self) -> None:
        if TopicThreadRepo._schema_checked:
            return
        try:
            for stmt in self._DDL.strip().split(";"):
                if stmt.strip():
                    await self.session.execute(text(stmt))
            TopicThreadRepo._schema_checked = True
        except Exception:  # noqa: BLE001 — Alembic 0004 should have created these
            pass

    async def upsert(self, session_id: str | uuid.UUID, content: str) -> Topic:
        """Compute a topic label from `content` and upsert it."""
        await self.ensure_schema()
        topic, kws = _label(content)
        sid = str(session_id)
        await self.session.execute(
            text(
                "INSERT INTO session_topics (session_id, topic, keywords) "
                "VALUES (:sid, :t, :kws) "
                "ON CONFLICT (session_id) DO UPDATE SET "
                "topic = EXCLUDED.topic, keywords = EXCLUDED.keywords, "
                "updated_at = NOW()"
            ),
            {"sid": sid, "t": topic, "kws": kws},
        )
        return Topic(session_id=sid, topic=topic, keywords=kws)

    async def thread_for(self, session_id: str | uuid.UUID) -> list[Topic]:
        """Return every session sharing the same topic label."""
        await self.ensure_schema()
        r = await self.session.execute(
            text(
                "SELECT topic FROM session_topics WHERE session_id = :sid"
            ),
            {"sid": str(session_id)},
        )
        row = r.first()
        if row is None:
            return []
        topic = row[0]
        rows = await self.session.execute(
            text(
                "SELECT session_id, topic, keywords FROM session_topics "
                "WHERE topic = :t ORDER BY updated_at DESC LIMIT 50"
            ),
            {"t": topic},
        )
        return [
            Topic(session_id=str(r[0]), topic=r[1], keywords=list(r[2] or []))
            for r in rows.all()
        ]


def _label(content: str) -> tuple[str, list[str]]:
    """Cheap topic-label extractor. Top-3 proper-noun-shaped terms."""
    if not content:
        return "general", []
    raw = [w for w in _PROPER_NOUN_RE.findall(content) if w not in _STOP and len(w) > 1]
    if not raw:
        return "general", []
    common = [t for t, _ in Counter(raw).most_common(3)]
    return " / ".join(common[:3]), common


__all__ = ["Topic", "TopicThreadRepo"]
