"""Explicit session-to-session link graph.

Schema (added by migration 0004):
    session_links(
        from_session UUID,
        to_session   UUID,
        kind         TEXT,   -- continues | references | clarifies | derived
        confidence   NUMERIC(4, 3),
        rationale    TEXT,
        created_at   TIMESTAMPTZ
    )

Links flow from older sessions to newer ones — `from_session` is the
*origin* the new session continues. The auto-link / suggest-confirm
flows the doc describes are layered on top of this primitive store.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class LinkKind(str, Enum):
    CONTINUES = "continues"
    REFERENCES = "references"
    CLARIFIES = "clarifies"
    DERIVED = "derived"


@dataclass
class Link:
    from_session: str
    to_session: str
    kind: LinkKind
    confidence: float
    rationale: str = ""


class LinkGraphRepo:
    """Tiny CRUD wrapper around `session_links`. We use raw SQL so
    the table can be created lazily — no ORM model is required and
    the migration stays small."""

    # Schema lives in Alembic 0004_continuity. The DDL below is a
    # belt-and-braces guard so the runtime path doesn't explode on a
    # database that's missing the table (e.g. a dev DB rolled back
    # past 0004 without re-running migrations).
    _DDL = """
        CREATE TABLE IF NOT EXISTS session_links (
            from_session UUID NOT NULL,
            to_session   UUID NOT NULL,
            kind         TEXT NOT NULL DEFAULT 'references',
            confidence   NUMERIC(4, 3) NOT NULL DEFAULT 0.5,
            rationale    TEXT NOT NULL DEFAULT '',
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (from_session, to_session, kind)
        );
        CREATE INDEX IF NOT EXISTS ix_session_links_from ON session_links(from_session);
        CREATE INDEX IF NOT EXISTS ix_session_links_to   ON session_links(to_session);
    """

    # Set True after a successful `ensure_schema` so we don't re-issue
    # the DDL on every call from the chat route.
    _schema_checked: bool = False

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def ensure_schema(self) -> None:
        if LinkGraphRepo._schema_checked:
            return
        try:
            for stmt in self._DDL.strip().split(";"):
                if stmt.strip():
                    await self.session.execute(text(stmt))
            LinkGraphRepo._schema_checked = True
        except Exception:  # noqa: BLE001 — Alembic 0004 should have created these
            pass

    async def add(self, link: Link) -> None:
        await self.ensure_schema()
        await self.session.execute(
            text(
                "INSERT INTO session_links "
                "(from_session, to_session, kind, confidence, rationale) "
                "VALUES (:fs, :ts, :k, :c, :r) "
                "ON CONFLICT (from_session, to_session, kind) DO UPDATE SET "
                "confidence = EXCLUDED.confidence, rationale = EXCLUDED.rationale"
            ),
            {
                "fs": link.from_session,
                "ts": link.to_session,
                "k": link.kind.value if isinstance(link.kind, LinkKind) else str(link.kind),
                "c": float(link.confidence),
                "r": link.rationale or "",
            },
        )

    async def neighbours(
        self, session_id: str | uuid.UUID, *, kind: LinkKind | None = None
    ) -> list[Link]:
        """Return both inbound and outbound links for one session."""
        await self.ensure_schema()
        clauses = ["(from_session = :sid OR to_session = :sid)"]
        params: dict[str, object] = {"sid": str(session_id)}
        if kind is not None:
            clauses.append("kind = :k")
            params["k"] = kind.value
        sql = (
            "SELECT from_session, to_session, kind, confidence, rationale "
            "FROM session_links WHERE " + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT 50"
        )
        result = await self.session.execute(text(sql), params)
        return [
            Link(
                from_session=str(r[0]),
                to_session=str(r[1]),
                kind=LinkKind(r[2]) if r[2] in {k.value for k in LinkKind} else LinkKind.REFERENCES,
                confidence=float(r[3] or 0.5),
                rationale=r[4] or "",
            )
            for r in result.all()
        ]


__all__ = ["Link", "LinkKind", "LinkGraphRepo"]
