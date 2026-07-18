"""Versioned artifact store — Phase 5 persistence (Document Generation roadmap).

CRUD over `generated_documents`: save a produced document as a new version, fetch
the latest, list an evolution timeline. The source Markdown is the single stored
representation; every export format + the structured model derive from it (Phase
1/1b), so an edit is a cheap new row, not a re-rendered blob.

Every entry point is FAIL-OPEN: a persistence error returns None / no-op and is
logged, never breaking the turn that produced the document (the store is an
enhancement, not on the critical path). The DB round-trip isn't exercised by the
unit suite (no test-DB harness); the version-increment logic is factored into the
pure `next_version` so it IS tested, and the queries mirror the codebase's
established SQLAlchemy patterns.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional, Union

log = logging.getLogger(__name__)


def next_version(max_existing: Optional[int]) -> int:
    """Version number for a new revision: 1 for a brand-new document, else one
    past the highest existing version. Pure — the DB layer just supplies the
    current max."""
    return int(max_existing or 0) + 1


def _as_uuid(value: Union[str, uuid.UUID, None]) -> Optional[uuid.UUID]:
    if value is None or isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


async def save_version(session, session_id, content_md: str, *,
                       title: str = "", fmt: str = "pdf",
                       goal: Optional[str] = None,
                       meta: Optional[dict] = None,
                       doc_key: Union[str, uuid.UUID, None] = None):
    """Add a new document version on ``session`` (caller commits). A ``doc_key``
    chains this as the next version of an existing document; without one a fresh
    document (version 1, new key) is created. Returns the ORM row."""
    from sqlalchemy import func, select
    from storage.models import GeneratedDocument

    key = _as_uuid(doc_key)
    if key is None:
        key = uuid.uuid4()
        version = 1
    else:
        res = await session.execute(
            select(func.max(GeneratedDocument.version))
            .where(GeneratedDocument.doc_key == key))
        version = next_version(res.scalar())

    row = GeneratedDocument(
        session_id=_as_uuid(session_id), doc_key=key, version=version,
        title=(title or "")[:500], doc_format=(fmt or "pdf")[:16],
        goal=(goal or None), content_md=content_md or "", meta=meta)
    session.add(row)
    await session.flush()
    return row


async def record_generation(session_id, content_md: str, *, title: str = "",
                            fmt: str = "pdf", goal: Optional[str] = None,
                            meta: Optional[dict] = None,
                            doc_key: Union[str, uuid.UUID, None] = None,
                            chain_latest: bool = False
                            ) -> Optional[uuid.UUID]:
    """Fail-open, self-contained persist: opens its own session, saves a version,
    commits. ``chain_latest`` (used on an UPDATE_EXISTING turn) chains this as the
    next version of the conversation's most recent document when no explicit
    ``doc_key`` is given. Returns the resolved ``doc_key`` (so the caller can
    thread it onto the message for future edits), or None on any error."""
    try:
        from storage.db import get_session_factory
        factory = get_session_factory()
        sid = _as_uuid(session_id)
        if factory is None or sid is None or not (content_md or "").strip():
            return None
        async with factory() as s:
            key = _as_uuid(doc_key)
            if key is None and chain_latest:
                prev = await latest_for_session(s, sid)
                if prev is not None:
                    key = prev.doc_key
            row = await save_version(s, sid, content_md, title=title, fmt=fmt,
                                     goal=goal, meta=meta, doc_key=key)
            resolved = row.doc_key
            await s.commit()
            return resolved
    except Exception as exc:  # noqa: BLE001 — persistence must never break a turn
        log.debug("record_generation failed (non-fatal): %s", exc)
        return None


async def latest_for_session(session, session_id):
    """The most recently created document in a conversation, or None."""
    from sqlalchemy import select
    from storage.models import GeneratedDocument
    res = await session.execute(
        select(GeneratedDocument)
        .where(GeneratedDocument.session_id == _as_uuid(session_id))
        .order_by(GeneratedDocument.created_at.desc()).limit(1))
    return res.scalar_one_or_none()


async def latest_version(session, doc_key):
    """The highest-numbered version of a document, or None."""
    from sqlalchemy import select
    from storage.models import GeneratedDocument
    res = await session.execute(
        select(GeneratedDocument)
        .where(GeneratedDocument.doc_key == _as_uuid(doc_key))
        .order_by(GeneratedDocument.version.desc()).limit(1))
    return res.scalar_one_or_none()


async def list_for_session(session, session_id, limit: int = 50) -> list:
    """A conversation's generated documents, newest first."""
    from sqlalchemy import select
    from storage.models import GeneratedDocument
    res = await session.execute(
        select(GeneratedDocument)
        .where(GeneratedDocument.session_id == _as_uuid(session_id))
        .order_by(GeneratedDocument.created_at.desc()).limit(max(1, int(limit))))
    return list(res.scalars().all())


async def list_versions(session, doc_key) -> list:
    """Every version of a document, oldest first (the evolution timeline)."""
    from sqlalchemy import select
    from storage.models import GeneratedDocument
    res = await session.execute(
        select(GeneratedDocument)
        .where(GeneratedDocument.doc_key == _as_uuid(doc_key))
        .order_by(GeneratedDocument.version.asc()))
    return list(res.scalars().all())


__all__ = [
    "next_version", "save_version", "record_generation",
    "latest_for_session", "latest_version", "list_versions",
    "list_for_session",
]
