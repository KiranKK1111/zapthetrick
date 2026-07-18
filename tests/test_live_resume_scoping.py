"""Per-session resume isolation on the LIVE socket.

A resume uploaded for session A must ground ONLY session A's answers — never
session B's. The socket used to trust the client's query-param resume_id,
which resolves to the globally "active" (last-uploaded) resume, so one
session's resume bled into every other. The fix makes each session's own
persisted Session.resume_id authoritative.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

from app.api import routes_ws


class _FakeSession:
    """Minimal async DB session double: .get(Model, key) → preset row."""

    def __init__(self, rows: dict):
        self._rows = rows

    async def get(self, _model, key):
        return self._rows.get(str(key))


class _Row:
    def __init__(self, resume_id):
        self.resume_id = resume_id


def _factory(rows: dict):
    @asynccontextmanager
    async def _cm():
        yield _FakeSession(rows)

    def factory():
        return _cm()

    return factory


def _install(monkeypatch, rows: dict):
    monkeypatch.setattr(routes_ws, "get_session_factory",
                        lambda: _factory(rows))


def test_session_with_linked_resume_is_authoritative(monkeypatch):
    sid = str(uuid.uuid4())
    rid = str(uuid.uuid4())
    _install(monkeypatch, {sid: _Row(uuid.UUID(rid))})
    has_row, resolved = asyncio.run(
        routes_ws._load_session_resume_id(sid))
    assert has_row is True
    assert resolved == rid


def test_session_with_no_resume_returns_null_not_fallback(monkeypatch):
    # A session that exists but has NO resume linked must resolve to None —
    # a stale global query-param resume must NOT bleed in.
    sid = str(uuid.uuid4())
    _install(monkeypatch, {sid: _Row(None)})
    has_row, resolved = asyncio.run(
        routes_ws._load_session_resume_id(sid))
    assert has_row is True
    assert resolved is None


def test_no_session_row_keeps_query_param(monkeypatch):
    # No row (brand-new ad-hoc session) → has_row False, so the caller keeps
    # whatever the client sent.
    _install(monkeypatch, {})
    has_row, resolved = asyncio.run(
        routes_ws._load_session_resume_id(str(uuid.uuid4())))
    assert has_row is False
    assert resolved is None


def test_two_sessions_resolve_to_their_own_resumes(monkeypatch):
    # The core isolation property: A and B each resolve to their OWN resume.
    a, ra = str(uuid.uuid4()), str(uuid.uuid4())
    b, rb = str(uuid.uuid4()), str(uuid.uuid4())
    _install(monkeypatch, {a: _Row(uuid.UUID(ra)), b: _Row(uuid.UUID(rb))})
    _, resolved_a = asyncio.run(routes_ws._load_session_resume_id(a))
    _, resolved_b = asyncio.run(routes_ws._load_session_resume_id(b))
    assert resolved_a == ra
    assert resolved_b == rb
    assert resolved_a != resolved_b


def test_bad_session_id_is_safe(monkeypatch):
    _install(monkeypatch, {})
    has_row, resolved = asyncio.run(
        routes_ws._load_session_resume_id("not-a-uuid"))
    assert has_row is False
    assert resolved is None


def test_no_factory_is_safe(monkeypatch):
    monkeypatch.setattr(routes_ws, "get_session_factory", lambda: None)
    has_row, resolved = asyncio.run(
        routes_ws._load_session_resume_id(str(uuid.uuid4())))
    assert has_row is False
    assert resolved is None
