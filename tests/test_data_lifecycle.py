"""Data lifecycle & privacy (Architecture §18 / #13)."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from app.memory import data_lifecycle as dl


# ---- provenance forget: KG node + edges (pure) ---------------------------

def test_forget_kg_node_removes_node_and_incident_edges():
    kg = {
        "nodes": [{"id": "jwt"}, {"id": "token"}, {"id": "cookie"}],
        "edges": [
            {"src": "jwt", "dst": "token"},   # touches jwt → drop
            {"src": "token", "dst": "cookie"},  # keep
            {"src": "cookie", "dst": "jwt"},   # touches jwt → drop
        ],
    }
    out = dl.forget_kg_node(kg, "JWT")          # case-insensitive
    assert [n["id"] for n in out["nodes"]] == ["token", "cookie"]
    assert out["edges"] == [{"src": "token", "dst": "cookie"}]


def test_forget_kg_node_missing_node_is_noop():
    kg = {"nodes": [{"id": "a"}], "edges": [{"src": "a", "dst": "b"}]}
    out = dl.forget_kg_node(kg, "zzz")
    assert out["nodes"] == [{"id": "a"}]
    assert out["edges"] == [{"src": "a", "dst": "b"}]


def test_forget_kg_node_handles_garbage():
    assert dl.forget_kg_node(None, "x") == {"nodes": [], "edges": []}
    assert dl.forget_kg_node({}, "x") == {"nodes": [], "edges": []}


# ---- fake async session --------------------------------------------------

class _Result:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    """Serves scripted results per model, records adds/deletes/commits."""

    def __init__(self, by_model=None, delete_rowcounts=None):
        self.by_model = by_model or {}
        self.delete_rowcounts = delete_rowcounts or {}
        self.deleted = []
        self.added = []
        self.committed = 0
        self.store = {}

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        # Stand in for the INSERT that assigns the primary key — import_bundle
        # needs the new id to re-link children.
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    def of(self, name):
        """Every added row of one model, in insertion order."""
        return [o for o in self.added if type(o).__name__ == name]

    async def execute(self, stmt):
        # crude model detection from the statement's target entity
        model = _stmt_model(stmt)
        if _is_delete(stmt):
            return _Result(rowcount=self.delete_rowcounts.get(model, 0))
        return _Result(rows=list(self.by_model.get(model, [])))

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.committed += 1

    async def get(self, model, pk):
        return self.store.get(pk)


def _stmt_model(stmt):
    try:
        return stmt.column_descriptions[0]["entity"].__name__
    except Exception:
        try:
            return stmt.entity_description["name"]
        except Exception:
            return None


def _is_delete(stmt):
    return stmt.__class__.__name__.lower().startswith("delete")


# ---- retention purge -----------------------------------------------------

def test_purge_disabled_is_noop():
    sess = _FakeSession()
    out = asyncio.run(dl.purge_expired(sess, retention_days=0))
    assert out == {"enabled": False, "episodes": 0, "skills": 0}
    assert sess.committed == 0


def test_purge_deletes_old_rows():
    old = datetime.now(timezone.utc) - timedelta(days=100)

    class _Ep:
        def __init__(self):
            self.id = uuid.uuid4()
            self.vector_point_id = uuid.uuid4()
            self.created_at = old

    class _Sk:
        def __init__(self):
            self.id = uuid.uuid4()
            self.vector_point_id = None
            self.created_at = old

    eps = [_Ep(), _Ep()]
    sks = [_Sk()]
    sess = _FakeSession(by_model={"Episode": eps, "SkillRow": sks})
    out = asyncio.run(dl.purge_expired(sess, retention_days=30))
    assert out["enabled"] is True
    assert out["episodes"] == 2 and out["skills"] == 1
    assert len(sess.deleted) == 3           # 2 episodes + 1 skill
    assert sess.committed == 1


# ---- export-all ----------------------------------------------------------

def test_export_all_shapes_bundle():
    class _S:
        def __init__(self):
            self.id = uuid.uuid4()
            self.title = "Chat"
            self.type = "chat"
            self.project_id = None
            self.session_metadata = {"kg": {"nodes": [{"id": "x"}]}}
            self.started_at = datetime.now(timezone.utc)

    class _Ep:
        def __init__(self):
            self.id = uuid.uuid4()
            self.session_tag = "s1"
            self.project_id = None
            self.question = "q"
            self.final = "a"
            self.intent = "knowledge"
            self.feedback = None

    sess = _FakeSession(by_model={"Session": [_S()], "Episode": [_Ep()],
                                  "SkillRow": [], "Project": [],
                                  "Message": []})
    out = asyncio.run(dl.export_all(sess, user_id=uuid.uuid4()))
    assert out["counts"]["conversations"] == 1
    assert out["counts"]["episodes"] == 1
    assert out["conversations"][0]["kg"] == {"nodes": [{"id": "x"}]}
    assert "exported_at" in out


# ---- delete-all ----------------------------------------------------------

def test_delete_all_counts_and_commits():
    class _S:
        def __init__(self):
            self.id = uuid.uuid4()

    sessions = [_S(), _S()]
    sess = _FakeSession(
        by_model={"Session": sessions, "Episode": [], "SkillRow": [],
                  "Message": []},
        delete_rowcounts={"Project": 3})
    out = asyncio.run(dl.delete_all(sess, user_id=uuid.uuid4()))
    assert out["deleted"] is True
    assert out["conversations"] == 2
    assert out["projects"] == 3
    assert sess.committed == 1
    assert len(sess.deleted) == 2           # the two sessions


# ---- forget one episode --------------------------------------------------

def test_forget_episode_deletes_row():
    class _Ep:
        def __init__(self):
            self.id = uuid.uuid4()
            self.vector_point_id = None
            self.user_id = None

    ep = _Ep()
    sess = _FakeSession()
    sess.store[ep.id] = ep
    ok = asyncio.run(dl.forget_episode(sess, str(ep.id)))
    assert ok is True
    assert ep in sess.deleted and sess.committed == 1


def test_forget_episode_missing_returns_false():
    assert asyncio.run(dl.forget_episode(_FakeSession(), str(uuid.uuid4()))) is False
    assert asyncio.run(dl.forget_episode(_FakeSession(), "not-a-uuid")) is False


# ---- import (the mirror of export-all) -----------------------------------

def _bundle():
    """One project, one chat + one live session, messages for both, and one
    episode + one skill — i.e. every section, so section selection is testable."""
    proj, chat, live = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    return {
        "projects": [{"id": proj, "name": "Interview prep",
                      "instructions": "be terse",
                      "kg": {"nodes": [{"id": "dp"}]}}],
        "conversations": [
            {"id": chat, "title": "Two Sum", "type": "chat",
             "project_id": proj, "started_at": "2026-07-01T10:00:00+00:00"},
            {"id": live, "title": "Mock round", "type": "live",
             "project_id": None},
        ],
        "messages": [
            {"conversation_id": chat, "role": "user", "content": "solve it",
             "created_at": "2026-07-01T10:00:01+00:00"},
            {"conversation_id": chat, "role": "assistant", "content": "here"},
            {"conversation_id": live, "role": "user", "content": "hello"},
            {"conversation_id": str(uuid.uuid4()), "role": "user",
             "content": "orphan"},
        ],
        "episodes": [{"question": "q", "final": "a", "project_id": proj}],
        "skills": [{"text": "always check bounds", "kind": "lesson",
                    "confidence": 0.8}],
    }


def test_sections_in_bundle_counts_each_section():
    out = dl.sections_in_bundle(_bundle())
    assert out == {"chat_sessions": 1, "live_sessions": 1, "projects": 1,
                   "memories": 2}


def test_sections_in_bundle_tolerates_garbage():
    assert dl.sections_in_bundle({}) == {"chat_sessions": 0, "live_sessions": 0,
                                         "projects": 0, "memories": 0}
    assert dl.sections_in_bundle(None)["projects"] == 0
    # A conversation with no `type` counts as a chat, not as nothing.
    assert dl.sections_in_bundle(
        {"conversations": [{"id": "x"}, "junk"]})["chat_sessions"] == 1


def test_import_everything_relinks_the_graph():
    sess = _FakeSession()
    uid = uuid.uuid4()
    out = asyncio.run(dl.import_bundle(sess, user_id=uid, bundle=_bundle()))

    assert out["imported"]["projects"] == 1
    assert out["imported"]["chat_sessions"] == 1
    assert out["imported"]["live_sessions"] == 1
    assert out["imported"]["memories"] == 2
    # 3 of the 4 messages have a parent in the bundle; the 4th is dropped.
    assert out["imported"]["messages"] == 3
    assert out["imported"]["orphan_messages"] == 1
    assert sess.committed == 1

    project = sess.of("Project")[0]
    chat = next(s for s in sess.of("Session") if s.type == "chat")
    live = next(s for s in sess.of("Session") if s.type == "live")
    # The bundle's ids are NOT reused — they only re-link parent → child.
    assert chat.project_id == project.id
    assert live.project_id is None
    assert {m.session_id for m in sess.of("Message")} == {chat.id, live.id}
    assert all(r.user_id == uid for r in sess.of("Session") + sess.of("Project"))
    # Timestamps survive the round trip so the thread keeps its order.
    assert chat.started_at.year == 2026


def test_import_respects_section_selection():
    sess = _FakeSession()
    out = asyncio.run(dl.import_bundle(
        sess, user_id=None, bundle=_bundle(), sections=["live_sessions"]))
    assert out["imported"]["live_sessions"] == 1
    assert out["imported"]["chat_sessions"] == 0
    assert out["imported"]["projects"] == 0
    assert out["imported"]["memories"] == 0
    # Only the live session's own message came across; the chat's are orphans
    # because their conversation wasn't selected.
    assert out["imported"]["messages"] == 1
    assert [s.type for s in sess.of("Session")] == ["live"]
    # A conversation whose project wasn't imported must not carry a dangling FK.
    assert sess.of("Session")[0].project_id is None


def test_import_reports_unknown_sections_and_skips_them():
    sess = _FakeSession()
    out = asyncio.run(dl.import_bundle(
        sess, user_id=None, bundle=_bundle(),
        sections=["projects", "passwords"]))
    assert out["ignored_sections"] == ["passwords"]
    assert out["imported"]["projects"] == 1
    assert out["imported"]["chat_sessions"] == 0


def test_import_empty_bundle_is_a_noop_that_still_answers():
    sess = _FakeSession()
    out = asyncio.run(dl.import_bundle(sess, user_id=None, bundle={}))
    assert out["imported"]["chat_sessions"] == 0
    assert out["available"]["projects"] == 0
    assert sess.added == []


def test_import_skips_blank_skills_and_contentless_messages():
    sess = _FakeSession()
    cid = str(uuid.uuid4())
    bundle = {
        "conversations": [{"id": cid, "title": "t", "type": "chat"}],
        "messages": [{"conversation_id": cid, "role": "user", "content": None},
                     {"conversation_id": cid, "role": "user", "content": "ok"}],
        "skills": [{"text": "   "}, {"text": "real"}],
    }
    out = asyncio.run(dl.import_bundle(sess, user_id=None, bundle=bundle))
    assert out["imported"]["messages"] == 1
    assert out["imported"]["orphan_messages"] == 1
    assert out["imported"]["memories"] == 1


def test_import_survives_a_garbage_timestamp():
    sess = _FakeSession()
    cid = str(uuid.uuid4())
    bundle = {
        "conversations": [{"id": cid, "title": "t", "type": "chat",
                           "started_at": "not-a-date"}],
        "messages": [{"conversation_id": cid, "role": "user", "content": "x",
                      "created_at": ""}],
    }
    out = asyncio.run(dl.import_bundle(sess, user_id=None, bundle=bundle))
    assert out["imported"]["chat_sessions"] == 1
    assert out["imported"]["messages"] == 1


# ---- routes --------------------------------------------------------------
# Asserted against the router object, NOT a TestClient over app.main — importing
# app.main loads the ML models and leaks state into later tests.

def test_import_routes_are_registered():
    from app.api.routes_agents import router

    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/agents/data/import" in paths
    assert "/api/agents/data/import/inspect" in paths
    assert "/api/agents/data/export" in paths   # the pair stays a pair
