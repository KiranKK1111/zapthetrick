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
    """Serves scripted results per model, records deletes/commits."""

    def __init__(self, by_model=None, delete_rowcounts=None):
        self.by_model = by_model or {}
        self.delete_rowcounts = delete_rowcounts or {}
        self.deleted = []
        self.committed = 0
        self.store = {}

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
