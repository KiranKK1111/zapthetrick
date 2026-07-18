"""Projects — model + migration registration and CRUD route wiring (§17 / #11B).

The ORM uses Postgres-only types, so like the other route tests the DB layer is
FAKED; these assert the model/migration are registered and the routes' wiring
(ownership scoping, assignment, 404s) is correct.
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import uuid

import pytest

from app.api import routes_projects as rp


# ── model + migration registration ───────────────────────────────────────

def test_project_model_registered():
    from storage.models import Base
    t = Base.metadata.tables["projects"]
    cols = set(t.columns.keys())
    assert {"id", "user_id", "name", "instructions", "metadata",
            "archived", "created_at", "updated_at"} <= cols


def test_session_has_project_fk():
    from storage.models import Base
    t = Base.metadata.tables["sessions"]
    assert "project_id" in t.columns
    assert any(fk.column.table.name == "projects"
               for fk in t.c.project_id.foreign_keys)


def test_episode_has_project_and_user_fk():
    from storage.models import Base
    t = Base.metadata.tables["episodes"]
    assert "project_id" in t.columns and "user_id" in t.columns
    assert any(fk.column.table.name == "projects"
               for fk in t.c.project_id.foreign_keys)


def test_skill_has_project_and_user_fk():
    from storage.models import Base
    t = Base.metadata.tables["skills"]
    assert "project_id" in t.columns and "user_id" in t.columns
    assert any(fk.column.table.name == "projects"
               for fk in t.c.project_id.foreign_keys)


def test_migration_0017_chain():
    p = (pathlib.Path(__file__).resolve().parent.parent
         / "storage" / "migrations" / "versions"
         / "0017_skill_project_scope.py")
    spec = importlib.util.spec_from_file_location("m0017", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.revision == "0017_skill_project_scope"
    assert mod.down_revision == "0016_episode_project_scope"
    assert callable(mod.upgrade) and callable(mod.downgrade)


def test_migration_0016_chain():
    p = (pathlib.Path(__file__).resolve().parent.parent
         / "storage" / "migrations" / "versions"
         / "0016_episode_project_scope.py")
    spec = importlib.util.spec_from_file_location("m0016", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.revision == "0016_episode_project_scope"
    assert mod.down_revision == "0015_projects"
    assert callable(mod.upgrade) and callable(mod.downgrade)


def test_migration_0015_chain():
    p = (pathlib.Path(__file__).resolve().parent.parent
         / "storage" / "migrations" / "versions" / "0015_projects.py")
    spec = importlib.util.spec_from_file_location("m0015", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.revision == "0015_projects"
    assert mod.down_revision == "0014_message_envelope"
    assert callable(mod.upgrade) and callable(mod.downgrade)


# ── pure helpers ──────────────────────────────────────────────────────────

def test_parse_uuid_rejects_garbage():
    with pytest.raises(Exception):
        rp._parse_uuid("not-a-uuid", "project")
    good = uuid.uuid4()
    assert rp._parse_uuid(str(good), "project") == good


def test_project_dict_shape():
    from storage.models import Project
    p = Project(name="Alpha", instructions="be terse", archived=False)
    p.id = uuid.uuid4()
    d = rp._project_dict(p, conversation_count=3)
    assert d["name"] == "Alpha" and d["instructions"] == "be terse"
    assert d["archived"] is False and d["conversation_count"] == 3
    assert d["id"] == str(p.id)


def test_project_dict_omits_count_when_none():
    from storage.models import Project
    p = Project(name="B")
    p.id = uuid.uuid4()
    assert "conversation_count" not in rp._project_dict(p)


# ── CRUD wiring (faked AsyncSession) ──────────────────────────────────────

class _FakeResult:
    def __init__(self, value):
        self._v = value

    def scalar(self):
        return self._v

    def all(self):
        return self._v


class _FakeSession:
    def __init__(self, store=None, scalar=0):
        self.store = store or {}
        self._scalar = scalar
        self.added = []
        self.deleted = []
        self.committed = 0

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        self.added.append(obj)
        self.store[obj.id] = obj

    async def commit(self):
        self.committed += 1

    async def refresh(self, obj):
        pass

    async def delete(self, obj):
        self.deleted.append(obj)
        self.store.pop(getattr(obj, "id", None), None)

    async def get(self, model, pk):
        return self.store.get(pk)

    async def execute(self, *_a, **_k):
        return _FakeResult(self._scalar)


def _patch_uid(monkeypatch, uid):
    async def _fake():
        return uid
    monkeypatch.setattr(rp, "ensure_device_user", _fake)


def test_create_project(monkeypatch):
    uid = uuid.uuid4()
    _patch_uid(monkeypatch, uid)
    sess = _FakeSession()
    out = asyncio.run(rp.create_project(
        {"name": "  My Project  ", "instructions": "x" * 10}, session=sess))
    assert out["name"] == "My Project"
    assert out["conversation_count"] == 0
    assert sess.added and sess.added[0].user_id == uid
    assert sess.committed == 1


def test_create_project_defaults_name(monkeypatch):
    _patch_uid(monkeypatch, uuid.uuid4())
    out = asyncio.run(rp.create_project({}, session=_FakeSession()))
    assert out["name"] == "New project"
    assert out["instructions"] == ""


def test_get_project_404_when_missing(monkeypatch):
    _patch_uid(monkeypatch, uuid.uuid4())
    with pytest.raises(Exception) as exc:
        asyncio.run(rp.get_project(str(uuid.uuid4()), session=_FakeSession()))
    assert "404" in str(getattr(exc.value, "status_code", "")) or \
        getattr(exc.value, "status_code", None) == 404


def test_owned_project_rejects_other_user(monkeypatch):
    from storage.models import Project
    mine = uuid.uuid4()
    other = uuid.uuid4()
    proj = Project(name="theirs")
    proj.id = uuid.uuid4()
    proj.user_id = other
    sess = _FakeSession(store={proj.id: proj})
    with pytest.raises(Exception) as exc:
        asyncio.run(rp._owned_project(sess, str(proj.id), mine))
    assert getattr(exc.value, "status_code", None) == 404


def test_owned_project_allows_null_owner(monkeypatch):
    from storage.models import Project
    proj = Project(name="shared")
    proj.id = uuid.uuid4()
    proj.user_id = None                      # ungrouped/legacy → visible
    sess = _FakeSession(store={proj.id: proj})
    got = asyncio.run(rp._owned_project(sess, str(proj.id), uuid.uuid4()))
    assert got is proj


def test_assign_conversation(monkeypatch):
    from storage.models import Project, Session as SessionRow
    uid = uuid.uuid4()
    _patch_uid(monkeypatch, uid)
    proj = Project(name="P"); proj.id = uuid.uuid4(); proj.user_id = uid
    convo = SessionRow(title="c"); convo.id = uuid.uuid4()
    sess = _FakeSession(store={proj.id: proj, convo.id: convo})
    out = asyncio.run(rp.assign_conversation(
        str(proj.id), str(convo.id), session=sess))
    assert out["ok"] is True
    assert convo.project_id == proj.id
    assert sess.committed == 1


def test_assign_conversation_404_missing_convo(monkeypatch):
    from storage.models import Project
    uid = uuid.uuid4()
    _patch_uid(monkeypatch, uid)
    proj = Project(name="P"); proj.id = uuid.uuid4(); proj.user_id = uid
    sess = _FakeSession(store={proj.id: proj})
    with pytest.raises(Exception) as exc:
        asyncio.run(rp.assign_conversation(
            str(proj.id), str(uuid.uuid4()), session=sess))
    assert getattr(exc.value, "status_code", None) == 404


def test_delete_project_detaches_and_deletes(monkeypatch):
    from storage.models import Project
    uid = uuid.uuid4()
    _patch_uid(monkeypatch, uid)
    proj = Project(name="P"); proj.id = uuid.uuid4(); proj.user_id = uid
    sess = _FakeSession(store={proj.id: proj})
    out = asyncio.run(rp.delete_project(str(proj.id), session=sess))
    assert out["deleted"] is True
    assert proj in sess.deleted
    assert sess.committed == 1
