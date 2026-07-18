"""Project context + KG scoping (Architecture §17 / #11B Phase 3)."""
from __future__ import annotations

import asyncio
import uuid

from app.personalization import projects as pc
from app.rag import documents as docs


# ---- project instructions framing ----------------------------------------

def test_frame_empty_is_blank():
    assert pc.frame_project_instructions("") == ""
    assert pc.frame_project_instructions(None) == ""


def test_frame_states_precedence_and_trust():
    block = pc.frame_project_instructions("Prefer Rust; cite RFCs.")
    assert "Prefer Rust; cite RFCs." in block
    assert "take precedence" in block          # user/safety win
    assert "trusted" in block.lower()
    assert "UNTRUSTED" not in block


def test_frame_caps_length():
    block = pc.frame_project_instructions("z" * (pc._INSTR_CAP + 500))
    assert block.count("z") == pc._INSTR_CAP


# ---- load_project_context (faked session) --------------------------------

class _Convo:
    def __init__(self, project_id):
        self.project_id = project_id


class _Proj:
    def __init__(self, pid, instructions):
        self.id = pid
        self.instructions = instructions


class _Session:
    def __init__(self, proj=None):
        self._proj = proj

    async def get(self, _model, _pk):
        return self._proj


def test_context_empty_when_ungrouped():
    ctx = asyncio.run(pc.load_project_context(_Session(), _Convo(None)))
    assert ctx == {"project_id": None, "instructions": ""}


def test_context_loads_project_instructions():
    pid = uuid.uuid4()
    sess = _Session(_Proj(pid, "  be terse  "))
    ctx = asyncio.run(pc.load_project_context(sess, _Convo(pid)))
    assert ctx["project_id"] == str(pid)
    assert ctx["instructions"] == "be terse"


def test_context_fail_open_on_error():
    class _Boom:
        async def get(self, *a):
            raise RuntimeError("db down")
    ctx = asyncio.run(pc.load_project_context(_Boom(), _Convo(uuid.uuid4())))
    assert ctx == {"project_id": None, "instructions": ""}


# ---- KG metadata dispatch (session vs project) ---------------------------

def test_kg_meta_reads_project_over_session():
    from storage.models import Project, Session
    p = Project(name="p")
    p.project_metadata = {"kg": {"nodes": 1}}
    s = Session(title="s")
    s.session_metadata = {"kg": {"nodes": 2}}
    assert docs._kg_meta(p) == {"kg": {"nodes": 1}}
    assert docs._kg_meta(s) == {"kg": {"nodes": 2}}


def test_set_kg_meta_targets_right_column():
    from storage.models import Project, Session
    p = Project(name="p")
    s = Session(title="s")
    docs._set_kg_meta(p, {"kg": "P"})
    docs._set_kg_meta(s, {"kg": "S"})
    assert p.project_metadata == {"kg": "P"}
    assert s.session_metadata == {"kg": "S"}
