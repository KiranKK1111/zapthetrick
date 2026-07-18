"""Phase 4 #9/#15 — runtime transaction/checkpoint persistence.

The DAG executor's checkpoint round-trips through a per-workspace sidecar,
wiring `orchestration/state.py::save_state`/`load_state`. Resume from a saved
checkpoint skips already-DONE work.
"""
from __future__ import annotations

import tempfile

from app.orchestration import checkpoint as ckpt
from app.orchestration.state import AgentState


def test_checkpoint_roundtrip_and_scope():
    with tempfile.TemporaryDirectory() as ws:
        scope = ckpt.scope_for(ws)
        st = AgentState(goal="build X", scope=scope)
        st.set_tasks(["a", "b", "c"])
        st.mark_done(0, "did a")
        assert ckpt.save_checkpoint(ws, st)

        loaded = ckpt.load_checkpoint(ws, scope)
        assert loaded is not None
        assert loaded.goal == "build X"
        assert loaded.is_done(0) and not loaded.is_done(1)
        assert loaded.tasks[0].output == "did a"
        # other scope isn't returned
        assert ckpt.load_checkpoint(ws, "workspace:other") is None


def test_clear_checkpoint():
    with tempfile.TemporaryDirectory() as ws:
        scope = ckpt.scope_for(ws)
        st = AgentState(goal="g", scope=scope)
        st.set_tasks(["a"])
        ckpt.save_checkpoint(ws, st)
        ckpt.clear_checkpoint(ws, scope)
        assert ckpt.load_checkpoint(ws, scope) is None


def test_load_missing_is_none_failopen():
    assert ckpt.load_checkpoint("/nonexistent/path/xyz", "workspace:x") is None
