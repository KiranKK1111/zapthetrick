"""Phase 14 — end-to-end integration of the chat-builder vertical.

Exercises the REAL pipeline offline — materialize → git baseline → the actual
agent loop (`run_agent`) driving the actual workspace tools (edit/write/verify)
→ diff → package — with ONLY the LLM scripted (so it's deterministic and needs
no provider keys). Covers both flows:
  • Flow B: fix an uploaded codebase.
  • Flow A: build a project from scratch + verify it.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import zipfile

from app.agent.loop import run_agent
from app.agent_workspace import (
    git_init_baseline,
    materialize_archive,
    package_workspace,
    workspace_path,
)
import app.core.llm_client as llm_mod


def _ws_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ZAPTHETRICK_WS_ROOT", str(tmp_path / "ws"))


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _script(monkeypatch, replies: list[str]):
    """Patch the LLM singleton to emit `replies` (one JSON action per loop step,
    last reply repeats)."""
    state = {"i": 0}

    async def _complete(messages, model=None, options=None):
        i = state["i"]
        state["i"] += 1
        return replies[min(i, len(replies) - 1)]

    monkeypatch.setattr(llm_mod.llm, "complete", _complete)


def _drive(task: str, workspace: str, *, mode: str = "acceptEdits") -> list[dict]:
    async def go():
        return [e async for e in run_agent(task, workspace=workspace, mode=mode)]
    return asyncio.run(go())


def _venv_on_path(monkeypatch):
    monkeypatch.setenv(
        "PATH", os.path.dirname(sys.executable) + os.pathsep
        + os.environ.get("PATH", ""))


# ── Flow B: fix an uploaded codebase ────────────────────────────────────────
def test_e2e_fix_uploaded_codebase(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    cid = "e2e-fix"
    res = materialize_archive(
        cid, _zip({"calc.py": b"def add(a, b):\n    return a - b\n"}),
        "buggy.zip")
    assert res.ok
    ws = workspace_path(cid)
    asyncio.run(git_init_baseline(ws))   # best-effort baseline

    _script(monkeypatch, [
        '{"thought":"the add() uses minus","tool":"read","args":{"path":"calc.py"}}',
        '{"thought":"fix it","tool":"edit","args":{"path":"calc.py",'
        '"old":"return a - b","new":"return a + b"}}',
        '{"tool":"final","args":{"message":"Fixed add() to use +."}}',
    ])
    events = _drive("fix the bug in add()", ws)

    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds and "final" in kinds
    # The real edit tool actually changed the file on disk.
    with open(os.path.join(ws, "calc.py"), encoding="utf-8") as f:
        assert "return a + b" in f.read()
    # The fix survives the round-trip into the downloadable project zip.
    zbytes = package_workspace(cid)
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        assert "calc.py" in z.namelist()
        assert b"a + b" in z.read("calc.py")


# ── Flow A: build a project from scratch + verify it ────────────────────────
def test_e2e_build_and_verify(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    _venv_on_path(monkeypatch)             # so the `verify` tool resolves pytest
    cid = "e2e-build"
    from app.agent_workspace import fresh_workspace
    ws = fresh_workspace(cid, reset=True)

    _script(monkeypatch, [
        '{"thought":"create the package marker","tool":"write",'
        '"args":{"path":"requirements.txt","content":""}}',
        '{"thought":"add a module","tool":"write","args":{"path":"mathx.py",'
        '"content":"def mul(a, b):\\n    return a * b\\n"}}',
        '{"thought":"add a test","tool":"write","args":{"path":"test_mathx.py",'
        '"content":"from mathx import mul\\n\\ndef test_mul():\\n    assert mul(2,3)==6\\n"}}',
        '{"thought":"verify","tool":"verify","args":{"steps":["test"]}}',
        '{"tool":"final","args":{"message":"Built and tested mathx.mul."}}',
    ])
    events = _drive("build a tiny math module with a passing test", ws)

    # Files were really written by the write tool.
    assert os.path.isfile(os.path.join(ws, "mathx.py"))
    assert os.path.isfile(os.path.join(ws, "test_mathx.py"))
    # The verify tool actually ran pytest in the workspace and it passed.
    verify_results = [
        e for e in events
        if e["type"] == "tool_result" and e.get("tool") == "verify"]
    assert verify_results, "verify tool did not run"
    assert "PASS" in verify_results[0]["result"]
    # Packaged project contains the generated files.
    with zipfile.ZipFile(io.BytesIO(package_workspace(cid))) as z:
        names = set(z.namelist())
        assert {"mathx.py", "test_mathx.py"} <= names


# ── the chat agent-run endpoint, end to end over a materialized workspace ───
def test_e2e_endpoint_over_real_workspace(monkeypatch, tmp_path):
    """The /api/chat/agent-run endpoint (no-DB path) driving the REAL loop +
    tools over a REAL materialized workspace, LLM scripted."""
    _ws_env(monkeypatch, tmp_path)
    from app.api import routes_chat_agent as rca

    cid = "e2e-ep"
    materialize_archive(
        cid, _zip({"greet.py": b"def hi():\n    return 'helo'\n"}), "p.zip")
    asyncio.run(git_init_baseline(workspace_path(cid)))

    monkeypatch.setattr("storage.db.get_session_factory", lambda: None)
    _script(monkeypatch, [
        '{"tool":"edit","args":{"path":"greet.py","old":"helo","new":"hello"}}',
        '{"tool":"final","args":{"message":"Fixed the typo."}}',
    ])
    # Keep the trust/council passes from making real LLM calls.
    from app.chat import council as cc, redteam as rt

    async def _no_review(*a, **k):
        return []
    async def _no_verdict(*a, **k):
        return cc.CouncilVerdict()
    monkeypatch.setattr(rt, "red_team_review", _no_review)
    monkeypatch.setattr(cc, "cross_model_verify", _no_verdict)

    resp = asyncio.run(rca.chat_agent_run(rca.ChatAgentRunBody(
        conversation_id=cid, task="fix the typo in hi()", kind="edit")))

    async def collect():
        return "".join([c if isinstance(c, str) else c.decode()
                        async for c in resp.body_iterator])
    joined = asyncio.run(collect())

    assert "event: tool_call" in joined
    assert "event: final" in joined
    assert "event: metrics" in joined
    assert "event: end" in joined
    with open(os.path.join(workspace_path(cid), "greet.py"), encoding="utf-8") as f:
        assert "hello" in f.read()
