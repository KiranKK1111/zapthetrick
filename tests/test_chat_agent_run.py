"""Phase 3 — chat intent router (detect.py) + /api/chat/agent-run endpoint.

Intent routing is pure/deterministic. The endpoint test fakes the DB layer
(Postgres-only ORM types) + the agent loop + the workspace resolver to assert
wiring: workspace resolved by kind, every event persisted as an ordered step,
the user + final turns saved as Messages, a `diff` + `done` frame emitted, and
the assistant turn flagged as a downloadable project zip.
"""
from __future__ import annotations

import asyncio

from app.documents.detect import detect_agentic_intent


# ── intent router ─────────────────────────────────────────────────────────
def test_intent_edit_on_archive():
    r = detect_agentic_intent("fix the login bug", has_archive=True)
    assert r["agentic"] and r["kind"] == "edit" and r["workspace_required"]


def test_intent_edit_optimize_refactor():
    for verb in ("optimize the query layer", "refactor the auth module",
                 "add pagination to /users", "migrate this to FastAPI"):
        r = detect_agentic_intent(verb, has_archive=True)
        assert r["agentic"] and r["kind"] == "edit", verb


def test_intent_build_from_spec_doc():
    r = detect_agentic_intent("build an app based on this spec",
                              has_spec_doc=True)
    assert r["agentic"] and r["kind"] == "build"


def test_intent_build_variants():
    for msg in ("create a REST api for todos",
                "scaffold a web app with auth",
                "generate a CLI tool"):
        r = detect_agentic_intent(msg, has_spec_doc=True)
        assert r["agentic"] and r["kind"] == "build", msg


def test_intent_readonly_stays_qa():
    for msg in ("explain this code", "review the codebase",
                "what does this function do", "how does the auth flow work",
                "summarize the project"):
        r = detect_agentic_intent(msg, has_archive=True)
        assert not r["agentic"], msg


def test_intent_no_workspace_no_edit():
    # An edit verb with NO archive / workspace → not agentic (nothing to edit).
    r = detect_agentic_intent("fix the bug")
    assert not r["agentic"]


def test_intent_followup_reuses_workspace():
    # Flow C: no new upload, but a workspace exists → edit routes agentically.
    r = detect_agentic_intent("now add unit tests", workspace_exists=True)
    assert r["agentic"] and r["kind"] == "edit"


def test_intent_empty():
    assert detect_agentic_intent("") == {
        "agentic": False, "kind": None, "workspace_required": False}


# ── /api/chat/agent-run — persistence path (DB + loop + workspace faked) ──
def _collect(resp) -> list[str]:
    async def go():
        out = []
        async for chunk in resp.body_iterator:
            out.append(chunk if isinstance(chunk, str) else chunk.decode())
        return out
    return asyncio.run(go())


async def _scripted(*_a, **_k):
    yield {"type": "goal_round", "round": 1, "of": 4}
    yield {"type": "thought", "text": "reading", "step": 1}
    yield {"type": "tool_call", "tool": "read", "args": {"path": "x.py"}, "step": 1}
    yield {"type": "tool_result", "tool": "read", "result": "x = 1"}
    yield {"type": "final", "message": "fixed the bug"}


def test_agent_run_persists_and_delivers(monkeypatch):
    from app.api import routes_chat_agent as rca

    steps: list[str] = []
    messages: list[str] = []
    saved_sources: list[dict] = []

    class _Obj:
        def __init__(self, i): self.id = i

    class _Sess:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def commit(self): pass
        async def rollback(self): pass

    class _SessionRepo:
        def __init__(self, s): pass
        async def get(self, sid): return _Obj(sid)
        async def record_message(self, *a, **k): pass

    class _MessageRepo:
        def __init__(self, s): pass
        async def append(self, **kw):
            messages.append(kw["role"])
            if kw.get("sources"):
                saved_sources.append(kw["sources"])
            return _Obj(f"M{len(messages)}")

    class _StepRepo:
        def __init__(self, s): pass
        async def append(self, **kw):
            steps.append(kw["event"])
            return _Obj("X")
        async def next_seq(self, sid): return (0, 0)

    monkeypatch.setattr("storage.db.get_session_factory",
                        lambda: (lambda: _Sess()))
    monkeypatch.setattr("storage.repos.SessionRepo", _SessionRepo)
    monkeypatch.setattr("storage.repos.MessageRepo", _MessageRepo)
    monkeypatch.setattr("storage.repos.AgentStepRepo", _StepRepo)
    monkeypatch.setattr(rca, "_resolve_kind", lambda body: "edit")
    monkeypatch.setattr(rca, "_resolve_workspace",
                        lambda cid, kind: ("/tmp/ws", ""))

    async def _diff(_ws): return "Changed files:\nM\tx.py"
    monkeypatch.setattr(rca, "_diff", _diff)

    # Patch the loop generator imported inside the endpoint.
    import app.agent.loop as loop
    monkeypatch.setattr(loop, "run_goal", _scripted)

    resp = asyncio.run(rca.chat_agent_run(rca.ChatAgentRunBody(
        conversation_id="11111111-1111-1111-1111-111111111111",
        task="fix the login bug", kind="edit")))
    frames = _collect(resp)
    joined = "".join(frames)

    assert frames[0].startswith("event: session")
    assert steps == ["user", "goal_round", "thought", "tool_call",
                     "tool_result", "final"]
    assert messages == ["user", "assistant"]
    assert "event: diff" in joined
    assert "event: done" in joined
    assert frames[-1].startswith("event: end")
    # Assistant turn flagged as a downloadable project zip.
    assert saved_sources and saved_sources[0]["format"] == "zip"
    assert saved_sources[0]["download"].endswith("/download")


def test_agent_run_missing_workspace_errors(monkeypatch):
    from app.api import routes_chat_agent as rca

    monkeypatch.setattr(rca, "_resolve_kind", lambda body: "edit")
    monkeypatch.setattr(
        rca, "_resolve_workspace",
        lambda cid, kind: (None, "Upload a code archive first."))

    resp = asyncio.run(rca.chat_agent_run(rca.ChatAgentRunBody(
        conversation_id="c1", task="fix the bug", kind="edit")))
    joined = "".join(_collect(resp))
    assert "event: error" in joined
    assert "Upload a code archive" in joined
    assert joined.rstrip().endswith("event: end\ndata: {}")
