"""Phase 13 — optional polish: semantic (AST) diff (#106), dependency-impact
tool (#63), and the distributed-systems / reliability domain (#42/#43).

Offline / deterministic (Python AST always available; tree-sitter for the
graph). LLM-backed pieces are faked.
"""
from __future__ import annotations

import asyncio

from app.codegraph.semantic_diff import (
    has_changes,
    semantic_diff,
    summarize_semantic_diff,
)


# ── semantic diff (#106) ────────────────────────────────────────────────────
OLD = (
    "def login(user):\n    return 1\n\n"
    "def legacy():\n    return 0\n\n"
    "class User:\n    def charge(self, amount):\n        return amount\n"
)
NEW = (
    "def login(user, remember=False):\n    return 1\n\n"
    "class User:\n    def charge(self, amount, currency):\n        return amount\n"
    "    def deactivate(self):\n        return True\n"
)


def test_semantic_diff_added_removed_changed():
    d = semantic_diff(OLD, NEW, path="m.py")
    assert "User.deactivate" in d["added"]
    assert "legacy" in d["removed"]
    changed = {c["symbol"] for c in d["changed"]}
    assert "login" in changed            # gained a parameter
    assert "User.charge" in changed      # gained a parameter
    assert has_changes(d)


def test_semantic_diff_no_change_on_formatting():
    # Same symbols + signatures, only comments/whitespace differ → no semantic change.
    a = "def f(x):\n    return x\n"
    b = "# a comment\ndef f(x):\n\n    return x  # inline\n"
    d = semantic_diff(a, b, path="m.py")
    assert not has_changes(d)
    assert summarize_semantic_diff(d) == ""


def test_summarize_semantic_diff_text():
    d = semantic_diff(OLD, NEW, path="m.py")
    s = summarize_semantic_diff(d, path="m.py")
    assert s.startswith("m.py:")
    assert "added" in s and "removed" in s and "changed signature" in s


# ── dependency-impact tool (#63) ────────────────────────────────────────────
def test_impact_of_tool(tmp_path):
    from app.agent.tools import HANDLERS, SPEC_BY_NAME

    (tmp_path / "lib.py").write_text(
        "def helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "from lib import helper\n\n"
        "def run():\n    return helper()\n\n"
        "def go():\n    return run()\n",
        encoding="utf-8")

    assert SPEC_BY_NAME["impact_of"].writes is False
    out = asyncio.run(HANDLERS["impact_of"](str(tmp_path), symbol="helper"))
    assert "helper" in out
    # `run` calls helper → it's an impacted caller.
    assert "run" in out


def test_impact_of_unknown_symbol(tmp_path):
    from app.agent.tools import HANDLERS

    (tmp_path / "a.py").write_text("def x():\n    return 1\n", encoding="utf-8")
    out = asyncio.run(HANDLERS["impact_of"](str(tmp_path), symbol="nope"))
    assert "not found" in out


def test_impact_tool_registered_and_readonly():
    from app.agent import permissions
    from app.agent.tools import HANDLERS, tools_doc

    assert "impact_of" in HANDLERS
    assert "impact_of" in tools_doc()
    # read-only → allowed even in plan mode.
    assert permissions.decide("impact_of", {}, "plan")[0] == "allow"


# ── distributed / reliability domain (#42/#43) ──────────────────────────────
def test_distributed_domain_classification():
    from app.technical_pipeline.dispatcher import DOMAINS, classify_domain

    assert "distributed" in DOMAINS
    assert classify_domain("explain raft consensus and quorum") == "distributed"
    assert classify_domain("design an idempotent exactly-once consumer") == \
        "distributed"
    assert classify_domain("add a circuit breaker with back-pressure") == \
        "distributed"


def test_distributed_domain_runs_structured(monkeypatch):
    from app.technical_pipeline import distributed, structured

    full = (
        "## Consistency & Consensus\nUse Raft for quorum-based linearizability.\n"
        "## Partitioning & Replication\nShard by key; leader/follower replication.\n"
        "## Failure Handling & Reliability\nRetries with timeout + circuit breaker; "
        "idempotent writes.\n"
        "## Coordination & Messaging\nKafka with an outbox; exactly-once.\n"
        "## Back-pressure & Flow Control\nBulkheads + rate limit + load shed.\n"
        "## Assumptions\nWe assume multi-region.\n"
        "## Recommended Pattern(s)\nThe pattern is leaderless replication.\n"
        "## Trade-offs\nPros and cons vs a single-leader alternative.\n"
        "## Governance & Operability\nObservability, cost, security.\n"
    )

    async def _complete(messages, **kw):
        return full
    monkeypatch.setattr(structured.llm, "complete", _complete)

    async def go():
        return [e async for e in distributed.run("design a distributed queue")]
    evts = asyncio.run(go())
    assert evts[0]["data"]["domain"] == "distributed"
    assert any(e["kind"] == "markdown" for e in evts)
    assert evts[-1]["data"]["missing"] == []   # all sections present → no gaps


# ── endpoint emits semantic_diff frame ──────────────────────────────────────
def test_agent_run_emits_semantic_diff(monkeypatch):
    from app.api import routes_chat_agent as rca
    from app.chat import redteam as rt
    from app.chat import council as cc

    monkeypatch.setattr(rca, "_resolve_kind", lambda b: "edit")
    monkeypatch.setattr(rca, "_resolve_workspace", lambda c, k: ("/tmp/ws", ""))
    monkeypatch.setattr("storage.db.get_session_factory", lambda: None)

    async def _diff(_ws):
        return "Changed files:\nM\tm.py"
    monkeypatch.setattr(rca, "_diff", _diff)

    async def _sem(_ws):
        return ["m.py: added `User.deactivate`"]
    monkeypatch.setattr(rca, "_semantic", _sem)

    async def _review(*a, **k):
        return []
    monkeypatch.setattr(rt, "red_team_review", _review)

    async def _verify(*a, **k):
        return cc.CouncilVerdict()
    monkeypatch.setattr(cc, "cross_model_verify", _verify)

    import app.agent.loop as loop

    async def _scripted(*_a, **_k):
        yield {"type": "final", "message": "done"}
    monkeypatch.setattr(loop, "run_goal", _scripted)

    resp = asyncio.run(rca.chat_agent_run(rca.ChatAgentRunBody(
        conversation_id="c1", task="add a method", kind="edit")))

    async def collect():
        return "".join([c if isinstance(c, str) else c.decode()
                        async for c in resp.body_iterator])
    joined = asyncio.run(collect())
    assert "event: semantic_diff" in joined
    assert "User.deactivate" in joined
