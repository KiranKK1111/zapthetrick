"""Phase 11 — project brain + decision ledger + learning-lite (#18/#59/#14).

Pure/offline file-based memory over a temp workspace.
"""
from __future__ import annotations

import asyncio
import os

from app.agent_workspace import brain
from app.agent_workspace.brain import (
    brain_context,
    brain_path,
    ensure_brain,
    preferred_model,
    read_brain,
    record_decision,
    remember_run,
)


# ── brain.md lifecycle ──────────────────────────────────────────────────────
def test_ensure_brain_creates_template(tmp_path):
    p = ensure_brain(str(tmp_path), project="MyApp")
    assert os.path.isfile(p)
    assert p.endswith(os.path.join(".zapthetrick", "brain.md"))
    text = read_brain(str(tmp_path))
    assert "# Project Brain" in text
    assert "## Decisions" in text
    assert "Project: MyApp" in text


def test_pristine_brain_yields_empty_context(tmp_path):
    ensure_brain(str(tmp_path))
    # No real facts/decisions → no memory preamble.
    assert brain_context(str(tmp_path)) == ""


# ── decision ledger ─────────────────────────────────────────────────────────
def test_record_decision_appends_to_ledger(tmp_path):
    ws = str(tmp_path)
    assert record_decision(ws, "Database", "Use Postgres",
                           "pgvector + relational needs")
    assert record_decision(ws, "Auth", "JWT with refresh tokens")
    text = read_brain(ws)
    assert "**Database**" in text and "Use Postgres" in text
    assert "_Rationale:_ pgvector" in text
    assert "**Auth**" in text
    # Both entries live under the Decisions section.
    decisions = text.split("## Decisions", 1)[1]
    assert "Use Postgres" in decisions and "JWT" in decisions


def test_record_decision_requires_content(tmp_path):
    assert record_decision(str(tmp_path), "", "") is False


def test_brain_context_includes_decisions(tmp_path):
    ws = str(tmp_path)
    record_decision(ws, "Framework", "FastAPI", "async + typing")
    ctx = brain_context(ws)
    assert "PROJECT MEMORY" in ctx
    assert "FastAPI" in ctx


def test_brain_context_includes_project_facts(tmp_path):
    ensure_brain(str(tmp_path), project="Acme")
    ctx = brain_context(str(tmp_path))
    assert "PROJECT MEMORY" in ctx
    assert "Acme" in ctx


# ── learning-lite ───────────────────────────────────────────────────────────
def test_remember_and_prefer_model(tmp_path):
    ws = str(tmp_path)
    remember_run(ws, model="model-A", success=True)
    remember_run(ws, model="model-A", success=True)
    remember_run(ws, model="model-B", success=False)
    assert preferred_model(ws) == "model-A"
    assert os.path.isfile(os.path.join(ws, ".zapthetrick", "learning.json"))


def test_preferred_model_none_without_success(tmp_path):
    ws = str(tmp_path)
    remember_run(ws, model="model-X", success=False)
    assert preferred_model(ws) is None


def test_learning_hint_in_brain_context(tmp_path):
    ws = str(tmp_path)
    record_decision(ws, "X", "did a thing")
    remember_run(ws, model="qwen-coder", success=True)
    ctx = brain_context(ws)
    assert "qwen-coder" in ctx


def test_remember_run_ignores_missing_model(tmp_path):
    remember_run(str(tmp_path), model=None, success=True)
    assert preferred_model(str(tmp_path)) is None


# ── agent tool wrapper ──────────────────────────────────────────────────────
def test_record_decision_tool(tmp_path):
    from app.agent.tools import HANDLERS, SPEC_BY_NAME
    from app.agent import permissions

    assert "record_decision" in HANDLERS
    assert SPEC_BY_NAME["record_decision"].writes is True
    assert permissions.decide("record_decision", {}, "plan")[0] == "deny"

    out = asyncio.run(HANDLERS["record_decision"](
        str(tmp_path), title="Caching", decision="Use Redis",
        rationale="hot keys"))
    assert "recorded decision" in out
    assert "Use Redis" in read_brain(str(tmp_path))
