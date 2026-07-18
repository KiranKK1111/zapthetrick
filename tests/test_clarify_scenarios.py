"""Phase 3 — deterministic scenario guards in app/agents/clarifier.

The scenario PLAYBOOK lives in the LLM prompt (not unit-testable), but the two
deterministic guards — the sample-popup fast-path and the risky-operation
confirmation — are pure and tested here.
"""
from __future__ import annotations

import pytest

_mod = pytest.importorskip("app.agents.clarifier")
is_sample = _mod.is_sample_popup_request
is_risky = _mod.is_risky_operation_request
_risky_payload = _mod._risky_payload


# ---- risky-operation detection (R15.1) ------------------------------------

@pytest.mark.parametrize("text", [
    "delete all customer data",
    "drop table users",
    "truncate table orders",
    "rm -rf /var/www",
    "wipe the production database",
    "purge every record in the accounts table",
    "format the drive",
])
def test_risky_detects_destructive_actions(text):
    assert is_risky(text) is True


@pytest.mark.parametrize("text", [
    "write a function to delete a node from a linked list",
    "how do I delete a file in python",
    "explain how to drop a table in SQL",
    "generate a query that removes duplicate records",
    "build me a todo app",
    "what is the difference between delete and truncate",
])
def test_risky_ignores_code_and_explanation_requests(text):
    assert is_risky(text) is False


def test_risky_payload_shape():
    qs, meta = _risky_payload()
    assert len(qs) == 1
    q = qs[0]
    assert q["kind"] == "single"
    labels = [o["label"] for o in q["options"]]
    assert len(labels) == 3
    recs = [o for o in q["options"] if o["recommended"]]
    assert len(recs) == 1                       # exactly one recommended
    assert meta["blocking"] is True
    assert meta["confidence"] < 0.4
    assert all(o.get("id") for o in q["options"])


# ---- sample popup detection (R21) -----------------------------------------

@pytest.mark.parametrize("text", [
    "can you provide a sample user question popup",
    "show me an example clarification popup",
    "give a demo question card",
])
def test_sample_detects(text):
    assert is_sample(text) is True


def test_sample_ignores_real_tasks():
    assert is_sample("build me a sample CRM app") is False
