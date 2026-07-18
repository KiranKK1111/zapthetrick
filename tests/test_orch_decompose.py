"""Request decomposition (agent-orchestration R1, task 1.2).

Pins Property 1: multi-goal split with deps; a single simple goal → [] (existing
single path).
"""
from __future__ import annotations

from app.orchestration.decompose import decompose, SubTask


def test_single_simple_goal_not_decomposed():
    assert decompose("fix the login bug") == []
    assert decompose("what is a hashmap?") == []
    assert decompose("") == []


def test_enumerated_goals_split():
    req = ("1. review the repo architecture\n"
           "2. write a migration plan\n"
           "3. generate the new code")
    subs = decompose(req)
    assert len(subs) == 3
    assert all(isinstance(s, SubTask) for s in subs)
    assert "review" in subs[0].text.lower()


def test_sequential_connectors_split():
    subs = decompose("analyze the codebase and then write unit tests")
    assert len(subs) >= 2


def test_dependencies_present_for_dependent_step():
    subs = decompose("1. build the API\n2. document it")
    # "document it" references prior work → depends on task 0.
    doc = subs[1]
    assert 0 in doc.deps


def test_capped_to_max(monkeypatch):
    import sys
    D = sys.modules["app.orchestration.decompose"]
    monkeypatch.setattr(D, "_cfg_max", lambda: 2)
    req = "\n".join(f"{i}. write module {i}" for i in range(1, 6))
    subs = decompose(req)
    assert len(subs) == 2
