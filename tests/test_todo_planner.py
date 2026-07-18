"""P2-4 — long-horizon + live TODO checklist (TodoWrite parity).

Pure/offline: todo normalization/persistence/summary, the planner (with a
scripted completer + fallback), the loop's `todo_write` → `todo` event +
persistence, run_goal injecting the checklist across rounds, and the RunMetrics
todo counters.
"""
from __future__ import annotations

import asyncio

from app.agent.planner import plan_todos
from app.agent.todos import (
    COMPLETED,
    IN_PROGRESS,
    PENDING,
    Todo,
    clear_todos,
    load_todos,
    normalize_todos,
    progress,
    save_todos,
    todos_summary,
)
from app.obs.metrics import RunMetrics


# ── normalization ─────────────────────────────────────────────────────────
def test_normalize_accepts_dicts_and_strings():
    raw = [
        {"content": "Step one", "status": "completed", "activeForm": "Doing one"},
        "Step two",
        {"task": "Step three", "status": "in_progress"},
    ]
    todos = normalize_todos(raw)
    assert [t.content for t in todos] == ["Step one", "Step two", "Step three"]
    assert todos[0].status == COMPLETED
    assert todos[1].status == PENDING
    assert todos[2].status == IN_PROGRESS


def test_normalize_enforces_single_in_progress():
    raw = [
        {"content": "a", "status": "in_progress"},
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "in_progress"},
    ]
    todos = normalize_todos(raw)
    active = [t for t in todos if t.status == IN_PROGRESS]
    assert len(active) == 1 and active[0].content == "a"


def test_normalize_bad_status_and_junk():
    todos = normalize_todos([{"content": "x", "status": "bogus"}, 42, {}, ""])
    assert len(todos) == 1 and todos[0].status == PENDING


def test_normalize_non_list():
    assert normalize_todos("nope") == []


# ── persistence + summary ───────────────────────────────────────────────────
def test_save_load_clear_roundtrip(tmp_path):
    ws = str(tmp_path)
    todos = [Todo("a", COMPLETED), Todo("b", IN_PROGRESS, "Doing b"),
             Todo("c")]
    assert save_todos(ws, todos)
    loaded = load_todos(ws)
    assert [t.content for t in loaded] == ["a", "b", "c"]
    assert progress(loaded) == (1, 3)
    clear_todos(ws)
    assert load_todos(ws) == []


def test_todos_summary_renders_marks():
    todos = [Todo("done it", COMPLETED), Todo("in flight", IN_PROGRESS,
                                              "Flying"), Todo("later")]
    s = todos_summary(todos)
    assert "TASK CHECKLIST (1/3 done" in s
    assert "[x] done it" in s
    assert "[~] Flying" in s     # active_form shown while in_progress
    assert "[ ] later" in s


def test_todos_summary_empty():
    assert todos_summary([]) == ""


# ── planner ──────────────────────────────────────────────────────────────────
def test_plan_todos_parses_model_json():
    async def completer(messages, options):
        assert "TASK:" in messages[0]["content"]
        return ('[{"content": "Set up routes", "activeForm": "Setting up routes"},'
                ' {"content": "Add tests", "activeForm": "Adding tests"}]')

    todos = asyncio.run(plan_todos("build an API", completer=completer))
    assert [t.content for t in todos] == ["Set up routes", "Add tests"]
    assert todos[0].active_form == "Setting up routes"


def test_plan_todos_fallback_on_garbage():
    async def completer(messages, options):
        return "I cannot make a plan, sorry."

    todos = asyncio.run(plan_todos("do the thing", completer=completer))
    assert len(todos) == 1 and "do the thing" in todos[0].content


def test_plan_todos_fallback_on_exception():
    async def completer(messages, options):
        raise RuntimeError("model down")

    todos = asyncio.run(plan_todos("fix the bug", completer=completer))
    assert len(todos) == 1 and "fix the bug" in todos[0].content


def test_plan_todos_empty_task():
    async def completer(messages, options):
        return "[]"
    assert asyncio.run(plan_todos("", completer=completer)) == []


def test_looks_multistep_heuristic():
    from app.agent.planner import looks_multistep
    # simple one-liners stay cheap (no plan)
    assert not looks_multistep("fix the typo in hi()")
    assert not looks_multistep("rename foo to bar")
    # long or multi-step tasks get a plan
    assert looks_multistep("build a REST API with auth, pagination and tests")
    assert looks_multistep("refactor the parser")
    assert looks_multistep("add the endpoint and then write tests")
    assert not looks_multistep("")


# ── loop: todo_write emits a structured `todo` event + persists ──────────────
def _drain(agen):
    async def go():
        return [e async for e in agen]
    return asyncio.run(go())


def test_loop_todo_write_emits_event_and_persists(tmp_path, monkeypatch):
    import app.core.llm_client as lc
    from app.agent import loop

    ws = str(tmp_path)
    replies = iter([
        '{"thought": "plan it", "tool": "todo_write", "args": {"todos": ['
        '{"content": "step A", "status": "in_progress", "activeForm": "Doing A"},'
        '{"content": "step B", "status": "pending"}]}}',
        '{"tool": "final", "args": {"message": "done"}}',
    ])

    async def fake_complete(messages, model=None, options=None):
        return next(replies)

    monkeypatch.setattr(lc.llm, "complete", fake_complete)

    events = _drain(loop.run_agent("do stuff", workspace=ws, mode="acceptEdits"))
    todo_evts = [e for e in events if e["type"] == "todo"]
    assert todo_evts, "expected a todo event"
    te = todo_evts[0]
    assert te["total"] == 2 and te["done"] == 0
    assert te["todos"][0]["content"] == "step A"
    assert te["todos"][0]["status"] == "in_progress"
    # persisted to the workspace
    assert [t.content for t in load_todos(ws)] == ["step A", "step B"]


# ── metrics: todo counters ────────────────────────────────────────────────────
def test_metrics_tracks_todo_progress():
    m = RunMetrics()
    m.on_event({"type": "todo", "total": 3, "done": 0, "todos": []})
    m.on_event({"type": "todo", "total": 3, "done": 2, "todos": []})
    m.finalize()
    d = m.to_dict()
    assert d["todos_total"] == 3 and d["todos_completed"] == 2


def test_metrics_todo_counts_from_list_without_done():
    m = RunMetrics()
    m.on_event({"type": "todo", "todos": [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "pending"}]})
    assert m.to_dict()["todos_total"] == 2
    assert m.to_dict()["todos_completed"] == 1
