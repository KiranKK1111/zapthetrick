"""P2-2 — graded Claude-comparison eval harness.

Pure/offline: gate primitives, task-bank integrity, the async model-comparison
runner + scoreboard aggregation, the LLM judge (with a scripted judge), and the
`make_llm_runner` wiring (monkeypatched LLMClient). No provider keys needed.
"""
from __future__ import annotations

import asyncio

import pytest

from app.eval.model_eval import (
    Scoreboard,
    compare_models,
    judge_compare,
    judge_ranking,
    make_llm_runner,
    rank_models,
    run_task_bank,
)
from app.eval.scoring import (
    GradedTask,
    contains_all,
    contains_any,
    contains_none,
    grade_output,
    has_code_block,
    has_sections,
    json_parseable,
    min_words,
    regex_present,
)
from app.eval.task_bank import categories, default_task_bank, task_bank


# ── gate primitives ─────────────────────────────────────────────────────────
def test_contains_gates():
    assert contains_all("a", "b").run("x a y b").passed
    assert not contains_all("a", "z").run("a only").passed
    assert contains_any("z", "b").run("has b").passed
    assert contains_none("bad").run("all good").passed
    assert not contains_none("bad").run("this is bad").passed


def test_regex_and_code_and_words():
    assert regex_present(r"range\(1, n\+1\)").run("for i in range(1, n+1):").passed
    assert has_code_block().run("text ``` code ```").passed
    assert has_code_block(lang="python").run("```python\nx=1\n```").passed
    assert not has_code_block(lang="python").run("```js\nx\n```").passed
    assert min_words(3).run("one two three").passed
    assert not min_words(5).run("one two").passed


def test_sections_and_json_gates():
    md = "# Installation\nstuff\n## Usage\nmore"
    assert has_sections("Installation", "Usage").run(md).passed
    assert has_sections("Context", "Decision").run(
        "**Context**\nx\nDecision: y").passed
    assert json_parseable().run('prefix {"a": 1} suffix').passed
    assert json_parseable().run("```json\n[1,2,3]\n```").passed
    assert not json_parseable().run("no json here").passed


def test_gate_handles_broken_grader():
    from app.eval.scoring import Gate
    g = Gate("boom", lambda out: (_ for _ in ()).throw(ValueError("x")))
    res = g.run("anything")
    assert not res.passed and "raised" in res.detail


# ── grading ──────────────────────────────────────────────────────────────────
def test_grade_output_weighted_score():
    task = GradedTask(
        id="t", prompt="p",
        gates=[contains_all("alpha", weight=1.0),
               contains_all("bravo", weight=3.0)],
    )
    # only the weight-1 gate passes → 1/4 = 0.25
    s = grade_output(task, "has alpha only", pass_threshold=0.7)
    assert s.score == 0.25 and not s.passed
    # both pass → 1.0
    s2 = grade_output(task, "has alpha and bravo", pass_threshold=0.7)
    assert s2.score == 1.0 and s2.passed


def test_grade_output_error_fails_even_if_gates_pass():
    task = GradedTask(id="t", prompt="p", gates=[contains_all("x")])
    s = grade_output(task, "x", error="boom")
    assert s.error == "boom" and not s.passed


# ── task bank integrity ──────────────────────────────────────────────────────
def test_task_bank_loads_and_is_well_formed():
    bank = default_task_bank()
    assert len(bank) >= 30
    ids = [t.id for t in bank]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    for t in bank:
        assert t.prompt.strip()
        assert t.gates, f"{t.id} has no gates"
        assert t.difficulty in {"easy", "standard", "hard", "expert"}


def test_task_bank_filters():
    arch = task_bank(categories=["architecture"])
    assert arch and all(t.category == "architecture" for t in arch)
    hard = task_bank(difficulties=["hard"])
    assert hard and all(t.difficulty == "hard" for t in hard)
    assert "architecture" in categories()


def test_reference_answer_gates_actually_pass_on_good_answers():
    # A spot check: the sql-injection task's gates pass on a correct fix.
    bank = {t.id: t for t in default_task_bank()}
    t = bank["bugfix/sql-injection"]
    good = ("```python\n"
            "def get_user(db, name):\n"
            "    return db.execute('SELECT * FROM users WHERE name = ?', "
            "(name,)).fetchone()\n```")
    s = grade_output(t, good)
    assert s.passed, s.to_dict()


# ── async runner + scoreboard ────────────────────────────────────────────────
def _good_runner_for(bank):
    """A runner that returns an answer satisfying each task's first contains_*
    term plus a code block — enough to pass most gates (used to test plumbing)."""
    async def _run(task):
        # echo the prompt + a code block; good enough for plumbing tests
        return f"answer\n```python\ncode\n```\n{task.prompt}"
    return _run


def test_run_task_bank_aggregates():
    tasks = [
        GradedTask(id="a", prompt="p", category="x", difficulty="easy",
                   gates=[contains_all("yes")]),
        GradedTask(id="b", prompt="p", category="x", difficulty="hard",
                   gates=[contains_all("affirmative")]),
    ]

    async def runner(task):
        return "yes" if task.id == "a" else "negative"

    board = asyncio.run(run_task_bank(tasks, runner, model="m"))
    assert isinstance(board, Scoreboard)
    assert board.total == 2 and board.passed == 1
    assert board.pass_rate == 0.5
    bc = board.by_category()["x"]
    assert bc["count"] == 2 and bc["passed"] == 1
    bd = board.by_difficulty()
    assert bd["easy"]["passed"] == 1 and bd["hard"]["passed"] == 0
    d = board.to_dict()
    assert d["model"] == "m" and "tasks" in d


def test_run_task_bank_runner_exception_is_caught():
    tasks = [GradedTask(id="a", prompt="p", gates=[contains_all("x")])]

    async def boom(task):
        raise RuntimeError("model down")

    board = asyncio.run(run_task_bank(tasks, boom))
    assert board.passed == 0
    assert "RuntimeError" in board.scores[0].error


def test_compare_models_and_rank():
    tasks = [GradedTask(id="a", prompt="p", gates=[contains_all("win")])]

    async def good(task):
        return "win"

    async def bad(task):
        return "lose"

    boards = asyncio.run(compare_models(tasks, {"good": good, "bad": bad}))
    board = rank_models(boards)
    assert board[0]["model"] == "good" and board[0]["rank"] == 1
    assert board[1]["model"] == "bad"


# ── LLM judge ────────────────────────────────────────────────────────────────
def test_judge_ranking_parses_json_verdict():
    task = GradedTask(id="t", prompt="explain X", gates=[])

    async def judge(prompt):
        assert "explain X" in prompt
        return '{"ranking": ["B", "A"], "winner": "B", "reason": "clearer"}'

    v = asyncio.run(judge_ranking(task, {"A": "ans a", "B": "ans b"}, judge))
    assert v["winner"] == "B" and v["ranking"] == ["B", "A"]
    assert v["reason"] == "clearer"


def test_judge_ranking_degrades_on_garbage():
    task = GradedTask(id="t", prompt="p", gates=[])

    async def judge(prompt):
        return "I think A is the best one honestly"

    v = asyncio.run(judge_ranking(task, {"A": "x", "B": "y"}, judge))
    assert v["winner"] == "A"


def test_judge_compare_tallies_wins():
    tasks = [
        GradedTask(id="t1", prompt="p", gates=[]),
        GradedTask(id="t2", prompt="p", gates=[]),
    ]
    boards = {
        "t1": {"A": "a1", "B": "b1"},
        "t2": {"A": "a2", "B": "b2"},
    }

    async def judge(prompt):
        return '{"ranking": ["A","B"], "winner": "A", "reason": "x"}'

    out = asyncio.run(judge_compare(tasks, boards, judge))
    assert out["wins"]["A"] == 2
    assert len(out["verdicts"]) == 2


# ── real-runner wiring (monkeypatched client) ────────────────────────────────
def test_make_llm_runner_calls_client(monkeypatch):
    captured = {}

    async def fake_complete(messages, model=None, options=None):
        captured["messages"] = messages
        captured["options"] = options
        return "routed answer"

    import app.core.llm_client as lc
    monkeypatch.setattr(lc.llm, "complete", fake_complete)

    runner = make_llm_runner()
    task = GradedTask(id="t", prompt="do it", difficulty="hard",
                      system="be terse", gates=[contains_all("routed")])
    out = asyncio.run(runner(task))
    assert out == "routed answer"
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][-1]["content"] == "do it"
    assert captured["options"]["difficulty"] == "hard"
