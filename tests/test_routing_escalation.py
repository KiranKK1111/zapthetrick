"""Confidence-based escalation (intelligent-model-routing R5, task 6.3).

Pins Property 5: escalate on low confidence / failed step, stop when the
threshold is met, and disabled → a single generation.
"""
from __future__ import annotations

import asyncio

from app.llm.escalation import run_with_escalation, escalation_chain


def _run(chain, gen, conf, **kw):
    return asyncio.run(run_with_escalation(chain, gen, conf, **kw))


def test_stops_when_threshold_met_on_first_step():
    async def gen(step):
        return f"answer-{step}"
    res = _run(["fast", "strong"], gen, lambda a: 0.9, threshold=0.6)
    assert res.steps_used == 1 and not res.escalated
    assert res.answer == "answer-fast"


def test_escalates_when_low_confidence():
    # fast model low, strong model high → escalates once, keeps the strong answer.
    scores = {"answer-fast": 0.3, "answer-strong": 0.85}

    async def gen(step):
        return f"answer-{step}"
    res = _run(["fast", "strong"], gen, lambda a: scores[a], threshold=0.6)
    assert res.steps_used == 2 and res.escalated
    assert res.answer == "answer-strong" and res.confidence == 0.85


def test_failed_step_escalates():
    async def gen(step):
        if step == "fast":
            raise RuntimeError("provider down")
        return "recovered"
    res = _run(["fast", "strong"], gen, lambda a: 0.9, threshold=0.6)
    assert res.answer == "recovered" and res.escalated


def test_disabled_is_single_generation():
    async def gen(step):
        return f"answer-{step}"
    res = _run(["fast", "strong"], gen, lambda a: 0.1, threshold=0.6,
               enabled=False)
    assert res.steps_used == 1 and not res.escalated


def test_keeps_best_when_none_meet_threshold():
    scores = {"answer-fast": 0.3, "answer-strong": 0.5}

    async def gen(step):
        return f"answer-{step}"
    res = _run(["fast", "strong"], gen, lambda a: scores[a], threshold=0.9)
    assert res.steps_used == 2
    assert res.answer == "answer-strong"      # best of the two


def test_escalation_chain_shape():
    assert escalation_chain("trivial")[0] == "trivial"
    assert escalation_chain("expert") == ["expert"]
    assert escalation_chain("standard") == ["standard", "hard"]
