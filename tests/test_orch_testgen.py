"""Automatic test generation (agent-orchestration R5, task 7.2).

Pins Property 5: generate+run+report, confidence signal, disabled skip.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.orchestration.tests_gen import generate_and_run


@dataclass
class _FakeRun:
    exit_code: int

    @property
    def ok(self):
        return self.exit_code == 0

    def summary(self):
        return f"[exit {self.exit_code}]"


async def _gen_ok(change):
    return "def test_x():\n    assert add(2,2)==4\n"


async def _gen_empty(change):
    return ""


def test_disabled_skips():
    res = asyncio.run(generate_and_run("ws", "diff", gen_fn=_gen_ok,
                                       test_cmd="pytest", force=False))
    assert res.status == "skipped" and res.confidence_delta == 0.0


def test_generate_run_pass_feeds_positive_confidence():
    async def runner(cmd, wid):
        return _FakeRun(0)
    res = asyncio.run(generate_and_run(
        "ws", "diff", gen_fn=_gen_ok, test_cmd="pytest -q",
        runner=runner, force=True))
    assert res.generated and res.ran and res.passed
    assert res.confidence_delta > 0


def test_generate_run_fail_feeds_negative_confidence():
    async def runner(cmd, wid):
        return _FakeRun(1)
    res = asyncio.run(generate_and_run(
        "ws", "diff", gen_fn=_gen_ok, test_cmd="pytest -q",
        runner=runner, force=True))
    assert res.generated and res.ran and not res.passed
    assert res.confidence_delta < 0


def test_no_generator_skips():
    res = asyncio.run(generate_and_run("ws", "diff", gen_fn=None, force=True))
    assert res.status == "skipped"


def test_empty_tests_reported_no_tests():
    res = asyncio.run(generate_and_run("ws", "diff", gen_fn=_gen_empty,
                                       test_cmd="pytest", force=True))
    assert res.status == "no_tests" and not res.passed
