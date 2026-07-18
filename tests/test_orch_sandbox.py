"""Execution sandbox (agent-orchestration R4, task 6.2).

Pins Property 4: in-sandbox execution, honest status (failure → repair, never
false-verified), and disabled/unavailable marking.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.orchestration.sandbox import run_code


@dataclass
class _FakeRun:
    exit_code: int
    timed_out: bool = False
    denied: bool = False

    @property
    def ok(self):
        return self.exit_code == 0 and not self.timed_out and not self.denied

    def summary(self):
        return f"[exit {self.exit_code}]"


def test_disabled_marks_status_not_verified():
    # force=False + flag off → disabled, never verified.
    res = asyncio.run(run_code("ws", "pytest", force=False))
    assert res.status == "disabled" and res.verified is False


def test_success_is_verified():
    async def runner(cmd, wid):
        return _FakeRun(exit_code=0)
    res = asyncio.run(run_code("ws", "pytest -q", runner=runner, force=True))
    assert res.ran and res.verified and res.status == "verified"


def test_failure_feeds_repair_never_verified():
    async def runner(cmd, wid):
        return _FakeRun(exit_code=1)
    res = asyncio.run(run_code("ws", "pytest -q", runner=runner, force=True))
    assert res.ran and not res.verified
    assert res.status == "failed" and res.repair_feedback


def test_empty_command_unavailable():
    res = asyncio.run(run_code("ws", "   ", force=True))
    assert res.status == "unavailable" and not res.verified


def test_runner_exception_is_unavailable():
    async def runner(cmd, wid):
        raise RuntimeError("boom")
    res = asyncio.run(run_code("ws", "pytest", runner=runner, force=True))
    assert res.status == "unavailable" and not res.verified
