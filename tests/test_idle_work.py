"""Idle-time background work (roadmap Phase 5 #10).

Pins: idle jobs run serially and best-effort; a failing job never sinks its
siblings; work is shed under memory pressure; and everything is a no-op when the
feature is disabled.
"""
from __future__ import annotations

import asyncio

from app.perceived.idle import (EMBED, SUMMARIZE, VERIFY, IdleJob, IdleScheduler)


class _Mem:
    def __init__(self, shed):
        self._shed = shed

    def sheds_background(self):
        return self._shed


def _run(sched, **kw):
    return asyncio.run(sched.run_once(**kw))


def test_runs_all_jobs_serially():
    log = []
    s = IdleScheduler(memory=_Mem(False))

    async def summarize():
        log.append("s")

    def embed():                       # sync job supported too
        log.append("e")

    s.register(IdleJob("summarize", summarize, SUMMARIZE))
    s.register(IdleJob("embed", embed, EMBED))
    report = _run(s)
    assert report.ran == ["summarize", "embed"]
    assert log == ["s", "e"]


def test_failing_job_isolated():
    s = IdleScheduler(memory=_Mem(False))

    async def ok():
        return None

    async def bad():
        raise RuntimeError("boom")

    s.register(IdleJob("bad", bad, VERIFY))
    s.register(IdleJob("ok", ok, SUMMARIZE))
    report = _run(s)
    assert "bad" in report.failed
    assert "ok" in report.ran            # sibling still ran


def test_shed_under_memory_pressure():
    ran = []
    s = IdleScheduler(memory=_Mem(True))   # sheds background
    s.register(IdleJob("x", lambda: ran.append("x")))
    report = _run(s)
    assert report.shed_for_pressure
    assert ran == [] and report.ran == []


def test_disabled_is_noop(monkeypatch):
    import app.perceived.idle as idle
    monkeypatch.setattr(idle, "idle_enabled", lambda: False)
    ran = []
    s = IdleScheduler(memory=_Mem(False))
    s.register(IdleJob("x", lambda: ran.append("x")))
    report = _run(s)
    assert ran == [] and report.ran == []


def test_timeout_marks_failed_not_hang():
    s = IdleScheduler(memory=_Mem(False))

    async def slow():
        await asyncio.sleep(10)

    s.register(IdleJob("slow", slow, SUMMARIZE, max_ms=20))
    report = _run(s)
    assert "slow" in report.failed


def test_spawn_returns_none_without_loop():
    s = IdleScheduler(memory=_Mem(False))
    s.register(IdleJob("x", lambda: None))
    assert s.spawn() is None              # no running loop → no task, no crash
