"""Resource scheduler — memory-pressure controller (roadmap Phase 5 #12).

Pins the shed behaviour: under high/critical pressure the optional lanes (P1/P2)
are shed so the P0 real-time path keeps headroom; fail-open when the probe can't
read memory.
"""
from __future__ import annotations

import asyncio

from app.blackboard.memory_pressure import (CRITICAL, HIGH, OK, MemoryPressure,
                                            Thresholds)
from app.blackboard import scheduler as sched

P0, P1, P2 = sched.P0, sched.P1, sched.P2


def _mp(frac):
    return MemoryPressure(sample=lambda: frac, ttl_s=0.0,
                          thresholds=Thresholds(0.75, 0.85, 0.93))


def test_levels_bucket_correctly():
    assert _mp(0.10).level() == OK
    assert _mp(0.80).level() == "elevated"
    assert _mp(0.88).level() == HIGH
    assert _mp(0.99).level() == CRITICAL


def test_admits_sheds_optional_lanes_under_pressure():
    high = _mp(0.90)                 # HIGH → shed P2, keep P0/P1
    assert high.admits(P0) and high.admits(P1)
    assert not high.admits(P2)

    crit = _mp(0.99)                 # CRITICAL → only P0
    assert crit.admits(P0)
    assert not crit.admits(P1) and not crit.admits(P2)


def test_ok_admits_everything():
    ok = _mp(0.10)
    assert ok.admits(P0) and ok.admits(P1) and ok.admits(P2)


def test_failopen_when_probe_returns_none():
    unknown = MemoryPressure(sample=lambda: None, ttl_s=0.0)
    assert unknown.level() == OK
    assert unknown.admits(P2)               # admit everything when unknowable


def test_failopen_when_probe_raises():
    def boom():
        raise RuntimeError("no /proc")
    mp = MemoryPressure(sample=boom, ttl_s=0.0)
    assert mp.level() == OK
    assert mp.admits(P2)


def test_scheduler_sheds_background_p2_under_pressure():
    """PriorityScheduler.run_p2_background must drop P2 work under high pressure."""
    ran = []

    class _Agent:
        name = "bg"
        priority = P2
        reads = set()
        writes = set()
        expected_latency_ms = 10

        async def run(self, board):
            ran.append(self.name)

    class _Board:
        def has(self, k):
            return True

    async def _go():
        s = sched.PriorityScheduler(_Board(), memory_pressure=_mp(0.99))
        s.run_p2_background([_Agent()])
        await asyncio.sleep(0.02)

    asyncio.run(_go())
    assert ran == []                        # shed, never launched
