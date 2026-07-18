"""Progressive artifact delivery (roadmap Phase 5 #18).

Pins: artifacts are emitted AS THEY COMPLETE (fastest first), not after the whole
batch; a failing producer yields an ok=False event without sinking siblings.
"""
from __future__ import annotations

import asyncio

from app.perceived.progressive import ArtifactEvent, deliver_artifacts


def _collect(producers):
    async def go():
        out = []
        async for ev in deliver_artifacts(producers):
            out.append(ev)
        return out
    return asyncio.run(go())


def test_emits_in_completion_order():
    # Deterministic (not wall-clock, which Windows' coarse timer can batch into
    # one loop tick): `fast` returns immediately; `slow` yields across many loop
    # ticks so it can never land in the same `wait()` batch as `fast`.
    async def slow():
        for _ in range(50):
            await asyncio.sleep(0)
        return "slow"

    async def fast():
        return "fast"

    events = _collect({"slow": slow, "fast": fast})
    assert [e.artifact for e in events] == ["fast", "slow"]     # fastest first
    assert [e.index for e in events] == [0, 1]
    assert all(isinstance(e, ArtifactEvent) and e.ok for e in events)


def test_sync_producers_supported():
    events = _collect([("a", lambda: 1), ("b", lambda: 2)])
    assert sorted(e.artifact for e in events) == [1, 2]


def test_failing_producer_isolated():
    async def good():
        return "ok"

    def bad():
        raise ValueError("nope")

    events = _collect({"good": good, "bad": bad})
    by_name = {e.name: e for e in events}
    assert by_name["good"].ok and by_name["good"].artifact == "ok"
    assert not by_name["bad"].ok and "nope" in by_name["bad"].error


def test_empty_producers_yields_nothing():
    assert _collect({}) == []
