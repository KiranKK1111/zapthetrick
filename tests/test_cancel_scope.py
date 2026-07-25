"""Per-turn cancellation scope (vNext §4.5)."""
from __future__ import annotations

import asyncio

from app.api import cancel_scope as CS
from app.api import replay as _replay


def setup_function():
    CS.reset_for_tests()
    _replay.clear_cancel("t1")


def test_scope_for_reuses_by_id():
    assert CS.scope_for("t1") is CS.scope_for("t1")
    assert CS.scope_for("t1") is not CS.scope_for("t2")


def test_cancel_sets_cooperative_flag():
    s = CS.scope_for("t1")
    assert s.cancelled is False
    CS.cancel("t1")
    assert s.cancelled is True
    assert _replay.is_cancelled("t1") is True   # legacy pollers still stop


def test_cancel_tears_down_registered_callbacks():
    s = CS.scope_for("t1")
    hits = []
    s.register_callback(lambda: hits.append("a"))
    s.register_callback(lambda: hits.append("b"))
    CS.cancel("t1")
    assert hits == ["a", "b"]


def test_cancel_cancels_registered_tasks():
    async def _run():
        s = CS.scope_for("t1")
        task = asyncio.ensure_future(asyncio.sleep(30))
        s.register_task(task)
        await asyncio.sleep(0)          # let it start
        CS.cancel("t1")
        await asyncio.sleep(0)
        return task.cancelled() or task.done()
    assert asyncio.run(_run()) is True


def test_register_after_cancel_tears_down_immediately():
    s = CS.scope_for("t1")
    CS.cancel("t1")
    hits = []
    s.register_callback(lambda: hits.append("late"))  # arrives after STOP
    assert hits == ["late"]


def test_sandbox_group_is_killed_on_cancel(monkeypatch):
    killed = []
    import app.sandbox.docker_exec as _dex
    monkeypatch.setattr(_dex, "cancel_group", lambda g: killed.append(g),
                        raising=False)
    s = CS.scope_for("t1")
    s.register_sandbox("grp-42")
    CS.cancel("t1")
    assert killed == ["grp-42"]


def test_cancel_is_idempotent():
    s = CS.scope_for("t1")
    hits = []
    s.register_callback(lambda: hits.append(1))
    CS.cancel("t1")
    CS.cancel("t1")                      # second STOP must not re-fire callbacks
    assert hits == [1]


def test_cancel_unknown_turn_still_sets_flag():
    CS.cancel("never-registered")
    assert _replay.is_cancelled("never-registered") is True


def test_clear_drops_scope_and_flag():
    CS.cancel("t1")
    CS.clear("t1")
    assert _replay.is_cancelled("t1") is False
    # A fresh scope after clear starts uncancelled.
    assert CS.scope_for("t1").cancelled is False
