"""Background maintenance loop (roadmap Phase 7 #10/#11/#13 + Phase 1 #24)."""
from __future__ import annotations

import asyncio

from app.obs import maintenance


def test_run_once_light_pass_touches_all_tasks():
    """One light pass (no benchmark) runs memory/self-heal/update fail-open."""
    rep = maintenance.run_maintenance_once(run_benchmark=False)
    assert "memory" in rep
    assert "self_heal" in rep
    assert "update" in rep
    assert "trend" not in rep               # benchmark skipped


def test_update_check_reports_up_to_date():
    rep = maintenance.run_maintenance_once(run_benchmark=False)
    upd = rep["update"]
    # Self-referential manifest (latest == running version) → up to date.
    assert upd.get("status") == "up_to_date"
    assert upd.get("update_available") is False


def test_run_once_with_benchmark_records_trend(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_TRENDS", str(tmp_path / "trends.jsonl"))
    rep = maintenance.run_maintenance_once(run_benchmark=True)
    assert "trend" in rep
    assert rep["trend"].get("points", 0) >= 1


def test_start_stop_loop_is_idempotent():
    async def _drive():
        started = maintenance.start_maintenance_loop()
        # Second start while running is a no-op.
        again = maintenance.start_maintenance_loop()
        maintenance.stop_maintenance_loop()
        return started, again

    started, again = asyncio.run(_drive())
    assert started is True
    assert again is False


def test_start_returns_false_when_disabled(monkeypatch):
    monkeypatch.setattr(maintenance, "_enabled", lambda: False)

    async def _drive():
        return maintenance.start_maintenance_loop()

    assert asyncio.run(_drive()) is False
