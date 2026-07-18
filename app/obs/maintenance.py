"""Background maintenance loop — the scheduler the learning/reliability loops
were missing (roadmap Phase 7 #10/#11/#13 + Phase 1 #24).

Several built + wired subsystems only ever ran on a user-triggered path
(session-end memory consolidation) or not at all (self-benchmark trends, update
check). This is the single periodic trigger that runs them on a timer, so the
roadmap's "nightly" cadence actually exists:

  * memory consolidation           (`memory.lifecycle.maintain_scheduled`)
  * self-healing diagnostics       (`obs.self_heal.heal`)
  * self-benchmark trend snapshot  (`eval.trends.run_and_record`)   — nightly
  * update check                   (`core.update_check` + `core.rollout`)

Every task is fail-open and independent — one failing never blocks the others,
and none ever blocks a request (the loop runs in its own asyncio task). Gated by
`obs.maintenance` (enabling default ON); the heavy nightly benchmark only runs
once per `benchmark_interval_s`.
"""
from __future__ import annotations

import asyncio
import logging
import time

log = logging.getLogger("obs.maintenance")

_task: "asyncio.Task | None" = None
# Cadence: the light tasks run every tick; the benchmark runs at most nightly.
_TICK_S = 3600.0                 # 1 hour
_BENCH_INTERVAL_S = 24 * 3600.0  # 24 hours
_last_benchmark: float = 0.0


def _enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        sec = getattr(cfg, "obs", None)
        if sec is None:
            return True
        return bool(getattr(sec, "maintenance", True))
    except Exception:  # noqa: BLE001
        return True


def _tick_seconds() -> float:
    try:
        from app.core.config_loader import cfg
        sec = getattr(cfg, "obs", None)
        return float(getattr(sec, "maintenance_tick_s", _TICK_S)) if sec else _TICK_S
    except Exception:  # noqa: BLE001
        return _TICK_S


def _bench_interval() -> float:
    try:
        from app.core.config_loader import cfg
        sec = getattr(cfg, "obs", None)
        return (float(getattr(sec, "benchmark_interval_s", _BENCH_INTERVAL_S))
                if sec else _BENCH_INTERVAL_S)
    except Exception:  # noqa: BLE001
        return _BENCH_INTERVAL_S


def _maybe_update_check() -> dict:
    """Opt-in periodic update check (Phase 1 #24). No network by default: builds
    a self-referential manifest (latest == running version → up_to_date) unless
    the config supplies one. Fail-open, decision-only (never downloads)."""
    try:
        from app.core.update_check import (
            APP_VERSION, ReleaseManifest, check_for_update,
        )
        manifest = ReleaseManifest(latest=APP_VERSION)
        try:
            from app.core.config_loader import cfg
            upd = getattr(cfg, "update", None)
            if upd is not None:
                manifest = ReleaseManifest(
                    latest=str(getattr(upd, "latest", APP_VERSION) or APP_VERSION),
                    minimum_supported=str(getattr(upd, "minimum_supported", "0.0.0")),
                    channel=str(getattr(upd, "channel", "stable")),
                    notes=str(getattr(upd, "notes", "")),
                    url=str(getattr(upd, "url", "")),
                )
        except Exception:  # noqa: BLE001
            pass
        result = check_for_update(APP_VERSION, manifest)
        return {"status": result.status.value,
                "update_available": result.update_available,
                "current": result.current, "latest": result.latest}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:160]}


def run_maintenance_once(*, run_benchmark: bool = True) -> dict:
    """Run one maintenance pass synchronously and return a per-task report. Used
    by the loop, the on-demand endpoint, and tests. Every task is fail-open."""
    report: dict = {}

    # 1) Memory consolidation (Phase 7 #11) — nightly-schedulable trigger.
    try:
        from app.memory.lifecycle import maintain_scheduled
        report["memory"] = maintain_scheduled()
    except Exception as exc:  # noqa: BLE001
        report["memory"] = {"error": str(exc)[:160]}

    # 2) Self-healing diagnostics (Phase 7 #13).
    try:
        from app.obs.self_heal import heal
        report["self_heal"] = heal(apply=True)
    except Exception as exc:  # noqa: BLE001
        report["self_heal"] = {"error": str(exc)[:160]}

    # 3) Update check (Phase 1 #24).
    report["update"] = _maybe_update_check()

    # 4) Self-benchmark trend snapshot (Phase 7 #10) — the expensive one.
    if run_benchmark:
        try:
            from app.eval.trends import run_and_record
            report["trend"] = run_and_record()
        except Exception as exc:  # noqa: BLE001
            report["trend"] = {"error": str(exc)[:160]}

    return report


async def _loop() -> None:
    global _last_benchmark
    while True:
        try:
            due = (time.time() - _last_benchmark) >= _bench_interval()
            await asyncio.to_thread(run_maintenance_once, run_benchmark=due)
            if due:
                _last_benchmark = time.time()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the loop must survive any task error
            log.debug("maintenance tick failed", exc_info=True)
        try:
            await asyncio.sleep(_tick_seconds())
        except asyncio.CancelledError:
            raise


def start_maintenance_loop() -> bool:
    """Start the periodic loop (idempotent). Returns True if it started, False if
    disabled or already running. Never raises."""
    global _task
    try:
        if not _enabled():
            log.info("maintenance loop disabled by config")
            return False
        if _task is not None and not _task.done():
            return False
        _task = asyncio.create_task(_loop(), name="obs-maintenance")
        log.info("maintenance loop started (tick=%.0fs)", _tick_seconds())
        return True
    except RuntimeError:
        # No running event loop (e.g. imported outside the app) — harmless.
        return False
    except Exception:  # noqa: BLE001
        return False


def stop_maintenance_loop() -> None:
    global _task
    try:
        if _task is not None and not _task.done():
            _task.cancel()
        _task = None
    except Exception:  # noqa: BLE001
        pass


__all__ = ["run_maintenance_once", "start_maintenance_loop",
           "stop_maintenance_loop"]
