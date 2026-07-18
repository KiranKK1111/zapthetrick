"""Task Center — list background jobs (answer generation, exports) so the FE
can show what's running and what recently finished."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.obs.jobs import jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
async def list_jobs() -> dict:
    return {"jobs": jobs().snapshot()}


@router.post("/clear")
async def clear_jobs() -> dict:
    """Drop finished/errored jobs; keep anything still running."""
    jobs().clear_finished()
    return {"ok": True}


# Architecture Health Dashboard (P1 #6) + benchmark leaderboard (P1 #1) live on
# the same status router.
health_router = APIRouter(prefix="/api/health", tags=["health"])


@health_router.get("/dashboard")
async def health_dashboard() -> dict:
    from app.obs.health_dashboard import snapshot
    return snapshot()


@health_router.get("/leaderboard")
async def benchmark_leaderboard() -> dict:
    from app.eval.leaderboard import run_all
    return run_all()


class CrashBody(BaseModel):
    message: str
    stack: str = ""
    version: str = ""
    platform: str = ""


@health_router.post("/crash")
async def report_crash(body: CrashBody) -> dict:
    """Client crash-report intake (release channel, P1 #24)."""
    from app.obs.crash_reports import crash_log
    crash_log().record(message=body.message, stack=body.stack,
                       version=body.version, platform=body.platform)
    return {"ok": True}


@health_router.get("/crashes")
async def list_crashes() -> dict:
    from app.obs.crash_reports import crash_log
    log = crash_log()
    return {"summary": log.summary(), "recent": log.recent()}


@health_router.get("/update")
async def check_update(device_id: str = "") -> dict:
    """Opt-in update / release-channel check (P1 #24). Decision-only — compares
    the running version against a manifest (self-referential by default → no
    network) and applies staged-rollout/rollback logic. Never downloads."""
    from app.core.update_check import (
        APP_VERSION, ReleaseManifest, check_for_update,
    )
    from app.core.rollout import rollout_decision

    manifest = ReleaseManifest(latest=APP_VERSION)
    percent = 100
    blocked: set[str] = set()
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
            percent = int(getattr(upd, "rollout_percent", 100) or 100)
            blocked = set(getattr(upd, "blocked_versions", None) or [])
    except Exception:  # noqa: BLE001
        pass

    result = check_for_update(APP_VERSION, manifest)
    out = {
        "status": result.status.value,
        "update_available": result.update_available,
        "current": result.current,
        "latest": result.latest,
        "notes": result.notes,
        "url": result.url,
    }
    if device_id:
        out["rollout"] = rollout_decision(
            device_id, manifest.latest, APP_VERSION,
            percent=percent, blocked=blocked)
    return out


@health_router.get("/replay")
async def flight_recorder(dry_run: bool = True) -> dict:
    """Flight-recorder status + replay (P1 #4). Replays every captured
    (inputs → trace) recording through the CURRENT `build_trace` and diffs — any
    mismatch is behavioral drift since the recording was made."""
    from app.obs import replay as _replay
    from app.obs.trace import replay_trace

    report = _replay.replay_captured(replay_trace)
    return {
        "captured": _replay.captured_count(),
        "total": report.total,
        "matched": report.matched,
        "match_rate": round(report.match_rate, 4),
        "mismatches": len(report.mismatches),
        "errors": report.errors[:8],
    }


@health_router.post("/self-heal")
async def self_heal(apply: bool = True) -> dict:
    """Diagnostics-driven self-healing (P7 #13). Inspects live health signals and
    takes bounded remediation actions (clear finished jobs, trim the recorder,
    surface a history-backed recovery). `apply=false` is a dry run."""
    from app.obs.self_heal import heal
    return heal(apply=apply)


@health_router.post("/maintenance")
async def run_maintenance(benchmark: bool = False) -> dict:
    """Run one maintenance pass on demand (P7 #10/#11/#13 + P1 #24): memory
    consolidation, self-heal, update check, and (optionally) the self-benchmark
    trend snapshot. The same pass the background loop runs on a timer."""
    from app.obs.maintenance import run_maintenance_once
    return run_maintenance_once(run_benchmark=benchmark)


@health_router.post("/memory-maintain")
async def memory_maintain() -> dict:
    """Trigger schedulable memory consolidation now (P7 #11)."""
    from app.memory.lifecycle import maintain_scheduled
    return maintain_scheduled()
