"""Self-healing + diagnostics (roadmap Phase 7 #13).

The health dashboard (Phase 1 #6) is read-only — it *shows* trouble but does
nothing about it. This closes that half of the loop: `diagnose()` inspects the
same live signals and names concrete problems; `heal()` takes bounded, safe
remediation actions and reports what it did.

Every action is conservative and reversible-by-nature (clearing finished jobs,
trimming an in-process ring, surfacing a history-backed recovery). It never
touches provider routing or user data. Deterministic + fail-open: a diagnostics
error must never make things worse than the fault it was inspecting.
"""
from __future__ import annotations

# Thresholds (kept here so they're testable / tunable in one place).
FINISHED_JOBS_HIGH = 25          # clear finished jobs once this many pile up
FAILURE_OCCURRENCE_HIGH = 5      # a failure class recurring this often is "hot"
RECORDER_TRIM_TO = 500           # cap the flight recorder at this many recordings


def diagnose() -> dict:
    """Inspect live health signals and return named issues (no side effects)."""
    issues: list[dict] = []

    # 1) Background jobs piling up finished/errored.
    try:
        from app.obs.jobs import jobs
        js = jobs().snapshot()
        finished = [j for j in js if j.get("status") in ("done", "error", "cancelled")]
        errored = [j for j in js if j.get("status") == "error"]
        if len(finished) >= FINISHED_JOBS_HIGH:
            issues.append({"kind": "jobs_backlog", "severity": "low",
                           "finished": len(finished),
                           "action": "clear_finished_jobs"})
        if errored:
            issues.append({"kind": "jobs_errored", "severity": "medium",
                           "count": len(errored)})
    except Exception:  # noqa: BLE001
        pass

    # 2) Recurring failure classes — and whether the KB knows a fix.
    try:
        from app.obs import failure_kb
        for fid, rec in (failure_kb.stats() or {}).items():
            occ = int(rec.get("occurrences", 0) or 0)
            if occ >= FAILURE_OCCURRENCE_HIGH:
                best = failure_kb.best_recovery(fid)
                issues.append({"kind": "recurring_failure", "severity": "high",
                               "failure_id": fid, "occurrences": occ,
                               "recommended_recovery": best,
                               "action": ("apply_recovery" if best else "investigate")})
    except Exception:  # noqa: BLE001
        pass

    # 3) Flight recorder overgrown (memory pressure).
    try:
        from app.obs import replay
        n = replay.captured_count()
        if n > RECORDER_TRIM_TO:
            issues.append({"kind": "recorder_overgrown", "severity": "low",
                           "captured": n, "action": "trim_recorder"})
    except Exception:  # noqa: BLE001
        pass

    return {"issues": issues, "healthy": not issues}


def heal(*, apply: bool = True) -> dict:
    """Diagnose, then take bounded remediation actions for the issues we can act
    on. `apply=False` is a dry run (report what *would* be done). Fail-open."""
    report = diagnose()
    actions: list[dict] = []
    for issue in report.get("issues", []):
        act = issue.get("action")
        try:
            if act == "clear_finished_jobs":
                if apply:
                    from app.obs.jobs import jobs
                    jobs().clear_finished()
                actions.append({"action": act, "applied": bool(apply)})
            elif act == "trim_recorder":
                if apply:
                    from app.obs import replay
                    with replay._REC_LOCK:  # bounded trim to the cap
                        recs = replay._RECORDER._recs
                        overflow = len(recs) - RECORDER_TRIM_TO
                        if overflow > 0:
                            del recs[0:overflow]
                actions.append({"action": act, "applied": bool(apply)})
            elif act == "apply_recovery":
                # We can't force a retry here (no in-flight request), but we
                # surface the history-backed recovery so the engine/operator can
                # act — this is the diagnostics → recovery bridge.
                actions.append({
                    "action": act, "applied": False,
                    "failure_id": issue.get("failure_id"),
                    "recommended_recovery": issue.get("recommended_recovery"),
                })
        except Exception:  # noqa: BLE001 — one failed remedy never aborts the rest
            actions.append({"action": act, "applied": False, "error": True})

    return {
        "diagnosed": len(report.get("issues", [])),
        "healthy": report.get("healthy", True),
        "actions": actions,
        "issues": report.get("issues", []),
    }


__all__ = ["diagnose", "heal", "FINISHED_JOBS_HIGH",
           "FAILURE_OCCURRENCE_HIGH", "RECORDER_TRIM_TO"]
