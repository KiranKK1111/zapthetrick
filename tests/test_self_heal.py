"""Self-healing + diagnostics (roadmap Phase 7 #13)."""
from __future__ import annotations

from app.obs import self_heal


def test_healthy_when_nothing_wrong(monkeypatch):
    # No jobs backlog, no recurring failures, small recorder → healthy.
    from app.obs import failure_kb, replay
    failure_kb.reset()
    replay.reset_recorder()
    rep = self_heal.diagnose()
    # (Other subsystems may add advisory issues, but none of ours should fire.)
    kinds = {i["kind"] for i in rep["issues"]}
    assert "jobs_backlog" not in kinds
    assert "recurring_failure" not in kinds


def test_diagnoses_and_clears_job_backlog():
    from app.obs.jobs import jobs
    reg = jobs()
    # Create + finish enough jobs to trip the backlog threshold.
    for i in range(self_heal.FINISHED_JOBS_HIGH + 2):
        jid = reg.start(f"j{i}", kind="task")
        reg.finish(jid, ok=True)
    rep = self_heal.diagnose()
    assert any(i["kind"] == "jobs_backlog" for i in rep["issues"])

    out = self_heal.heal(apply=True)
    assert any(a["action"] == "clear_finished_jobs" and a["applied"]
               for a in out["actions"])
    # After healing, the finished jobs are gone.
    remaining = [j for j in reg.snapshot() if j["status"] != "running"]
    assert not remaining


def test_recurring_failure_surfaces_learned_recovery():
    from app.obs import failure_kb
    failure_kb.reset()
    for _ in range(self_heal.FAILURE_OCCURRENCE_HIGH + 1):
        failure_kb.record_occurrence("provider_rate_limit")
    # Give the KB a proven recovery so `heal` can recommend it.
    for _ in range(3):
        failure_kb.record_outcome("provider_rate_limit", "cooldown_wait", True)

    out = self_heal.heal(apply=True)
    rec = [a for a in out["actions"] if a["action"] == "apply_recovery"]
    assert rec and rec[0]["recommended_recovery"] == "cooldown_wait"


def test_dry_run_takes_no_action():
    from app.obs.jobs import jobs
    reg = jobs()
    for i in range(self_heal.FINISHED_JOBS_HIGH + 2):
        jid = reg.start(f"d{i}", kind="task")
        reg.finish(jid, ok=True)
    before = len([j for j in reg.snapshot() if j["status"] != "running"])
    out = self_heal.heal(apply=False)
    after = len([j for j in reg.snapshot() if j["status"] != "running"])
    assert after >= before                    # nothing cleared on a dry run
    assert all(not a.get("applied") for a in out["actions"])
    reg.clear_finished()


def test_trims_overgrown_recorder():
    from app.obs import replay
    replay.reset_recorder()
    for i in range(self_heal.RECORDER_TRIM_TO + 20):
        replay.capture("k", {"i": i}, i, cap=10_000)   # bypass the ring bound
    assert replay.captured_count() > self_heal.RECORDER_TRIM_TO
    self_heal.heal(apply=True)
    assert replay.captured_count() <= self_heal.RECORDER_TRIM_TO
