"""Staged rollout + rollback + crash intake (P1 #24 — software half)."""
from __future__ import annotations

from app.core.rollout import in_rollout, rollout_decision
from app.obs.crash_reports import CrashLog


def test_rollout_cohort_is_deterministic_and_monotone():
    dev = "device-abc"
    v = "2.0.0"
    # Same inputs → same answer.
    assert in_rollout(dev, v, percent=50) == in_rollout(dev, v, percent=50)
    # 0% offers nobody, 100% offers everybody.
    assert in_rollout(dev, v, percent=0) is False
    assert in_rollout(dev, v, percent=100) is True
    # A device inside a small rollout stays inside a larger one (monotone).
    if in_rollout(dev, v, percent=20):
        assert in_rollout(dev, v, percent=60) is True


def test_rollout_distribution_is_roughly_the_percentage():
    inside = sum(1 for i in range(1000)
                 if in_rollout(f"dev-{i}", "2.0.0", percent=30))
    assert 250 <= inside <= 350            # ~30% ± tolerance


def test_decision_blocks_rolled_back_and_older():
    # Rolled back → never offered.
    d = rollout_decision("dev", "2.0.0", "1.0.0", percent=100,
                         blocked={"2.0.0"})
    assert d["offer"] is False and "rolled back" in d["reason"]
    # Already current/newer → not offered.
    d2 = rollout_decision("dev", "1.0.0", "1.0.0", percent=100)
    assert d2["offer"] is False and "current" in d2["reason"]
    # Eligible + in cohort → offered.
    d3 = rollout_decision("dev", "2.0.0", "1.0.0", percent=100)
    assert d3["offer"] is True


def test_crash_log_bounded_and_summarized():
    log = CrashLog(max_reports=3)
    for i in range(5):
        log.record(message=f"boom {i}", version="2.0.0", platform="android")
    log.record(message="", version="x")     # empty ignored
    s = log.summary()
    assert s["total"] == 3                   # bounded
    assert s["by_version"]["2.0.0"] == 3
    assert len(log.recent()) == 3
