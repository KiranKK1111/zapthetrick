"""Tests for ops resilience (vNext §6.4/§11.3/§11.4, Stage 9 Component E)."""
from __future__ import annotations

import app.obs.resilience as R


def _drain_on(monkeypatch):
    monkeypatch.setattr(R, "_deploy_drain", lambda: True)


def _gc_on(monkeypatch):
    monkeypatch.setattr(R, "_ops_gc", lambda: True)


# ---- zero-downtime drain --------------------------------------------------
def test_accepts_before_drain(monkeypatch):
    _drain_on(monkeypatch)
    assert R.DrainController().should_accept()


def test_rejects_new_turns_while_draining(monkeypatch):
    _drain_on(monkeypatch)
    d = R.DrainController()
    d.begin(now=1000)
    assert not d.should_accept()


def test_drains_when_inflight_finishes(monkeypatch):
    _drain_on(monkeypatch)
    d = R.DrainController()
    d.add_inflight("t1"); d.add_inflight("t2")
    d.begin(now=1000)
    assert not d.is_drained()
    d.finish_inflight("t1")
    assert not d.is_drained()
    d.finish_inflight("t2")
    assert d.is_drained() and d.state == R.DRAINED


def test_deadline_forces_restart(monkeypatch):
    _drain_on(monkeypatch)
    d = R.DrainController()
    d.add_inflight("stuck")
    d.begin(now=1000)
    assert not d.deadline_exceeded(now=1010, deadline_s=30)
    assert d.deadline_exceeded(now=1031, deadline_s=30)
    assert d.can_restart(now=1031)             # straggler → restart anyway


def test_can_restart_when_drained(monkeypatch):
    _drain_on(monkeypatch)
    d = R.DrainController()
    d.begin(now=1000)
    assert d.can_restart(now=1000)             # no inflight → immediately drained


def test_drain_disabled_always_accepts(monkeypatch):
    monkeypatch.setattr(R, "_deploy_drain", lambda: False)
    d = R.DrainController()
    d.begin(now=1000)
    assert d.should_accept()                   # byte-identical when off


def test_deadline_false_before_draining():
    assert R.DrainController().deadline_exceeded(now=9999) is False


# ---- retention-as-data ----------------------------------------------------
def test_retention_policies():
    assert R.retention_for("eval") == 90 * 86400
    assert R.retention_for("screen_state") == 86400
    assert R.retention_for("voice") == -1.0
    assert R.retention_for("pre_answer") == 0.0
    assert R.retention_for("unknown-kind") == 30 * 86400   # default TTL


def test_referenced_blob_always_kept():
    assert not R.should_purge("eval", age_s=1e9, referenced=True)


def test_voice_never_purged():
    assert not R.should_purge("voice", age_s=1e12, referenced=False)


def test_ttl_purge():
    assert R.should_purge("eval", age_s=100 * 86400, referenced=False)
    assert not R.should_purge("eval", age_s=10 * 86400, referenced=False)


def test_screen_state_24h():
    assert not R.should_purge("screen_state", age_s=12 * 3600, referenced=False)
    assert R.should_purge("screen_state", age_s=25 * 3600, referenced=False)


def test_session_scoped_pre_answers():
    assert not R.should_purge("pre_answer", age_s=1, referenced=False, session_active=True)
    assert R.should_purge("pre_answer", age_s=1, referenced=False, session_active=False)


def test_should_purge_never_raises():
    assert R.should_purge(None, age_s=None, referenced=False) is False  # type: ignore[arg-type]


# ---- ref-counted blob GC --------------------------------------------------
def test_plan_gc_ref_counted_and_retention(monkeypatch):
    _gc_on(monkeypatch)
    blobs = [
        {"id": "old_eval", "kind": "eval", "created_at": 0, "refs": 0, "size": 100},
        {"id": "referenced", "kind": "eval", "created_at": 0, "refs": 2, "size": 50},
        {"id": "voice", "kind": "voice", "created_at": 0, "refs": 0, "size": 10},
        {"id": "old_screen", "kind": "screen_state", "created_at": 0, "refs": 0, "size": 7},
    ]
    plan = R.plan_gc(blobs, now=100 * 86400)
    assert set(plan.purge) == {"old_eval", "old_screen"}
    assert set(plan.keep) == {"referenced", "voice"}
    assert plan.freed_bytes == 107


def test_plan_gc_disabled_is_empty(monkeypatch):
    monkeypatch.setattr(R, "_ops_gc", lambda: False)
    blobs = [{"id": "x", "kind": "eval", "created_at": 0, "refs": 0, "size": 1}]
    plan = R.plan_gc(blobs, now=1e9)
    assert plan.purge == [] and plan.keep == []


def test_plan_gc_created_at_zero_is_old(monkeypatch):
    # created_at=0 (epoch) must be treated as old, not "now" (the falsy bug).
    _gc_on(monkeypatch)
    plan = R.plan_gc([{"id": "z", "kind": "eval", "created_at": 0, "refs": 0}],
                     now=100 * 86400)
    assert plan.purge == ["z"]


# ---- auto-resurrection ----------------------------------------------------
def _redirector_on(monkeypatch, threshold=3):
    from app.core import config_loader as C
    monkeypatch.setattr(C.cfg.ops, "redirector", True, raising=False)
    monkeypatch.setattr(C.cfg.ops, "resurrect_after_failures", threshold, raising=False)


def test_resurrect_after_threshold_failures(monkeypatch):
    _redirector_on(monkeypatch, threshold=3)
    m = R.ResurrectionMonitor()
    m.record_probe(False); m.record_probe(False)
    assert not m.should_resurrect()            # only 2 failures
    m.record_probe(False)
    assert m.should_resurrect()                # 3rd → trigger


def test_success_resets_failure_count(monkeypatch):
    _redirector_on(monkeypatch, threshold=2)
    m = R.ResurrectionMonitor()
    m.record_probe(False)
    m.record_probe(True)                       # recovered
    assert m.consecutive_failures == 0
    m.record_probe(False)
    assert not m.should_resurrect()            # counter reset → only 1


def test_resurrect_fires_once_per_outage(monkeypatch):
    _redirector_on(monkeypatch, threshold=1)
    m = R.ResurrectionMonitor()
    m.record_probe(False)
    assert m.should_resurrect()
    m.mark_triggered()
    m.record_probe(False)
    assert not m.should_resurrect()            # already triggered this outage


def test_resurrect_disabled_when_redirector_off(monkeypatch):
    from app.core import config_loader as C
    monkeypatch.setattr(C.cfg.ops, "redirector", False, raising=False)
    m = R.ResurrectionMonitor(threshold=1)
    m.record_probe(False)
    assert not m.should_resurrect()
