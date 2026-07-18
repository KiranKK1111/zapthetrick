"""Outcome-driven threshold calibration (gap G1)."""
from __future__ import annotations

from app.core import calibration as cal


def _enable(monkeypatch, *, min_samples=4):
    from app.core import config_loader as cl
    _ms = min_samples

    class _C:
        enabled = True
        min_samples = _ms
    monkeypatch.setattr(cl.cfg, "calibration", _C(), raising=False)
    cal._reset_for_test()


def test_default_when_disabled(monkeypatch):
    from app.core import config_loader as cl

    class _C:
        enabled = False
        min_samples = 4
    monkeypatch.setattr(cl.cfg, "calibration", _C(), raising=False)
    cal._reset_for_test()
    cal.observe("t", 0.9, True, persist=False)
    assert cal.calibrated("t", 0.5) == 0.5      # off → the configured default


def test_default_until_min_samples(monkeypatch):
    _enable(monkeypatch, min_samples=4)
    cal.observe("t", 0.9, True, persist=False)
    cal.observe("t", 0.2, False, persist=False)
    assert cal.calibrated("t", 0.5) == 0.5      # < min_samples → default


def test_calibrates_toward_good_bad_midpoint(monkeypatch):
    _enable(monkeypatch, min_samples=4)
    # good verdicts cluster at 0.8, bad at 0.4 → learned midpoint 0.6, blended
    # 50/50 with default 0.5 → 0.55
    for _ in range(3):
        cal.observe("t", 0.8, True, persist=False)
    for _ in range(3):
        cal.observe("t", 0.4, False, persist=False)
    got = cal.calibrated("t", 0.5)
    assert abs(got - 0.55) < 1e-6


def test_default_when_only_one_class(monkeypatch):
    _enable(monkeypatch, min_samples=2)
    cal.observe("t", 0.8, True, persist=False)
    cal.observe("t", 0.9, True, persist=False)   # all good → can't separate
    assert cal.calibrated("t", 0.5) == 0.5


def test_ring_is_bounded(monkeypatch):
    _enable(monkeypatch, min_samples=2)
    for i in range(cal._MAX_SAMPLES + 50):
        cal.observe("t", 0.5, i % 2 == 0, persist=False)
    assert cal.stats()["t"] == cal._MAX_SAMPLES


def test_observe_fail_open():
    # no config / disabled path must not raise
    cal.observe("t", 0.5, True, persist=False)
    assert isinstance(cal.calibrated("t", 0.42), float)


def test_enabled_by_default_when_flag_absent(monkeypatch):
    """P7 #12/#14: the getattr default is ENABLING — an absent flag means
    calibration is ON so learned thresholds adapt out of the box."""
    from app.core import config_loader as cl

    class _NoCalib:      # a config object with no `enabled` attribute at all
        min_samples = 4
    monkeypatch.setattr(cl.cfg, "calibration", _NoCalib(), raising=False)
    assert cal.enabled() is True


def test_learned_threshold_influences_decision(monkeypatch):
    """P7 #14: prove the learned value actually MOVES a decision boundary."""
    _enable(monkeypatch, min_samples=4)
    default = 0.5
    # Good decisions clustered high, bad ones lower → learned boundary rises.
    # mean(good)=0.9, mean(bad)=0.6 → midpoint 0.75, blended 50/50 with the
    # 0.5 default → 0.625, strictly above the static default.
    for _ in range(4):
        cal.observe("primary_threshold", 0.9, True, persist=False)
    for _ in range(4):
        cal.observe("primary_threshold", 0.6, False, persist=False)
    learned = cal.calibrated("primary_threshold", default)
    assert learned != default                 # the threshold genuinely shifted
    assert learned > default                  # learned bar is higher than static
    assert abs(learned - 0.625) < 1e-6


def test_clamps_to_bounds(monkeypatch):
    _enable(monkeypatch, min_samples=2)
    for _ in range(2):
        cal.observe("t", 5.0, True, persist=False)      # out-of-range scores
        cal.observe("t", -3.0, False, persist=False)
    got = cal.calibrated("t", 0.5, lo=0.0, hi=1.0)
    assert 0.0 <= got <= 1.0
