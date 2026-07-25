"""Tests for the eval flywheel + canary (vNext §8.9, Stage 11 Component A)."""
from __future__ import annotations

import app.eval.flywheel as F


def _fly(monkeypatch):
    monkeypatch.setattr(F, "flywheel_enabled", lambda: True)


def _canary(monkeypatch):
    monkeypatch.setattr(F, "canary_enabled", lambda: True)


# ---- case capture ---------------------------------------------------------
def test_should_capture_known_signals(monkeypatch):
    _fly(monkeypatch)
    for s in (F.THUMBS_DOWN, F.FORCED_ANSWER, F.VERIFY_FAIL, F.SCHEMA_RETRY,
              F.CLARIFIER_OTHER):
        assert F.should_capture(s)
    assert not F.should_capture("random_event")


def test_capture_redacts_pii(monkeypatch):
    _fly(monkeypatch)
    c = F.capture_case(F.THUMBS_DOWN,
                       prompt="mail bob@x.com about card 4111111111111111")
    assert c is not None
    assert "bob@x.com" not in c.prompt and "<email>" in c.prompt
    assert "4111111111111111" not in c.prompt


def test_capture_uses_injected_redactor(monkeypatch):
    _fly(monkeypatch)
    c = F.capture_case(F.VERIFY_FAIL, prompt="secret data",
                       redact_fn=lambda s: s.replace("secret", "***"))
    assert c.prompt == "*** data"


def test_capture_stable_id(monkeypatch):
    _fly(monkeypatch)
    a = F.capture_case(F.THUMBS_DOWN, prompt="same prompt")
    b = F.capture_case(F.THUMBS_DOWN, prompt="same prompt")
    assert a.id == b.id                          # dedup key


def test_capture_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(F, "flywheel_enabled", lambda: False)
    assert F.capture_case(F.THUMBS_DOWN, prompt="x") is None


def test_capture_non_signal_returns_none(monkeypatch):
    _fly(monkeypatch)
    assert F.capture_case("not_a_signal", prompt="x") is None


def test_capture_empty_prompt_returns_none(monkeypatch):
    _fly(monkeypatch)
    assert F.capture_case(F.THUMBS_DOWN, prompt="   ") is None


def test_capture_never_raises(monkeypatch):
    _fly(monkeypatch)
    assert F.capture_case(F.THUMBS_DOWN, prompt="ok", redact_fn=lambda s: 1 / 0) is None


# ---- golden set -----------------------------------------------------------
def test_golden_set_dedups(monkeypatch):
    _fly(monkeypatch)
    gs = F.GoldenSet()
    c = F.capture_case(F.THUMBS_DOWN, prompt="p")
    assert gs.add(c) is True
    assert gs.add(F.capture_case(F.THUMBS_DOWN, prompt="p")) is False  # dup id
    assert len(gs) == 1


def test_golden_set_signal_histogram(monkeypatch):
    _fly(monkeypatch)
    gs = F.GoldenSet()
    gs.add(F.capture_case(F.THUMBS_DOWN, prompt="a"))
    gs.add(F.capture_case(F.VERIFY_FAIL, prompt="b"))
    gs.add(F.capture_case(F.VERIFY_FAIL, prompt="c"))
    assert gs.signals() == {"thumbs_down": 1, "verify_fail": 2}


def test_golden_set_ignores_none():
    assert F.GoldenSet().add(None) is False


# ---- canary assignment ----------------------------------------------------
def test_canary_assigns_roughly_fraction(monkeypatch):
    _canary(monkeypatch)
    cc = F.CanaryController()
    n = sum(cc.assign(f"user{i}", fraction=0.1) for i in range(2000))
    assert 150 <= n <= 250                        # ~10% of 2000, tolerant


def test_canary_assignment_is_stable(monkeypatch):
    _canary(monkeypatch)
    cc = F.CanaryController()
    assert cc.assign("user42") == cc.assign("user42")


def test_canary_fraction_extremes(monkeypatch):
    _canary(monkeypatch)
    cc = F.CanaryController()
    assert cc.assign("x", fraction=1.0) is True
    assert cc.assign("x", fraction=0.0) is False


def test_canary_disabled_never_assigns(monkeypatch):
    monkeypatch.setattr(F, "canary_enabled", lambda: False)
    assert F.CanaryController().assign("user1", fraction=1.0) is False


# ---- canary verdict -------------------------------------------------------
def test_promote_when_canary_beats_baseline(monkeypatch):
    _canary(monkeypatch)
    v = F.CanaryController().evaluate(baseline_score=0.80, canary_score=0.85,
                                      samples=100, min_samples=50, margin=0.02)
    assert v.decision == F.PROMOTE


def test_rollback_when_canary_worse(monkeypatch):
    _canary(monkeypatch)
    v = F.CanaryController().evaluate(baseline_score=0.80, canary_score=0.74,
                                      samples=100, min_samples=50, margin=0.02)
    assert v.decision == F.ROLLBACK


def test_hold_within_margin(monkeypatch):
    _canary(monkeypatch)
    v = F.CanaryController().evaluate(baseline_score=0.80, canary_score=0.805,
                                      samples=100, min_samples=50, margin=0.02)
    assert v.decision == F.HOLD


def test_hold_when_insufficient_samples(monkeypatch):
    _canary(monkeypatch)
    v = F.CanaryController().evaluate(baseline_score=0.80, canary_score=0.99,
                                      samples=10, min_samples=50, margin=0.02)
    assert v.decision == F.HOLD and "samples" in v.reason


def test_verdict_disabled_holds(monkeypatch):
    monkeypatch.setattr(F, "canary_enabled", lambda: False)
    v = F.CanaryController().evaluate(baseline_score=0.5, canary_score=0.99,
                                      samples=1000)
    assert v.decision == F.HOLD


def test_verdict_never_raises(monkeypatch):
    _canary(monkeypatch)
    v = F.CanaryController().evaluate(baseline_score=None, canary_score=None,  # type: ignore[arg-type]
                                      samples=100, min_samples=1)
    assert isinstance(v, F.CanaryVerdict)


def test_end_to_end_change_ships_behind_canary(monkeypatch):
    # §8.9 acceptance: a change rides a 10% canary and auto-rolls-back on a
    # metric regression.
    _canary(monkeypatch)
    cc = F.CanaryController()
    assert cc.assign("some-user", fraction=0.1) in (True, False)   # deterministic
    regressed = cc.evaluate(baseline_score=0.90, canary_score=0.82,
                            samples=200, min_samples=50, margin=0.02)
    assert regressed.decision == F.ROLLBACK
