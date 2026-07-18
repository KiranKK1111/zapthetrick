"""Tests for the four built-out Phase 2 gaps:
  #13 Answer Readiness Score   (app/live/readiness.py)
  #23 Conversation Contract    (app/live/contract.py)
  #31 Rhythm / Fatigue         (app/live/rhythm.py)
  #32 Conversation Steering    (app/live/steer.py)
All deterministic + offline.
"""
from __future__ import annotations

import pytest

from app.live import contract as C
from app.live import readiness as R
from app.live import rhythm as RH
from app.live import steer as S


# ── #13 readiness ──────────────────────────────────────────────────────────
def test_readiness_bounds_and_monotonic_in_confidence():
    lo = R.readiness_score(confidence=0.1, answer_chars=40)
    hi = R.readiness_score(confidence=0.9, answer_chars=40)
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert hi > lo


def test_readiness_rewards_evidence_and_length_and_verification():
    base = R.readiness_score(confidence=0.5, answer_chars=0)
    with_ev = R.readiness_score(confidence=0.5, answer_chars=0, has_evidence=True)
    with_len = R.readiness_score(confidence=0.5, answer_chars=100)
    verified = R.readiness_score(confidence=0.5, answer_chars=100, verified=True)
    unverified = R.readiness_score(confidence=0.5, answer_chars=100, verified=False)
    assert with_ev > base
    assert with_len > base
    assert verified > unverified


def test_ready_to_stream_threshold():
    assert R.ready_to_stream(0.6, 0.5) is True
    assert R.ready_to_stream(0.4, 0.5) is False


def test_readiness_fail_open_on_bad_input():
    assert R.readiness_score(confidence="oops") == 0.0  # type: ignore[arg-type]


# ── #23 contract ───────────────────────────────────────────────────────────
def test_contract_length_budget_and_validation():
    c = C.Contract(max_answer_seconds=30)
    short = "This is a concise answer."
    long = "word " * 400  # ~2000 chars >> 30s budget
    assert C.validate(short, c).ok
    chk = C.validate(long, c)
    assert not chk.ok and "too_long" in chk.violations


def test_contract_hr_phase_is_tighter():
    tech = C.derive_contract(phase="technical_screening")
    hr = C.derive_contract(phase="hr")
    assert hr.max_answer_seconds < tech.max_answer_seconds


def test_contract_registry_lifecycle():
    C.forget_session("s")
    c1 = C.ensure_contract("s", phase="hr")
    c2 = C.ensure_contract("s")  # idempotent — same contract
    assert c1 is c2
    C.forget_session("s")
    assert C.get_contract("s") is None


# ── #31 rhythm ─────────────────────────────────────────────────────────────
def test_rhythm_cadence_classes():
    rapid = RH.RhythmTracker()
    for _ in range(4):
        rapid.observe(3.0)  # 3s gaps → rapid fire
    assert rapid.cadence() == "rapid_fire" and rapid.is_rapid_fire()

    slow = RH.RhythmTracker()
    for _ in range(4):
        slow.observe(60.0)
    assert slow.cadence() == "slow"

    steady = RH.RhythmTracker()
    for _ in range(4):
        steady.observe(20.0)
    assert steady.cadence() == "steady"


def test_rhythm_fatigue_grows():
    t = RH.RhythmTracker()
    f0 = t.fatigue()
    for _ in range(30):
        t.observe(30.0)
    assert t.fatigue() > f0
    assert t.fatigue() == 1.0  # past the saturation count


def test_rhythm_registry_isolated():
    RH.forget_session("a"); RH.forget_session("b")
    a, b = RH.get_rhythm("a"), RH.get_rhythm("b")
    a.observe(5.0)
    assert b.count == 0 and a.count == 1


# ── #32 steering ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("q,open_", [
    ("Tell me about yourself", True),
    ("Walk me through your background", True),
    ("What is a Kafka partition?", False),
    ("Explain the CAP theorem", False),
])
def test_is_open_prompt(q, open_):
    assert S.is_open_prompt(q) is open_


def test_steering_leads_with_strength_only_when_open():
    d = S.steering_directive("Tell me about yourself",
                             ["led a 40% latency cut", "scaled to 1M users"])
    assert d and "led a 40% latency cut" in d and "bridge to" in d
    # Not an open prompt -> no steering.
    assert S.steering_directive("What is a mutex?", ["x"]) is None
    # Open but no strengths -> a generic steer (still useful), not None.
    generic = S.steering_directive("Tell me about yourself", [])
    assert generic and "strongest" in generic


def test_rhythm_observe_now_is_live_safe():
    t = RH.RhythmTracker()
    t.observe_now()   # first — no gap
    t.observe_now()   # second — derives a tiny monotonic gap
    assert t.count == 2  # never raises, always counts
