"""Tests for clarification fatigue + trust adaptation
(advanced-intent-reasoning Phase 2). Pure functions — no DB, no LLM."""
from __future__ import annotations

from app.clarify.adaptation import (
    adapted_answer_band,
    fatigue_threshold,
    trust_factor,
)


class TestTrustFactor:
    def test_neutral_with_no_history(self):
        assert trust_factor(0, 0) == 1.0

    def test_more_skips_raises_toward_ceiling(self):
        # All skips → high suppression multiplier.
        assert trust_factor(10, 0) >= 1.4

    def test_more_answers_lowers_toward_floor(self):
        assert trust_factor(0, 10) <= 0.7

    def test_always_bounded(self):
        for s in (0, 1, 5, 50):
            for a in (0, 1, 5, 50):
                assert 0.5 <= trust_factor(s, a) <= 1.5

    def test_monotonic_in_skip_rate(self):
        low = trust_factor(1, 9)
        high = trust_factor(9, 1)
        assert high > low


class TestFatigueThreshold:
    def test_no_recent_is_identity(self):
        assert fatigue_threshold(0.90, 0) == 0.90

    def test_lowers_band_with_recent(self):
        assert fatigue_threshold(0.90, 3) < 0.90

    def test_monotonic_decrease(self):
        b1 = fatigue_threshold(0.90, 1)
        b3 = fatigue_threshold(0.90, 3)
        assert b3 < b1

    def test_floored(self):
        # Many recent clarifications can't drop the band below the floor.
        assert fatigue_threshold(0.90, 100) >= 0.5

    def test_trust_scales_adjustment(self):
        # Higher trust multiplier (eroded trust) lowers the band more.
        eroded = fatigue_threshold(0.90, 3, trust=1.5)
        engaged = fatigue_threshold(0.90, 3, trust=0.5)
        assert eroded < engaged

    def test_bad_base_safe(self):
        assert 0.0 <= fatigue_threshold("x", 3) <= 1.0


class TestAdaptedAnswerBand:
    def test_recovery_to_baseline(self):
        # Quiet conversation (recent 0) → exactly the base band.
        assert adapted_answer_band(0.90, 0, 5, 5) == 0.90

    def test_fatigue_plus_eroded_trust_suppresses_most(self):
        base = 0.90
        # Recent volume + all skips → lowest band (ask least).
        suppressed = adapted_answer_band(base, 4, 8, 0)
        mild = adapted_answer_band(base, 1, 0, 8)
        assert suppressed < mild <= base

    def test_stays_in_range(self):
        for recent in (0, 1, 4, 20):
            v = adapted_answer_band(0.90, recent, 3, 3)
            assert 0.0 <= v <= 1.0
