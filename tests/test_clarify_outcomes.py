"""Tests for the clarification outcome store + confidence calibration
(advanced-intent-reasoning Phase 1). Pure / dict-backed — no DB, no LLM."""
from __future__ import annotations

from app.clarify.calibration import calibrate
from app.clarify.outcomes import OutcomeStore, confidence_bucket


class TestOutcomeStore:
    def test_decision_then_answered_resolves_to_ring(self):
        prefs: dict = {}
        s = OutcomeStore(prefs)
        s.record_decision("c1", "project_build", 0.3, asked=True)
        assert s.has_pending("c1")
        s.record_response("c1", "answered")
        assert not s.has_pending("c1")
        buckets = s.calibration_buckets()
        # bucket 3 (0.3) → one "needed" (user answered the question)
        assert buckets[3]["needed"] == 1
        assert buckets[3]["answerable"] == 0

    def test_overridden_counts_as_answerable(self):
        s = OutcomeStore({})
        s.record_decision("c1", "code_generation", 0.3, asked=True)
        s.record_response("c1", "overridden")
        buckets = s.calibration_buckets()
        assert buckets[3]["answerable"] == 1
        assert buckets[3]["needed"] == 0

    def test_answer_only_decision_is_not_recorded(self):
        s = OutcomeStore({})
        s.record_decision("c1", "code_generation", 0.95, asked=False)
        assert not s.has_pending("c1")
        assert s.calibration_buckets() == {}

    def test_no_raw_text_stored(self):
        prefs: dict = {}
        s = OutcomeStore(prefs)
        s.record_decision("c1", "project_build", 0.3, asked=True)
        s.record_response("c1", "answered")
        blob = repr(prefs)
        # Only intent label + bucket ints; assert the ring entry keys are minimal.
        entry = prefs["clarify_outcomes"]["ring"][0]
        assert set(entry.keys()) == {"b", "i", "r"}
        assert "project_build" in blob  # intent label allowed
        assert isinstance(entry["b"], int)

    def test_ring_is_capped(self):
        s = OutcomeStore({})
        for i in range(250):
            cid = f"c{i}"
            s.record_decision(cid, "x", 0.5, asked=True)
            s.record_response(cid, "answered")
        ring = s._o["ring"]  # noqa: SLF001 — white-box cap check
        assert len(ring) == 200

    def test_counters_and_decay(self):
        s = OutcomeStore({})
        s.record_decision("c1", "x", 0.5, asked=True)
        s.record_response("c1", "answered")
        assert s.counters()["answered"] == 1
        assert s.counters()["recent"] == 1
        s.decay_recent()
        assert s.counters()["recent"] == 0
        s.decay_recent()  # floors at 0
        assert s.counters()["recent"] == 0

    def test_shares_root_with_other_prefs(self):
        # The outcome store must not clobber sibling keys (preferences share it).
        prefs = {"clarify": {"durable": {"Language": "Python"}}}
        s = OutcomeStore(prefs)
        s.record_decision("c1", "x", 0.4, asked=True)
        s.record_response("c1", "answered")
        assert prefs["clarify"]["durable"]["Language"] == "Python"
        assert "clarify_outcomes" in prefs

    def test_unknown_response_kind_ignored(self):
        s = OutcomeStore({})
        s.record_decision("c1", "x", 0.5, asked=True)
        s.record_response("c1", "bogus")
        assert s.has_pending("c1")  # not resolved
        assert s.calibration_buckets() == {}


class TestConfidenceBucket:
    def test_bucketing(self):
        assert confidence_bucket(0.0) == 0
        assert confidence_bucket(0.34) == 3
        assert confidence_bucket(0.35) == 4   # rounds
        assert confidence_bucket(1.0) == 10
        assert confidence_bucket(5.0) == 10   # clamps
        assert confidence_bucket(-1.0) == 0
        assert confidence_bucket("x") == 0


class TestCalibration:
    def test_identity_below_min_samples(self):
        s = OutcomeStore({})
        s.record_decision("c1", "x", 0.3, asked=True)
        s.record_response("c1", "answered")
        buckets = s.calibration_buckets()
        # Only 1 sample in bucket 3 → identity.
        assert calibrate(0.3, buckets, min_samples=8) == 0.3

    def test_no_data_is_identity(self):
        assert calibrate(0.42, {}, min_samples=8) == 0.42

    def test_blends_toward_observed_when_enough_samples(self):
        s = OutcomeStore({})
        # 10 asks at bucket 3, all OVERRIDDEN → observed answerable rate 1.0.
        for i in range(10):
            cid = f"c{i}"
            s.record_decision(cid, "x", 0.3, asked=True)
            s.record_response(cid, "overridden")
        buckets = s.calibration_buckets()
        out = calibrate(0.3, buckets, min_samples=8)
        # blend 0.5*0.3 + 0.5*1.0 = 0.65 → calibration raised confidence
        assert out > 0.3
        assert abs(out - 0.65) < 1e-9

    def test_blend_lowers_when_questions_were_needed(self):
        s = OutcomeStore({})
        # 10 asks at bucket 7, all ANSWERED → observed answerable rate 0.0.
        for i in range(10):
            cid = f"c{i}"
            s.record_decision(cid, "x", 0.7, asked=True)
            s.record_response(cid, "answered")
        buckets = s.calibration_buckets()
        out = calibrate(0.7, buckets, min_samples=8)
        assert out < 0.7   # 0.5*0.7 + 0.5*0.0 = 0.35

    def test_always_in_range(self):
        for raw in (-1.0, 0.0, 0.5, 1.0, 2.0):
            assert 0.0 <= calibrate(raw, {}, 8) <= 1.0
