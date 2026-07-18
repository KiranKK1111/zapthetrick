"""Phase-6 (ArchitectureVerdict.md): the behavioral benchmark corpus is the
CI regression gate for orchestration behavior — SeveralFeatures.md's own
targets, asserted on every run:

    unnecessary clarification rate  < 5%
    missed clarification rate       < 2%

Any decision-layer change that breaks a doc scenario fails here with the
exact prompt + expected/got in the report.
"""
from __future__ import annotations

from app.eval.behavior_bench import load_corpus, run_corpus
from app.obs import decision_metrics as dm


class TestBehaviorBench:
    def test_corpus_loads_and_is_labeled(self):
        rows = load_corpus()
        assert len(rows) >= 40
        assert all(r.get("prompt") and r.get("expected") for r in rows)

    def test_doc_targets_hold(self):
        report = run_corpus()
        assert report["unnecessary_clarification_rate"] < 0.05, \
            report["failures"]
        assert report["missed_clarification_rate"] < 0.02, report["failures"]

    def test_full_accuracy_on_seed_corpus(self):
        # The seed corpus encodes deterministic doc scenarios — they must ALL
        # hold. New, harder rows may relax this to the rate targets above.
        report = run_corpus()
        assert report["accuracy"] == 1.0, report["failures"]


class TestDecisionMetrics:
    def test_gate_counters_and_rates(self):
        dm.reset_for_tests()
        dm.record_gate_decision("answer", "answer_direct")
        dm.record_gate_decision("clarify", "clarify_missing_required")
        dm.record_gate_decision("clarify", "clarify_missing_required")
        dm.record_gate_decision("defer", "defer_to_gate")
        s = dm.snapshot()
        assert s["gate"]["total"] == 4
        assert s["gate"]["clarify_rate"] == 0.5
        assert s["policy_rules"]["clarify_missing_required"] == 2

    def test_artifact_counters(self):
        dm.reset_for_tests()
        dm.record_artifact_validation({"validated": True, "method": "pypdf",
                                       "repaired": False,
                                       "degraded_from": None})
        dm.record_artifact_validation({"validated": True, "method": "zip",
                                       "repaired": True,
                                       "degraded_from": None})
        dm.record_artifact_validation({"validated": True, "method": "zip",
                                       "repaired": False,
                                       "degraded_from": "pdf"})
        dm.record_artifact_validation({"validated": False, "method": "pypdf",
                                       "repaired": False,
                                       "degraded_from": None})
        dm.record_artifact_validation({"validated": False,
                                       "method": "disabled"})
        s = dm.snapshot()["artifacts"]
        assert s == {"validated": 1, "repaired": 1, "degraded": 1,
                     "failed": 1, "skipped": 0}

    def test_assess_feeds_metrics(self):
        from app.clarify.intent_pipeline import assess
        dm.reset_for_tests()
        assess("what is a hash map?")
        assess("write a login api")
        s = dm.snapshot()
        assert s["gate"]["total"] == 2
        assert s["gate"]["counts"].get("answer", 0) >= 1
        assert s["gate"]["counts"].get("clarify", 0) >= 1

    def test_recording_never_raises(self):
        dm.record_gate_decision(None, None)          # type: ignore[arg-type]
        dm.record_artifact_validation("garbage")     # type: ignore[arg-type]
        assert isinstance(dm.snapshot(), dict)
