"""Stage-5 §2.7 F — Live session model plan: pin primary+standby, reserve, failover."""
from __future__ import annotations

import pytest

from app.llm import live_plan as LP
from app.llm import quota_plan as Q
from app.llm.live_plan import Candidate, LivePlanner


@pytest.fixture(autouse=True)
def _fresh():
    LP.reset_for_tests()
    Q.reset_for_tests()
    yield
    LP.reset_for_tests()
    Q.reset_for_tests()


def _cands():
    # groq is the best; cerebras serves the SAME model; nvidia a different one.
    return [
        Candidate("llama|70|3.3|-", "groq", model_db_id=1, key_id=10),
        Candidate("llama|70|3.3|-", "cerebras", model_db_id=2, key_id=20),
        Candidate("qwen|72|2.5|-", "nvidia", model_db_id=3, key_id=30),
    ]


class TestPinning:
    def test_pins_primary_and_standby(self):
        p = LivePlanner()
        plan = p.plan("s1", "live_answer", _cands(), reserve=False)
        assert plan is not None
        assert plan.primary.provider == "groq"
        assert plan.standby is not None

    def test_standby_prefers_same_model_different_provider(self):
        p = LivePlanner()
        plan = p.plan("s1", "live_answer", _cands(), reserve=False)
        # Standby = same canonical model on a DIFFERENT provider (voice
        # consistency + failover diversity), not the different qwen model.
        assert plan.standby.provider == "cerebras"
        assert plan.standby.cid_key == plan.primary.cid_key

    def test_standby_falls_to_a_different_model_if_no_same_model_elsewhere(self):
        p = LivePlanner()
        cands = [Candidate("llama|70|3.3|-", "groq", key_id=1),
                 Candidate("qwen|72|2.5|-", "nvidia", key_id=2)]
        plan = p.plan("s1", "live_answer", cands, reserve=False)
        assert plan.standby.cid_key == "qwen|72|2.5|-"

    def test_no_standby_when_only_one_provider(self):
        p = LivePlanner()
        plan = p.plan("s1", "live_answer",
                      [Candidate("llama|70|3.3|-", "groq", key_id=1)],
                      reserve=False)
        assert plan.standby is None

    def test_no_healthy_candidate_returns_none(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "gauntlet", True, raising=False)
        # Every candidate is unproven → quarantined → no pin (ordinary ladder).
        from app.llm import gauntlet as G
        G.reset_for_tests()
        p = LivePlanner()
        assert p.plan("s1", "live_answer", _cands(), reserve=False) is None
        G.reset_for_tests()


class TestFailover:
    def test_sticky_to_primary(self):
        p = LivePlanner()
        p.plan("s1", "live_answer", _cands(), reserve=False)
        assert p.next_model("s1").provider == "groq"
        assert p.next_model("s1").provider == "groq"     # sticky, no re-eval

    def test_primary_failure_hands_off_to_standby(self):
        p = LivePlanner()
        p.plan("s1", "live_answer", _cands(), reserve=False)
        nxt = p.next_model("s1", primary_failed=True)
        assert nxt.provider == "cerebras"                # standby, zero re-eval
        # The handoff persists across turns (no flip-flop back to primary).
        assert p.next_model("s1").provider == "cerebras"

    def test_both_failed_returns_none(self):
        p = LivePlanner()
        p.plan("s1", "live_answer", _cands(), reserve=False)
        p.next_model("s1", primary_failed=True)
        p.note_standby_failed("s1")
        assert p.next_model("s1") is None                # → ordinary ladder

    def test_unknown_session_returns_none(self):
        assert LivePlanner().next_model("nope") is None


class TestReservation:
    def test_reserve_reduces_headroom(self):
        p = LivePlanner()
        # groq daily limit is 14_400; reserve 100 across primary(key 10) + standby.
        before = Q.quota_planner().headroom("groq", 10)
        p.plan("s1", "live_answer", _cands(), expected_requests=100)
        after = Q.quota_planner().headroom("groq", 10)
        assert after == before - 100

    def test_release_refunds_the_reservation(self):
        p = LivePlanner()
        before = Q.quota_planner().headroom("groq", 10)
        p.plan("s1", "live_answer", _cands(), expected_requests=100)
        p.release("s1")
        assert Q.quota_planner().headroom("groq", 10) == before

    def test_reserve_holds_on_both_pinned_models(self):
        p = LivePlanner()
        cer_before = Q.quota_planner().headroom("cerebras", 20)
        p.plan("s1", "live_answer", _cands(), expected_requests=50)
        # cerebras is the standby → its ledger is reserved too.
        assert Q.quota_planner().headroom("cerebras", 20) == cer_before - 50

    def test_replan_releases_the_old_reservation(self):
        p = LivePlanner()
        before = Q.quota_planner().headroom("groq", 10)
        p.plan("s1", "live_answer", _cands(), expected_requests=100)
        p.plan("s1", "live_answer", _cands(), expected_requests=30)  # re-plan
        # Only the new reservation stands (old 100 released, new 30 held).
        assert Q.quota_planner().headroom("groq", 10) == before - 30


class TestGauntletHealth:
    def test_quarantined_primary_is_skipped(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "gauntlet", True, raising=False)
        from app.llm import gauntlet as G
        from app.llm.gauntlet import Scorecard
        G.reset_for_tests()
        # Only cerebras's llama is probed-healthy; groq's is quarantined.
        G.gauntlet().record("llama|70|3.3|-", "cerebras", Scorecard(probed_at=1.0))
        p = LivePlanner()
        plan = p.plan("s1", "live_answer", _cands(), reserve=False)
        assert plan.primary.provider == "cerebras"       # groq skipped
        G.reset_for_tests()
