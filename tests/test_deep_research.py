"""Tests for deep research / subagent isolation (vNext §8.6, Stage 11 E)."""
from __future__ import annotations

import app.agents.deep_research as R


def _on(monkeypatch):
    monkeypatch.setattr(R, "enabled", lambda: True)


# ---- isolation contract / distill ----------------------------------------
def test_distill_keeps_grounded_result():
    d = R.distill("a grounded finding", ["src1", "src2"])
    assert not d.dropped and d.summary == "a grounded finding"
    assert d.citations == ["src1", "src2"]


def test_distill_drops_uncited():
    d = R.distill("finding with no source", [])
    assert d.dropped and d.summary == ""


def test_distill_truncates_to_cap():
    d = R.distill("word " * 5000, ["src"],
                  contract=R.IsolationContract(result_cap_tokens=100))
    assert d.truncated
    assert len(d.summary) <= 100 * 4 + 4


def test_distill_allows_uncited_when_not_required():
    c = R.IsolationContract(require_citations=False)
    d = R.distill("finding", [], contract=c)
    assert not d.dropped and d.summary == "finding"


def test_distill_never_raises():
    assert isinstance(R.distill(None, None), R.DistilledResult)  # type: ignore[arg-type]


# ---- planner --------------------------------------------------------------
def test_plan_template_makes_2_to_4_workers():
    plan = R.plan_research("how does raft consensus work")
    assert 2 <= len(plan.workers) <= 4
    assert plan.shared_prefix                     # L0 prefix present


def test_plan_uses_injected_planner():
    plan = R.plan_research("X", planner_fn=lambda q: ["angle a", "angle b"])
    assert plan.workers == ["angle a", "angle b"]


def test_plan_clamps_to_max_workers():
    plan = R.plan_research("X", planner_fn=lambda q: ["a", "b", "c", "d", "e", "f"],
                           max_workers=4)
    assert len(plan.workers) == 4


def test_plan_min_two_workers():
    plan = R.plan_research("X", planner_fn=lambda q: ["only one"])
    assert len(plan.workers) >= 2


def test_plan_falls_back_on_planner_error():
    plan = R.plan_research("Y", planner_fn=lambda q: 1 / 0)
    assert 2 <= len(plan.workers) <= 4            # template fallback ran


# ---- synthesis ------------------------------------------------------------
def _results():
    return [R.distill("Finding A about raft", ["src1"]),
            R.distill("Finding B about elections", ["src2"]),
            R.distill("uncited noise", [])]       # dropped


def test_synthesize_merges_grounded_only(monkeypatch):
    _on(monkeypatch)
    s = R.synthesize("raft", _results())
    assert s.workers_used == 2                    # the uncited one dropped
    assert "Finding A" in s.answer and "Finding B" in s.answer
    assert set(s.citations) == {"src1", "src2"}


def test_synthesize_uses_injected_synth(monkeypatch):
    _on(monkeypatch)
    s = R.synthesize("raft", _results(),
                     synth_fn=lambda q, rs: "one coherent answer")
    assert s.answer == "one coherent answer"


def test_synthesize_disabled_is_empty(monkeypatch):
    monkeypatch.setattr(R, "enabled", lambda: False)
    assert R.synthesize("q", _results()).answer == ""


def test_synthesize_never_raises(monkeypatch):
    _on(monkeypatch)
    assert isinstance(R.synthesize("q", [None, None]), R.Synthesis)


# ---- one-follow-up-wave gate ---------------------------------------------
def test_followup_needed_when_too_few_grounded():
    need, gaps = R.needs_followup([R.distill("only one", ["src"])], min_workers=2)
    assert need is True


def test_followup_not_needed_with_enough_grounded():
    res = [R.distill("a", ["s1"]), R.distill("b", ["s2"])]
    need, _ = R.needs_followup(res, min_workers=2)
    assert need is False


def test_followup_capped_at_one_wave():
    need, _ = R.needs_followup([R.distill("only one", ["src"])],
                               min_workers=2, followups_done=1)
    assert need is False                          # one wave max


def test_followup_triggered_by_worker_gap():
    class G:
        summary = "x"; citations = ["s"]; dropped = False
        gaps = ["what about byzantine faults?"]
    need, gaps = R.needs_followup([G()])
    assert need and gaps == ["what about byzantine faults?"]


def test_end_to_end_bounded_cited_research(monkeypatch):
    # §8.6 acceptance: bounded (≤4 workers, one follow-up wave), cited, fail-soft.
    _on(monkeypatch)
    plan = R.plan_research("compare raft vs paxos")
    assert len(plan.workers) <= 4
    results = [R.distill(f"finding {i}", [f"src{i}"]) for i in range(len(plan.workers))]
    s = R.synthesize("compare raft vs paxos", results)
    assert s.answer and s.citations                # cited
    assert not R.needs_followup(results, followups_done=1)[0]   # bounded
