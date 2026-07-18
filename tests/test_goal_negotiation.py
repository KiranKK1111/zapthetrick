"""General goal negotiation (roadmap Phase 5 #22).

Pins the request → closest-achievable-goal downgrade beyond document format: an
unavailable capability walks the fallback ladder to something we CAN do, with a
reason; an achievable goal passes through unchanged.
"""
from __future__ import annotations

from app.quality.goal_negotiation import (DEPLOY, EXPLAIN, RUN_CODE,
                                          WEB_RESEARCH, WRITE_CODE,
                                          NegotiatedGoal, negotiate_goal)


def _none_available(_cap):
    return False


def _all_available(_cap):
    return True


def test_achievable_goal_passes_through():
    r = negotiate_goal(WRITE_CODE, available=_all_available)
    assert isinstance(r, NegotiatedGoal)
    assert not r.downgraded and r.achievable == WRITE_CODE


def test_run_code_without_sandbox_downgrades_to_write_code():
    r = negotiate_goal(RUN_CODE, available=_none_available)
    assert r.downgraded and r.achievable == WRITE_CODE
    assert "sandbox" in r.reason.lower() or "write" in r.reason.lower()


def test_web_research_without_search_downgrades_to_explain():
    r = negotiate_goal(WEB_RESEARCH, available=_none_available)
    assert r.downgraded and r.achievable == EXPLAIN


def test_deploy_walks_ladder_to_something_achievable():
    # deploy → run_code → write_code (write_code has no requirement).
    r = negotiate_goal(DEPLOY, available=_none_available)
    assert r.downgraded and r.achievable == WRITE_CODE


def test_deploy_stays_when_capability_present():
    r = negotiate_goal(DEPLOY, available=_all_available)
    assert not r.downgraded and r.achievable == DEPLOY


def test_unknown_goal_passes_through():
    r = negotiate_goal("teleport", available=_none_available)
    assert not r.downgraded and r.achievable == "teleport"


def test_failopen_on_bad_available():
    def boom(_c):
        raise RuntimeError("nope")
    r = negotiate_goal(RUN_CODE, available=boom)
    assert isinstance(r, NegotiatedGoal)
