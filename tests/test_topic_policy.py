"""Topic-risk policy gate (personalization-and-governance R4, task 6.3).

Pins Property 4: sensitive → caveats + recommend professional + prohibited
advice listed; general → today (no-op); deterministic (no LLM call).
"""
from __future__ import annotations

import inspect

from app.personalization import policy as P


def test_general_is_noop():
    assert P.classify("how do I reverse a list in python") == P.GENERAL
    strat = P.strategy_for(P.GENERAL)
    assert strat.directive == "" and not strat.add_caveat


def test_medical_classified_and_caveated():
    assert P.classify("should I take 800mg of ibuprofen for my fever?") == P.MEDICAL
    strat = P.strategy_for(P.MEDICAL)
    assert strat.add_caveat
    assert "medical professional" in strat.recommend_professional
    assert any("diagnosis" in p for p in strat.prohibited)
    assert "general" in strat.directive.lower()    # still offers general help


def test_legal_and_financial():
    assert P.classify("can I sue my landlord for this?") == P.LEGAL
    assert P.classify("should I invest my retirement savings in crypto?") == P.FINANCIAL
    assert P.strategy_for(P.LEGAL).recommend_professional
    assert P.strategy_for(P.FINANCIAL).recommend_professional


def test_engineering_mention_is_general_not_medical():
    # "build a medical app" is engineering, not personal medical advice.
    assert P.classify("build a medical records app in Flutter") == P.GENERAL


def test_classify_is_deterministic_no_llm():
    assert not inspect.iscoroutinefunction(P.classify)
    a = P.classify("should I take this medication?")
    b = P.classify("should I take this medication?")
    assert a == b == P.MEDICAL


def test_classify_failopen_to_general():
    assert P.classify(None) == P.GENERAL
