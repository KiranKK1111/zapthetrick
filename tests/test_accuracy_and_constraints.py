"""Accuracy ledger, performance-constraint directives, and profile-question
detection (this round's chat/live accuracy work)."""
from __future__ import annotations

import json

import pytest

from app.chat.perf_constraints import extract_performance_constraints
from app.live import ledger
from app.live.profile import (CandidateProfile, first_person_directive,
                              is_profile_question, profile_summary)


# ── Performance constraints ──────────────────────────────────────────────────

def test_time_bound_extracted():
    d = extract_performance_constraints(
        "give me the code for this problem statement which can execute "
        "within 500 millisecond")
    assert "within 500ms" in d
    assert "PERFORMANCE REQUIREMENTS" in d


def test_time_range_extracted():
    d = extract_performance_constraints(
        "get me the solution in python for this for which execute time "
        "should be between 500 millisecond to 1000 milliseconds")
    assert "between 500ms and 1000ms" in d


def test_big_o_and_case_and_space():
    d = extract_performance_constraints(
        "solve it in O(n log n) worst case with constant space")
    assert "O(n log n)" in d
    assert "worst-case" in d
    assert "constant space" in d


def test_no_constraints_no_directive():
    assert extract_performance_constraints(
        "write a python function to reverse a string") == ""
    assert extract_performance_constraints("") == ""


def test_language_clarify_still_required_without_language():
    # "give me the code … within 500 ms" names NO language → the required-slot
    # pre-gate must still ask which language (the perf directive doesn't
    # swallow the clarify).
    from app.clarify.intent_pipeline import CLARIFY, assess
    a = assess("give me the code for this problem statement which can "
               "execute within 500 millisecond", [], {})
    assert a.decision == CLARIFY
    assert "language" in a.missing_required


def test_fully_specified_prompt_has_no_required_slot_missing():
    # Language named ("in python") → nothing REQUIRED is missing, so the
    # blocking language card must not appear (the turn answers; the perf
    # directive rides along).
    from app.clarify.intent_pipeline import CLARIFY, assess
    a = assess("get me the solution in python for this problem for which "
               "execute time should be between 500 and 1000 milliseconds",
               [], {})
    assert a.decision != CLARIFY
    assert "language" not in (a.missing_required or [])


# ── Profile-question detection ───────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "Tell me about yourself.",
    "Walk me through your experience.",
    "What projects have you worked on recently?",
    "Tell us about your current role.",
    "Why should we hire you?",
    "What's your tech stack?",
])
def test_profile_questions_detected(q):
    assert is_profile_question(q), q


@pytest.mark.parametrize("q", [
    "What is a hash map?",
    "How does Kafka guarantee ordering?",
    "Explain dependency injection.",
])
def test_general_questions_are_not_profile(q):
    assert not is_profile_question(q), q


def test_profile_summary_covers_all_sections():
    p = CandidateProfile(
        skills=["Java", "Kafka"],
        projects=[{"name": "Billing platform", "tech": ["Spring"],
                   "company": "Acme"}],
        achievements=["Cut p99 latency 40%"],
        experience="6 years backend engineering",
        metrics=["1M req/day"],
    )
    lines = profile_summary(p)
    blob = "\n".join(lines)
    assert "6 years" in blob
    assert "Billing platform" in blob and "Spring" in blob and "Acme" in blob
    assert "Cut p99" in blob
    assert "first-person" in first_person_directive().lower()


# ── Accuracy ledger ──────────────────────────────────────────────────────────

def test_ledger_records_and_summarizes(tmp_path, monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "ledger_path",
                        str(tmp_path / "ledger.jsonl"), raising=False)
    ledger.reset_for_tests()
    ledger.record("s1", "q1", "How does Kafka work?", ledger.ANSWERED,
                  qtype="technical_concept")
    ledger.record("s1", "q2", "nice weather", ledger.SKIPPED,
                  reason="not_a_question")
    ledger.record("s1", "q3", "Suppose the DB dies.", ledger.PROMOTED,
                  reason="hypothetical", signals={"promoted": "hypothetical"})
    ledger.feedback("s1", "q2", "should_have_answered",
                    utterance="nice weather")
    s = ledger.summary()
    assert s["decisions"]["answered"] == 1
    assert s["decisions"]["skipped"] == 1
    assert s["decisions"]["promoted"] == 1
    assert s["feedback"]["should_have_answered"] == 1
    # The JSONL is the durable labeled corpus.
    lines = [json.loads(x) for x in
             (tmp_path / "ledger.jsonl").read_text(
                 encoding="utf-8").strip().splitlines()]
    assert len(lines) == 4
    kinds = {x["kind"] for x in lines}
    assert kinds == {"decision", "feedback"}


def test_ledger_disabled_is_noop(tmp_path, monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "accuracy_ledger", False, raising=False)
    monkeypatch.setattr(cfg.live, "ledger_path",
                        str(tmp_path / "off.jsonl"), raising=False)
    ledger.reset_for_tests()
    ledger.record("s1", "q1", "anything", ledger.ANSWERED)
    assert not (tmp_path / "off.jsonl").exists()
    assert ledger.summary()["decisions"] == {}
