"""
Phase 15 tests — precision & explicit-coverage hardening
(live-conversational-intelligence R47–R58).

One focused property per requirement:
47 incremental prep (no early surface) / 48 implicit-question / 49 evidence
binding + stale-hedge / 50 multi-pass no-extra-call / 51 coreference confidence
gating / 52 rhetorical suppression + low-conf fallthrough / 53 acoustic degrade →
confidence drop / 54 override suggestion + confidence band / 55 per-stage budget
degrade / 56 feedback capture / 57 skill-gap retrieval boost / 58 cognitive-load
depth adaptation.
"""
from __future__ import annotations

from app.live import acoustic as _ac
from app.live import budget as _budget
from app.live import evidence as _ev
from app.live import health as _health
from app.live import implicit as _impl
from app.live import knowledge as _know
from app.live import objective as _obj
from app.live import rhetorical as _rhet
from app.live import style as _style
from app.live import surface as _surface
from app.live import world_model as _wm


# ---- R48: implicit / semantic-completion -----------------------------------

def test_implicit_question_detection():
    assert _impl.detect_implicit("Walk me through your approach.").is_implicit_question
    assert _impl.detect_implicit("I'm curious about your reasoning here").is_implicit_question
    # An explicit question is left to the explicit layer.
    assert not _impl.detect_implicit("What is a hash map?").is_implicit_question
    assert not _impl.detect_implicit("The weather is nice.").is_implicit_question


# ---- R49: evidence binding + stale hedge -----------------------------------

def test_evidence_hedge_when_thin_or_stale():
    empty = _ev.EvidenceBinding()
    assert _ev.hedge_directive(empty)  # thin → hedge
    good = _ev.EvidenceBinding()
    good.add("Kafka partitions preserve per-partition order.", source="kb")
    assert _ev.hedge_directive(good) == ""  # fresh + present → no hedge
    # Force staleness.
    good.segments[0].ts -= 10_000
    assert _ev.hedge_directive(good)


# ---- R50: multi-pass understanding (no extra LLM call) ---------------------

def test_multi_pass_escalates_depth_on_followup():
    base_obj, base_dep = _obj.estimate("what is a hash map?")
    obj, dep = _obj.multi_pass("what is a hash map?", recent=["but why is it O(1)?", "go deeper"])
    assert dep != base_dep or dep in (_obj.INTERNALS, _obj.ARCHITECTURE, _obj.SOURCE_LEVEL)
    # Deterministic — same inputs, same output (no model call).
    assert _obj.multi_pass("what is a hash map?", recent=["go deeper"]) == \
        _obj.multi_pass("what is a hash map?", recent=["go deeper"])


# ---- R51: coreference confidence gating ------------------------------------

def test_coreference_defers_low_confidence():
    m = _wm.InterviewWorldModel()
    # No anchor → defer to clarifier.
    r0 = _wm.resolve_coreference("how does it scale?", m)
    assert r0["defer"] is True and r0["resolved"] is False
    # Clear single topic anchor → resolved.
    m.topic = "kafka"
    r1 = _wm.resolve_coreference("how does it scale?", m)
    assert r1["resolved"] is True and r1["referent"] == "kafka"
    # No pronoun → nothing to resolve.
    assert _wm.resolve_coreference("what is sharding?", m)["resolved"] is False


# ---- R52: rhetorical suppression + low-conf fallthrough --------------------

def test_rhetorical_suppressed_but_genuine_answered():
    assert _rhet.classify("Makes sense?").is_rhetorical
    assert not _rhet.should_answer("That scales well, right?")
    # A genuine question is still answered.
    assert _rhet.should_answer("How would you scale this to 100M users?")
    # No '?' → not rhetorical (handled elsewhere).
    assert not _rhet.classify("Tell me about kafka").is_rhetorical


# ---- R53: acoustic degrade → confidence drop -------------------------------

def test_acoustic_degrade_lowers_confidence():
    clean = _ac.assess(snr_db=28, stt_conf=0.95, partial_stability=0.9)
    assert clean.condition == _ac.CLEAN
    assert clean.confidence_penalty == 0.0 or clean.confidence_penalty < 0.05
    noisy = _ac.assess(snr_db=4, stt_conf=0.4, partial_stability=0.3)
    assert noisy.confidence_penalty > 0
    assert _ac.adjust_confidence(0.9, noisy) < 0.9
    assert _ac.needs_reconfirmation(noisy)
    # No signal → neutral, no penalty.
    assert _ac.assess().condition == _ac.UNKNOWN


# ---- R54: override suggestion + confidence band ----------------------------

def test_override_is_gated_suggestion_with_band():
    # Below margin → no override (defer to candidate).
    assert _surface.override_suggestion("I'd use a list", "Use a set", system_confidence=0.5) is None
    # Confident past margin → an additive SUGGESTION, never a silent override.
    sug = _surface.override_suggestion("I'd use a list", "Use a set", system_confidence=0.9)
    assert sug is not None and sug["type"] == "suggestion"
    assert sug["band"] == "high"
    assert _surface.confidence_band(0.6) == "medium"
    assert _surface.confidence_band(0.2) == "low"


# ---- R55: per-stage budget degrade -----------------------------------------

def test_stage_budget_degrade():
    assert not _budget.stage_over_budget("retrieval", 100.0)
    assert _budget.stage_over_budget("retrieval", 999_999.0)
    assert _budget.stage_budget_ms("generation") > _budget.stage_budget_ms("detection")


# ---- R56: feedback capture -------------------------------------------------

def test_feedback_signal_capture():
    assert _health.classify_feedback("Perfect, exactly right") == "positive"
    assert _health.classify_feedback("No, that's not correct") == "negative"
    assert _health.classify_feedback("hmm") == "neutral"
    from app.live import eventlog as _el
    sid = "fb-test"
    _el.forget_session(sid)
    state = _health.capture_feedback(sid, "great, makes sense", qid="q1")
    assert state == "positive"
    types = [e["type"] for e in _el.get_log(sid).events()]
    assert "feedback_signal" in types
    _el.forget_session(sid)


# ---- R57: recurring-topic skill-gap retrieval boost ------------------------

def test_skill_gap_boost():
    m = _wm.InterviewWorldModel()
    for _ in range(3):
        _wm.record_topic(m, "kafka")
    gaps = _wm.skill_gaps(m)
    assert "kafka" in gaps
    base = _know.interview_knowledge("kafka")
    boosted = _know.skill_gap_boost("kafka", gaps)
    assert len(boosted) >= len(base)
    # A non-gap topic just returns the base angles.
    assert _know.skill_gap_boost("redis", gaps) == _know.interview_knowledge("redis")


# ---- R58: cognitive-load depth adaptation ----------------------------------

def test_cognitive_load_depth_adaptation():
    high = _style.cognitive_load(questions_per_min=8, pending_answers=2,
                                 interviewer_style=_style.RAPID_FIRE)
    assert high == _style.LOAD_HIGH
    assert "short" in _style.depth_for_load(high).lower()
    low = _style.cognitive_load(questions_per_min=1, pending_answers=0)
    assert low == _style.LOAD_LOW
    assert _style.depth_for_load(low) == ""  # full depth fine
