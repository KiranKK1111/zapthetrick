"""
Phase 14 tests — outcome analytics, replay, simulation & multimodal
(live-conversational-intelligence R44, R45, R46).

Property 44: the outcome estimate is advisory and explicitly labeled
not-a-hiring-decision; unknown when there is no signal.
Property 45: replay is built read-only from the event log with no new schema and
ordered chronologically.
Property 46: mock mode generates labeled practice questions; the multimodal
adapter is additive and audio-only when no source is present.
"""
from __future__ import annotations

from app.live import eventlog as _eventlog
from app.live import mock as _mock
from app.live import multimodal as _mm
from app.live import outcome as _outcome
from app.live import replay as _replay
from app.live.org import build_org
from app.live.profile import build_profile


# ---- Property 44: outcome (advisory) ---------------------------------------

def test_outcome_unknown_without_signal():
    o = _outcome.estimate()
    assert o.band == _outcome.UNKNOWN
    assert o.advisory is True
    assert "NOT a hiring decision" in o.disclaimer


def test_outcome_bands_and_disclaimer():
    strong = _outcome.estimate(answered=10, total=10, avg_confidence=0.9, satisfaction=0.9)
    assert strong.band == _outcome.STRONG
    assert strong.to_dict()["disclaimer"] == _outcome.DISCLAIMER
    weak = _outcome.estimate(answered=2, total=10, avg_confidence=0.3, satisfaction=0.2)
    assert weak.band == _outcome.NEEDS_WORK


def test_outcome_penalizes_contradictions():
    base = _outcome.estimate(answered=8, total=10, avg_confidence=0.8)
    pen = _outcome.estimate(answered=8, total=10, avg_confidence=0.8,
                            contradictions=3, health_warnings=2)
    assert pen.score < base.score


# ---- Property 45: replay (read-only from event log) ------------------------

def test_replay_built_from_event_log_ordered():
    sid = "replay-test-1"
    _eventlog.forget_session(sid)
    log = _eventlog.get_log(sid)
    log.append("event", {"kind": "QUESTION", "questions": 1})
    log.append("answer", {"topic": "kafka"})
    log.append("feedback", {"state": "satisfied"})
    r = _replay.build_replay(sid)
    assert r["count"] == 3
    types = [s["type"] for s in r["steps"]]
    assert types == ["event", "answer", "feedback"]
    # offsets are non-decreasing (chronological)
    offsets = [s["offset"] for s in r["steps"]]
    assert offsets == sorted(offsets)
    _eventlog.forget_session(sid)


def test_replay_summary_counts_by_type():
    sid = "replay-test-2"
    _eventlog.forget_session(sid)
    log = _eventlog.get_log(sid)
    log.append("answer", {})
    log.append("answer", {})
    log.append("feedback", {"state": "satisfied"})
    s = _replay.summary(sid)
    assert s["by_type"]["answer"] == 2
    assert s["by_type"]["feedback"] == 1
    _eventlog.forget_session(sid)


def test_replay_empty_when_no_events():
    r = _replay.build_replay("never-seen-session")
    assert r["count"] == 0
    assert r["steps"] == []


# ---- Property 46: mock mode + multimodal -----------------------------------

def test_mock_generates_labeled_practice_questions():
    prof = build_profile({"skills": ["python", "kafka"],
                          "projects": [{"name": "Billing", "tech": ["redis"]}]})
    org = build_org("Acme", "We need python and kubernetes.", "Backend")
    qs = _mock.generate_questions(prof, org, limit=12)
    assert qs
    assert all(_mock.is_practice(q) for q in qs)
    blob = " ".join(q["question"].lower() for q in qs)
    assert "python" in blob       # skill-targeted
    assert "billing" in blob      # project-targeted
    assert "kubernetes" in blob   # JD-targeted


def test_mock_empty_inputs_still_safe():
    qs = _mock.generate_questions(None, None)
    assert all(_mock.is_practice(q) for q in qs)


def test_multimodal_audio_only_when_absent():
    # Audio modality returns None (uses the existing path) → audio-only.
    assert _mm.to_utterance(_mm.AUDIO, b"...") is None
    # Empty / unknown also returns None.
    assert _mm.to_utterance(_mm.TYPED, "") is None
    assert _mm.to_utterance("hologram", "x") is None


def test_multimodal_normalizes_text_and_code():
    typed = _mm.to_utterance(_mm.TYPED, "  what is a hash map?  ")
    assert typed is not None and typed.text == "what is a hash map?"
    code = _mm.to_utterance(_mm.PASTED_CODE, "def f(): pass")
    assert code is not None and code.text.startswith("[shared code]")


def test_multimodal_register_extends_without_touching_pipeline():
    _mm.register_modality("whiteboard", lambda raw: f"WB:{raw}")
    out = _mm.to_utterance("whiteboard", "diagram")
    assert out is not None and out.text == "WB:diagram"
    assert "whiteboard" in _mm.supported()
