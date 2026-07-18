"""Audio capture & transport reliability
(live-conversational-intelligence R18, R23, R26; tasks 17.2).

Pins Properties 18, 23, 26: topology routing + speaker labelling, qid-registry
exact cancellation + out-of-order final tolerance + resume without duplicate
answers, and mobile error classification / degrade.
"""
from __future__ import annotations

from app.live import mobile
from app.live.capture_topology import (
    BOTH,
    CANDIDATE,
    CANDIDATE_SPEAKER,
    INTERVIEWER_SPEAKER,
    LOOPBACK,
    CaptureTopology,
)
from app.live.resume import QidRegistry, get_registry


# ---- capture topology --------------------------------------------------
def test_topology_speaker_labelling():
    t = CaptureTopology(mode=BOTH)
    assert t.speaker_for("mic") == CANDIDATE_SPEAKER
    assert t.speaker_for("system_loopback") == INTERVIEWER_SPEAKER
    assert t.is_candidate_source("microphone") is True
    assert sorted(t.sources()) == sorted([CANDIDATE, LOOPBACK])


def test_topology_single_source():
    assert CaptureTopology(mode=LOOPBACK).sources() == [LOOPBACK]


# ---- qid registry / resume --------------------------------------------
def test_cancel_targets_only_active_qid():
    reg = QidRegistry()
    reg.open("q1")
    reg.open("q2")
    assert reg.cancel("q1") is True       # active → cancelled
    assert reg.cancel("q1") is False      # already cancelled → no-op
    assert reg.cancel("unknown") is False
    assert reg.is_active("q2") is True


def test_out_of_order_final_after_answered_is_noop():
    # should_process_final was removed (dead code — the QuestionDeduper is
    # the real duplicate guard); answered-state tracking is what remains.
    reg = QidRegistry()
    reg.open("q1")
    reg.close("q1")                       # answered
    assert reg.is_answered("q1") is True
    assert reg.is_answered("q2") is False


def test_resume_registry_is_per_session():
    a = get_registry("s1")
    a.open("q1"); a.close("q1")
    assert get_registry("s1") is a
    assert get_registry("s1").is_answered("q1") is True
    assert get_registry("s2") is not a


# ---- mobile ------------------------------------------------------------
def test_mobile_error_classification():
    assert mobile.classify_audio_error("Microphone permission denied")["state"] == "mic_permission"
    assert mobile.classify_audio_error("device in use by another app")["state"] == "mic_contention"
    assert mobile.classify_audio_error("audio output routing changed")["state"] == "audio_routing"


def test_mobile_degrade_on_pressure():
    assert mobile.degrade_for_pressure(battery_pct=10) is True
    assert mobile.degrade_for_pressure(thermal="critical") is True
    assert mobile.degrade_for_pressure(battery_pct=80, thermal="nominal") is False
    assert mobile.degrade_for_pressure() is False
