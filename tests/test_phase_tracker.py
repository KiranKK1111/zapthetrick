"""Tests for the per-session interview-phase progression tracker (Phase 2 #3/#14)."""
from __future__ import annotations

from app.live import phase as _phase
from app.live.phase_tracker import (
    PHASE_ORDER,
    PhaseTracker,
    forget_session,
    get_phase_tracker,
)


def test_seeds_on_first_observation():
    t = PhaseTracker()
    assert t.observe(_phase.TECHNICAL_SCREENING) == _phase.TECHNICAL_SCREENING
    assert t.current == _phase.TECHNICAL_SCREENING
    assert t.history == [_phase.TECHNICAL_SCREENING]


def test_single_stray_cue_does_not_transition():
    t = PhaseTracker()
    t.observe(_phase.TECHNICAL_SCREENING)
    # One lone HR detection amid technical questions must NOT move to HR.
    assert t.observe(_phase.HR) == _phase.TECHNICAL_SCREENING
    assert t.current == _phase.TECHNICAL_SCREENING
    assert t.transitions == []


def test_sustained_signal_commits_transition():
    t = PhaseTracker()
    t.observe(_phase.TECHNICAL_SCREENING)
    t.observe(_phase.HR)          # 1 vote — not yet
    got = t.observe(_phase.HR)    # 2 votes in window — commit
    assert got == _phase.HR
    assert t.current == _phase.HR
    assert t.transitions == [(_phase.TECHNICAL_SCREENING, _phase.HR)]


def test_progress_monotonic_ish_and_bounds():
    t = PhaseTracker()
    t.observe(_phase.INTRODUCTION)
    p0 = t.progress()
    t.observe(_phase.CLOSING); t.observe(_phase.CLOSING)
    p1 = t.progress()
    assert p0 == 0.0
    assert p1 == 1.0
    assert p1 > p0


def test_late_stage_detection():
    t = PhaseTracker()
    t.observe(_phase.CODING)
    assert not t.is_late_stage()
    t.observe(_phase.HR); t.observe(_phase.HR)
    assert t.is_late_stage()


def test_unknown_phase_ignored():
    t = PhaseTracker()
    t.observe(_phase.TECHNICAL_SCREENING)
    assert t.observe("not_a_phase") == _phase.TECHNICAL_SCREENING


def test_snapshot_shape():
    t = PhaseTracker()
    t.observe(_phase.INTRODUCTION)
    t.observe(_phase.TECHNICAL_SCREENING); t.observe(_phase.TECHNICAL_SCREENING)
    snap = t.snapshot()
    assert snap["phase"] == _phase.TECHNICAL_SCREENING
    assert 0.0 <= snap["phase_progress"] <= 1.0
    assert snap["phase_history"][0] == _phase.INTRODUCTION
    assert snap["late_stage"] is False
    assert isinstance(snap["phase_transitions"], int)


def test_history_has_no_consecutive_dupes_in_snapshot():
    t = PhaseTracker()
    for p in [_phase.INTRODUCTION, _phase.CODING, _phase.CODING, _phase.HR, _phase.HR]:
        t.observe(p)
    hist = t.snapshot()["phase_history"]
    assert hist == list(dict.fromkeys(hist))  # de-duped, ordered


def test_registry_is_per_session():
    forget_session("s1"); forget_session("s2")
    a = get_phase_tracker("s1")
    b = get_phase_tracker("s2")
    assert a is not b
    assert get_phase_tracker("s1") is a  # stable per session
    a.observe(_phase.HR)
    assert not b.history  # isolation
    forget_session("s1")
    assert get_phase_tracker("s1") is not a  # forgotten -> fresh


def test_phase_order_covers_all_phases():
    assert set(PHASE_ORDER) == _phase.PHASES


def test_predict_next_forward_trajectory():
    from app.live.phase_tracker import PhaseTracker
    t = PhaseTracker()
    t.observe(_phase.INTRODUCTION)
    # next unvisited forward phase after introduction
    assert t.predict_next() == _phase.RESUME_DISCUSSION
    for _ in range(2):
        t.observe(_phase.HR)
    assert t.predict_next() == _phase.CLOSING  # after HR → closing
    for _ in range(2):
        t.observe(_phase.CLOSING)
    assert t.predict_next() is None  # at the end


def test_snapshot_has_predicted_next():
    from app.live.phase_tracker import PhaseTracker
    t = PhaseTracker()
    t.observe(_phase.CODING)
    assert "predicted_next" in t.snapshot()
