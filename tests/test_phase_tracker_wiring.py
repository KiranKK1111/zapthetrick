"""Wiring test — the phase tracker is actually USED in the live WS pipeline
(Phase 2 #3/#14), updated per answered utterance and cleaned up on teardown.
Static assertion (AST/source) so it needs no live socket/audio/models.
"""
from __future__ import annotations

import pathlib

_WS = pathlib.Path(__file__).resolve().parents[1] / "app" / "api" / "routes_ws.py"


def test_pipeline_observes_phase_tracker():
    src = _WS.read_text(encoding="utf-8")
    assert "from app.live.phase_tracker import get_phase_tracker" in src, (
        "routes_ws.py must import get_phase_tracker."
    )
    assert ".observe(d.phase)" in src, (
        "routes_ws.py must feed the detected phase into the tracker (.observe)."
    )
    assert 'extra["phase_progress"]' in src, (
        "routes_ws.py must surface phase_progress in the meta frame."
    )


def test_pipeline_forgets_tracker_on_teardown():
    src = _WS.read_text(encoding="utf-8")
    assert "from app.live.phase_tracker import forget_session" in src, (
        "routes_ws.py teardown must forget the phase tracker (no per-session leak)."
    )


def test_tracker_progression_end_to_end():
    # Simulate a realistic noisy interview arc and assert the smoothed snapshot
    # is sensible — the behavior the pipeline relies on.
    from app.live import phase as _phase
    from app.live.phase_tracker import PhaseTracker

    arc = [
        _phase.INTRODUCTION,
        _phase.TECHNICAL_SCREENING, _phase.TECHNICAL_SCREENING,
        _phase.HR,                       # stray single HR cue mid-technical...
        _phase.CODING, _phase.CODING,    # ...actually moved to coding
        _phase.HR, _phase.HR,            # then genuinely into HR
        _phase.CLOSING, _phase.CLOSING,
    ]
    t = PhaseTracker()
    for p in arc:
        t.observe(p)
    snap = t.snapshot()
    assert snap["phase"] == _phase.CLOSING
    assert snap["late_stage"] is True
    assert snap["phase_progress"] == 1.0
    # The stray single HR must NOT appear as a committed phase before coding.
    hist = snap["phase_history"]
    assert hist.index(_phase.CODING) < hist.index(_phase.HR)
