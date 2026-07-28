"""B6 — STT-dropout gap detection + recovery."""
from app.core.config_loader import cfg
from app.live.recovery import SttGapTracker


def test_gap_after_threshold_then_recovery(monkeypatch):
    monkeypatch.setattr(cfg.live, "stt_gap_recovery", True, raising=False)
    monkeypatch.setattr(cfg.live, "stt_gap_threshold", 3, raising=False)
    g = SttGapTracker()
    assert g.note_status("empty") is False       # 1
    assert g.note_status("error") is False       # 2
    assert g.note_status("empty") is True        # 3 → gap declared ONCE
    assert g.note_status("empty") is False       # still in gap, no re-fire
    assert g.in_gap is True
    # a real transcript recovers exactly once.
    assert g.note_transcript("what is kafka") is True
    assert g.in_gap is False
    assert g.note_transcript("more text") is False


def test_non_failure_kinds_do_not_count(monkeypatch):
    monkeypatch.setattr(cfg.live, "stt_gap_threshold", 2, raising=False)
    g = SttGapTracker()
    assert g.note_status("ok") is False
    assert g.note_status("listening") is False
    assert g.in_gap is False


def test_disabled_never_gaps(monkeypatch):
    monkeypatch.setattr(cfg.live, "stt_gap_recovery", False, raising=False)
    g = SttGapTracker()
    for _ in range(10):
        assert g.note_status("empty") is False
    assert g.in_gap is False
