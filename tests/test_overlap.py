"""B4 — overlap/crosstalk detection over the dual-source channels."""
from app.core.config_loader import cfg
from app.live.overlap import OverlapMonitor, INTERVIEWER, CANDIDATE


def test_overlap_latched_when_both_speak(monkeypatch):
    monkeypatch.setattr(cfg.live, "overlap_detection", True, raising=False)
    m = OverlapMonitor()
    assert m.update(INTERVIEWER, True) is False    # only one channel
    assert m.saw_overlap() is False
    assert m.update(CANDIDATE, True) is True        # edge INTO overlap
    assert m.overlapping is True
    assert m.saw_overlap() is True
    # candidate stops → no longer overlapping, but the latch persists for the turn
    m.update(CANDIDATE, False)
    assert m.overlapping is False
    assert m.saw_overlap() is True
    # reset clears the latch for the next utterance
    m.reset()
    assert m.saw_overlap() is False


def test_turn_taking_never_overlaps(monkeypatch):
    monkeypatch.setattr(cfg.live, "overlap_detection", True, raising=False)
    m = OverlapMonitor()
    m.update(INTERVIEWER, True); m.update(INTERVIEWER, False)
    m.update(CANDIDATE, True); m.update(CANDIDATE, False)
    assert m.saw_overlap() is False


def test_disabled_never_overlaps(monkeypatch):
    monkeypatch.setattr(cfg.live, "overlap_detection", False, raising=False)
    m = OverlapMonitor()
    m.update(INTERVIEWER, True); m.update(CANDIDATE, True)
    assert m.saw_overlap() is False


def test_unknown_role_ignored(monkeypatch):
    monkeypatch.setattr(cfg.live, "overlap_detection", True, raising=False)
    m = OverlapMonitor()
    assert m.update("panelist_3", True) is False
