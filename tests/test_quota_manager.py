"""Free-tier quota & rotation manager (P5 #16): windows, reset, ranking."""
from __future__ import annotations

from app.llm.quota_manager import DAY, QuotaManager


def test_headroom_and_exhaustion():
    clock = [1000.0]
    qm = QuotaManager(now=lambda: clock[0])
    qm.configure("p", limit=3, window_s=DAY)
    assert qm.headroom("p") == 3
    qm.record("p"); qm.record("p")
    assert qm.headroom("p") == 1
    assert qm.exhausted("p") is False
    qm.record("p")
    assert qm.headroom("p") == 0 and qm.exhausted("p") is True


def test_window_resets_after_elapse():
    clock = [0.0]
    qm = QuotaManager(now=lambda: clock[0])
    qm.configure("p", limit=2, window_s=DAY)
    qm.record("p"); qm.record("p")
    assert qm.exhausted("p") is True
    clock[0] += DAY + 1          # window rolls over
    assert qm.exhausted("p") is False and qm.headroom("p") == 2


def test_rank_prefers_headroom_sinks_exhausted():
    clock = [0.0]
    qm = QuotaManager(now=lambda: clock[0])
    qm.configure("full", limit=10, window_s=DAY)
    qm.configure("low", limit=10, window_s=DAY)
    qm.configure("dead", limit=1, window_s=DAY)
    for _ in range(8):
        qm.record("low")
    qm.record("dead")            # exhausted
    order = qm.rank(["dead", "low", "full", "unknown"])
    assert order[0] == "unknown"     # unlimited/unknown first
    assert order.index("full") < order.index("low")
    assert order[-1] == "dead"       # exhausted last


def test_next_reset_is_window_end():
    clock = [100.0]
    qm = QuotaManager(now=lambda: clock[0])
    qm.configure("p", limit=5, window_s=DAY)
    assert qm.next_reset("p") == 100.0 + DAY


def test_snapshot_shape():
    qm = QuotaManager(now=lambda: 0.0)
    snap = qm.snapshot()
    assert any(row["provider"] == "groq" for row in snap)
    assert all("headroom" in row and "resets_at" in row for row in snap)
