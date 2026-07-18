"""Multi-level memory + session summary
(live-conversational-intelligence R6; task 5.2).

Pins Property 7: L1/L2/L3 level selection, deterministic non-blocking summary
(no LLM call), and fail-open to recent-Q+A. Uses a lightweight fake tracker.
"""
from __future__ import annotations

from app.live.memory import MultiLevelMemory, for_tracker, refresh_summary
from app.question_detection.context_tracker import Turn


class _FakeTracker:
    def __init__(self, turns):
        self._turns = list(turns)


def _tracker():
    return _FakeTracker([
        Turn(question="What is Kafka?", answer="...", topic="kafka", qtype="technical_concept"),
        Turn(question="How do partitions work?", answer="...", topic="kafka partitions", qtype="technical_concept"),
        Turn(question="What is Redis?", answer="...", topic="redis", qtype="technical_concept"),
    ])


def test_l1_returns_recent_questions():
    m = MultiLevelMemory(_tracker())
    l1 = m.l1(n=2)
    assert l1 == ["How do partitions work?", "What is Redis?"]


def test_l2_filters_by_topic():
    m = MultiLevelMemory(_tracker())
    l2 = m.l2("kafka")
    qs = [t.question for t in l2]
    assert "What is Kafka?" in qs
    assert "How do partitions work?" in qs   # related sub-topic
    assert "What is Redis?" not in qs


def test_context_for_includes_topic_then_recent_then_summary():
    m = MultiLevelMemory(_tracker())
    m.set_summary("Topics covered: kafka, redis")
    ctx = m.context_for("more on kafka", "kafka")
    assert any("Kafka" in c for c in ctx)
    assert ctx[-1].startswith("Summary:")


def test_refresh_summary_is_deterministic_no_llm():
    m = MultiLevelMemory(_tracker())
    summary = refresh_summary(m)
    assert "Topics covered:" in summary
    assert "kafka" in summary
    assert "Recent questions:" in summary
    # Idempotent / stored.
    assert m.l3() == summary


def test_refresh_summary_failopen_empty_tracker():
    m = MultiLevelMemory(_FakeTracker([]))
    assert refresh_summary(m) == ""
    assert m.l3() == ""


def test_for_tracker_is_attached_and_stable():
    t = _tracker()
    a = for_tracker(t)
    b = for_tracker(t)
    assert a is b
