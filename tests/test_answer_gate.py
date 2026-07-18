"""Answer-first gate v2 (roadmap #12 / Phase D)."""
from __future__ import annotations

from app.clarify import answer_gate as ag


def test_aggregate_answerability():
    buckets = {
        "high": {"answerable": 6, "needed": 1},
        "low": {"answerable": 2, "needed": 1},
    }
    rate, n = ag.aggregate_answerability(buckets)
    assert n == 10
    assert abs(rate - 0.8) < 1e-9


def test_aggregate_empty():
    assert ag.aggregate_answerability({}) == (0.0, 0)
    assert ag.aggregate_answerability(None) == (0.0, 0)


def test_upgrade_when_history_shows_overasking():
    buckets = {"b": {"answerable": 9, "needed": 1}}   # 90% answerable, n=10
    assert ag.should_upgrade_to_answer(
        "defer", 0.7, buckets, enabled=True) is True


def test_no_upgrade_when_disabled():
    buckets = {"b": {"answerable": 9, "needed": 1}}
    assert ag.should_upgrade_to_answer(
        "defer", 0.7, buckets, enabled=False) is False


def test_no_upgrade_for_non_defer():
    buckets = {"b": {"answerable": 9, "needed": 1}}
    assert ag.should_upgrade_to_answer(
        "clarify", 0.9, buckets, enabled=True) is False
    assert ag.should_upgrade_to_answer(
        "answer", 0.9, buckets, enabled=True) is False


def test_no_upgrade_below_sample_floor():
    buckets = {"b": {"answerable": 3, "needed": 0}}    # only 3 samples
    assert ag.should_upgrade_to_answer(
        "defer", 0.9, buckets, enabled=True) is False


def test_no_upgrade_when_user_often_needs_clarification():
    buckets = {"b": {"answerable": 3, "needed": 7}}    # 30% answerable
    assert ag.should_upgrade_to_answer(
        "defer", 0.9, buckets, enabled=True) is False


def test_no_upgrade_below_confidence_floor():
    buckets = {"b": {"answerable": 9, "needed": 1}}
    assert ag.should_upgrade_to_answer(
        "defer", 0.3, buckets, enabled=True) is False


def test_fail_open_on_bad_buckets():
    assert ag.should_upgrade_to_answer(
        "defer", 0.9, {"b": "not-a-dict"}, enabled=True) is False


def test_enabled_reads_config(monkeypatch):
    from app.core import config_loader as cl

    class _L:
        answer_first_v2 = True
    monkeypatch.setattr(cl.cfg, "learning", _L(), raising=False)
    assert ag.enabled() is True

    class _L2:
        answer_first_v2 = False
    monkeypatch.setattr(cl.cfg, "learning", _L2(), raising=False)
    assert ag.enabled() is False
