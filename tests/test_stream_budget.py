"""Unified stream budget controller (P6 #17)."""
from __future__ import annotations

from types import SimpleNamespace

from app.response_arch.budget import StreamBudget, load_budget


def test_milestones():
    b = StreamBudget(ack_threshold_s=0.8, first_visible_s=5.0, total_s=300.0)
    # before ack threshold, nothing visible → no ack
    assert b.should_ack(0.5, first_seen=False) is False
    # past ack threshold, still nothing visible → ack
    assert b.should_ack(1.0, first_seen=False) is True
    # once something is visible, never ack
    assert b.should_ack(10.0, first_seen=True) is False
    # first-visible deadline
    assert b.first_visible_overdue(6.0, first_seen=False) is True
    assert b.first_visible_overdue(6.0, first_seen=True) is False
    # total ceiling
    assert b.exhausted(299.0) is False
    assert b.exhausted(300.0) is True
    assert b.remaining_s(280.0) == 20.0


def test_uncapped_total():
    b = StreamBudget(0.0, 0.0, 0.0)
    assert b.exhausted(9999) is False
    assert b.remaining_s(5) == 0.0
    assert b.should_ack(5, first_seen=False) is False


def test_load_budget_reads_config_locations():
    cfg = SimpleNamespace(
        perceived=SimpleNamespace(ttft_ack_threshold_s=0.8),
        llm=SimpleNamespace(
            chat_stream_budget_s=120.0,
            routing=SimpleNamespace(first_token_deadline_s=4.0)),
    )
    b = load_budget(cfg)
    assert b.ack_threshold_s == 0.8
    assert b.first_visible_s == 4.0
    assert b.total_s == 120.0


def test_load_budget_clamps_ack_after_first_visible():
    cfg = SimpleNamespace(
        perceived=SimpleNamespace(ttft_ack_threshold_s=10.0),
        llm=SimpleNamespace(
            chat_stream_budget_s=300.0,
            routing=SimpleNamespace(first_token_deadline_s=5.0)),
    )
    b = load_budget(cfg)
    assert b.ack_threshold_s < b.first_visible_s


def test_load_budget_fail_open_on_missing_cfg():
    b = load_budget(SimpleNamespace())
    assert b.first_visible_s == 5.0 and b.total_s == 300.0
