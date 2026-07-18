"""Speculation budget + kill-switch (perceived-speed R19, task 2.3).

Pins Property 4: concurrency cap, period exhaustion suspends new work, prompt
cancellation (scope release), and disabled = no speculative work.
"""
from __future__ import annotations

from app.core.config_loader import cfg
from app.perceived.budget import SpeculationBudget, speculation_enabled


def _enable(monkeypatch, **over):
    monkeypatch.setattr(cfg.perceived, "speculation_enabled", True, raising=False)
    for k, v in over.items():
        monkeypatch.setattr(cfg.perceived, k, v, raising=False)


def test_disabled_means_no_speculative_work(monkeypatch):
    monkeypatch.setattr(cfg.perceived, "speculation_enabled", False, raising=False)
    b = SpeculationBudget()
    assert speculation_enabled() is False
    assert b.allow() is False
    assert b.allow(kind="draft") is False


def test_enabled_allows_work(monkeypatch):
    _enable(monkeypatch, speculation_period_budget=0, max_concurrent_drafts=2)
    b = SpeculationBudget()
    assert b.allow() is True
    assert b.allow(kind="draft") is True


def test_concurrency_cap(monkeypatch):
    _enable(monkeypatch, max_concurrent_drafts=2, speculation_period_budget=0)
    b = SpeculationBudget()
    with b.cancel_scope(kind="draft"):
        assert b.active_drafts == 1
        with b.cancel_scope(kind="draft"):
            assert b.active_drafts == 2
            # Cap reached → no new draft allowed.
            assert b.allow(kind="draft") is False
        # One scope released → a draft is allowed again (prompt cancellation).
        assert b.active_drafts == 1
        assert b.allow(kind="draft") is True
    assert b.active_drafts == 0


def test_period_budget_exhaustion_suspends(monkeypatch):
    _enable(monkeypatch, speculation_period_budget=3, speculation_period_seconds=3600)
    b = SpeculationBudget()
    assert b.allow() is True
    b.account(3)                      # exhaust the period budget
    assert b.allow() is False         # new speculative work suspended
    b.reset()
    assert b.allow() is True          # recovers on a fresh period


def test_account_is_floored(monkeypatch):
    _enable(monkeypatch, speculation_period_budget=5)
    b = SpeculationBudget()
    b.account(-10)                    # never goes negative
    assert b.allow() is True
