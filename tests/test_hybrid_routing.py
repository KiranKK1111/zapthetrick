"""Phase P2-1 — hybrid strong-tier routing + monthly paid budget guard.

The decision logic (`_allow_paid_tier`) and the budget counter are pure/offline;
the DB-backed `route_request` isn't exercised here (covered by router tests).
"""
from __future__ import annotations

from types import SimpleNamespace

from app.llm import budget
from app.llm import router as rt


def setup_function():
    budget.reset()


# ── budget guard ────────────────────────────────────────────────────────────
def test_budget_unlimited_when_cap_zero():
    assert budget.can_use_paid(0)
    budget.record_paid(100)
    assert budget.can_use_paid(0)            # 0 = unlimited
    assert budget.paid_this_month() == 100


def test_budget_cap_enforced():
    assert budget.can_use_paid(3)
    budget.record_paid()
    budget.record_paid()
    assert budget.can_use_paid(3)            # 2 < 3
    budget.record_paid()
    assert not budget.can_use_paid(3)        # 3 >= 3 → blocked


def test_budget_record_ignores_nonpositive():
    budget.record_paid(0)
    budget.record_paid(-5)
    assert budget.paid_this_month() == 0


# ── strong-tier decision (config-driven) ────────────────────────────────────
def _cfg(**routing):
    base = dict(strong_tier_for_hard=False,
                strong_tier_difficulties=["expert"],
                monthly_paid_request_cap=0)
    base.update(routing)
    return SimpleNamespace(routing=SimpleNamespace(**base))


def _patch_cfg(monkeypatch, cfg):
    monkeypatch.setattr("app.core.config_loader.get_config", lambda: cfg)


def test_strong_tier_off_by_default(monkeypatch):
    _patch_cfg(monkeypatch, _cfg())                     # strong_tier_for_hard=False
    assert rt._allow_paid_tier("expert") is False
    assert rt._allow_paid_tier("standard") is False


def test_strong_tier_on_for_listed_difficulty(monkeypatch):
    _patch_cfg(monkeypatch, _cfg(strong_tier_for_hard=True,
                                 strong_tier_difficulties=["hard", "expert"]))
    assert rt._allow_paid_tier("expert") is True
    assert rt._allow_paid_tier("hard") is True
    assert rt._allow_paid_tier("standard") is False     # not listed
    assert rt._allow_paid_tier("trivial") is False


def test_strong_tier_respects_budget_cap(monkeypatch):
    _patch_cfg(monkeypatch, _cfg(strong_tier_for_hard=True,
                                 strong_tier_difficulties=["expert"],
                                 monthly_paid_request_cap=2))
    assert rt._allow_paid_tier("expert") is True
    budget.record_paid(2)                               # hit the cap
    assert rt._allow_paid_tier("expert") is False       # over cap → free-only


def test_maybe_record_paid_only_counts_nonfree():
    budget.reset()
    rt._maybe_record_paid({"free": True})
    assert budget.paid_this_month() == 0
    rt._maybe_record_paid({"free": False})
    assert budget.paid_this_month() == 1
    rt._maybe_record_paid({})                # missing key defaults to free
    assert budget.paid_this_month() == 1


def test_default_config_keeps_free_first(monkeypatch):
    """With real defaults, hard/expert turns must NOT be granted the paid tier
    (free-first preserved unless explicitly opted in)."""
    from app.core.config_loader import RoutingSection
    monkeypatch.setattr("app.core.config_loader.get_config",
                        lambda: SimpleNamespace(routing=RoutingSection()))
    assert rt._allow_paid_tier("expert") is False
    assert rt._allow_paid_tier("hard") is False
