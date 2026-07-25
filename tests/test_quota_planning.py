"""Stage-5 §2.7 — proactive free-quota planning: ledgers, reserve, spread, headers."""
from __future__ import annotations

import pytest

from app.llm import quota_plan as Q
from app.llm.quota_manager import DAY


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def planner(clock):
    return Q.QuotaPlanner(now=clock)


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
    return cfg


# --------------------------------------------------------------------------- #
class TestLedgers:
    def test_per_key_isolation(self, planner):
        # groq is in DEFAULTS (14_400/day). Two keys drain independently.
        planner.record("groq", key_id=1, n=100)
        assert planner.headroom("groq", 1) == 14_400 - 100
        assert planner.headroom("groq", 2) == 14_400   # untouched

    def test_unknown_provider_has_no_signal(self, planner):
        planner.record("mystery", key_id=1)
        assert planner.headroom("mystery", 1) is None      # unlimited/unknown
        assert planner.headroom_fraction("mystery", 1) == 1.0

    def test_window_rolls_on_boundary(self, planner, clock):
        planner.record("groq", 1, n=14_400)
        assert planner.headroom("groq", 1) == 0
        clock.advance(DAY + 1)
        assert planner.headroom("groq", 1) == 14_400      # reset

    def test_exhausted(self, planner):
        planner.record("gemini", 1, n=1_500)              # gemini free daily
        assert planner.exhausted("gemini", 1, for_live=True) is True


class TestReserve:
    def test_non_live_withholds_reserve(self, planner, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        monkeypatch.setattr(cfg.routing, "live_reserve", True, raising=False)
        monkeypatch.setattr(cfg.routing, "live_reserve_fraction", 0.30,
                            raising=False)
        # Fresh ledger: full headroom. A non-Live turn sees 1.0 - 0.30 = 0.70;
        # a Live turn sees the full 1.0.
        assert planner.headroom_fraction("groq", 1, for_live=False) == \
            pytest.approx(0.70)
        assert planner.headroom_fraction("groq", 1, for_live=True) == 1.0

    def test_reserve_released_near_reset(self, planner, clock, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        monkeypatch.setattr(cfg.routing, "live_reserve", True, raising=False)
        # Seed the ledger now, then jump to the last 3h before reset.
        planner.headroom_fraction("groq", 1)              # seed
        clock.advance(DAY - 3600)                          # 1h before reset
        # Within the release window → the reserve is freed (non-Live sees full).
        assert planner.headroom_fraction("groq", 1, for_live=False) == 1.0

    def test_reserve_off_no_withholding(self, planner, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        monkeypatch.setattr(cfg.routing, "live_reserve", False, raising=False)
        assert planner.headroom_fraction("groq", 1, for_live=False) == 1.0


class TestSpread:
    def test_rank_keys_by_headroom(self, planner):
        planner.record("groq", 1, n=10_000)   # heavily drained
        planner.record("groq", 2, n=100)       # nearly full
        assert planner.rank_keys("groq", [1, 2]) == [2, 1]

    def test_spread_penalty_grows_as_key_drains(self, planner, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        monkeypatch.setattr(cfg.routing, "spread", True, raising=False)
        assert planner.spread_penalty("groq", 1) == pytest.approx(0.0)
        planner.record("groq", 1, n=14_400)   # fully drained
        assert planner.spread_penalty("groq", 1) == pytest.approx(1.0)

    def test_spread_penalty_zero_when_off(self, planner, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        monkeypatch.setattr(cfg.routing, "spread", False, raising=False)
        planner.record("groq", 1, n=14_400)
        assert planner.spread_penalty("groq", 1) == 0.0


class TestHeaderCorrection:
    def test_remaining_header_reconciles_used(self, planner):
        planner.record("groq", 1, n=50)        # our estimate: used=50
        planner.reconcile_headers("groq", 1,
                                  {"x-ratelimit-remaining-requests": "9000"})
        # Provider says 9000 remaining of 14_400 → used = 5_400 (truth wins).
        assert planner.headroom("groq", 1) == 9000

    def test_retry_after_blocks(self, planner, clock):
        planner.reconcile_headers("groq", 1, {"Retry-After": "60"})
        assert planner.headroom("groq", 1) == 0            # blocked
        clock.advance(61)
        assert planner.headroom("groq", 1) == 14_400       # unblocked

    def test_missing_headers_leave_estimate(self, planner):
        planner.record("groq", 1, n=42)
        planner.reconcile_headers("groq", 1, {"content-type": "application/json"})
        assert planner.headroom("groq", 1) == 14_400 - 42

    def test_case_insensitive_headers(self, planner):
        planner.reconcile_headers("groq", 1,
                                  {"X-RateLimit-Remaining": "100"})
        assert planner.headroom("groq", 1) == 100


class TestModuleHooks:
    def test_record_success_noop_when_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", False, raising=False)
        Q.reset_for_tests()
        Q.record_success("groq", 1)
        assert Q.quota_planner().headroom("groq", 1) == 14_400  # untouched

    def test_record_success_records_when_on(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        Q.reset_for_tests()
        Q.record_success("groq", 1)
        assert Q.quota_planner().headroom("groq", 1) == 14_399

    def test_reconcile_fail_open_on_bad_headers(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        Q.reset_for_tests()
        Q.reconcile("groq", 1, None)          # never raises
        Q.reconcile("groq", 1, {"retry-after": "not-a-number"})


class TestProviderSignal:
    def test_best_key_wins(self, planner):
        planner.record("groq", 1, n=14_000)   # nearly drained
        planner.record("groq", 2, n=100)       # nearly full
        frac, exhausted = planner.provider_signal("groq", for_live=True)
        # The router will pick the best key → the signal reflects key 2.
        assert frac == pytest.approx((14_400 - 100) / 14_400)
        assert exhausted is False

    def test_all_exhausted_flag(self, planner):
        planner.record("gemini", 1, n=1_500)
        planner.record("gemini", 2, n=1_500)
        frac, exhausted = planner.provider_signal("gemini", for_live=True)
        assert frac == 0.0 and exhausted is True

    def test_unknown_provider_returns_none(self, planner):
        planner.record("mystery", 1)           # unlimited/unknown ledger
        assert planner.provider_signal("mystery") is None
        assert planner.provider_signal("never-seen") is None


class TestUsageHeaderSink:
    def test_keeps_only_rate_limit_headers(self):
        from app.llm import usage
        usage.reset()
        usage.record_headers({
            "content-type": "application/json",
            "X-RateLimit-Remaining": "42",
            "Retry-After": "30",
        })
        h = usage.rate_limit_headers()
        assert h == {"x-ratelimit-remaining": "42", "retry-after": "30"}
        usage.reset()
        assert usage.rate_limit_headers() is None

    def test_record_headers_fail_open(self):
        from app.llm import usage
        usage.record_headers(None)             # never raises
        assert usage.rate_limit_headers() is None


class TestPersistence:
    def test_serialize_round_trips_state(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        Q.reset_for_tests()
        Q.quota_planner().record("groq", 1, n=77)
        blob = Q._serialize()
        # Simulate a restart: wipe + reload from the blob (rehydrate's inner path).
        import json
        Q.reset_for_tests()
        for r in json.loads(blob):
            Q.quota_planner()._l[r["k"]] = Q._Ledger(
                limit=r["l"], window_s=r["w"], used=r["u"],
                window_start=r["s"], blocked_until=r["b"])
        assert Q.quota_planner().headroom("groq", 1) == 14_400 - 77

    def test_rehydrate_no_db_returns_zero(self, monkeypatch):
        # No session factory (dev/unit) → rehydrate is a fail-open no-op.
        import asyncio
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        assert asyncio.run(Q.rehydrate()) == 0


class TestRouterOverlay:
    def test_quota_state_overlays_planner_when_on(self, monkeypatch):
        from app.core.config_loader import cfg
        from app.llm import router
        monkeypatch.setattr(cfg.routing, "quota_planning", True, raising=False)
        Q.reset_for_tests()
        # Drain groq key 1 to 25% headroom on the planner's per-key ledger.
        Q.quota_planner().record("groq", 1, n=int(14_400 * 0.75))
        state = router._quota_state(for_live=True)
        assert "groq" in state
        frac, _ = state["groq"]
        assert frac == pytest.approx(0.25, abs=0.02)

    def test_quota_state_off_ignores_planner(self, monkeypatch):
        from app.core.config_loader import cfg
        from app.llm import router
        monkeypatch.setattr(cfg.routing, "quota_planning", False, raising=False)
        Q.reset_for_tests()
        Q.quota_planner().record("groq", 1, n=14_000)
        # Planner disabled → the reactive per-provider manager's value stands
        # (fresh manager = full headroom), NOT the drained planner ledger.
        state = router._quota_state(for_live=True)
        assert state.get("groq", (1.0, False))[0] > 0.5
