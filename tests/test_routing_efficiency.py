"""Routing efficiency enhancements (2026-07-14): latency-aware scoring, a
per-model circuit breaker, hot-path precomputed weights, and a hard free-only
cost mode. Every signal is additive + flag-gated, so with the weights/flags at
their defaults the ranking is byte-for-byte today's."""
from __future__ import annotations

import app.llm.router as R
from app.perceived.health import ProviderHealth


class TestCircuitBreaker:
    def test_opens_after_threshold_consecutive_hard_failures(self):
        h = ProviderHealth()
        assert h.is_open("m", 3, 30.0) is False
        h.record("m", ok=False, hard_failure=True)
        h.record("m", ok=False, hard_failure=True)
        assert h.is_open("m", 3, 30.0) is False       # only 2 — not yet
        h.record("m", ok=False, hard_failure=True)
        assert h.is_open("m", 3, 30.0) is True         # 3rd trips it

    def test_success_closes_breaker(self):
        h = ProviderHealth()
        for _ in range(5):
            h.record("m", ok=False, hard_failure=True)
        assert h.is_open("m", 3, 30.0) is True
        h.record("m", ok=True)
        assert h.is_open("m", 3, 30.0) is False        # a success closes it

    def test_rate_limit_is_not_a_hard_failure(self):
        h = ProviderHealth()
        for _ in range(5):
            h.record("m", ok=False)                    # 429-style, NOT hard
        assert h.is_open("m", 3, 30.0) is False         # never trips the breaker

    def test_half_open_after_cooldown(self, monkeypatch):
        import app.perceived.health as H
        clock = {"t": 1000.0}
        monkeypatch.setattr(H.time, "monotonic", lambda: clock["t"])
        h = ProviderHealth()
        for _ in range(3):
            h.record("m", ok=False, hard_failure=True)
        assert h.is_open("m", 3, 30.0) is True          # open during cooldown
        clock["t"] += 31.0
        assert h.is_open("m", 3, 30.0) is False          # half-open: probe allowed

    def test_probe_failure_reopens(self, monkeypatch):
        import app.perceived.health as H
        clock = {"t": 1000.0}
        monkeypatch.setattr(H.time, "monotonic", lambda: clock["t"])
        h = ProviderHealth()
        for _ in range(3):
            h.record("m", ok=False, hard_failure=True)
        clock["t"] += 31.0
        assert h.is_open("m", 3, 30.0) is False          # half-open
        h.record("m", ok=False, hard_failure=True)       # probe fails → re-arm
        assert h.is_open("m", 3, 30.0) is True


class TestLatencyFactor:
    def test_no_samples_is_neutral(self):
        assert ProviderHealth().latency_factor("m", 8.0) == 1.0

    def test_fast_is_one_slow_is_low(self):
        h = ProviderHealth()
        h.record("fast", latency_s=1.0)
        h.record("slow", latency_s=40.0)
        assert h.latency_factor("fast", 8.0) == 1.0
        assert h.latency_factor("slow", 8.0) < 0.3


class TestScoreTerm:
    def test_latency_weight_zero_is_byte_identical(self):
        base = R._candidate_score(0, 1.0, 10, 10, "standard")
        z = R._candidate_score(0, 1.0, 10, 10, "standard",
                               latency_factor=0.1, latency_w=0.0)
        assert base == z

    def test_latency_penalizes_slow_when_weighted(self):
        fast = R._candidate_score(0, 1.0, 10, 10, "standard",
                                  latency_factor=1.0, latency_w=8.0)
        slow = R._candidate_score(0, 1.0, 10, 10, "standard",
                                  latency_factor=0.2, latency_w=8.0)
        assert slow > fast

    def test_precomputed_weights_are_applied_additively(self):
        # Hot-path: an explicitly-passed task_w is used verbatim (config-
        # independent). task_match=0 → the full weight is added.
        lo = R._candidate_score(0, 1.0, 10, 10, "standard",
                                task_match=0.0, learned=1.0,
                                task_w=0.0, learn_w=0.0)
        hi = R._candidate_score(0, 1.0, 10, 10, "standard",
                                task_match=0.0, learned=1.0,
                                task_w=100.0, learn_w=0.0)
        assert hi == lo + 100.0

    def test_precomputed_weight_matches_helper_when_equal(self):
        # The router hoists these weights out of the loop; they must equal what
        # the in-score helpers would read, so the ranking is byte-identical.
        tw, lw = R._task_weight(), R._learn_weight()
        helper = R._candidate_score(0, 1.0, 10, 10, "standard",
                                    task_match=0.5, learned=0.5)
        hoisted = R._candidate_score(0, 1.0, 10, 10, "standard",
                                     task_match=0.5, learned=0.5,
                                     task_w=tw, learn_w=lw)
        assert helper == hoisted


class TestCostPolicy:
    """Free-only cost mode vs the existing free-first behavior."""

    def _pool(self):
        return [
            {"model": "paid-strong", "free": False, "intel": 5},
            {"model": "free-a", "free": True, "intel": 20},
            {"model": "free-b", "free": True, "intel": 30},
        ]

    def test_free_only_keeps_only_free(self):
        out = R._cost_pool(self._pool(), free_only=True,
                           prefer_free=False, allow_paid=False)
        assert {c["model"] for c in out} == {"free-a", "free-b"}
        assert all(c["free"] for c in out)

    def test_free_only_empty_when_no_free(self):
        paid_only = [{"model": "p", "free": False}]
        # Empty result → route_request raises NoRouteAvailable (never spends).
        assert R._cost_pool(paid_only, free_only=True,
                            prefer_free=False, allow_paid=False) == []

    def test_free_first_collapses_to_free_when_available(self):
        out = R._cost_pool(self._pool(), free_only=False,
                           prefer_free=True, allow_paid=False)
        assert all(c["free"] for c in out)          # paid dropped (last resort)

    def test_free_first_allows_paid_when_no_free(self):
        paid_only = [{"model": "p", "free": False}]
        out = R._cost_pool(paid_only, free_only=False,
                           prefer_free=True, allow_paid=False)
        assert out == paid_only                      # paid IS the last resort

    def test_allow_paid_tier_keeps_full_pool(self):
        out = R._cost_pool(self._pool(), free_only=False,
                           prefer_free=True, allow_paid=True)
        assert len(out) == 3                         # paid strong tier competes


class TestProactiveQuota:
    """P5 #16 — the free-tier quota LEDGER is now read by candidate selection,
    not just written on dispatch. A provider whose free window is drained is
    rotated away from BEFORE it 429s; one with more quota left outranks one
    that's nearly dry. Fail-open: no ledger data → today's ranking exactly."""

    def _qm(self, **windows):
        """A QuotaManager holding only the given providers (limit, used)."""
        from app.llm.quota_manager import DAY, QuotaManager
        qm = QuotaManager(now=lambda: 0.0)
        qm._q.clear()                      # drop the built-in DEFAULTS
        for prov, (limit, used) in windows.items():
            qm.configure(prov, limit=limit, window_s=DAY)
            for _ in range(used):
                qm.record(prov)
        return qm

    def _patch_qm(self, monkeypatch, qm):
        monkeypatch.setattr("app.llm.quota_manager.quota_manager", lambda: qm)

    # ── the ledger snapshot the router reads once per request ────────────
    def test_state_reports_fraction_left_and_exhaustion(self, monkeypatch):
        self._patch_qm(monkeypatch, self._qm(
            fresh=(100, 0), dry=(100, 90), dead=(100, 100)))
        st = R._quota_state()
        assert st["fresh"] == (1.0, False)
        assert st["dry"] == (0.1, False)
        assert st["dead"] == (0.0, True)          # exhausted() → skip it

    def test_state_omits_unlimited_and_unknown_providers(self, monkeypatch):
        # limit<=0 = unlimited/unknown → headroom() is None → NO quota signal,
        # so such a provider is scored exactly as it is today.
        self._patch_qm(monkeypatch, self._qm(unlimited=(0, 0), known=(10, 1)))
        st = R._quota_state()
        assert "unlimited" not in st and "never-heard-of-it" not in st
        assert st["known"] == (0.9, False)

    def test_state_fails_open_when_ledger_errors(self, monkeypatch):
        def boom():
            raise RuntimeError("ledger is on fire")
        monkeypatch.setattr("app.llm.quota_manager.quota_manager", boom)
        assert R._quota_state() == {}            # → every candidate scores as today

    # ── scoring: headroom feeds the score ────────────────────────────────
    def test_more_headroom_outranks_less_all_else_equal(self):
        roomy = R._candidate_score(0, 1.0, 10, 10, "standard",
                                   quota_headroom=0.9, quota_w=R._W_QUOTA)
        dry = R._candidate_score(0, 1.0, 10, 10, "standard",
                                 quota_headroom=0.05, quota_w=R._W_QUOTA)
        assert roomy < dry                        # lower score = picked first

    def test_quota_weight_zero_is_byte_identical(self):
        base = R._candidate_score(0, 1.0, 10, 10, "standard")
        z = R._candidate_score(0, 1.0, 10, 10, "standard",
                               quota_headroom=0.0, quota_w=0.0)
        assert base == z

    def test_unknown_quota_is_neutral(self):
        # The router passes quota_headroom=1.0 for a provider with no ledger row.
        base = R._candidate_score(0, 1.0, 10, 10, "standard")
        unknown = R._candidate_score(0, 1.0, 10, 10, "standard",
                                     quota_headroom=1.0, quota_w=R._W_QUOTA)
        assert base == unknown

    # ── pool filter: exhausted providers are skipped, never to zero ──────
    def test_exhausted_dropped_when_a_healthy_one_exists(self):
        pool = [{"model": "dead", "quota_exhausted": True},
                {"model": "alive", "quota_exhausted": False}]
        out = R._quota_pool(pool)
        assert [c["model"] for c in out] == ["alive"]

    def test_all_exhausted_falls_back_to_full_pool(self):
        # NEVER leave the router with zero candidates: a drained-everything pool
        # is returned intact (reactive 429 / emergency-paid / backoff take over)
        # rather than manufacturing a NoRouteAvailable.
        pool = [{"model": "a", "quota_exhausted": True},
                {"model": "b", "quota_exhausted": True}]
        assert R._quota_pool(pool) == pool

    def test_missing_quota_key_is_treated_as_routable(self):
        # Fail-open: candidates scored before this change (no quota key at all).
        pool = [{"model": "a"}, {"model": "b"}]
        assert R._quota_pool(pool) == pool

    # ── the two together, as route_request composes them ─────────────────
    def test_exhausted_provider_is_not_chosen_over_a_healthy_one(self, monkeypatch):
        """End-to-end of the selection math: the STRONGER model is on a drained
        provider, the weaker one is fresh. Today the router would happily send to
        the drained one (guaranteed 429); now it doesn't."""
        self._patch_qm(monkeypatch, self._qm(dead=(100, 100), fresh=(100, 0)))
        st = R._quota_state()

        def candidate(name, platform, intel):
            frac, exh = st.get(platform, (1.0, False))
            score = R._candidate_score(0, 1.0, intel, 10, "standard",
                                       quota_headroom=frac, quota_w=R._W_QUOTA)
            if exh:
                score += R._QUOTA_EXHAUSTED_PENALTY
            return {"model": name, "score": score, "quota_exhausted": exh}

        pool = [candidate("strong-but-drained", "dead", 5),
                candidate("ok-and-fresh", "fresh", 12)]
        pool = R._quota_pool(pool)                       # filter, then rank
        pool.sort(key=lambda c: c["score"])
        assert pool[0]["model"] == "ok-and-fresh"
        assert all(not c["quota_exhausted"] for c in pool)

    def test_ranking_unchanged_when_no_quota_data(self, monkeypatch):
        """Routing still succeeds — and picks the same model as today — when the
        ledger errors out entirely."""
        def boom():
            raise RuntimeError("no ledger")
        monkeypatch.setattr("app.llm.quota_manager.quota_manager", boom)
        st = R._quota_state()
        assert st == {}
        frac, exh = st.get("groq", (1.0, False))
        with_quota = R._candidate_score(0, 1.0, 5, 10, "standard",
                                        quota_headroom=frac, quota_w=R._W_QUOTA)
        today = R._candidate_score(0, 1.0, 5, 10, "standard")
        assert with_quota == today and exh is False


class TestConfigDefaults:
    def test_new_flags_default_off(self):
        from app.core.config_loader import RoutingSection
        r = RoutingSection()
        assert r.latency_aware is False and r.latency_weight == 0.0
        assert r.circuit_breaker is False and r.circuit_fail_threshold == 3
        assert r.circuit_cooldown_s == 30.0
        assert r.free_only is False
        # Emergency paid fallback defaults ON so enabling free_only is SAFE (a
        # transient free-tier outage degrades gracefully instead of erroring);
        # set it False for the strict never-spend guarantee.
        assert r.free_only_emergency_paid is True
