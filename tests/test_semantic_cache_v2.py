"""Stage-4 §3.6 Component D — semantic answer cache v2.

Additive over the exact-only cache (byte-identical when `semantic_cache` is off):
per-user + context-fingerprint keying, a VOLATILE freshness bypass, a SLOW-class
shorter TTL, and a near (embedding) tier keyed by a caller-supplied vector.
"""
from __future__ import annotations

import time

import pytest

from app.llm import cache
from storage.context import current_user_id_var

_A = "11111111-1111-1111-1111-111111111111"
_B = "22222222-2222-2222-2222-222222222222"
_MSGS = [{"role": "user", "content": "explain a hashmap"}]


@pytest.fixture(autouse=True)
def _clean():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def semantic_on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.advanced_rag, "semantic_cache", True, raising=False)
    monkeypatch.setattr(cfg.advanced_rag, "cognitive_cache", True, raising=False)
    yield


def _as_user(uid):
    return current_user_id_var.set(uid)


# --------------------------------------------------------------------------- #
class TestFreshness:
    def test_volatile(self):
        assert cache.freshness_class("what is the latest news today") == "volatile"
        assert cache.freshness_class("current stock price of X") == "volatile"

    def test_slow(self):
        assert cache.freshness_class("the modern way to do X") == "slow"

    def test_stable(self):
        assert cache.freshness_class("explain how a hashmap works") == "stable"


class TestBackwardCompatOff:
    def test_exact_roundtrip_unchanged(self):
        # Flag OFF (default) → process-wide exact cache, no scope folding.
        k = cache.maybe_key(_MSGS, {"temperature": 0.0})
        assert k is not None
        cache.put(k, "an answer")
        assert cache.get(k) == "an answer"

    def test_key_ignores_user_when_off(self):
        k1 = cache.maybe_key(_MSGS, {})
        tok = _as_user(_A)
        try:
            k2 = cache.maybe_key(_MSGS, {})
        finally:
            current_user_id_var.reset(tok)
        assert k1 == k2  # no per-user scoping while the flag is off

    def test_high_temp_not_cached(self):
        assert cache.maybe_key(_MSGS, {"temperature": 0.9}) is None


class TestPerUserAndContext:
    def test_per_user_keys_differ(self, semantic_on):
        tok = _as_user(_A)
        try:
            ka = cache.maybe_key(_MSGS, {})
        finally:
            current_user_id_var.reset(tok)
        tok = _as_user(_B)
        try:
            kb = cache.maybe_key(_MSGS, {})
        finally:
            current_user_id_var.reset(tok)
        assert ka is not None and kb is not None and ka != kb

    def test_context_fingerprint_keys_differ(self, semantic_on):
        ka = cache.maybe_key(_MSGS, {}, context_fp="filesetA")
        kb = cache.maybe_key(_MSGS, {}, context_fp="filesetB")
        assert ka != kb

    def test_isolation_a_cannot_read_b(self, semantic_on):
        tok = _as_user(_A)
        try:
            ka = cache.maybe_key(_MSGS, {})
            cache.put(ka, "A's answer")
        finally:
            current_user_id_var.reset(tok)
        tok = _as_user(_B)
        try:
            kb = cache.maybe_key(_MSGS, {})
            assert cache.get(kb) is None      # B misses A's entry
        finally:
            current_user_id_var.reset(tok)


class TestFreshnessGate:
    def test_volatile_bypasses_cache(self, semantic_on):
        assert cache.maybe_key(
            [{"role": "user", "content": "latest news today"}], {}) is None

    def test_slow_gets_shorter_ttl(self, semantic_on, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.advanced_rag, "semantic_cache_slow_ttl_s", 600,
                            raising=False)
        k = cache.maybe_key(
            [{"role": "user", "content": "the modern way to do X"}], {})
        assert k is not None
        assert cache._pending_ttl.get(k) == 600   # SLOW ttl staged
        cache.put(k, "v")
        expiry = cache._store[k][0]
        assert expiry <= time.time() + 601        # ~600s, not the 3600s default
        assert k not in cache._pending_ttl        # consumed by put


class TestNearTier:
    def _near_on(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.advanced_rag, "semantic_cache", True,
                            raising=False)
        monkeypatch.setattr(cfg.advanced_rag, "semantic_cache_near", True,
                            raising=False)
        monkeypatch.setattr(cfg.advanced_rag, "semantic_cache_near_threshold",
                            0.97, raising=False)

    def test_near_hit_above_threshold(self, monkeypatch):
        self._near_on(monkeypatch)
        k = cache.maybe_key(_MSGS, {})
        cache.put(k, "the stored answer")
        cache.near_index(_MSGS, key=k, query_vec=[1.0, 0.0, 0.0])
        hit = cache.near_get(_MSGS, {}, query_vec=[0.999, 0.02, 0.0])
        assert hit is not None
        assert hit[0] == "the stored answer"
        assert hit[1]["near"] is True and hit[1]["similarity"] >= 0.97

    def test_near_miss_below_threshold(self, monkeypatch):
        self._near_on(monkeypatch)
        k = cache.maybe_key(_MSGS, {})
        cache.put(k, "ans")
        cache.near_index(_MSGS, key=k, query_vec=[1.0, 0.0, 0.0])
        assert cache.near_get(_MSGS, {}, query_vec=[0.0, 1.0, 0.0]) is None

    def test_near_none_vector_skips(self, monkeypatch):
        self._near_on(monkeypatch)
        assert cache.near_get(_MSGS, {}, query_vec=None) is None

    def test_near_off_returns_none(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.advanced_rag, "semantic_cache_near", False,
                            raising=False)
        cache.near_index(_MSGS, key="k", query_vec=[1.0, 0.0])
        assert cache.near_get(_MSGS, {}, query_vec=[1.0, 0.0]) is None

    def test_near_respects_user_scope(self, monkeypatch):
        self._near_on(monkeypatch)
        tok = _as_user(_A)
        try:
            ka = cache.maybe_key(_MSGS, {})
            cache.put(ka, "A only")
            cache.near_index(_MSGS, key=ka, query_vec=[1.0, 0.0, 0.0])
        finally:
            current_user_id_var.reset(tok)
        tok = _as_user(_B)
        try:
            assert cache.near_get(_MSGS, {}, query_vec=[1.0, 0.0, 0.0]) is None
        finally:
            current_user_id_var.reset(tok)

    def test_near_enabled_flag(self, monkeypatch):
        assert cache.near_enabled() is False
        self._near_on(monkeypatch)
        assert cache.near_enabled() is True
