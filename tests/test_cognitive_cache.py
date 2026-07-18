"""P2-10 — cognitive cache + recently-successful-model routing bias.

Pure/offline: the cache key/maybe_key gating, TTL + LRU eviction, stats, the
LLMClient.complete_routed cache wiring (provider monkeypatched), and the
engine's per-difficulty success bias helpers.
"""
from __future__ import annotations

import asyncio

import pytest

from app.llm import cache


@pytest.fixture(autouse=True)
def _clean_cache():
    cache.clear()
    yield
    cache.clear()


# ── key + gating ──────────────────────────────────────────────────────────
def test_cache_key_is_stable_and_option_sensitive():
    msgs = [{"role": "user", "content": "hello"}]
    k1 = cache.cache_key(msgs, {"temperature": 0.0, "difficulty": "standard"})
    k2 = cache.cache_key(msgs, {"temperature": 0.0, "difficulty": "standard"})
    k3 = cache.cache_key(msgs, {"temperature": 0.0, "difficulty": "expert"})
    assert k1 == k2 and k1 != k3
    # an option that doesn't shape the answer is ignored
    k4 = cache.cache_key(msgs, {"temperature": 0.0, "difficulty": "standard",
                                "avoid_model_db_id": 9})
    assert k4 == k1


def test_maybe_key_skips_high_temp_and_multimodal():
    msgs = [{"role": "user", "content": "hi"}]
    assert cache.maybe_key(msgs, {"temperature": 0.2}) is not None
    assert cache.maybe_key(msgs, {"temperature": 0.9}) is None
    assert cache.maybe_key([], {}) is None
    multimodal = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    assert cache.maybe_key(multimodal, {"temperature": 0.0}) is None
    # This app's vision convention: a separate `images` key with string content
    # must NOT be cached (the image isn't part of the key).
    vision = [{"role": "user", "content": "can you solve this problem",
               "images": ["data:image/png;base64,AAAA"]}]
    assert cache.maybe_key(vision, {"temperature": 0.0}) is None


def test_maybe_key_respects_disable(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.advanced_rag, "cognitive_cache", False)
    assert cache.maybe_key([{"role": "user", "content": "x"}], {}) is None


# ── put/get + TTL + LRU ──────────────────────────────────────────────────────
def test_put_get_roundtrip_and_stats():
    k = cache.cache_key([{"role": "user", "content": "q"}], {"temperature": 0})
    assert cache.get(k) is None          # miss
    cache.put(k, "the answer")
    assert cache.get(k) == "the answer"  # hit
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["entries"] == 1


def test_empty_value_not_stored():
    k = cache.cache_key([{"role": "user", "content": "q"}], {})
    cache.put(k, "   ")
    assert cache.get(k) is None


def test_ttl_expiry(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.advanced_rag, "cognitive_cache_ttl_s", 0)
    k = cache.cache_key([{"role": "user", "content": "q"}], {})
    cache.put(k, "v")
    # ttl 0 → already expired on read
    assert cache.get(k) is None


def test_lru_eviction(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.advanced_rag, "cognitive_cache_max", 2)
    for i in range(3):
        cache.put(f"k{i}", f"v{i}")
    # oldest (k0) evicted
    assert cache.get("k0") is None
    assert cache.get("k2") == "v2"


# ── complete_routed wiring (provider monkeypatched) ──────────────────────────
def test_complete_routed_serves_from_cache(monkeypatch):
    import app.core.llm_client as lc

    calls = {"n": 0}

    async def fake_auto(messages, model, options):
        calls["n"] += 1

        class _R:
            model_db_id = 7
        return "computed answer", _R()

    monkeypatch.setattr(lc.cfg.llm, "provider", "auto")
    monkeypatch.setattr(lc.llm, "_auto_complete", fake_auto)

    msgs = [{"role": "user", "content": "explain caching"}]
    opts = {"temperature": 0.0, "difficulty": "standard"}
    t1, mid1 = asyncio.run(lc.llm.complete_routed(msgs, None, opts))
    t2, mid2 = asyncio.run(lc.llm.complete_routed(msgs, None, opts))
    assert t1 == t2 == "computed answer"
    assert calls["n"] == 1            # second call served from cache
    assert mid1 == 7 and mid2 is None  # cache hit has no model id


def test_complete_routed_not_cached_high_temp(monkeypatch):
    import app.core.llm_client as lc

    calls = {"n": 0}

    async def fake_auto(messages, model, options):
        calls["n"] += 1

        class _R:
            model_db_id = 1
        return f"answer {calls['n']}", _R()

    monkeypatch.setattr(lc.cfg.llm, "provider", "auto")
    monkeypatch.setattr(lc.llm, "_auto_complete", fake_auto)

    msgs = [{"role": "user", "content": "be creative"}]
    opts = {"temperature": 0.9}
    asyncio.run(lc.llm.complete_routed(msgs, None, opts))
    asyncio.run(lc.llm.complete_routed(msgs, None, opts))
    assert calls["n"] == 2            # high temp → never cached


# ── engine: prefer recently-successful model per difficulty ──────────────────
def test_engine_diff_success_bias(monkeypatch):
    from app.llm import engine

    engine._diff_success.clear()
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.advanced_rag, "prefer_recent_model", True)

    assert engine._diff_pref("hard") is None
    engine._record_diff_success("hard", 42)
    assert engine._diff_pref("hard") == 42
    assert engine._diff_pref("expert") is None   # per-difficulty


def test_engine_diff_success_respects_flag(monkeypatch):
    from app.core.config_loader import cfg
    from app.llm import engine

    engine._diff_success.clear()
    monkeypatch.setattr(cfg.advanced_rag, "prefer_recent_model", False)
    engine._record_diff_success("hard", 5)
    assert engine._diff_pref("hard") is None      # disabled → no bias
