"""Answer reuse cache + privacy scoping (perceived-speed R14/R21, task 16.3).

Pins Property 8: exact + semantic hit, revalidate/invalidate, per-user
isolation, clear_user, and LRU bound. Embeddings are injected (no model load).
"""
from __future__ import annotations

from app.perceived.cache import AnswerCache


def _embed(text: str):
    # Toy embedding: bag-of-3-buckets by first char, deterministic + comparable.
    v = [0.0, 0.0, 0.0]
    for ch in text.lower():
        v[ord(ch) % 3] += 1.0
    return v


def test_exact_hit_and_scope_isolation():
    c = AnswerCache(max_entries=8)
    c.store("u1", "what is a hashmap?", "A hashmap is…")
    assert c.get_exact("u1", "What is a hashmap?") == "A hashmap is…"
    assert c.get_exact("u2", "what is a hashmap?") is None   # other user (R21.1)


def test_semantic_hit_same_scope_only():
    c = AnswerCache(max_entries=8, similarity=0.99)
    c.store("u1", "explain hashmaps", "ans", embedding=_embed("explain hashmaps"))
    # Near-identical phrasing → cosine ~1.0 ≥ threshold.
    got = c.serve("u1", "explain hashmaps", embed_fn=_embed)
    assert got == "ans"
    # Same prompt, different user → no cross-user reuse (R21.2).
    assert c.serve("u2", "explain hashmaps", embed_fn=_embed) is None


def test_revalidate_discards_on_failure():
    c = AnswerCache(max_entries=8)
    c.store("u1", "q", "stale")
    got = c.serve("u1", "q", validate=lambda a: a == "fresh")
    assert got is None                       # rejected (R14.3)
    assert c.get_exact("u1", "q") is None     # invalidated (R14.4/R21.4)


def test_quality_gate_skips_low_quality():
    c = AnswerCache(max_entries=8)
    c.store("u1", "q", "meh", quality_ok=False)
    assert c.get_exact("u1", "q") is None


def test_clear_user_removes_only_that_user():
    c = AnswerCache(max_entries=8)
    c.store("u1", "a", "1")
    c.store("u1", "b", "2")
    c.store("u2", "a", "3")
    removed = c.clear_user("u1")
    assert removed == 2
    assert c.get_exact("u1", "a") is None and c.get_exact("u1", "b") is None
    assert c.get_exact("u2", "a") == "3"      # other user untouched


def test_lru_bound():
    c = AnswerCache(max_entries=2)
    c.store("u1", "a", "1")
    c.store("u1", "b", "2")
    c.get_exact("u1", "a")        # touch a → b least-recent
    c.store("u1", "c", "3")       # evicts b
    assert c.get_exact("u1", "a") == "1"
    assert c.get_exact("u1", "c") == "3"
    assert c.get_exact("u1", "b") is None
    assert len(c) == 2


# ── Wiring pass: process-wide singleton + flag defaults ─────────────────────


def test_answer_cache_singleton_is_shared():
    """The route's serve() and store() must hit the SAME instance so an answer
    stored on one turn is served on a later one (perceived-speed R14 wiring)."""
    from app.perceived.cache import answer_cache

    a = answer_cache()
    b = answer_cache()
    assert a is b
    a.store("u1", "what is a trie?", "A trie is…")
    assert b.get_exact("u1", "what is a trie?") == "A trie is…"
    # Clean up so the shared singleton doesn't leak across tests.
    a.clear_user("u1")


def test_perceived_flags_defaults():
    """Latency batch 2026-07-11 (#4): the perceived-speed features default ON
    (answer cache, speculation, drafting); observatory/TTFT ack stay off."""
    from app.core.config_loader import PerceivedSection

    p = PerceivedSection()
    assert p.answer_cache is True
    assert p.speculation_enabled is True
    assert p.speculative_drafting is True
    assert p.observatory_enabled is False
    assert p.ttft_ack_threshold_s == 0.0
