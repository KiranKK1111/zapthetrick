"""Knowledge Freshness wired into episodic recall (P3 #17): among equally
relevant memories, the fresher one ranks first."""
from __future__ import annotations

import time

from app.memory.episodic import Episode, EpisodicMemory


def _ep(q: str, age_days: float) -> Episode:
    return Episode(question=q,
                   ts_ms=int((time.time() - age_days * 86400) * 1000))


def test_fresher_memory_ranks_first_when_relevance_ties():
    mem = EpisodicMemory()
    old = _ep("how does kafka partition work", age_days=200)
    new = _ep("how does kafka partition work", age_days=1)
    mem.record(old)
    mem.record(new)
    hits = mem.search_similar("how does kafka partition work", top_k=2)
    assert hits[0] is new          # fresher wins the tie


def test_strong_relevance_still_beats_a_slightly_fresher_weak_match():
    mem = EpisodicMemory()
    strong_old = _ep("explain kafka consumer group rebalancing in detail",
                     age_days=60)
    weak_new = _ep("kafka basics", age_days=0)
    mem.record(strong_old)
    mem.record(weak_new)
    hits = mem.search_similar(
        "explain kafka consumer group rebalancing in detail", top_k=2)
    # freshness_weight is small (0.15) → high relevance still wins.
    assert hits[0] is strong_old


def test_no_overlap_returns_nothing():
    mem = EpisodicMemory()
    mem.record(_ep("something unrelated", age_days=1))
    assert mem.search_similar("totally different query xyz", top_k=3) == []
