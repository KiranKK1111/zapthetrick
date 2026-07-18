"""Knowledge Freshness scoring (roadmap Phase 3 #17).

Assigns a 0..1 freshness to a piece of knowledge from its age (exponential
decay by a half-life), flags staleness past a TTL, and blends freshness with a
base relevance score so retrieval can prefer *current* facts without ignoring
relevance. Deterministic — the caller supplies ages (seconds), so it stays
time-free and testable. Fail-open.
"""
from __future__ import annotations

import math

_DAY = 86_400.0


def freshness_score(age_seconds: float, *, half_life_seconds: float = 30 * _DAY) -> float:
    """1.0 for brand-new, decaying to 0.5 at one half-life, → 0 as it ages.
    Negative/zero age clamps to 1.0."""
    try:
        if age_seconds <= 0:
            return 1.0
        hl = max(1.0, float(half_life_seconds))
        return round(2.0 ** (-float(age_seconds) / hl), 4)
    except Exception:  # noqa: BLE001
        return 1.0


def is_stale(age_seconds: float, *, ttl_seconds: float = 180 * _DAY) -> bool:
    try:
        return float(age_seconds) > float(ttl_seconds)
    except Exception:  # noqa: BLE001
        return False


def blend(relevance: float, freshness: float, *, freshness_weight: float = 0.2) -> float:
    """Combine a base relevance score with freshness. freshness_weight in [0,1]
    controls how much recency matters (0 = pure relevance)."""
    try:
        w = min(1.0, max(0.0, float(freshness_weight)))
        return round((1 - w) * float(relevance) + w * float(freshness), 4)
    except Exception:  # noqa: BLE001
        return float(relevance)


def rerank_by_freshness(items, *, freshness_weight: float = 0.2):
    """Re-rank [(id, relevance, age_seconds), ...] by blended score, descending.
    Returns [(id, blended_score), ...]. Stable, deterministic, fail-open."""
    try:
        scored = []
        for i, (cid, rel, age) in enumerate(items):
            fr = freshness_score(age)
            scored.append((cid, blend(rel, fr, freshness_weight=freshness_weight), i))
        scored.sort(key=lambda t: (-t[1], t[2]))  # score desc, original order tiebreak
        return [(cid, score) for cid, score, _ in scored]
    except Exception:  # noqa: BLE001
        return [(cid, rel) for (cid, rel, _age) in items]


__all__ = [
    "freshness_score", "is_stale", "blend", "rerank_by_freshness",
]
