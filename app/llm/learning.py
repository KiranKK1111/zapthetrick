"""Learning router (intelligent-model-routing R8).

Records per-(Task_Category, model) outcomes and exposes ``learned_success`` as a
0..1 signal the router blends in (additive ``_W_LEARN`` term, weight 0 by
default). Bounded + persisted in ``User.preferences`` (no new table / migration);
neutral (0.5 → no bias) when there's no history (R8.3). Never overrides the
capability floor / free-first / paid-budget / safety constraints — it only
nudges the score (R8.4, Property 7).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Process-wide in-memory mirror of the learning store so `learned_success` (a
# hot, synchronous call inside scoring) never blocks on the DB. Loaded lazily
# and updated on each `record`. Shape: {category: {model_db_id: {"s": int, "n": int}}}
_STATS: dict[str, dict[int, dict]] = {}
_MAX_MODELS_PER_CAT = 200


def learned_success(task_category: str, model_key) -> float:
    """0..1 historical success for (category, model). Neutral 0.5 with no
    history so it never biases an unseen model (R8.3). Never raises. `model_key`
    may be a db id or a model_id string — the same key must be used by record()."""
    try:
        cat = _STATS.get(task_category or "")
        if not cat:
            return 0.5
        rec = cat.get(model_key)
        if not rec or rec.get("n", 0) <= 0:
            return 0.5
        # Laplace-smoothed success rate so a single sample isn't 0 or 1.
        return max(0.0, min(1.0, (rec["s"] + 1) / (rec["n"] + 2)))
    except Exception:  # noqa: BLE001
        return 0.5


def record(task_category: str, model_key, success: bool,
           latency_ms: float | None = None) -> None:
    """Record one routed outcome (in-memory + best-effort persist). Bounded per
    category with oldest/LRU-ish trim. Never raises (Property 9)."""
    try:
        if model_key is None:
            return
        cat = _STATS.setdefault(task_category or "general", {})
        rec = cat.setdefault(model_key, {"s": 0, "n": 0})
        rec["n"] += 1
        if success:
            rec["s"] += 1
        if latency_ms is not None:
            rec["lat"] = latency_ms
        # Bound the per-category map.
        if len(cat) > _MAX_MODELS_PER_CAT:
            # Drop the entry with the fewest samples (least informative).
            victim = min(cat, key=lambda k: cat[k].get("n", 0))
            cat.pop(victim, None)
    except Exception:  # noqa: BLE001
        pass


def load_from(prefs: dict | None) -> None:
    """Hydrate the in-memory mirror from a persisted ``User.preferences`` blob
    under the ``llm_routing_stats`` key. Best-effort."""
    try:
        if not isinstance(prefs, dict):
            return
        data = prefs.get("llm_routing_stats")
        if not isinstance(data, dict):
            return
        _STATS.clear()
        for cat, models in data.items():
            if not isinstance(models, dict):
                continue
            _STATS[cat] = {k: dict(v) for k, v in models.items()
                           if isinstance(v, dict)}
    except Exception:  # noqa: BLE001
        pass


def export_to(prefs: dict) -> None:
    """Write the in-memory mirror into a ``User.preferences`` blob (caller
    persists). Keys are stringified for JSON."""
    try:
        prefs["llm_routing_stats"] = {
            cat: {str(mid): rec for mid, rec in models.items()}
            for cat, models in _STATS.items()
        }
    except Exception:  # noqa: BLE001
        pass


def reset() -> None:
    _STATS.clear()


__all__ = ["learned_success", "record", "load_from", "export_to", "reset"]
