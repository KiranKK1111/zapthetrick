"""Embedding-clustered learned routing (Phase 2 — "which model works on turns
like this one").

The category-keyed learning router (`learning.py`) asks "which model did well on
*coding* turns." That's coarse. This asks it *semantically*: it clusters past
turns by their query embedding (bge-m3, already computed by the Understanding
pass) and tracks per-cluster, per-model success. At routing time the query's
nearest cluster yields a per-model success signal that the router folds into its
score — so a free model that reliably nails questions *like this* gets picked,
even across intent/category boundaries.

Online, bounded, fail-open:
  * up to `_MAX_CLUSTERS` clusters; a turn joins its nearest cluster when cosine
    >= `_JOIN_SIM`, else seeds a new one (or joins the nearest once full);
  * each cluster keeps a running-mean centroid + `{model_id: [successes, n]}`;
  * `success_for` returns a Laplace-smoothed rate (neutral 0.5 with no data, so
    an unseen model is never penalized);
  * persisted to `~/.zapthetrick/semantic_routing.json` (compact: 32 centroids,
    not raw history). Injectable/pure enough to unit-test with plain vectors.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import threading

log = logging.getLogger(__name__)

_MAX_CLUSTERS = 32
_JOIN_SIM = 0.55          # cosine >= this → same cluster
_ROUND = 5                # centroid float precision on disk

_LOCK = threading.Lock()
# each cluster: {"c": list[float] (unit centroid), "n": int, "m": {mid: [s, n]}}
_CLUSTERS: list[dict] = []
_LOADED = False


def _store_path() -> pathlib.Path:
    override = os.environ.get("ZAPTHETRICK_SEMANTIC_ROUTING")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".zapthetrick" / "semantic_routing.json"


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.routing, "semantic_learning", False))
    except Exception:  # noqa: BLE001
        return False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        try:
            p = _store_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for c in data.get("clusters") or []:
                    if isinstance(c, dict) and c.get("c"):
                        _CLUSTERS.append({
                            "c": [float(x) for x in c["c"]],
                            "n": int(c.get("n", 1)),
                            "m": {str(k): [int(v[0]), int(v[1])]
                                  for k, v in (c.get("m") or {}).items()},
                        })
        except Exception as exc:  # noqa: BLE001
            log.info("semantic routing: load failed (%s); starting empty", exc)
        _LOADED = True


def _persist() -> None:
    try:
        p = _store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        out = {"clusters": [{
            "c": [round(x, _ROUND) for x in c["c"]],
            "n": c["n"], "m": c["m"],
        } for c in _CLUSTERS]}
        p.write_text(json.dumps(out), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.info("semantic routing: persist failed (%s)", exc)


def _unit(vec) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return list(vec)
    return [x / n for x in vec]


def _cos(a, b) -> float:
    # both expected unit-length; guard anyway
    return sum(x * y for x, y in zip(a, b))


def _nearest(unit_vec) -> tuple[int, float]:
    best_i, best_s = -1, -1.0
    for i, c in enumerate(_CLUSTERS):
        s = _cos(unit_vec, c["c"])
        if s > best_s:
            best_i, best_s = i, s
    return best_i, best_s


def record(embedding, model_id, success: bool) -> None:
    """Fold one turn's outcome into its semantic cluster. Never raises."""
    if not embedding or not model_id:
        return
    try:
        _ensure_loaded()
        uv = _unit([float(x) for x in embedding])
        mid = str(model_id)
        with _LOCK:
            idx, sim = _nearest(uv)
            if idx < 0 or (sim < _JOIN_SIM and len(_CLUSTERS) < _MAX_CLUSTERS):
                _CLUSTERS.append({"c": uv, "n": 0, "m": {}})
                idx = len(_CLUSTERS) - 1
            c = _CLUSTERS[idx]
            # running-mean centroid on the unit sphere
            n = c["n"]
            merged = [(c["c"][k] * n + uv[k]) for k in range(len(uv))]
            c["c"] = _unit(merged)
            c["n"] = n + 1
            rec = c["m"].setdefault(mid, [0, 0])
            rec[0] += 1 if success else 0
            rec[1] += 1
            _persist()
    except Exception as exc:  # noqa: BLE001
        log.info("semantic routing record failed: %s", exc)


def success_for(embedding, model_id) -> float:
    """Laplace-smoothed success of `model_id` in the query's nearest cluster.
    Neutral 0.5 when there's no cluster or no data for that model (so an unseen
    model isn't biased). Never raises."""
    try:
        if not embedding or not model_id:
            return 0.5
        _ensure_loaded()
        with _LOCK:
            if not _CLUSTERS:
                return 0.5
            uv = _unit([float(x) for x in embedding])
            idx, sim = _nearest(uv)
            if idx < 0 or sim < _JOIN_SIM:
                return 0.5
            rec = _CLUSTERS[idx]["m"].get(str(model_id))
            if not rec or rec[1] <= 0:
                return 0.5
            return max(0.0, min(1.0, (rec[0] + 1) / (rec[1] + 2)))
    except Exception:  # noqa: BLE001
        return 0.5


# Bounded map episode_id -> (model_id, embedding), so an explicit 👍/👎 arriving
# later (via the feedback route) can fold answer QUALITY into the same cluster —
# not just the completeness signal recorded at answer time (G9).
_TURN_CACHE: dict[str, tuple[str, list[float]]] = {}
_TURN_MAX = 256


def remember_turn(episode_id, model_id, embedding) -> None:
    if not episode_id or not model_id or not embedding:
        return
    with _LOCK:
        if len(_TURN_CACHE) >= _TURN_MAX:
            _TURN_CACHE.pop(next(iter(_TURN_CACHE)), None)
        _TURN_CACHE[str(episode_id)] = (str(model_id), list(embedding))


def record_feedback(episode_id, positive: bool) -> bool:
    """Fold an explicit 👍/👎 on a past turn into its semantic cluster (G9).
    Returns False when the turn isn't cached. Never raises."""
    try:
        with _LOCK:
            ent = _TURN_CACHE.get(str(episode_id))
        if not ent:
            return False
        record(ent[1], ent[0], bool(positive))
        return True
    except Exception:  # noqa: BLE001
        return False


def stats() -> dict:
    _ensure_loaded()
    with _LOCK:
        return {
            "clusters": len(_CLUSTERS),
            "observations": sum(sum(v[1] for v in c["m"].values())
                                for c in _CLUSTERS),
        }


def clear(*, persist: bool = True) -> None:
    with _LOCK:
        _CLUSTERS.clear()
        if persist:
            _persist()


def _reset_for_test() -> None:
    global _LOADED
    with _LOCK:
        _CLUSTERS.clear()
        _LOADED = True          # skip disk load


__all__ = ["record", "success_for", "enabled", "stats", "clear",
           "remember_turn", "record_feedback"]
