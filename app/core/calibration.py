"""Outcome-driven threshold calibration (gap G1).

Several decisions use hand-set thresholds (the semantic-intent `primary_threshold`,
the topic-shift similarity, clarification bands). This lets them ADAPT: each named
parameter records (score_at_decision, was_it_good) samples, and `calibrated()`
returns the score that best separates good from bad outcomes — the midpoint of the
good-score and bad-score means — blended with the configured default for
stability, and guarded by a minimum sample count so it never swings on thin data.

Bounded per parameter, file-persisted, config-gated (`calibration.enabled`,
default off → callers keep their static defaults). Pure + fail-open.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import threading

log = logging.getLogger(__name__)

_MAX_SAMPLES = 400          # ring size per parameter
_BLEND = 0.5               # weight of the learned value vs the configured default

_LOCK = threading.Lock()
# param -> list[(score, good)]
_SAMPLES: dict[str, list[tuple[float, bool]]] = {}
_LOADED = False


def _store_path() -> pathlib.Path:
    override = os.environ.get("ZAPTHETRICK_CALIBRATION")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".zapthetrick" / "calibration.json"


def enabled() -> bool:
    # Enabling default (Phase 7 #12/#14 — continuous calibration + threshold
    # auto-tuning): when the flag is absent, calibration is ON so learned
    # thresholds adapt from real outcomes out of the box. The config owner sets
    # `calibration.enabled` in config.yaml to override (report: flip it to true).
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.calibration, "enabled", True))
    except Exception:  # noqa: BLE001
        return True


def _min_samples() -> int:
    try:
        from app.core.config_loader import cfg
        return int(getattr(cfg.calibration, "min_samples", 20))
    except Exception:  # noqa: BLE001
        return 20


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
                for k, rows in (data or {}).items():
                    _SAMPLES[k] = [(float(v), bool(g)) for v, g in rows]
        except Exception as exc:  # noqa: BLE001
            log.info("calibration load failed (%s); starting empty", exc)
        _LOADED = True


def _persist() -> None:
    try:
        p = _store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({k: [[round(v, 4), g] for v, g in rows]
                                 for k, rows in _SAMPLES.items()}),
                     encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.info("calibration persist failed (%s)", exc)


def observe(param: str, score: float, good: bool, *, persist: bool = True) -> None:
    """Record that a decision made at `score` turned out good/bad. Never raises."""
    try:
        _ensure_loaded()
        with _LOCK:
            ring = _SAMPLES.setdefault(str(param), [])
            ring.append((float(score), bool(good)))
            if len(ring) > _MAX_SAMPLES:
                del ring[0:len(ring) - _MAX_SAMPLES]
            if persist:
                _persist()
    except Exception as exc:  # noqa: BLE001
        log.info("calibration observe failed: %s", exc)


def calibrated(param: str, default: float, *, lo: float = 0.0,
               hi: float = 1.0) -> float:
    """The calibrated threshold for `param`, or `default` when disabled / too few
    samples. Learned value = midpoint of mean(good scores) and mean(bad scores),
    clamped to [lo, hi] and blended with `default`. Never raises."""
    try:
        if not enabled():
            return default
        _ensure_loaded()
        with _LOCK:
            rows = list(_SAMPLES.get(str(param), []))
        if len(rows) < _min_samples():
            return default
        good = [s for s, g in rows if g]
        bad = [s for s, g in rows if not g]
        if not good or not bad:
            return default
        learned = (sum(good) / len(good) + sum(bad) / len(bad)) / 2.0
        learned = max(lo, min(hi, learned))
        return _BLEND * learned + (1.0 - _BLEND) * default
    except Exception:  # noqa: BLE001
        return default


def stats() -> dict:
    _ensure_loaded()
    with _LOCK:
        return {k: len(v) for k, v in _SAMPLES.items()}


def clear(*, persist: bool = True) -> None:
    with _LOCK:
        _SAMPLES.clear()
        if persist:
            _persist()


def _reset_for_test() -> None:
    global _LOADED
    with _LOCK:
        _SAMPLES.clear()
        _LOADED = True


__all__ = ["observe", "calibrated", "enabled", "stats", "clear"]
