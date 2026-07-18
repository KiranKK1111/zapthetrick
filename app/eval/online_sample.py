"""Online eval sampling (Architecture §14 / gap G12.1).

Samples a small fraction of LIVE turns and records them as structured "eval
sample" records, so real production traffic can be fed into the offline grading
harness (accuracy tracking + regression over time) instead of relying only on a
tiny static golden set. Off by default (`learning.online_eval_sample_rate = 0`).

Pure + injectable (`rng`, `sink`) for testing; fail-open (never breaks a turn).
"""
from __future__ import annotations

import json
import logging
import random

log = logging.getLogger("eval.online")


def _default_sink(rec: dict) -> None:
    # One structured line per sample — greppable, and ready to ship to the eval
    # harness / a store. Paired with the turn's trace via `trace_id`.
    log.info("eval sample %s", json.dumps(rec, default=str))


def maybe_record(
    *,
    question: str,
    answer: str,
    intent: str | None = None,
    trace_id: str | None = None,
    rate: float = 0.0,
    rng=None,
    sink=None,
) -> bool:
    """With probability `rate` (0..1) record the turn as an eval sample. Returns
    True when recorded. `rng()` → float in [0,1) and `sink(rec)` are injectable."""
    try:
        r = float(rate or 0.0)
        if r <= 0.0:
            return False
        draw = (rng or random.random)()
        if draw >= r:
            return False
        rec = {
            "kind": "eval_sample",
            "question": (question or "")[:2000],
            "answer": (answer or "")[:4000],
            "intent": intent,
            "trace_id": trace_id,
        }
        (sink or _default_sink)(rec)
        return True
    except Exception:  # noqa: BLE001 — sampling must never break a turn
        return False


__all__ = ["maybe_record"]
