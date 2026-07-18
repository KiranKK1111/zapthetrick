"""Confidence calibration (advanced-intent-reasoning R2).

A pure function that corrects the deterministic pre-gate's raw confidence so a
predicted confidence matches observed answerability: if, historically, turns in
a confidence bucket turned out answerable (the user skipped/overrode the
clarification) X% of the time, nudge the raw confidence toward X.

Calibration only engages once a bucket has enough samples; below that it returns
the raw confidence unchanged, so behavior is identical to the pre-calibration
pre-gate until real data exists (R2.2 / R2.5).
"""
from __future__ import annotations

from .outcomes import confidence_bucket

# How strongly observed data pulls the raw confidence (0 = ignore data, 1 =
# fully trust observed rate). A measured blend keeps a single noisy bucket from
# swinging the decision.
_BLEND = 0.5


def calibrate(raw: float, buckets: dict, min_samples: int = 8) -> float:
    """Return a calibrated confidence in [0,1].

    `buckets` is [OutcomeStore.calibration_buckets()] output:
        {bucket_int: {"answerable": n, "needed": n}}.
    With >= `min_samples` observations in the raw value's bucket, blend the raw
    confidence with the observed answerable-rate; otherwise return `raw`.
    """
    try:
        r = float(raw)
    except (TypeError, ValueError):
        return 0.0
    r = min(1.0, max(0.0, r))
    if not isinstance(buckets, dict) or min_samples <= 0:
        return r
    b = confidence_bucket(r)
    slot = buckets.get(b) or buckets.get(str(b))
    if not isinstance(slot, dict):
        return r
    answerable = int(slot.get("answerable", 0))
    needed = int(slot.get("needed", 0))
    total = answerable + needed
    if total < min_samples:
        return r
    observed = answerable / total
    calibrated = (1.0 - _BLEND) * r + _BLEND * observed
    return min(1.0, max(0.0, calibrated))


__all__ = ["calibrate"]
