"""Staged rollout + rollback decision logic (roadmap Phase 1 #24).

The `update_check` foundation already compares versions + reads a manifest. This
adds the SOFTWARE half of a release channel: deterministic staged-rollout cohort
assignment (a device is offered an update only if it falls inside the rollout
percentage) and rollback (a version marked bad is never offered). The actual
binary DELIVERY / hosting is an ops/packaging concern, not code — this decides
WHETHER to offer, from a manifest the server controls.

Deterministic: the same device + version always lands in the same cohort bucket,
so a rollout can be widened smoothly without re-shuffling who already has it.
"""
from __future__ import annotations

import hashlib


def _bucket(device_id: str, version: str) -> int:
    """Stable 0..99 bucket for (device, version). Same inputs → same bucket."""
    h = hashlib.sha256(f"{device_id}|{version}".encode()).hexdigest()
    return int(h[:8], 16) % 100


def in_rollout(device_id: str, version: str, *, percent: int) -> bool:
    """True when this device is inside a [percent]% staged rollout of [version].
    percent<=0 → nobody yet; >=100 → everybody."""
    p = max(0, min(100, int(percent)))
    if p >= 100:
        return True
    if p <= 0:
        return False
    return _bucket(device_id, version) < p


def rollout_decision(
    device_id: str,
    latest_version: str,
    current_version: str,
    *,
    percent: int = 100,
    blocked: set[str] | None = None,
) -> dict:
    """Whether to OFFER [latest_version] to this device.

    Not offered when: the version is blocked (rolled back), the device is already
    on it or newer, or the device isn't in the staged-rollout cohort yet.
    """
    blocked = blocked or set()
    if latest_version in blocked:
        return {"offer": False, "reason": "version rolled back",
                "version": latest_version}
    if _cmp(latest_version, current_version) <= 0:
        return {"offer": False, "reason": "already current",
                "version": latest_version}
    if not in_rollout(device_id, latest_version, percent=percent):
        return {"offer": False, "reason": "not in staged-rollout cohort",
                "version": latest_version, "bucket": _bucket(device_id, latest_version)}
    return {"offer": True, "reason": "eligible", "version": latest_version}


def _cmp(a: str, b: str) -> int:
    """Compare dotted numeric versions. Returns -1/0/1 (a<b / a==b / a>b)."""
    def parts(v: str) -> list[int]:
        out = []
        for p in (v or "0").split("."):
            digits = "".join(ch for ch in p if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return out
    pa, pb = parts(a), parts(b)
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    return (pa > pb) - (pa < pb)


__all__ = ["in_rollout", "rollout_decision"]
