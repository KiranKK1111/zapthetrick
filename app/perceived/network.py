"""Network-condition adaptivity (perceived-speed R22).

Pure policy the client + server consult based on the current connection quality:
while slow/metered, prefer compact payloads + smaller chunks + local models
(R22.1); while metered/offline, suppress speculative work to conserve data
(R22.2); a return to normal restores standard behavior (R22.3, within the
SpeculationBudget). The client owns detection and gates its prefetch; this module
is the shared, testable decision.
"""
from __future__ import annotations

NORMAL = "normal"
SLOW = "slow"
METERED = "metered"
OFFLINE = "offline"

_VALID = {NORMAL, SLOW, METERED, OFFLINE}


def normalize(condition: str | None) -> str:
    c = (condition or NORMAL).strip().lower()
    return c if c in _VALID else NORMAL


def should_suppress_speculation(condition: str | None) -> bool:
    """Metered/offline → suppress speculative work (prefetch/draft/precompute)."""
    return normalize(condition) in (METERED, OFFLINE)


def prefers_compact_payloads(condition: str | None) -> bool:
    """Slow/metered/offline → compressed payloads + smaller chunks + local models."""
    return normalize(condition) in (SLOW, METERED, OFFLINE)


__all__ = [
    "NORMAL", "SLOW", "METERED", "OFFLINE",
    "normalize", "should_suppress_speculation", "prefers_compact_payloads",
]
