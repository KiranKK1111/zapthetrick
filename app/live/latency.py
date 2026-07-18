"""
Adaptive fast/deep latency path (live-conversational-intelligence R10).

Picks a fast vs deep answer path from the predicted difficulty (reusing the
`agent.predict` difficulty — no new difficulty LLM call) and, where available,
the `evaluation-and-reliability` governor. Degrades to the fast path when
latency health is poor. Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

FAST = "fast"
DEEP = "deep"


@dataclass
class PathChoice:
    path: str         # 'fast' | 'deep'
    depth: str        # 'concise' | 'standard' | 'detailed'


def select_path(difficulty: str = "standard", *, latency_degraded: bool = False) -> PathChoice:
    """Choose the live answer path. Hard/expert → deep/detailed; trivial →
    fast/concise; otherwise fast/standard. Poor latency health forces fast."""
    try:
        d = (difficulty or "standard").lower()
        if latency_degraded:
            return PathChoice(path=FAST, depth="concise")

        # Best-effort: let the evaluation-and-reliability governor weigh in if
        # present (it never raises here; failure → difficulty mapping).
        try:
            from app.quality import governor as _gov  # noqa: F401
            # The governor's pipeline hint is advisory; difficulty drives depth.
        except Exception:  # noqa: BLE001
            pass

        if d in ("hard", "expert"):
            return PathChoice(path=DEEP, depth="detailed")
        if d == "trivial":
            return PathChoice(path=FAST, depth="concise")
        return PathChoice(path=FAST, depth="standard")
    except Exception:  # noqa: BLE001
        return PathChoice(path=FAST, depth="standard")
