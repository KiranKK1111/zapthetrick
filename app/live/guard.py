"""
Knowledge-gap / hallucination guard (live-conversational-intelligence R9).

When the answer confidence is low (a hard/expert question, weak retrieval, or
poor STT) the guard caps the answer to concise + hedged rather than a long,
confident invention. Deterministic; composes with the existing
`quality.critic` / verification (it does not duplicate them). Fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

# Base answer confidence by predicted difficulty.
_BASE = {"trivial": 0.92, "standard": 0.8, "hard": 0.55, "expert": 0.42}


@dataclass
class GapVerdict:
    gap: bool
    max_depth: str          # 'concise' | 'standard' | 'detailed'
    hedge: bool
    confidence: float       # 0..1 surfaced answer confidence


def assess(
    difficulty: str = "standard",
    *,
    retrieval_conf: float | None = None,
    stt_conf: float | None = None,
    threshold: float = 0.5,
) -> GapVerdict:
    """Assess the knowledge-gap risk. Below `threshold` → concise + hedged.
    Never raises — returns a neutral standard verdict on bad input."""
    try:
        conf = _BASE.get((difficulty or "standard").lower(), 0.8)
        # Low upstream confidence drags the answer confidence down (uncertainty
        # propagation): weight each available signal.
        if stt_conf is not None:
            conf = min(conf, 0.5 + 0.5 * max(0.0, min(1.0, stt_conf)))
        if retrieval_conf is not None:
            conf = min(conf, 0.4 + 0.6 * max(0.0, min(1.0, retrieval_conf)))
        conf = max(0.0, min(1.0, conf))
        if conf < threshold:
            return GapVerdict(gap=True, max_depth="concise", hedge=True, confidence=conf)
        depth = "detailed" if conf >= 0.85 else "standard"
        return GapVerdict(gap=False, max_depth=depth, hedge=False, confidence=conf)
    except Exception:  # noqa: BLE001
        return GapVerdict(gap=False, max_depth="standard", hedge=False, confidence=0.8)


def hedge_directive() -> str:
    """Prompt directive used when the guard flags a knowledge gap."""
    return ("If you are not certain of specific facts, keep the answer concise and "
            "high-level, briefly acknowledge the uncertainty, and do NOT invent "
            "names, numbers, or details.")
