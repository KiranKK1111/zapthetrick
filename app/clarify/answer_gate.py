"""Answer-first gate v2 (roadmap #12).

The v1 pre-gate decides ANSWER / CLARIFY / DEFER purely from the current turn.
v2 adds a *learned* nudge: when the device user's own history shows they rarely
needed the clarifications we asked (calibration buckets record, per confidence
bucket, how often an ask turned out unnecessary), a borderline DEFER — which
otherwise falls through to the LLM clarification gate — is upgraded to
answer-first. This cuts latency and over-asking for users who consistently don't
want to be interrupted, WITHOUT touching required-slot CLARIFY decisions (a
missing language/format is a real gap and is still asked).

Pure + fail-open; gated by `learning.answer_first_v2` (default off).
"""
from __future__ import annotations

# Decision constant this module reasons about (mirrors intent_pipeline.DEFER).
_DEFER = "defer"


def aggregate_answerability(buckets: dict | None) -> tuple[float, int]:
    """Collapse the per-bucket calibration into (answerable_rate, sample_count).

    Each bucket is ``{"answerable": n, "needed": n}`` — an ask that the user
    skipped/overrode is "answerable" (we could have just answered), an ask they
    answered was "needed". Returns (rate in [0,1], total observations)."""
    ans = need = 0
    for b in (buckets or {}).values():
        if isinstance(b, dict):
            ans += int(b.get("answerable", 0) or 0)
            need += int(b.get("needed", 0) or 0)
    total = ans + need
    return (ans / total if total else 0.0, total)


def should_upgrade_to_answer(
    decision: str,
    confidence: float,
    buckets: dict | None,
    *,
    enabled: bool,
    min_samples: int = 8,
    min_answerable: float = 0.75,
    min_confidence: float = 0.5,
) -> bool:
    """True when a borderline DEFER should become answer-first, based on the
    user's learned answerability. Never raises."""
    try:
        if not enabled or (decision or "").lower() != _DEFER:
            return False
        rate, n = aggregate_answerability(buckets)
        return (n >= min_samples
                and rate >= min_answerable
                and float(confidence) >= min_confidence)
    except Exception:  # noqa: BLE001 — fail-open to "don't upgrade"
        return False


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.learning, "answer_first_v2", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = ["aggregate_answerability", "should_upgrade_to_answer", "enabled"]
