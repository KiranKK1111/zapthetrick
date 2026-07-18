"""
Consent gate + disclaimer (live-conversational-intelligence R17).

Surfaces an explicit acknowledgement + an AI-suggestion disclaimer before a live
session captures audio, and exposes a candidate-audio-only capture mode. All
additive and config-gated: with `cfg.live.consent` off the Live module behaves
exactly as today (no gate). The recommended production posture is consent on.
Fail-open.
"""
from __future__ import annotations

from app.core.config_loader import cfg

_DISCLAIMER = (
    "These are AI-generated suggestions and may be wrong — verify before relying "
    "on them. You are responsible for any applicable consent/recording rules for "
    "the other party in this session."
)


def requires_consent() -> bool:
    """True when the consent gate is enabled."""
    return bool(getattr(cfg.live, "consent", False))


def candidate_audio_only() -> bool:
    """True when only the candidate's own audio should be captured (no other
    party)."""
    return bool(getattr(cfg.live, "candidate_audio_only", False))


def disclaimer() -> str:
    return _DISCLAIMER


def consent_frame() -> dict | None:
    """The additive `consent` WebSocket frame to send before capture, or None
    when the gate is disabled."""
    if not requires_consent():
        return None
    return {
        "type": "consent",
        "required": True,
        "disclaimer": _DISCLAIMER,
        "candidate_audio_only": candidate_audio_only(),
    }
