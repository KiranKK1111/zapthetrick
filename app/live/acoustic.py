"""
Acoustic adaptation: accent / noise / room (live-conversational-intelligence R53).

Estimates the acoustic difficulty of the incoming audio (background noise,
reverberation/room, non-native accent proxy) from cheap signal stats the capture
path can supply (SNR, transcript stability across partials, STT confidence). A
degraded acoustic condition LOWERS the answer/decision confidence (so the system
hedges rather than confidently mis-hears) and can request a re-confirmation.
Deterministic + fail-open → neutral when no signal.
"""
from __future__ import annotations

from dataclasses import dataclass

CLEAN = "clean"
NOISY = "noisy"
REVERBERANT = "reverberant"
DEGRADED = "degraded"
UNKNOWN = "unknown"


@dataclass
class AcousticProfile:
    condition: str = UNKNOWN
    quality: float = 1.0          # [0,1] — 1 is pristine
    confidence_penalty: float = 0.0

    def to_dict(self) -> dict:
        return {"condition": self.condition, "quality": round(self.quality, 3),
                "confidence_penalty": round(self.confidence_penalty, 3)}


def assess(
    *,
    snr_db: float | None = None,           # signal-to-noise ratio in dB
    stt_conf: float | None = None,         # [0,1]
    partial_stability: float | None = None, # [0,1] — how stable partials were
) -> AcousticProfile:
    """Assess acoustic condition from cheap stats. Never raises → UNKNOWN.

    Lower SNR / STT confidence / partial stability → lower quality → a positive
    confidence penalty the caller subtracts from the answer confidence."""
    p = AcousticProfile()
    try:
        signals = []
        if snr_db is not None:
            # Map ~ [0dB..30dB] → [0..1].
            signals.append(max(0.0, min(1.0, snr_db / 30.0)))
        if stt_conf is not None:
            signals.append(max(0.0, min(1.0, stt_conf)))
        if partial_stability is not None:
            signals.append(max(0.0, min(1.0, partial_stability)))
        if not signals:
            return p
        quality = sum(signals) / len(signals)
        p.quality = quality
        p.confidence_penalty = round(max(0.0, (1.0 - quality)) * 0.3, 3)
        if quality >= 0.8:
            p.condition = CLEAN
        elif snr_db is not None and snr_db < 10:
            p.condition = NOISY
        elif partial_stability is not None and partial_stability < 0.5:
            p.condition = REVERBERANT
        else:
            p.condition = DEGRADED
        return p
    except Exception:  # noqa: BLE001
        return p


def adjust_confidence(confidence: float, profile: AcousticProfile) -> float:
    """Lower a decision/answer confidence by the acoustic penalty. Never raises."""
    try:
        return max(0.0, min(1.0, confidence - profile.confidence_penalty))
    except Exception:  # noqa: BLE001
        return confidence


def needs_reconfirmation(profile: AcousticProfile, threshold: float = 0.45) -> bool:
    """Whether the audio is degraded enough to warrant re-confirming the question."""
    try:
        return profile.condition in (NOISY, REVERBERANT, DEGRADED) and profile.quality < threshold
    except Exception:  # noqa: BLE001
        return False
