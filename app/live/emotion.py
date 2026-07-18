"""
Advisory Emotion_Signal from prosody (live-conversational-intelligence R43).

Reuses the acoustic/prosody features the question-detection path already
computes (pitch / energy / rate proxies) to produce an ADVISORY emotion signal —
calm / stressed / rushed / hesitant — surfaced as additive `meta.emotion`. It is
purely advisory: it never alters a safety or answer decision on its own (only the
answer text may soften delivery). Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

CALM = "calm"
STRESSED = "stressed"
RUSHED = "rushed"
HESITANT = "hesitant"
NEUTRAL = "neutral"


@dataclass
class EmotionSignal:
    label: str = NEUTRAL
    confidence: float = 0.0
    advisory: bool = True   # always advisory — never decisive

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3),
                "advisory": True}


def analyze(
    *,
    energy: float | None = None,       # normalized loudness [0,1]
    pitch_var: float | None = None,    # pitch variability [0,1]
    speech_rate: float | None = None,  # words/sec or normalized [0,1]
    filler_ratio: float | None = None, # disfluency ratio [0,1]
) -> EmotionSignal:
    """Map prosody proxies to an advisory emotion label. Never raises → NEUTRAL.

    None inputs are simply ignored (fail-open to NEUTRAL when nothing is known)."""
    try:
        scores: dict[str, float] = {}
        if filler_ratio is not None and filler_ratio >= 0.25:
            scores[HESITANT] = filler_ratio
        if speech_rate is not None and speech_rate >= 0.8:
            scores[RUSHED] = speech_rate
        if energy is not None and pitch_var is not None and energy >= 0.7 and pitch_var >= 0.7:
            scores[STRESSED] = (energy + pitch_var) / 2.0
        if not scores:
            # Calm when we have signal and it's all low; else neutral.
            known = [v for v in (energy, pitch_var, speech_rate, filler_ratio) if v is not None]
            if known and max(known) <= 0.4:
                return EmotionSignal(label=CALM, confidence=1.0 - max(known))
            return EmotionSignal(label=NEUTRAL, confidence=0.0)
        label = max(scores, key=scores.get)
        return EmotionSignal(label=label, confidence=max(0.0, min(1.0, scores[label])))
    except Exception:  # noqa: BLE001
        return EmotionSignal(label=NEUTRAL, confidence=0.0)


def delivery_note(signal: EmotionSignal) -> str:
    """Advisory delivery note for the answer call (never decisive). '' when
    neutral/calm or low confidence."""
    try:
        if signal.advisory is not True:
            return ""
        if signal.label in (NEUTRAL, CALM) or signal.confidence < 0.5:
            return ""
        notes = {
            STRESSED: "Candidate may sound stressed — keep the answer calm, clear, and reassuring.",
            RUSHED: "Candidate may be rushing — encourage a measured, structured delivery.",
            HESITANT: "Candidate may be hesitant — keep the answer confident and concrete.",
        }
        return notes.get(signal.label, "")
    except Exception:  # noqa: BLE001
        return ""
