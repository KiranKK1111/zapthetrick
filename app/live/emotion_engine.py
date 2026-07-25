"""Emotion engine — first-class pipeline stage (vNext §10.6.4, Stage 10 B).

`emotion.py` gives an advisory prosody label; §10.6.4 promotes emotion to a
first-class stage with a real speech-emotion-recognition head and a careful
response policy. The pipeline:

  prosodic features → **SER head** (injected, ~80 MB resident) → a distribution
  over {calm, nervous, frustrated, confused, excited, confident, sad, urgent} +
  arousal/valence + confidence → **FUSION** with transcript sentiment (WHAT) and
  the §4.15 situation class (prosody=HOW; low agreement WIDENS the band) →
  **hysteretic rolling session state** (one shaky sentence ≠ a "nervous session";
  a flip needs persistence) → **response policy**.

Response policy is the delicate part:
  * a **register** shift (tone / pace / length) ALWAYS applies when confident;
  * an **open acknowledgement** ("take your time", "let's break it down") fires
    ONLY when the read is strong AND prosody + content agree — else the register
    shifts SILENTLY (a wrong guess claims nothing);
  * it's **tentative, never clinical** (no "you sound anxious" diagnosis);
  * it's **instantly correctable** — "I'm fine" drops the read for the session;
  * it's context-weighted by §4.15 situation × surface (nervousness top-weighted
    in Live/coach).

Raw audio / voiceprints are NEVER stored — only the derived label. Fail-open:
emotion off / low-confidence → today's neutral register. Flag-gated
(`voice.emotion`, default OFF).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# The §10.6.4 emotion taxonomy.
CALM = "calm"
NERVOUS = "nervous"
FRUSTRATED = "frustrated"
CONFUSED = "confused"
EXCITED = "excited"
CONFIDENT = "confident"
SAD = "sad"
URGENT = "urgent"
NEUTRAL = "neutral"
_LABELS = (CALM, NERVOUS, FRUSTRATED, CONFUSED, EXCITED, CONFIDENT, SAD, URGENT)

# Coarse valence sign per label (for prosody↔content agreement).
_VALENCE = {CALM: +1, CONFIDENT: +1, EXCITED: +1, NEUTRAL: 0,
            CONFUSED: -1, NERVOUS: -1, FRUSTRATED: -1, SAD: -1, URGENT: 0}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.voice, "emotion", False))
    except Exception:  # noqa: BLE001
        return False


def mode() -> str:
    try:
        from app.core.config_loader import cfg
        m = str(getattr(cfg.voice, "emotion_mode", "local") or "local").lower()
        return m if m in ("local", "cloud") else "local"
    except Exception:  # noqa: BLE001
        return "local"


# --------------------------------------------------------------------------- #
# SER read (injected head)
# --------------------------------------------------------------------------- #
@dataclass
class EmotionRead:
    top: str = NEUTRAL
    distribution: dict = field(default_factory=dict)
    arousal: float = 0.0          # 0..1 (calm→activated)
    valence: float = 0.0          # -1..1 (negative→positive)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"top": self.top, "arousal": round(self.arousal, 3),
                "valence": round(self.valence, 3),
                "confidence": round(self.confidence, 3)}


def ser_read(features, *, ser_fn=None) -> EmotionRead:
    """Run the SER head over prosodic `features` → a typed emotion read. `ser_fn`
    is INJECTED (the wav2vec2/ECAPA-class head runs on GPU; a stub in tests):
    `ser_fn(features) -> {label: prob}`. Never raises → a NEUTRAL read (fail-open
    to today's neutral register). Raw audio is not retained — only this read."""
    try:
        if ser_fn is None:
            return EmotionRead()          # no head available → neutral
        dist = ser_fn(features) or {}
        dist = {k: float(v) for k, v in dist.items()
                if k in _LABELS and float(v) >= 0}
        if not dist:
            return EmotionRead()
        total = sum(dist.values()) or 1.0
        dist = {k: v / total for k, v in dist.items()}
        top = max(dist, key=dist.get)
        conf = dist[top]
        arousal = min(1.0, dist.get(EXCITED, 0) + dist.get(URGENT, 0)
                      + dist.get(NERVOUS, 0) + dist.get(FRUSTRATED, 0))
        valence = sum(_VALENCE.get(k, 0) * v for k, v in dist.items())
        return EmotionRead(top=top, distribution=dist, arousal=arousal,
                           valence=max(-1.0, min(1.0, valence)), confidence=conf)
    except Exception:  # noqa: BLE001
        return EmotionRead()


# --------------------------------------------------------------------------- #
# Fusion (prosody × transcript sentiment × situation)
# --------------------------------------------------------------------------- #
@dataclass
class FusedEmotion:
    label: str = NEUTRAL
    confidence: float = 0.0
    agree: bool = True            # prosody ↔ content agreement
    arousal: float = 0.0

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3),
                "agree": self.agree, "arousal": round(self.arousal, 3)}


def fuse(prosody: EmotionRead, *, transcript_sentiment: float | None = None,
         situation: str | None = None) -> FusedEmotion:
    """Fuse the prosody read (HOW) with transcript sentiment (WHAT, a valence in
    [-1,1]) and the §4.15 situation. Agreement between prosody valence and
    transcript sentiment BOOSTS confidence; disagreement WIDENS the band (lowers
    confidence — prosody says one thing, words another). Never raises."""
    try:
        label = prosody.top
        conf = prosody.confidence
        agree = True
        if transcript_sentiment is not None and label != NEUTRAL:
            # Same sign (both negative / both positive) → agree.
            pv = prosody.valence
            ts = float(transcript_sentiment)
            if pv * ts < 0 and abs(pv) >= 0.15 and abs(ts) >= 0.25:
                agree = False
                conf *= 0.5                      # widen the band
            elif pv * ts > 0:
                conf = min(1.0, conf * 1.15)     # reinforce
        # Situation prior: a stress/conviction situation nudges toward nervous.
        if situation in ("stress", "harshness", "conviction_trap") and \
                label in (NERVOUS, NEUTRAL):
            conf = min(1.0, conf + 0.1)
        return FusedEmotion(label=label, confidence=conf, agree=agree,
                            arousal=prosody.arousal)
    except Exception:  # noqa: BLE001
        return FusedEmotion()


# --------------------------------------------------------------------------- #
# Hysteretic rolling session state
# --------------------------------------------------------------------------- #
@dataclass
class SessionEmotionState:
    """One shaky sentence is not a nervous session. The session label only flips
    after `persistence` consecutive agreeing reads above `min_conf`; a strong
    dismissal ('I'm fine') drops the read for the rest of the session."""
    persistence: int = 3
    min_conf: float = 0.55
    label: str = NEUTRAL
    confidence: float = 0.0
    overridden: bool = False
    _recent: deque = field(default_factory=lambda: deque(maxlen=5))

    def update(self, fused: FusedEmotion) -> str:
        """Fold a fused read into the session state → the current session label.
        Never raises."""
        try:
            if self.overridden:
                return NEUTRAL
            self._recent.append(fused.label if fused.confidence >= self.min_conf
                                else NEUTRAL)
            # Flip only when the last `persistence` reads agree on a non-neutral.
            if len(self._recent) >= self.persistence:
                tail = list(self._recent)[-self.persistence:]
                if tail[0] != NEUTRAL and all(x == tail[0] for x in tail):
                    self.label = tail[0]
                    self.confidence = fused.confidence
                    return self.label
            # Not yet persistent → keep the prior session label (hysteresis).
            return self.label
        except Exception:  # noqa: BLE001
            return self.label

    def override(self) -> None:
        """Candidate said they're fine → drop the emotion read for the session."""
        self.overridden = True
        self.label = NEUTRAL
        self.confidence = 0.0

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3),
                "overridden": self.overridden}


# Dismissal detection ("I'm fine") — semantic-first with a cue fallback.
_DISMISSAL_CUES = (
    "i'm fine", "im fine", "i am fine", "i'm good", "im good", "i'm okay",
    "im okay", "i'm ok", "no i'm fine", "don't worry", "dont worry",
    "i'm alright", "im alright", "all good",
)


def is_dismissal(text: str, *, classify_fn=None) -> bool:
    """Whether the candidate dismissed a felt-emotion read ('I'm fine'). Semantic
    first (injectable `classify_fn` → bool), cue fallback. Never raises → False."""
    try:
        t = (text or "").strip().lower()
        if not t:
            return False
        if classify_fn is not None:
            try:
                return bool(classify_fn(t))
            except Exception:  # noqa: BLE001
                pass
        return any(c in t for c in _DISMISSAL_CUES)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Response policy
# --------------------------------------------------------------------------- #
# Register shifts (tone / pace / length) per session label — ALWAYS applied.
_REGISTER = {
    NERVOUS: "Warmer and slower; shorter sentences; steady, reassuring pacing.",
    FRUSTRATED: "Calm and direct; acknowledge the difficulty implicitly; no filler.",
    CONFUSED: "Slower; one idea at a time; concrete examples; check understanding.",
    SAD: "Gentle and unhurried; warm; low pressure.",
    URGENT: "Tighter and faster; lead with the answer; trim elaboration.",
    EXCITED: "Match the energy; keep it crisp and forward-moving.",
    CONFIDENT: "Concise and peer-level; skip the basics.",
    CALM: "Normal, even pacing.",
}
# Open acknowledgements — tentative, never clinical. Fire only when gated.
_ACKNOWLEDGE = {
    NERVOUS: "Take your time — there's no rush.",
    CONFUSED: "Let's break this down step by step.",
    FRUSTRATED: "Let's tackle this piece by piece.",
    SAD: "No pressure — we can take this slowly.",
}
# Surfaces where an emotion acknowledgement is welcome (vs. distracting).
_ACK_SURFACES = ("live", "coach", "interview")


@dataclass
class EmotionPolicy:
    register: str = ""            # ALWAYS applied when confident
    acknowledge: bool = False     # whether to open-acknowledge
    acknowledgement: str = ""     # the tentative line (never clinical)
    reason: str = ""

    def to_dict(self) -> dict:
        return {"register": self.register, "acknowledge": self.acknowledge,
                "acknowledgement": self.acknowledgement, "reason": self.reason}


def response_policy(state: SessionEmotionState, *, situation: str | None = None,
                    surface: str = "chat", ack_min_conf: float = 0.72,
                    agree: bool = True) -> EmotionPolicy:
    """The response policy: a register shift ALWAYS (when confident); an open
    acknowledgement ONLY when strong AND prosody/content agree AND the surface
    welcomes it. Disabled / overridden / low-confidence → neutral (silent). Never
    raises → a neutral policy."""
    try:
        if not enabled() or state.overridden or state.label == NEUTRAL:
            return EmotionPolicy(reason="disabled/neutral/overridden")
        register = _REGISTER.get(state.label, "")
        # Acknowledge only when: strong confidence, prosody/content agree, and the
        # surface welcomes it (nervousness top-weighted in live/coach).
        strong = state.confidence >= ack_min_conf
        surface_ok = (surface or "chat").lower() in _ACK_SURFACES
        ack_line = _ACKNOWLEDGE.get(state.label, "")
        acknowledge = bool(strong and agree and surface_ok and ack_line)
        if acknowledge:
            return EmotionPolicy(register=register, acknowledge=True,
                                 acknowledgement=ack_line,
                                 reason="strong + agreeing + welcome surface")
        # Otherwise: shift register SILENTLY (a wrong guess claims nothing).
        return EmotionPolicy(register=register, acknowledge=False,
                             reason="silent register shift")
    except Exception:  # noqa: BLE001
        return EmotionPolicy()


__all__ = ["CALM", "NERVOUS", "FRUSTRATED", "CONFUSED", "EXCITED", "CONFIDENT",
           "SAD", "URGENT", "NEUTRAL", "enabled", "mode", "EmotionRead",
           "ser_read", "FusedEmotion", "fuse", "SessionEmotionState",
           "is_dismissal", "EmotionPolicy", "response_policy"]
