"""Conversational intelligence — turn-taking + barge-in (vNext §10.6.2, Stage 10 C).

Two decisions make a voice loop feel human instead of walkie-talkie:

  * **Adaptive endpointing** — how long to wait after the interviewer stops
    talking before answering. A finished sentence with falling pitch → answer
    fast (~300 ms); a trailing-off or rising-pitch clause ("what is… Kafka?") →
    wait longer (up to 1.5 s) so we don't cut them off. The window is a function
    of VAD silence × a completeness gate × prosody, clamped to 300 ms–1.5 s.
  * **Barge-in intent** — when the candidate speaks WHILE the copilot's TTS is
    playing, classify the interruption in ~150 ms: a **backchannel** ("mhm",
    "right") must NOT stop TTS; a **continuation**, a **correction** ("no wait,
    actually…"), or a **stop** ("hold on") each duck/stop the audio.

Both are pure decision logic (VAD/prosody are injected numbers; the barge-in
classifier is semantic-first with a fast cue fallback). Fail-open. Flag-gated
(`voice.conversational_intel`, default OFF → today's fixed endpoint, no barge-in).
"""
from __future__ import annotations

from dataclasses import dataclass

# Barge-in intents.
BACKCHANNEL = "backchannel"   # "mhm", "yeah" — do NOT stop TTS
CONTINUATION = "continuation"  # the candidate adds to their turn
CORRECTION = "correction"     # "no wait", "actually" — redo
STOP = "stop"                 # "hold on", "stop" — halt
_INTENTS = (BACKCHANNEL, CONTINUATION, CORRECTION, STOP)

_MIN_MS = 300
_MAX_MS = 1500


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.voice, "conversational_intel", False))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Adaptive endpointing
# --------------------------------------------------------------------------- #
@dataclass
class EndpointDecision:
    endpoint: bool                # True → the turn is over, answer now
    window_ms: int                # the silence window required (300..1500)
    complete: bool                # the completeness gate
    reason: str = ""

    def to_dict(self) -> dict:
        return {"endpoint": self.endpoint, "window_ms": self.window_ms,
                "complete": self.complete, "reason": self.reason}


def endpoint_decision(*, silence_ms: float, completeness: float = 0.5,
                      pitch_contour: str = "flat", min_ms: int = _MIN_MS,
                      max_ms: int = _MAX_MS) -> EndpointDecision:
    """Decide whether the interviewer's turn has ended, and the silence WINDOW to
    require. `completeness` (0..1, from the §4.8 completeness gate) and
    `pitch_contour` ('falling'|'rising'|'flat', from prosody) set the window:
      * high completeness + falling pitch → short window (answer fast);
      * low completeness OR rising pitch (a question still forming) → long window.
    `endpoint` = silence has met the window. Disabled → a fixed mid window. Never
    raises."""
    try:
        if not enabled():
            w = int((min_ms + max_ms) / 2)         # fixed window, today's behaviour
            return EndpointDecision(silence_ms >= w, w, completeness >= 0.5,
                                    "disabled: fixed window")
        # Base: more complete → shorter window.
        span = max_ms - min_ms
        window = max_ms - span * max(0.0, min(1.0, completeness))
        # Prosody: falling pitch = done (shorten); rising = still going (lengthen).
        c = (pitch_contour or "flat").lower()
        if c == "falling":
            window -= span * 0.25
        elif c == "rising":
            window += span * 0.25
        window = int(max(min_ms, min(max_ms, window)))
        complete = completeness >= 0.5 and c != "rising"
        return EndpointDecision(silence_ms >= window, window, complete,
                                f"completeness={completeness:.2f} pitch={c}")
    except Exception:  # noqa: BLE001
        w = int((min_ms + max_ms) / 2)
        return EndpointDecision(silence_ms >= w, w, False, "error → mid window")


# --------------------------------------------------------------------------- #
# Barge-in intent classifier
# --------------------------------------------------------------------------- #
# Fast cue fallback (the ~150 ms path — semantic is optional/async).
_BACKCHANNEL_CUES = ("mhm", "mm-hmm", "uh-huh", "yeah", "yep", "right", "okay",
                     "ok", "sure", "got it", "i see", "makes sense", "cool", "nice")
_CORRECTION_CUES = ("no wait", "actually", "no no", "sorry", "i mean", "let me",
                    "hold on wait", "that's not", "not quite", "rather")
_STOP_CUES = ("stop", "hold on", "wait", "pause", "hang on", "one sec",
              "never mind", "forget it")


@dataclass
class BargeInIntent:
    intent: str
    should_stop_tts: bool
    source: str = "cue"           # "semantic" | "cue"

    def to_dict(self) -> dict:
        return {"intent": self.intent, "should_stop_tts": self.should_stop_tts,
                "source": self.source}


def _cue_classify(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return CONTINUATION
    # Correction / stop cues take priority over a backchannel word inside them.
    if any(c in t for c in _CORRECTION_CUES):
        return CORRECTION
    if any(c in t for c in _STOP_CUES):
        return STOP
    # A SHORT utterance made only of backchannel tokens → backchannel.
    words = t.split()
    if len(words) <= 3 and any(t == c or t.startswith(c + " ") or t == c
                               for c in _BACKCHANNEL_CUES):
        return BACKCHANNEL
    if len(words) <= 2 and all(
            any(w.strip(".,!?") == c for c in _BACKCHANNEL_CUES) for w in words):
        return BACKCHANNEL
    return CONTINUATION


def classify_bargein(text: str, *, classify_fn=None) -> BargeInIntent:
    """Classify a barge-in utterance → intent + whether it should stop the TTS.
    A BACKCHANNEL never stops TTS (an encouraging "mhm" isn't an interruption);
    continuation/correction/stop all halt it. Semantic-first (injectable
    `classify_fn` → one of the four intents), fast cue fallback. Never raises."""
    try:
        intent = None
        source = "cue"
        if classify_fn is not None:
            try:
                c = classify_fn(text or "")
                if c in _INTENTS:
                    intent, source = c, "semantic"
            except Exception:  # noqa: BLE001
                intent = None
        if intent is None:
            intent = _cue_classify(text)
        return BargeInIntent(intent=intent,
                             should_stop_tts=(intent != BACKCHANNEL),
                             source=source)
    except Exception:  # noqa: BLE001 — fail SAFE: treat as a stop (don't talk over)
        return BargeInIntent(STOP, True, "error")


__all__ = ["BACKCHANNEL", "CONTINUATION", "CORRECTION", "STOP", "enabled",
           "EndpointDecision", "endpoint_decision", "BargeInIntent",
           "classify_bargein"]
