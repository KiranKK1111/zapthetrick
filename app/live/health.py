"""
Live session-health monitoring (live-conversational-intelligence R37).

Computes real-time operational warnings for a session — low STT confidence,
dropped audio, speaker confusion, and high latency — surfaced as an additive,
non-blocking `health` warning frame. Reuses the perceived-speed
Latency_Observatory + uncertainty signals; a degraded condition NEVER tears down
the WebSocket. Deterministic + fail-open.
"""
from __future__ import annotations

_LOW_STT = 0.5
_LOW_SPEAKER = 0.5
_SLOW_MS = 4000


def session_health(*, stt_conf: float | None = None, dropped_audio: bool = False,
                   speaker_conf: float | None = None,
                   latency_ms: float | None = None) -> dict | None:
    """Return an additive `health` warning frame when degraded, else None.
    Never raises."""
    try:
        warnings: list[str] = []
        if stt_conf is not None and stt_conf < _LOW_STT:
            warnings.append("low_stt_confidence")
        if dropped_audio:
            warnings.append("dropped_audio")
        if speaker_conf is not None and speaker_conf < _LOW_SPEAKER:
            warnings.append("speaker_confusion")
        if latency_ms is not None and latency_ms > _SLOW_MS:
            warnings.append("high_latency")
        if not warnings:
            return None
        return {"type": "health", "warnings": warnings, "ok": False}
    except Exception:  # noqa: BLE001
        return None


def latency_ms_estimate() -> float | None:
    """Best-effort recent live latency from the perceived-speed observatory
    (None when unavailable). Never raises."""
    try:
        from app.perceived import observatory as _obs  # type: ignore
        getter = getattr(_obs, "recent_latency_ms", None)
        if callable(getter):
            return float(getter())
    except Exception:  # noqa: BLE001
        pass
    return None


# ── Feedback_Signal capture (R56) ──────────────────────────────────────
# Capture a lightweight feedback signal (the candidate/interviewer reaction to
# an answer) into the event log so the existing feedback/learning loop can use
# it. Deterministic + fail-open. No new schema — appends to the per-session
# Event_Log.

_POSITIVE = ("great", "perfect", "exactly", "good", "makes sense", "nice",
             "that's right", "correct", "yes", "love it", "awesome")
_NEGATIVE = ("no", "not quite", "wrong", "incorrect", "that's not", "actually no",
             "i disagree", "doesn't make sense", "confusing")


def classify_feedback(reaction: str) -> str:
    """Classify a reaction into positive / negative / neutral. Never raises."""
    try:
        t = (reaction or "").strip().lower()
        if not t:
            return "neutral"
        if any(c in t for c in _NEGATIVE):
            return "negative"
        if any(c in t for c in _POSITIVE):
            return "positive"
        return "neutral"
    except Exception:  # noqa: BLE001
        return "neutral"


def capture_feedback(session_id: str, reaction: str, *, qid: str = "") -> str:
    """Capture a Feedback_Signal into the event log; returns the classified
    state. Fail-open — never breaks the live turn."""
    state = classify_feedback(reaction)
    try:
        from app.live.eventlog import get_log
        get_log(session_id).append("feedback_signal",
                                   {"state": state, "qid": qid, "text": (reaction or "")[:80]})
    except Exception:  # noqa: BLE001
        pass
    return state
