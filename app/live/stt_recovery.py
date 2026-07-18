"""
Live confidence recovery — auto ASR switch on degradation
(roadmap Phase 2 #30 / 2D-30).

`app/stt/switch.py` can switch the live ASR engine, but nothing triggered it
automatically when transcription quality decayed mid-interview — the user had to
notice and change engines in Settings. This tracks a per-session rolling STT
confidence and, when it stays degraded across several utterances, PLANS a
one-time switch to the configured fallback engine. This module owns only the
DECISION (which engine + the single-shot latch) — the caller (the WS layer,
which already couples to `app.stt`) performs the actual switch via
`app.stt.switch.start_switch`. Advisory + fail-open, single-shot per session so
we never thrash between engines.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

_LOCK = threading.RLock()


@dataclass
class _Tracker:
    window: deque = field(default_factory=lambda: deque(maxlen=5))
    recovered: bool = False   # single-shot latch

    def observe(self, conf: float | None) -> None:
        if conf is None:
            return
        try:
            self.window.append(max(0.0, min(1.0, float(conf))))
        except Exception:  # noqa: BLE001
            pass

    def degraded(self, *, threshold: float, min_samples: int) -> bool:
        if len(self.window) < max(1, min_samples):
            return False
        avg = sum(self.window) / len(self.window)
        return avg < threshold


_TRACKERS: dict[str, _Tracker] = {}


def _tracker(session_id: str) -> _Tracker:
    with _LOCK:
        t = _TRACKERS.get(session_id)
        if t is None:
            t = _Tracker()
            _TRACKERS[session_id] = t
        return t


def observe(session_id: str, stt_conf: float | None) -> None:
    """Record an STT confidence sample for the session. Never raises."""
    try:
        _tracker(session_id or "").observe(stt_conf)
    except Exception:  # noqa: BLE001
        pass


def should_recover(session_id: str, *, threshold: float = 0.45,
                   min_samples: int = 3) -> bool:
    """True once the rolling confidence has stayed degraded — and only once per
    session (single-shot latch). Never raises → False."""
    try:
        t = _tracker(session_id or "")
        if t.recovered:
            return False
        return t.degraded(threshold=threshold, min_samples=min_samples)
    except Exception:  # noqa: BLE001
        return False


def _fallback_target() -> str | None:
    """Pick a fallback ASR engine id different from the active one. Best-effort;
    returns None when nothing better is knowable. Never raises."""
    try:
        from app.core.config_loader import cfg
        active = str(getattr(cfg.stt, "provider", "") or "").strip().lower()
        # Ordered preference of local engines; skip the active one.
        for cand in ("parakeet", "qwen_asr", "faster_whisper"):
            if cand != active:
                return cand
        return None
    except Exception:  # noqa: BLE001
        return None


def recover(session_id: str) -> str | None:
    """PLAN a one-time switch to the fallback ASR engine: latch the session so it
    fires at most once, and return the target engine id for the caller to switch
    to (or None when nothing better is knowable / already recovered). Never
    raises. The caller performs the actual `app.stt.switch.start_switch`."""
    try:
        t = _tracker(session_id or "")
        if t.recovered:
            return None
        t.recovered = True     # latch BEFORE returning → never double-fire
        return _fallback_target()
    except Exception:  # noqa: BLE001
        return None


def forget_session(session_id: str) -> None:
    with _LOCK:
        _TRACKERS.pop(session_id or "", None)


__all__ = ["observe", "should_recover", "recover", "forget_session"]
