"""Interviewer-silence watchdog (vNext §4.7, Stage 6 Component C).

An interview can silently break: the interviewer's audio channel drops (a muted
share, a lost loopback) and the candidate stares at a dead app. This watchdog
times the gap since the last interviewer-audio chunk WHILE the session is active
and, once it exceeds `silence_watchdog_s` (default 30 s), surfaces a one-shot
"interviewer_silent" signal the UI shows as a chip ("No interviewer audio for
30 s — check the capture source"). The pre-flight board (Component A) seeds it at
session start via `silence.seed_baseline`.

Per-session, injectable clock (monotonic), single-shot per silence EPISODE (fires
once, re-arms when audio returns) so it never spams. Advisory + fail-open — the
watchdog never gates the pipeline, it only raises a chip. Flag-gated
(`live.audio_watchdog`, default OFF).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

SILENCE = "interviewer_silent"

_LOCK = threading.RLock()


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "audio_watchdog", False))
    except Exception:  # noqa: BLE001
        return False


def _threshold_s() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.live, "silence_watchdog_s", 30.0))
    except Exception:  # noqa: BLE001
        return 30.0


@dataclass
class _Sess:
    last_audio: float          # monotonic ts of the last interviewer-audio chunk
    seen: bool = False         # interviewer audio ever observed
    active: bool = True        # session live (paused/ended → no watch)
    fired: bool = False        # chip fired for the CURRENT silence episode


class AudioWatchdog:
    """Per-session interviewer-silence timer. Deterministic + fail-open."""

    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._s: dict[str, _Sess] = {}
        self._now = now or time.monotonic

    def seed(self, session_id: str, *, interviewer_audio: bool | None = None,
             active: bool = True) -> None:
        """Establish the session baseline at pre-flight (Component A). A True
        `interviewer_audio` marks the channel as already seen."""
        now = self._now()
        with _LOCK:
            self._s[session_id] = _Sess(
                last_audio=now, seen=bool(interviewer_audio), active=active)

    def mark_audio(self, session_id: str) -> None:
        """Record an interviewer-audio chunk — resets the silence timer and
        re-arms the one-shot so a NEW silence episode can fire again."""
        now = self._now()
        with _LOCK:
            s = self._s.get(session_id)
            if s is None:
                s = _Sess(last_audio=now, seen=True)
                self._s[session_id] = s
            else:
                s.last_audio = now
                s.seen = True
                s.fired = False

    def set_active(self, session_id: str, active: bool) -> None:
        with _LOCK:
            s = self._s.get(session_id)
            if s is not None:
                s.active = active
                if active:
                    s.last_audio = self._now()   # don't count paused time as silence
                    s.fired = False

    def check(self, session_id: str, *, silence_s: float | None = None) -> str | None:
        """Return `SILENCE` once when the interviewer has been quiet longer than
        the threshold on an ACTIVE session that has seen audio; else None.
        One-shot per episode. Never raises."""
        try:
            thr = _threshold_s() if silence_s is None else silence_s
            with _LOCK:
                s = self._s.get(session_id)
                if s is None or not s.active or not s.seen or s.fired:
                    return None
                if self._now() - s.last_audio >= thr:
                    s.fired = True
                    return SILENCE
            return None
        except Exception:  # noqa: BLE001
            return None

    def forget(self, session_id: str) -> None:
        with _LOCK:
            self._s.pop(session_id, None)

    def clear(self) -> None:
        with _LOCK:
            self._s.clear()


_watchdog = AudioWatchdog()


def watchdog() -> AudioWatchdog:
    return _watchdog


# Module-level convenience wrappers (the WS layer calls these). All flag-gated:
# with `audio_watchdog` off they are cheap no-ops, so the Live path is unchanged.
def seed_baseline(session_id: str, *, interviewer_audio: bool | None = None) -> None:
    if not enabled():
        return
    _watchdog.seed(session_id, interviewer_audio=interviewer_audio)


def mark_interviewer_audio(session_id: str) -> None:
    if not enabled():
        return
    _watchdog.mark_audio(session_id)


def check_silence(session_id: str) -> str | None:
    if not enabled():
        return None
    return _watchdog.check(session_id)


def forget_session(session_id: str) -> None:
    _watchdog.forget(session_id)


def reset_for_tests() -> None:
    _watchdog.clear()


__all__ = ["SILENCE", "AudioWatchdog", "watchdog", "enabled", "seed_baseline",
           "mark_interviewer_audio", "check_silence", "forget_session",
           "reset_for_tests"]
