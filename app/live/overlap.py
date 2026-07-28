"""
B4 — overlap / crosstalk detection.

The dual-source live path runs TWO segmenters — the interviewer (system
loopback / role 0) and the candidate (mic / role 1). Normal turn-taking means
only one is voiced at a time. When BOTH are voiced together — the interviewer
talks over the candidate, a panelist interjects, or a barge-in — the audio of
the answered (interviewer) channel is partly masked, so its transcript is less
trustworthy. `OverlapMonitor` tracks each channel's speaking state and reports
whether overlap occurred during the current interviewer utterance, so the
handler can flag it (lower confidence / note it) instead of answering a
half-masked question as if it were clean.

Deterministic, in-process, fail-open. A heuristic (concurrent voice-activity),
not full speaker separation — but it catches the crosstalk case the single-
segmenter turn-taking model silently mishandles.
"""
from __future__ import annotations

from app.core.config_loader import cfg

INTERVIEWER = "interviewer"
CANDIDATE = "candidate"


def enabled() -> bool:
    return bool(getattr(cfg.live, "overlap_detection", True))


class OverlapMonitor:
    """Tracks per-role speaking state and latches whether overlap happened since
    the last reset (i.e. during the utterance now finalizing)."""

    def __init__(self) -> None:
        self._speaking = {INTERVIEWER: False, CANDIDATE: False}
        self._saw_overlap = False

    def update(self, role: str, speaking: bool) -> bool:
        """Record a channel's current speaking state. Returns True on the edge
        INTO overlap (both channels voiced). Fail-open → False when disabled."""
        if not enabled():
            return False
        r = (role or "").strip().lower()
        if r not in self._speaking:
            return False
        was_overlap = self._both()
        self._speaking[r] = bool(speaking)
        now_overlap = self._both()
        if now_overlap:
            self._saw_overlap = True
        return now_overlap and not was_overlap

    def _both(self) -> bool:
        return self._speaking[INTERVIEWER] and self._speaking[CANDIDATE]

    @property
    def overlapping(self) -> bool:
        return self._both()

    def saw_overlap(self) -> bool:
        return self._saw_overlap

    def reset(self) -> None:
        """Clear the latch at the start of a fresh interviewer utterance."""
        self._saw_overlap = False


__all__ = ["OverlapMonitor", "enabled", "INTERVIEWER", "CANDIDATE"]
