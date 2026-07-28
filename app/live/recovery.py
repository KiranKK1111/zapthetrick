"""
B6 — conversation recovery on STT dropout / network loss.

Real interviews drop audio: the mic cuts, the network stalls, STT returns a
run of empties. When that happens the assistant is flying blind — it must NOT
confidently answer half-heard questions, and it should tell the user a gap
happened (the transcript/summary is preserved, so context can be rebuilt).

`SttGapTracker` counts consecutive STT failures/empties and declares a GAP once
they cross a threshold; the first real transcript clears it. Deterministic,
in-process, fail-open. The live handler emits a one-time `context_gap` frame on
entry so the FE can show a calm "audio unclear — context preserved" notice.
"""
from __future__ import annotations

from app.core.config_loader import cfg

# STT status kinds that indicate NO usable transcript came back.
_FAILURE_KINDS = frozenset({"error", "empty"})


def _threshold() -> int:
    return int(getattr(cfg.live, "stt_gap_threshold", 3) or 3)


def enabled() -> bool:
    return bool(getattr(cfg.live, "stt_gap_recovery", True))


class SttGapTracker:
    """Per-session run-length counter over STT status events. `note_status`
    returns True on the transition INTO a gap (so the caller emits one frame);
    `note_transcript` clears the run when real text arrives."""

    def __init__(self) -> None:
        self._consecutive = 0
        self._in_gap = False

    def note_status(self, kind: str) -> bool:
        """Feed an stt_status kind. Returns True exactly once — on the event that
        crosses the threshold into a gap (so the handler emits `context_gap`)."""
        if not enabled():
            return False
        if (kind or "").strip().lower() not in _FAILURE_KINDS:
            return False
        self._consecutive += 1
        if not self._in_gap and self._consecutive >= _threshold():
            self._in_gap = True
            return True
        return False

    def note_transcript(self, text: str = "x") -> bool:
        """Feed a successful transcript. Returns True on RECOVERY (a gap just
        closed) so the handler can clear the notice."""
        if not (text or "").strip():
            return False
        recovered = self._in_gap
        self._consecutive = 0
        self._in_gap = False
        return recovered

    @property
    def in_gap(self) -> bool:
        return self._in_gap


__all__ = ["SttGapTracker", "enabled"]
