"""
Capture topology (live-conversational-intelligence R18).

Tracks which audio source(s) a live session captures — the candidate's mic, the
system-loopback (the other party), or both — and labels each stream's speaker so
downstream diarization (Phase 10) and the answer gate can tell the interviewer
from the candidate. Deterministic + fail-open. With the flag off the Live module
uses today's single-source (system-loopback) behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

CANDIDATE = "candidate"
LOOPBACK = "loopback"
BOTH = "both"

# Speaker roles a source maps to.
CANDIDATE_SPEAKER = "candidate"
INTERVIEWER_SPEAKER = "interviewer"

_MIC_SOURCES = {"mic", "microphone", "candidate"}
_LOOPBACK_SOURCES = {"system_loopback", "loopback", "system", "interviewer"}


@dataclass
class CaptureTopology:
    mode: str = LOOPBACK          # candidate | loopback | both

    @classmethod
    def from_config(cls) -> "CaptureTopology":
        from app.core.config_loader import cfg
        # candidate-audio-only (R17) forces the candidate source.
        if bool(getattr(cfg.live, "candidate_audio_only", False)):
            return cls(mode=CANDIDATE)
        src = (getattr(cfg.audio, "source", "") or "").strip().lower()
        if src in _MIC_SOURCES:
            return cls(mode=CANDIDATE)
        if src == "both":
            return cls(mode=BOTH)
        return cls(mode=LOOPBACK)

    def sources(self) -> list[str]:
        if self.mode == BOTH:
            return [CANDIDATE, LOOPBACK]
        return [self.mode]

    @staticmethod
    def speaker_for(source: str) -> str:
        """Map a raw capture source to its speaker role. Unknown → interviewer
        (today's assumption: the loopback is the other party)."""
        s = (source or "").strip().lower()
        if s in _MIC_SOURCES:
            return CANDIDATE_SPEAKER
        return INTERVIEWER_SPEAKER

    def is_candidate_source(self, source: str) -> bool:
        return self.speaker_for(source) == CANDIDATE_SPEAKER
