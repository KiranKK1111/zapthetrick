"""
Speaker diarization, roles & panel threads (live-conversational-intelligence R28).

Attributes each utterance to a Speaker_Role (candidate / primary or secondary
interviewer / recruiter / hiring manager / panel member), extending the
capture-topology speaker labelling with light textual cues, and keeps a
per-interviewer Panel_Thread so a multi-interviewer session stays coherent.
Candidate-attributed utterances are not answered. Deterministic + fail-open:
low confidence or no signal → single primary-interviewer behavior (today), and
low speaker confidence lowers the surfaced answer confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core import lexicons
from app.live.capture_topology import (
    CANDIDATE_SPEAKER,
    INTERVIEWER_SPEAKER,
    CaptureTopology,
)

CANDIDATE = CANDIDATE_SPEAKER
PRIMARY = "primary_interviewer"
SECONDARY = "secondary_interviewer"
RECRUITER = "recruiter"
HIRING_MANAGER = "hiring_manager"
PANEL = "panel_member"

# Textual cues that hint at a role/hand-off within an interviewer turn.
_HANDOFF_CUES = lexicons.LIVE_DIARIZE_HANDOFF_CUES
_RECRUITER_CUES = lexicons.LIVE_DIARIZE_RECRUITER_CUES
_HIRING_MGR_CUES = lexicons.LIVE_DIARIZE_HIRING_MGR_CUES


@dataclass
class Diarizer:
    topology: CaptureTopology = field(default_factory=CaptureTopology)
    _current_interviewer: str = PRIMARY
    threads: dict = field(default_factory=dict)   # role -> list[str] (turns)

    def attribute(self, source: str = "", text: str = "",
                  speaker_confidence: float | None = None) -> tuple[str, float]:
        """Return (role, confidence). Source maps candidate vs interviewer;
        textual cues refine the interviewer role / panel hand-off."""
        try:
            base = self.topology.speaker_for(source) if source else INTERVIEWER_SPEAKER
            if base == CANDIDATE_SPEAKER:
                return CANDIDATE, (speaker_confidence if speaker_confidence is not None else 0.9)
            low = (text or "").lower()
            role = self._current_interviewer
            if any(c in low for c in _HANDOFF_CUES):
                # A hand-off → the NEXT interviewer turn is a secondary panelist.
                self._current_interviewer = SECONDARY if role == PRIMARY else PRIMARY
            elif any(c in low for c in _RECRUITER_CUES):
                role = RECRUITER
            elif any(c in low for c in _HIRING_MGR_CUES):
                role = HIRING_MANAGER
            conf = speaker_confidence if speaker_confidence is not None else 0.6
            self.threads.setdefault(role, [])
            if text:
                self.threads[role].append(text.strip())
                if len(self.threads[role]) > 50:
                    self.threads[role].pop(0)
            return role, conf
        except Exception:  # noqa: BLE001 — fall back to single interviewer
            return PRIMARY, 0.5

    def is_candidate(self, role: str) -> bool:
        return role == CANDIDATE

    def panel_size(self) -> int:
        return len([r for r in self.threads if r != CANDIDATE]) or 1


def for_tracker(tracker) -> Diarizer:
    d = getattr(tracker, "_live_diarizer", None)
    if d is None:
        d = Diarizer(topology=CaptureTopology.from_config())
        try:
            setattr(tracker, "_live_diarizer", d)
        except Exception:  # noqa: BLE001
            pass
    return d
