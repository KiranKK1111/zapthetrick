"""
Supporting_Segments binding + staleness hedge
(live-conversational-intelligence R49).

Binds an answer to the supporting evidence segments (retrieved snippets /
profile slices / knowledge angles) it drew from, so the answer is grounded and
auditable. When the bound evidence is stale or thin, emit a knowledge-gap hedge
directive so the answer is appropriately tentative instead of confidently wrong.
Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass
class SupportingSegment:
    text: str
    source: str = ""
    ts: float = field(default_factory=time)
    score: float = 1.0


@dataclass
class EvidenceBinding:
    segments: list = field(default_factory=list)

    def add(self, text: str, source: str = "", score: float = 1.0) -> None:
        t = (text or "").strip()
        if t:
            self.segments.append(SupportingSegment(text=t, source=source, score=score))

    def is_thin(self, min_segments: int = 1) -> bool:
        return len([s for s in self.segments if s.text]) < min_segments

    def is_stale(self, max_age_s: float = 600.0) -> bool:
        """True when ALL segments are older than max_age_s (so the evidence may
        no longer reflect the current turn)."""
        if not self.segments:
            return True
        now = time()
        return all((now - s.ts) > max_age_s for s in self.segments)

    def to_dict(self) -> dict:
        return {"count": len(self.segments),
                "sources": sorted({s.source for s in self.segments if s.source}),
                "segments": [s.text[:120] for s in self.segments[:6]]}


# ── Evidence-strength ranking (roadmap Phase 2 #18/#19 / 2C-18/2C-19) ───────
# How much each evidence SOURCE is worth. Grounded first-party sources (the
# candidate's own resume/profile) outrank generic knowledge angles, which
# outrank bare directives. Used to rank supporting segments and to derive an
# overall STRENGTH label the answer can calibrate its confidence against.
_SOURCE_WEIGHT = {
    "profile": 1.0, "resume": 1.0, "candidate": 1.0,
    "retrieval": 0.8, "knowledge": 0.7, "org": 0.7,
    "world_model": 0.6, "directive": 0.4, "": 0.4,
}

STRONG = "strong"
MODERATE = "moderate"
WEAK = "weak"


def segment_strength(seg: SupportingSegment) -> float:
    """A single segment's strength = source weight × its own score. Never raises."""
    try:
        w = _SOURCE_WEIGHT.get((seg.source or "").strip().lower(), 0.4)
        return max(0.0, min(1.0, w * float(seg.score)))
    except Exception:  # noqa: BLE001
        return 0.0


def rank_segments(binding: EvidenceBinding) -> list[SupportingSegment]:
    """Supporting segments sorted strongest-first. Never raises → []."""
    try:
        return sorted(binding.segments, key=segment_strength, reverse=True)
    except Exception:  # noqa: BLE001
        return []


def strength_label(binding: EvidenceBinding) -> str:
    """Overall evidence-strength label for an answer: rewards BOTH the best
    source and having corroborating segments. Never raises → WEAK."""
    try:
        segs = binding.segments
        if not segs:
            return WEAK
        best = max(segment_strength(s) for s in segs)
        corroboration = min(0.2, 0.05 * (len(segs) - 1))
        score = min(1.0, best + corroboration)
        if score >= 0.75:
            return STRONG
        if score >= 0.5:
            return MODERATE
        return WEAK
    except Exception:  # noqa: BLE001
        return WEAK


def strength_directive(binding: EvidenceBinding) -> str:
    """A confidence-calibration directive keyed to evidence strength. Strong →
    answer assertively; weak → hedge. '' for moderate. Never raises."""
    try:
        label = strength_label(binding)
        if label == STRONG:
            return ("Evidence is strong and grounded — answer assertively with "
                    "the specifics.")
        if label == WEAK:
            return ("Evidence is weak — answer conservatively, flag assumptions, "
                    "and avoid over-claiming specifics.")
        return ""
    except Exception:  # noqa: BLE001
        return ""


def hedge_directive(binding: EvidenceBinding, *, max_age_s: float = 600.0) -> str:
    """Return a knowledge-gap hedge directive when evidence is thin/stale, else
    ''. Folded into the SAME answer call. Never raises."""
    try:
        if binding.is_thin() or binding.is_stale(max_age_s):
            return ("Evidence for this answer is thin or stale — answer tentatively, "
                    "state assumptions, and avoid over-confident claims.")
        return ""
    except Exception:  # noqa: BLE001
        return ""
