"""
Glanceable surfacing + candidate awareness (live-conversational-intelligence R19).

`talking_points` distils an answer into a few concise bullets a candidate can
glance at while speaking (additive — the full answer is unchanged).
`CandidateAwareness` tracks what the candidate has said so a competing full
answer can be suppressed/deferred while they are answering adequately, and so
answers build on (not repeat) their own words. Deterministic + fail-open; with
the flags off the full answer surfaces exactly as today.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_MAX_POINTS = 4
_MAX_POINT_CHARS = 110


def talking_points(answer: str, max_points: int = _MAX_POINTS) -> list[str]:
    """Distil an answer into <= max_points concise talking-point bullets.
    Prefers existing markdown bullets / numbered lines; else the first sentence
    of each paragraph. Never raises — returns [] on empty/error."""
    text = (answer or "").strip()
    if not text:
        return []
    try:
        points: list[str] = []
        # 1. Existing bullet / numbered lines.
        for line in text.splitlines():
            s = line.strip()
            m = re.match(r"^(?:[-*•]\s+|\d+[.)]\s+)(.+)$", s)
            if m:
                points.append(m.group(1).strip())
        # 2. Fallback: first sentence of each paragraph.
        if not points:
            for para in re.split(r"\n\s*\n", text):
                para = para.strip()
                if not para:
                    continue
                first = re.split(r"(?<=[.?!])\s+", para)[0].strip()
                if first:
                    points.append(first)
        # Clean markdown emphasis, cap length + count.
        out: list[str] = []
        for p in points:
            p = re.sub(r"[*_`#]+", "", p).strip()
            if not p:
                continue
            if len(p) > _MAX_POINT_CHARS:
                p = p[:_MAX_POINT_CHARS].rstrip() + "…"
            out.append(p)
            if len(out) >= max_points:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


@dataclass
class CandidateAwareness:
    """Rolling memory of the candidate's own recent utterances."""
    recent: list[str] = field(default_factory=list)
    _last_words: int = 0

    def observe_candidate(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        self.recent.append(t)
        if len(self.recent) > 10:           # bounded
            self.recent.pop(0)
        self._last_words = len(t.split())

    def is_answering_adequately(self) -> bool:
        """Heuristic: the candidate just gave a substantial answer (>= ~12
        words), so the assistant need not surface a competing full answer."""
        return self._last_words >= 12

    def should_surface(self) -> bool:
        """Whether to surface a full answer now (suppress while the candidate is
        answering adequately)."""
        return not self.is_answering_adequately()

    def reset_turn(self) -> None:
        self._last_words = 0


def for_tracker(tracker) -> CandidateAwareness:
    """Per-session CandidateAwareness attached to the context tracker (lazily)."""
    c = getattr(tracker, "_live_candidate", None)
    if c is None:
        c = CandidateAwareness()
        try:
            setattr(tracker, "_live_candidate", c)
        except Exception:  # noqa: BLE001
            pass
    return c


# ── Override gating + confidence indicator (R54) ───────────────────────
# When the system's answer disagrees with what the candidate just said, it does
# NOT silently override. It surfaces a SUGGESTION (additive) gated by a
# confidence margin, and tags every surfaced answer with a confidence band the
# candidate can glance at.

def confidence_band(confidence: float | None) -> str:
    """Map a [0,1] confidence to a glanceable band. Never raises."""
    try:
        if confidence is None:
            return "unknown"
        c = max(0.0, min(1.0, confidence))
        if c >= 0.8:
            return "high"
        if c >= 0.55:
            return "medium"
        return "low"
    except Exception:  # noqa: BLE001
        return "unknown"


def override_suggestion(
    candidate_text: str,
    system_answer: str,
    *,
    system_confidence: float = 0.0,
    margin: float = 0.7,
) -> dict | None:
    """When the system answer contradicts the candidate's statement AND the
    system is confident past `margin`, return an additive suggestion frame (never
    a silent override). Below margin → None (defer to the candidate). Never
    raises."""
    try:
        cand = (candidate_text or "").strip()
        ans = (system_answer or "").strip()
        if not cand or not ans:
            return None
        if system_confidence < margin:
            return None
        return {
            "type": "suggestion",
            "kind": "possible_correction",
            "confidence": round(system_confidence, 3),
            "band": confidence_band(system_confidence),
            "note": "The assistant's answer differs from what you said — review before relying on it.",
        }
    except Exception:  # noqa: BLE001
        return None
