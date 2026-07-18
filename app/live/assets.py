"""
Pre-generated interview assets + resume-reality
(live-conversational-intelligence R40).

Common questions ("tell me about yourself", "why should we hire you", behavioral
STAR stories, project/skill explanations) can be prepared from the
`CandidateProfile` and served near-instantly. `match_asset` maps a live question
to a prepared asset key; the actual pre-generation + serve reuse the
perceived-speed answer-cache (opt-in). `resume_reality` enforces that a generated
claim is supported by the profile (no inflation). Deterministic + fail-open.
"""
from __future__ import annotations

from app.live.profile import CandidateProfile, reality_terms

# Asset keys → trigger phrases.
_ASSETS = {
    "self_intro": ("tell me about yourself", "introduce yourself", "walk me through your background",
                   "walk me through your resume"),
    "why_hire": ("why should we hire you", "why are you a good fit", "what makes you a good"),
    "biggest_challenge": ("biggest challenge", "a challenge you faced", "difficult problem you"),
    "biggest_achievement": ("biggest achievement", "proudest", "accomplishment you"),
    "why_leaving": ("why are you leaving", "why do you want to leave"),
    "strengths": ("your strengths", "your greatest strength"),
    "weaknesses": ("your weakness", "areas to improve"),
}


def asset_keys() -> list[str]:
    return list(_ASSETS.keys())


def match_asset(question: str) -> str | None:
    """Map a live question to a prepared asset key, or None. Never raises."""
    try:
        t = (question or "").lower()
        if not t.strip():
            return None
        for key, triggers in _ASSETS.items():
            if any(tr in t for tr in triggers):
                return key
        return None
    except Exception:  # noqa: BLE001
        return None


def supports_claim(claim: str, profile: CandidateProfile) -> bool:
    """Resume_Reality: True when the claimed technology/skill is present in the
    profile (so the answer doesn't inflate beyond it). Never raises → True for an
    empty/ungrounded claim (caller decides)."""
    try:
        c = (claim or "").strip().lower()
        if not c:
            return True
        terms = reality_terms(profile)
        if not terms:
            return True
        # If the claim names a known term, it must be in the profile.
        return any(term in c for term in terms)
    except Exception:  # noqa: BLE001
        return True


def reality_directive(profile: CandidateProfile) -> str:
    """A directive grounding resume-based answers in the candidate's real
    experience — never inflating ("used X" must not become "designed X")."""
    try:
        terms = sorted(reality_terms(profile))
        if not terms:
            return ""
        shown = ", ".join(terms[:18])
        return ("Ground resume claims ONLY in the candidate's real experience "
                f"({shown}). Do not inflate (e.g. 'used X' is not 'designed X internals') "
                "or invent technologies not listed.")
    except Exception:  # noqa: BLE001
        return ""
