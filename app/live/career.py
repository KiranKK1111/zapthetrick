"""
Advisory Career_Intelligence (live-conversational-intelligence R61).

A STANDALONE, read-only, off-the-critical-path layer that turns the candidate
profile + fit analysis + session replay into advisory interview/career-prep
coaching: which skill gaps to shore up before the next round, role-fit notes,
and a coarse promotion/seniority-readiness signal. It is DISABLED by default,
scoped strictly to interview/career preparation, and explicitly NOT
professional / legal / financial advice. Deterministic + fail-open → nothing on
error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.live.org import OrganizationProfile
from app.live.profile import CandidateProfile, reality_terms

DISCLAIMER = ("Advisory interview/career-prep coaching only — NOT professional, "
              "legal, or financial advice.")


@dataclass
class CareerIntelligence:
    coaching: list[str] = field(default_factory=list)
    skill_gaps: list[str] = field(default_factory=list)
    role_fit: str = ""
    readiness: str = "unknown"   # entry / mid / senior_ready / unknown
    disclaimer: str = DISCLAIMER
    advisory: bool = True

    def to_dict(self) -> dict:
        return {"coaching": self.coaching, "skill_gaps": self.skill_gaps,
                "role_fit": self.role_fit, "readiness": self.readiness,
                "disclaimer": self.disclaimer, "advisory": True}


def _readiness(profile: CandidateProfile) -> str:
    try:
        n_proj = len(profile.projects)
        n_skill = len(profile.skills)
        n_ach = len(profile.achievements)
        score = n_proj + n_skill * 0.5 + n_ach
        if score >= 12:
            return "senior_ready"
        if score >= 6:
            return "mid"
        return "entry"
    except Exception:  # noqa: BLE001
        return "unknown"


# Capability-over-title framing (BandSpecific.md lines 843-859: engineers are
# increasingly evaluated by capability, not title). These short directives let
# the seniority-calibration layer fold a readiness signal into the live answer
# guidance so the answer leans on demonstrated capability, not the nominal band.
READINESS_HINTS: dict[str, str] = {
    "entry": ("Capability over title: lean on solid fundamentals, fast learning, and the "
              "concrete things already built rather than on years or a senior label."),
    "mid": ("Capability over title: let demonstrated ownership and delivery carry the "
            "answer — proven scope speaks louder than the title on paper."),
    "senior_ready": ("Capability over title: the depth of projects and impact signals "
                     "readiness beyond the nominal title — show senior-level judgement and "
                     "system thinking directly, backed by real evidence."),
}


def readiness_signal(cp: CandidateProfile | None) -> str:
    """Public, deterministic readiness signal (entry / mid / senior_ready /
    unknown) for a candidate profile. Never raises."""
    if cp is None:
        return "unknown"
    try:
        return _readiness(cp)
    except Exception:  # noqa: BLE001
        return "unknown"


def capability_directive(readiness: str | None) -> str:
    """One capability-over-title framing line for a readiness signal, or '' when
    the signal is unknown/unrecognised. Never raises."""
    try:
        return READINESS_HINTS.get((readiness or "").strip(), "")
    except Exception:  # noqa: BLE001
        return ""


def analyze(
    profile: CandidateProfile | None,
    org: OrganizationProfile | None = None,
    *,
    fit: dict | None = None,
    replay_summary: dict | None = None,
) -> CareerIntelligence:
    """Build advisory career intelligence. DISABLED-by-default gating happens at
    the call site; this function is pure. Never raises → empty (advisory)."""
    ci = CareerIntelligence()
    try:
        if profile is None:
            return ci
        ci.readiness = _readiness(profile)
        # Skill gaps from the fit analysis (preferred) or empty.
        if fit and isinstance(fit, dict):
            ci.skill_gaps = list(fit.get("gaps", []))[:8]
            if fit.get("matching"):
                ci.role_fit = ("Strong overlap on: " + ", ".join(fit["matching"][:6])
                               + (f". Gaps to close: {', '.join(ci.skill_gaps[:4])}."
                                  if ci.skill_gaps else "."))
        # Coaching notes.
        if ci.skill_gaps:
            ci.coaching.append("Before the next round, prepare concrete examples for: "
                               + ", ".join(ci.skill_gaps[:4]) + ".")
        if not reality_terms(profile):
            ci.coaching.append("Add concrete technologies/metrics to the profile to strengthen answers.")
        if replay_summary and isinstance(replay_summary, dict):
            answered = (replay_summary.get("by_type", {}) or {}).get("answer", 0)
            if answered:
                ci.coaching.append(f"Last session answered {answered} question(s) — "
                                   "review the replay for pacing and depth.")
        if org is not None and org.company:
            ci.coaching.append(f"Research {org.company}-specific context for 'Why us?' answers.")
        return ci
    except Exception:  # noqa: BLE001
        return ci
