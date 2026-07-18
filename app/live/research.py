"""
Pre-interview research brief — minimal, no-browsing version
(roadmap Phase 2 #36 / 2E-36).

FULL computer-use pre-interview research (autonomously browsing the company,
Glassdoor, the interviewer's LinkedIn) is a large agentic/browser feature and
remains deferred. What is genuinely useful and self-contained is a deterministic
PREP BRIEF assembled from the intake the session already has — company, role and
the pasted JD skills: a prioritized checklist of what to review, likely question
themes, and smart questions to ask back. No network, no LLM. Fail-open.

When the opt-in web-search/RAG tools are enabled elsewhere, they can enrich this
brief; absent them it degrades to a solid deterministic checklist.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResearchBrief:
    company: str = ""
    role: str = ""
    review_topics: list[str] = field(default_factory=list)
    question_themes: list[str] = field(default_factory=list)
    questions_to_ask: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"company": self.company, "role": self.role,
                "review_topics": self.review_topics,
                "question_themes": self.question_themes,
                "questions_to_ask": self.questions_to_ask}


_ROLE_THEMES = {
    "backend": ("system design", "databases", "APIs", "concurrency", "scaling"),
    "frontend": ("rendering", "state management", "accessibility", "performance"),
    "data": ("pipelines", "SQL", "modeling", "warehousing", "quality"),
    "ml": ("modeling", "evaluation", "data pipelines", "deployment"),
    "devops": ("CI/CD", "observability", "infrastructure as code", "incident response"),
    "mobile": ("lifecycle", "state", "offline", "performance"),
}


def _role_bucket(role: str) -> str:
    r = (role or "").lower()
    for k in _ROLE_THEMES:
        if k in r:
            return k
    if "full" in r and "stack" in r:
        return "backend"
    return ""


def build_brief(company: str = "", role: str = "",
                jd_skills: list[str] | None = None) -> ResearchBrief:
    """Assemble a deterministic prep brief from intake. Never raises."""
    try:
        b = ResearchBrief(company=(company or "").strip(), role=(role or "").strip())
        skills = [s for s in (jd_skills or []) if s][:10]
        # Review topics: the JD skills first, then role-bucket themes.
        b.review_topics = list(skills)
        bucket = _role_bucket(role)
        themes = list(_ROLE_THEMES.get(bucket, ()))
        for th in themes:
            if th not in b.review_topics:
                b.review_topics.append(th)
        b.review_topics = b.review_topics[:12]
        # Likely question themes.
        b.question_themes = (themes or ["fundamentals", "past projects",
                                        "problem solving", "collaboration"])[:6]
        # Smart questions to ask back (company-grounded when named).
        c = b.company or "the team"
        b.questions_to_ask = [
            f"What does success in this role look like at {c} in the first 90 days?",
            "How is the team structured and how do decisions get made?",
            "What are the biggest technical challenges the team is facing now?",
        ]
        return b
    except Exception:  # noqa: BLE001
        return ResearchBrief(company=(company or "").strip(), role=(role or "").strip())


def brief_directive(brief: ResearchBrief) -> str:
    """A compact directive seeding early answers with the prep focus. '' when
    there's nothing to say. Never raises."""
    try:
        if not (brief.company or brief.review_topics):
            return ""
        parts = []
        if brief.company:
            parts.append(f"Interview prep for {brief.company}"
                         + (f" ({brief.role})" if brief.role else ""))
        if brief.review_topics:
            parts.append("Likely focus areas: " + ", ".join(brief.review_topics[:6]))
        return ". ".join(parts) + "."
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["ResearchBrief", "build_brief", "brief_directive"]
