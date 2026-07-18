"""
Mock_Mode question generation (live-conversational-intelligence R46).

Generates practice interview questions deterministically from the candidate
profile + org/JD (reusing `app/live/profile.py` + `app/live/org.py`), so a user
can rehearse through the SAME live pipeline. Every generated turn is explicitly
labeled practice. Deterministic + fail-open. The actual answering reuses the
existing live answer path (no separate engine).
"""
from __future__ import annotations

from app.live.org import OrganizationProfile
from app.live.profile import CandidateProfile

PRACTICE_LABEL = "practice"

# Generic question templates by category.
_BEHAVIORAL = (
    "Tell me about a time you faced a difficult technical challenge.",
    "Describe a situation where you disagreed with a teammate.",
    "Walk me through your proudest achievement.",
)
_INTRO = (
    "Tell me about yourself.",
    "Why are you interested in this role?",
)


def generate_questions(
    profile: CandidateProfile | None = None,
    org: OrganizationProfile | None = None,
    *,
    limit: int = 8,
) -> list[dict]:
    """Generate labeled practice questions from the profile + org. Each item is
    {question, category, label}. Never raises."""
    out: list[dict] = []
    try:
        for q in _INTRO:
            out.append({"question": q, "category": "introduction", "label": PRACTICE_LABEL})
        # Skill-targeted technical questions from the profile.
        if profile is not None:
            for s in profile.skills[:4]:
                out.append({"question": f"Can you explain how you've used {s}?",
                            "category": "technical", "label": PRACTICE_LABEL})
            for proj in profile.projects[:3]:
                name = proj.get("name") if isinstance(proj, dict) else str(proj)
                if name:
                    out.append({"question": f"Walk me through the {name} project.",
                                "category": "resume", "label": PRACTICE_LABEL})
        # JD-targeted questions from the org.
        if org is not None:
            for s in org.jd_skills[:3]:
                out.append({"question": f"This role needs {s} — what's your experience with it?",
                            "category": "fit", "label": PRACTICE_LABEL})
        for q in _BEHAVIORAL:
            out.append({"question": q, "category": "behavioral", "label": PRACTICE_LABEL})
        return out[:limit]
    except Exception:  # noqa: BLE001
        return out[:limit]


def is_practice(item: dict) -> bool:
    """True when a generated turn is labeled practice (so the UI flags it)."""
    try:
        return item.get("label") == PRACTICE_LABEL
    except Exception:  # noqa: BLE001
        return False
