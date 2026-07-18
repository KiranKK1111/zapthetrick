"""
Structured candidate profile + resume knowledge graph
(live-conversational-intelligence R39).

Builds a structured `CandidateProfile` (skills / projects / achievements /
experience / metrics) and a small knowledge graph (company → project → tech)
from the resume profile dict the Live WS already loads, optionally merging
GitHub / LinkedIn / notes. `scoped_retrieve` returns just the profile slices
relevant to a topic so a resume-grounded answer draws from the profile, not the
whole resume. Deterministic + fail-open; persists in the existing `Resume` row /
`User.preferences` (no new schema).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateProfile:
    skills: list[str] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)   # {name, tech:[...], company?}
    achievements: list[str] = field(default_factory=list)
    experience: str = ""
    metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skills": self.skills, "projects": self.projects,
            "achievements": self.achievements, "experience": self.experience,
            "metrics": self.metrics,
        }


def _as_list(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip():
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


def build_profile(resume_profile: dict | None, *, github=None, linkedin=None,
                  notes=None) -> CandidateProfile:
    """Build a structured profile from the resume profile dict (+ optional
    sources). Defensive about the resume schema. Never raises."""
    p = CandidateProfile()
    try:
        d = resume_profile or {}
        p.skills = [str(s) for s in _as_list(d.get("skills"))]
        for proj in _as_list(d.get("projects")):
            if isinstance(proj, dict):
                p.projects.append({
                    "name": str(proj.get("name", "")),
                    "tech": [str(t) for t in _as_list(proj.get("tech"))],
                    "company": str(proj.get("company", "")),
                })
            elif isinstance(proj, str):
                p.projects.append({"name": proj, "tech": [], "company": ""})
        p.achievements = [str(a) for a in _as_list(d.get("achievements"))]
        p.metrics = [str(m) for m in _as_list(d.get("metrics"))]
        p.experience = str(d.get("experience") or d.get("summary") or "")
        # Optional source merges (additive).
        if github:
            for repo in _as_list(github):
                name = repo.get("name") if isinstance(repo, dict) else str(repo)
                if name:
                    p.projects.append({"name": str(name), "tech": [], "company": "github"})
        if linkedin:
            p.skills.extend(str(s) for s in _as_list(linkedin))
        if notes:
            p.experience = (p.experience + "\n" + str(notes)).strip()
        # De-dup skills.
        seen, sk = set(), []
        for s in p.skills:
            k = s.lower()
            if k and k not in seen:
                seen.add(k)
                sk.append(s)
        p.skills = sk
        return p
    except Exception:  # noqa: BLE001
        return p


def knowledge_graph(profile: CandidateProfile) -> dict:
    """company → {projects:[...], tech:[...]} from the profile. Never raises."""
    g: dict = {}
    try:
        for proj in profile.projects:
            company = (proj.get("company") or "other").strip() or "other"
            node = g.setdefault(company, {"projects": [], "tech": []})
            if proj.get("name"):
                node["projects"].append(proj["name"])
            for t in proj.get("tech", []):
                if t and t not in node["tech"]:
                    node["tech"].append(t)
        return g
    except Exception:  # noqa: BLE001
        return g


def scoped_retrieve(profile: CandidateProfile, topic: str, k: int = 4) -> list[str]:
    """Profile slices relevant to `topic` (matching skills/projects). Never
    raises → []."""
    try:
        t = (topic or "").strip().lower()
        if not t:
            return []
        out: list[str] = []
        for s in profile.skills:
            if t in s.lower() or s.lower() in t:
                out.append(f"Skill: {s}")
        for proj in profile.projects:
            blob = (proj.get("name", "") + " " + " ".join(proj.get("tech", []))).lower()
            if t in blob:
                tech = ", ".join(proj.get("tech", []))
                out.append(f"Project: {proj.get('name','')}" + (f" ({tech})" if tech else ""))
        return out[:k]
    except Exception:  # noqa: BLE001
        return []


def reality_terms(profile: CandidateProfile) -> set[str]:
    """The set of technologies/skills the candidate can truthfully claim."""
    terms = {s.lower() for s in profile.skills}
    for proj in profile.projects:
        terms.update(t.lower() for t in proj.get("tech", []))
    return {t for t in terms if t}


# Questions ABOUT THE CANDIDATE ("tell me about yourself", "walk me through
# your experience", "what projects have you worked on") must be answered
# from the resume/profile — detailed and in the first person — not from
# general knowledge. Cue-based; "your" alone is too broad ("your opinion on
# Kafka"), so cues pair the possessive with a personal subject.
_PROFILE_CUES = (
    "about yourself", "introduce yourself", "your background",
    "your experience", "your resume", "your cv", "your profile",
    "your projects", "your project", "your work history", "your role",
    "your current role", "your current company", "your last company",
    "your previous company", "your responsibilities", "your day to day",
    "your day-to-day", "your strengths", "your weaknesses",
    "your achievements", "your career", "your journey", "your skills",
    "your tech stack", "your expertise", "worked on", "have you built",
    "you have worked", "you've worked", "why should we hire you",
    "walk me through your", "take me through your", "tell us about your",
    "tell me about your",
)


def is_profile_question(text: str) -> bool:
    """Is this question about the CANDIDATE themselves (experience, projects,
    background, self-introduction)? Deterministic; never raises."""
    try:
        t = (text or "").strip().lower()
        if not t:
            return False
        if any(cue in t for cue in _PROFILE_CUES):   # zero-latency fast-path
            return True
        # SEMANTIC gate (2026-07-09): paraphrases the cue list can't
        # anticipate ("what did you do at your last company", "which project
        # are you most proud of"). Fail-open → cue-list verdict stands.
        from app.semantics import gates as _gates
        return _gates.matches("profile_question", t) is True
    except Exception:  # noqa: BLE001
        return False


def profile_summary(profile: CandidateProfile, k: int = 12) -> list[str]:
    """The FULL structured profile as grounding lines — used for questions
    about the candidate themselves, where topic-scoped slices are too thin
    ("tell me about yourself" has no topic). Never raises → []."""
    out: list[str] = []
    try:
        if profile.experience:
            out.append(f"Experience: {profile.experience[:600]}")
        if profile.skills:
            out.append("Skills: " + ", ".join(profile.skills[:20]))
        for proj in profile.projects[:6]:
            tech = ", ".join(proj.get("tech", []))
            company = proj.get("company", "")
            line = f"Project: {proj.get('name', '')}"
            if tech:
                line += f" ({tech})"
            if company and company != "github":
                line += f" at {company}"
            out.append(line)
        for a in profile.achievements[:4]:
            out.append(f"Achievement: {a}")
        for m in profile.metrics[:4]:
            out.append(f"Metric: {m}")
        return out[:k]
    except Exception:  # noqa: BLE001
        return out[:k]


def first_person_directive() -> str:
    """Directive for answering a question about the candidate themselves.
    The answer is read ALOUD verbatim by the candidate — spoken sentences,
    crisp and specific, never a formatted essay."""
    return (
        "This question is about the CANDIDATE THEMSELVES, and the candidate "
        "will READ YOUR ANSWER ALOUD word for word. Write clean spoken "
        "first-person sentences (\"I\", \"my\") grounded ONLY in the "
        "profile/resume context provided. Be specific: real project names, "
        "tech stack, responsibilities, and quantified results from the "
        "profile — pick the 2-3 most relevant items, never everything. "
        "Shape: one strong opening sentence that answers directly, then the "
        "specifics, then one sentence on why it fits. Keep it to roughly "
        "120-220 words (60-90 seconds spoken). NO headings, tables, bold "
        "labels, or meta commentary — only speakable sentences. NEVER "
        "invent employers, dates, projects, or numbers not in the profile; "
        "if a detail is missing, speak to what IS there instead."
    )
