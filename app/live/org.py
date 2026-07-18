"""
Organization & JD intelligence + fit analysis
(live-conversational-intelligence R41).

Builds an `OrganizationProfile` from the company name + (optional) pasted job
description, computes a `Fit_Analysis` against the `CandidateProfile`, and grounds
"Why us? / Why hire you? / What do you know about us?" answers. JD parsing is
deterministic (built-in skills lexicon + capitalized/quoted-term heuristics — no
LLM call); live company research is opt-in via the existing web-search/RAG
tools (fail-open). No new schema. Fail-open → today's generic behavior when
absent.

Integration contract (called from the live WS layer):

    org = build_org(org_name, jd_text, role, *, job_role="", notes="")
    text = fit_directive(org, candidate_profile)   # compact, < 500 chars

`job_role` is a keyword alias for the positional `role` (the session metadata
key is `job_role`); `notes` is free-form recruiter/user context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.live.profile import CandidateProfile, reality_terms

# A built-in skills lexicon to pull required skills out of a JD deterministically
# (~80 common languages / frameworks / clouds / databases / tools). Multi-word
# and symbol-bearing entries (c++, c#, ci/cd, .net, next.js) are supported by
# the custom boundary regex in `_extract_jd_skills`.
_SKILL_LEXICON = (
    # Languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "c++", "c#", "kotlin", "swift", "scala", "ruby", "php", "sql", "bash",
    "html", "css",
    # Frameworks / runtimes
    "react", "react native", "angular", "vue", "svelte", "next.js", "node",
    "express", "spring", "spring boot", "django", "flask", "fastapi", ".net",
    "rails", "laravel", "flutter",
    # Data / ML
    "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
    "pandas", "numpy", "spark", "hadoop", "airflow", "dbt", "etl",
    "snowflake", "databricks",
    # Messaging / databases / search
    "kafka", "rabbitmq", "redis", "postgres", "postgresql", "mysql",
    "mongodb", "cassandra", "dynamodb", "elasticsearch", "sqlite", "oracle",
    "neo4j",
    # Cloud / infra / tooling
    "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "ansible",
    "jenkins", "helm", "prometheus", "grafana", "linux", "git",
    "github actions", "gitlab", "ci/cd", "devops", "serverless", "lambda",
    # Architecture / practice
    "microservices", "distributed systems", "rest", "graphql", "grpc",
    "websockets", "oauth", "system design", "agile", "scrum", "tdd",
)

# Skills whose lowercase form is a common English word — require the cased form
# in the ORIGINAL text to count ("Go" the language vs. "go fast").
_CASE_SENSITIVE = {
    "go": r"\bGo\b|\bgolang\b",
}

# ALLCAPS tokens that are boilerplate, not skills.
_ACRONYM_STOP = {
    "AND", "THE", "FOR", "WITH", "YOU", "OUR", "ARE", "NOT", "ALL", "ANY",
    "PER", "USA", "LLC", "INC", "JD", "EEO", "IT", "HR", "US", "PTO", "USD",
    "EST", "PST", "FAQ", "CEO", "CTO", "VP",
}

_MAX_JD_SKILLS = 24


def _boundary_pattern(skill: str) -> str:
    """Word-boundary-ish regex that also works for c++, c#, ci/cd, .net."""
    return r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9+#])"


def _extract_jd_skills(jd_text: str) -> list[str]:
    """Deterministic skill extraction: lexicon match + quoted-term and
    capitalized-term heuristics. Never raises → []."""
    out: list[str] = []
    try:
        raw = jd_text or ""
        low = raw.lower()
        if not low.strip():
            return out
        # 1) Lexicon.
        for s in _SKILL_LEXICON:
            cased = _CASE_SENSITIVE.get(s)
            if cased is not None:
                if re.search(cased, raw):
                    out.append(s)
            elif re.search(_boundary_pattern(s), low):
                out.append(s)

        def _add(term: str) -> None:
            t = term.strip().lower()
            if not (1 < len(t) <= 30) or len(t.split()) > 3:
                return
            if t in out or any(t in have for have in out):
                return
            out.append(t)

        # 2) Quoted terms — '"Temporal" experience required'.
        for m in re.findall(r'["“‘\']([A-Za-z][A-Za-z0-9 .+#/_-]{1,29})["”’\']', raw):
            _add(m)
        # 3) CamelCase tokens (GraphQL, DevOps, JavaScript) — internal capital
        #    keeps this conservative (skips sentence-start words).
        for m in re.findall(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b", raw):
            _add(m)
        # 4) ALLCAPS acronyms (AWS, GCP, SDET) minus boilerplate.
        for m in re.findall(r"\b[A-Z]{2,6}\b", raw):
            if m not in _ACRONYM_STOP:
                _add(m)
        return out[:_MAX_JD_SKILLS]
    except Exception:  # noqa: BLE001
        return out[:_MAX_JD_SKILLS]


# ── Named-company interview-style bias (roadmap Phase 2 #20 / 2C-20) ────────
# A small, honest library of well-known companies' interview emphases. Keys are
# matched as substrings of the (lowercased) company name. Deterministic bias
# only — a directive nudging the answer's SHAPE, never fabricated facts about
# the company. Absent/unknown company → no bias (today's behavior).
_COMPANY_STYLE = {
    "amazon": ("Amazon weighs the Leadership Principles heavily — structure "
               "behavioral answers as STAR with clear ownership, data, and "
               "customer obsession."),
    "google": ("Google emphasizes algorithmic depth and scalable design — show "
               "clean problem-solving, complexity awareness, and trade-offs."),
    "meta": ("Meta values speed and impact — be direct, quantify impact, and "
             "show pragmatic execution at scale."),
    "facebook": ("Meta values speed and impact — be direct, quantify impact, "
                 "and show pragmatic execution at scale."),
    "microsoft": ("Microsoft looks for a growth mindset and collaboration — "
                  "show learning from setbacks and cross-team empathy."),
    "apple": ("Apple prizes craftsmanship and detail — emphasize quality, "
              "user experience, and rigorous attention to detail."),
    "netflix": ("Netflix expects high judgment and candor — show independent "
                "decision-making and a bias for the best idea, not seniority."),
    "stripe": ("Stripe values rigor and clarity of thought — reason carefully, "
               "state assumptions, and show craftsmanship in API/systems design."),
    "uber": ("Uber optimizes for scale and speed — emphasize operating at scale, "
             "real-time systems, and pragmatic trade-offs."),
    "startup": ("Early-stage teams value breadth and ownership — show you can "
                "wear many hats and ship end-to-end with limited resources."),
}


def company_style_directive(company: str) -> str:
    """A deterministic interview-style bias for a well-known company. '' when
    the company is unknown/empty. Never raises."""
    try:
        c = (company or "").strip().lower()
        if not c:
            return ""
        for key, note in _COMPANY_STYLE.items():
            if key in c:
                return note
        return ""
    except Exception:  # noqa: BLE001
        return ""


def known_company(company: str) -> bool:
    """Whether a named-company style bias exists. Never raises → False."""
    try:
        c = (company or "").strip().lower()
        return any(k in c for k in _COMPANY_STYLE) if c else False
    except Exception:  # noqa: BLE001
        return False


@dataclass
class OrganizationProfile:
    company: str = ""
    role: str = ""
    jd_skills: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {"company": self.company, "role": self.role,
                "jd_skills": self.jd_skills, "notes": self.notes}


def build_org(company: str = "", jd_text: str = "", role: str = "", *,
              job_role: str = "", notes: str = "") -> OrganizationProfile:
    """Build an org profile from the company name + optional JD text, role and
    notes. `job_role` is a keyword alias for `role` (metadata key name); `role`
    wins if both are given. Never raises."""
    o = OrganizationProfile(
        company=(company or "").strip(),
        role=(role or "").strip() or (job_role or "").strip(),
        notes=(notes or "").strip(),
    )
    try:
        o.jd_skills = _extract_jd_skills(jd_text or "")
        return o
    except Exception:  # noqa: BLE001
        return o


def fit_analysis(profile: CandidateProfile, org: OrganizationProfile) -> dict:
    """Matching skills / gaps / role-relevant strengths. Never raises."""
    try:
        cand = reality_terms(profile)
        need = {s.lower() for s in org.jd_skills}
        if not need:
            return {"matching": [], "gaps": [], "strengths": sorted(cand)[:8]}
        matching = sorted(need & cand)
        gaps = sorted(need - cand)
        return {"matching": matching, "gaps": gaps,
                "strengths": matching[:8] or sorted(cand)[:8]}
    except Exception:  # noqa: BLE001
        return {"matching": [], "gaps": [], "strengths": []}


def directive(org: OrganizationProfile, fit: dict) -> str:
    """Ground company-specific answers (Why us / fit) in the org + fit data."""
    try:
        parts = []
        if org.company:
            parts.append(f"Target company: {org.company}"
                         + (f"; role: {org.role}" if org.role else ""))
        if fit.get("matching"):
            parts.append("Emphasize matching strengths: " + ", ".join(fit["matching"][:8]))
        if fit.get("gaps"):
            parts.append("Be honest about gaps: " + ", ".join(fit["gaps"][:5]))
        return (" ".join(parts) + ".") if parts else ""
    except Exception:  # noqa: BLE001
        return ""


def fit_directive(org: OrganizationProfile,
                  profile: CandidateProfile | None = None) -> str:
    """Compact (< 500 chars) fit directive for the live answer prompt: target
    company + role, matching skills, gap skills, and one line of guidance.
    Degrades gracefully (empty JD → still names the org). Never raises → ''."""
    try:
        fit = fit_analysis(profile or CandidateProfile(), org)
        matching = [s for s in (fit.get("matching") or []) if s]
        gaps = [s for s in (fit.get("gaps") or []) if s]
        parts: list[str] = []
        if org.company:
            parts.append("Target: " + org.company
                         + (f" ({org.role})" if org.role else ""))
        elif org.role:
            parts.append("Target role: " + org.role)
        if matching:
            parts.append("Matching skills: " + ", ".join(matching[:6]))
        if gaps:
            parts.append("JD gaps: " + ", ".join(gaps[:5]))
        if matching and gaps:
            parts.append(f"Emphasize {', '.join(matching[:3])}; "
                         f"be ready to address {', '.join(gaps[:3])}")
        elif matching:
            parts.append("Emphasize " + ", ".join(matching[:3]))
        elif gaps:
            parts.append("Be ready to address " + ", ".join(gaps[:3]))
        if org.notes:
            parts.append("Notes: " + org.notes[:120])
        out = ". ".join(p.rstrip(".") for p in parts if p)
        if not out:
            return ""
        out += "."
        if len(out) >= 500:
            out = out[:496].rstrip(" ,;.") + "..."
        return out
    except Exception:  # noqa: BLE001
        return ""
