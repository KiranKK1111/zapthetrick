"""Candidate Career Graph — the authoritative envelope (vNext §4.18, Stage 6 J).

A live answer about the candidate must be grounded in what the resume ACTUALLY
says — never an impressive-sounding fabrication. §4.18 makes the resume a typed
**Career Graph**: every role / project / metric / skill is a `CareerFact` carrying
a **source span** (the char range in the resume text it came from). A fact WITHOUT
a source span is ungrounded and flagged — so the answer layer can refuse to claim
it. This is the anti-hallucination envelope the prepared-answer library
(`prepared.py`) and the live answer both draw from.

At session setup, tailoring to the JD is a cheap DELTA — `tailor_to_jd` marks
which grounded facts to EMPHASIZE (they match the JD) and which JD requirements
are GAPS (no supporting fact) — rather than regenerating anything.

Profile-question detection is SEMANTIC (`is_profile_question` → the existing
`profile_question` gate). Pure + fail-open. Flag-gated (`live.profile_library`).
The prepared-answer generation + cosine match already live in `prepared.py`; this
adds the grounded graph + the JD delta.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "profile_library", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class SourceSpan:
    text: str
    start: int
    end: int


@dataclass
class CareerFact:
    kind: str                       # role | project | metric | skill
    value: str
    source: SourceSpan | None = None

    @property
    def grounded(self) -> bool:
        return self.source is not None

    def as_dict(self) -> dict:
        d = {"kind": self.kind, "value": self.value, "grounded": self.grounded}
        if self.source is not None:
            d["source"] = {"start": self.source.start, "end": self.source.end}
        return d


@dataclass
class CareerGraph:
    roles: list[CareerFact] = field(default_factory=list)
    projects: list[CareerFact] = field(default_factory=list)
    metrics: list[CareerFact] = field(default_factory=list)
    skills: list[CareerFact] = field(default_factory=list)

    def all_facts(self) -> list[CareerFact]:
        return self.roles + self.projects + self.metrics + self.skills

    def grounded_facts(self) -> list[CareerFact]:
        return [f for f in self.all_facts() if f.grounded]

    def ungrounded_facts(self) -> list[CareerFact]:
        return [f for f in self.all_facts() if not f.grounded]

    def as_dict(self) -> dict:
        return {k: [f.as_dict() for f in v] for k, v in (
            ("roles", self.roles), ("projects", self.projects),
            ("metrics", self.metrics), ("skills", self.skills))}


def _find_span(value: str, resume_lower: str, resume_text: str) -> SourceSpan | None:
    """Locate `value` in the resume text (case-insensitive) → its source span, or
    None when the fact isn't literally supported by the resume."""
    v = (value or "").strip()
    if not v:
        return None
    idx = resume_lower.find(v.lower())
    if idx < 0:
        # Try the first significant token (a project/skill name) as a weaker anchor.
        tok = next((w for w in re.findall(r"[A-Za-z0-9+#.]{3,}", v)), "")
        if tok:
            idx = resume_lower.find(tok.lower())
            if idx >= 0:
                return SourceSpan(text=resume_text[idx:idx + len(tok)],
                                  start=idx, end=idx + len(tok))
        return None
    return SourceSpan(text=resume_text[idx:idx + len(v)], start=idx,
                      end=idx + len(v))


def build_career_graph(profile: dict, resume_text: str = "") -> CareerGraph:
    """Build the typed Career Graph from a profile dict + the raw resume text,
    attaching a SOURCE SPAN to every fact literally supported by the resume (the
    rest are kept but flagged ungrounded). Never raises."""
    g = CareerGraph()
    try:
        rt = resume_text or ""
        rl = rt.lower()

        def _fact(kind: str, value: str) -> CareerFact:
            return CareerFact(kind=kind, value=str(value).strip(),
                              source=_find_span(str(value), rl, rt) if rt else None)

        p = profile or {}
        for r in (p.get("experience") or p.get("roles") or []):
            name = r.get("title") or r.get("role") or r if isinstance(r, dict) else r
            if name:
                g.roles.append(_fact("role", name))
        for proj in (p.get("projects") or []):
            name = proj.get("name") if isinstance(proj, dict) else proj
            if name:
                g.projects.append(_fact("project", name))
        for m in (p.get("metrics") or []):
            if m:
                g.metrics.append(_fact("metric", m))
        for s in (p.get("skills") or []):
            if s:
                g.skills.append(_fact("skill", s))
        return g
    except Exception:  # noqa: BLE001
        return g


# --------------------------------------------------------------------------- #
# JD tailoring delta
# --------------------------------------------------------------------------- #
@dataclass
class Tailoring:
    emphasize: list[str] = field(default_factory=list)   # grounded facts matching the JD
    gaps: list[str] = field(default_factory=list)        # JD terms with no fact


def tailor_to_jd(graph: CareerGraph, jd_terms) -> Tailoring:
    """Session-setup DELTA: which GROUNDED career facts to emphasize (they match a
    JD term) and which JD terms are gaps (no supporting fact). Cheap + pure."""
    t = Tailoring()
    try:
        terms = [str(x).strip().lower() for x in (jd_terms or []) if str(x).strip()]
        if not terms:
            return t
        grounded = graph.grounded_facts()
        fact_blob = " ".join(f.value.lower() for f in grounded)
        emphasized_terms: set[str] = set()
        for term in terms:
            if term in fact_blob:
                # Emphasize the specific grounded facts that mention this term.
                for f in grounded:
                    if term in f.value.lower() and f.value not in t.emphasize:
                        t.emphasize.append(f.value)
                emphasized_terms.add(term)
            else:
                t.gaps.append(term)
        return t
    except Exception:  # noqa: BLE001
        return t


# --------------------------------------------------------------------------- #
def is_profile_question(text: str) -> bool:
    """Whether the interviewer's question is about the CANDIDATE's own background
    (→ answer from the Career Graph / prepared library). SEMANTIC-first via the
    existing `profile_question` gate; a cue fallback covers cold start. Never
    raises."""
    try:
        t = (text or "").strip()
        if not t:
            return False
        try:
            from app.semantics import gates
            verdict = gates.matches("profile_question", t)
            if verdict is not None:
                return bool(verdict)
        except Exception:  # noqa: BLE001
            pass
        return bool(re.search(
            r"\b(tell me about yourself|your (?:experience|background|strengths|"
            r"weakness|projects?)|walk me through|why did you leave|a time you)\b",
            t, re.I))
    except Exception:  # noqa: BLE001
        return False


__all__ = ["enabled", "SourceSpan", "CareerFact", "CareerGraph",
           "build_career_graph", "Tailoring", "tailor_to_jd",
           "is_profile_question"]
