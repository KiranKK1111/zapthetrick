"""
Domain-context builder for STT question repair.

The live question-cleaner (`app/question_detection/agent.py`) normally distrusts
corrections — it will not swap a real word ("spring") or introduce a proper noun
("kube-proxy"), because without context those "fixes" are usually hallucinations.
But an interview HAS a domain: the candidate's resume skills/tech, the target
role + job description, and the topics already discussed. Given that domain, the
right correction is obvious — "spring" -> "string" in a Java coding question,
"Q proxy" -> "kube-proxy" in a Kubernetes session.

`build_domain` assembles that domain into a compact vocabulary + one-line
summary the cleaner can use to repair mis-transcriptions with confidence.
Deterministic + fail-open: never raises; empty context on any error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_MAX_VOCAB = 40           # cap the prompt block size
_MIN_LEN, _MAX_LEN = 2, 40


@dataclass
class DomainContext:
    vocab: list[str] = field(default_factory=list)   # domain terms (deduped)
    role: str = ""
    topics: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.vocab or self.role or self.topics)

    def prompt_block(self) -> str:
        """A compact block for the cleaner prompt ('' when there's no context)."""
        if self.empty:
            return ""
        parts: list[str] = []
        if self.role:
            parts.append(f"- Target role: {self.role}")
        if self.topics:
            parts.append("- Topics discussed so far: " + ", ".join(self.topics[:8]))
        if self.vocab:
            parts.append("- Technologies in play (the interview is about these): "
                         + ", ".join(self.vocab[:_MAX_VOCAB]))
        return ("This interview's DOMAIN (use it to repair the transcript — a "
                "word that is phonetically close to one of these terms, or an "
                "obvious component of them, is very likely what was said):\n"
                + "\n".join(parts))


def _split_terms(v) -> list[str]:
    """Coerce a skills/tech field (list or comma/slash string) into terms."""
    out: list[str] = []
    if isinstance(v, list):
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
    elif isinstance(v, str):
        for s in re.split(r"[,/|;]", v):
            s = s.strip()
            if s:
                out.append(s)
    return out


def _jd_terms(jd_text: str) -> list[str]:
    """Best-effort tech keywords from a job description. Reuses the org JD skill
    extractor when available; falls back to capitalized / techy tokens."""
    jd = (jd_text or "").strip()
    if not jd:
        return []
    try:
        from app.live import org as _org
        skills = _org.build_org("", jd, "").jd_skills
        if skills:
            return [str(s) for s in skills]
    except Exception:  # noqa: BLE001
        pass
    # Fallback: tokens that look like tech (Capitalized, ALLCAPS, or with a
    # digit/dot/dash — e.g. "Kafka", "CI/CD", "K8s", "gRPC", "PostgreSQL").
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9.+#/-]{1,29}", jd):
        if (tok[0].isupper() or tok.isupper()
                or any(ch.isdigit() for ch in tok) or "/" in tok):
            out.append(tok)
    return out[:60]


def build_domain(profile: dict | None = None, org_ctx: dict | None = None,
                 recent: list[str] | None = None) -> DomainContext:
    """Assemble the interview's domain context from the resume profile, the org
    intake (role + JD), and recent question topics. Never raises → empty."""
    dc = DomainContext()
    try:
        seen: set[str] = set()

        def _add(term: str) -> None:
            t = (term or "").strip().strip(".,")
            k = t.lower()
            if t and k not in seen and _MIN_LEN <= len(t) <= _MAX_LEN:
                seen.add(k)
                dc.vocab.append(t)

        prof = profile if isinstance(profile, dict) else {}
        for s in _split_terms(prof.get("skills")):
            _add(s)
        for proj in (prof.get("projects") or []):
            if isinstance(proj, dict):
                for t in _split_terms(proj.get("tech")):
                    _add(t)
            elif isinstance(proj, str):
                # "Foo — built with React/Node" — pull the techy tail tokens.
                for t in _jd_terms(proj):
                    _add(t)

        oc = org_ctx if isinstance(org_ctx, dict) else {}
        dc.role = str(oc.get("job_role") or "").strip()[:60]
        for t in _jd_terms(str(oc.get("job_description") or "")):
            _add(t)

        if recent:
            # Short recent utterances double as topic hints for the domain.
            topics: list[str] = []
            for q in recent[-4:]:
                q = str(q).strip()
                if q:
                    topics.append(q[:60])
            dc.topics = topics
        return dc
    except Exception:  # noqa: BLE001
        return DomainContext()
