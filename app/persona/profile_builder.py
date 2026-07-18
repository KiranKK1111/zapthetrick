"""
Resume text -> structured profile dict via the configured LLM.

The persona-mode answer endpoint reads the profile dict to ground its
first-person replies in real facts. We accept whatever the LLM emits and
fall back to a minimal {summary: ...} dict if JSON parsing fails — the
app stays usable either way.

Why this is its own module: in the full architecture, the profile is the
"persona" of the candidate. RAG (Phase 2) will add a vector store keyed
off the same upload, but the structured profile is what every prompt
template embeds, so it lives next to voice.py.
"""
import json
import logging
import re

from app.core.llm_client import LLMError, llm
from app.core.prompt import fill

_log = logging.getLogger(__name__)

# Keys the answer endpoint expects to find in a profile. Missing values
# become null / [] — the LLM is told this explicitly in the prompt.
PROFILE_KEYS = (
    "name",
    "headline",
    "years_experience",
    "current_role",
    "summary",
    "skills",
    "work_history",
    "education",
    "projects",
)

_EXTRACTION_PROMPT = """You convert resumes into JSON profiles for an interview-prep app.

Read the resume below and return a JSON object with these keys:
- name: full name (string)
- headline: a one-line professional tagline (string)
- years_experience: total years of professional experience (number or string like "5+")
- current_role: most recent job title (string)
- summary: 2-3 sentence professional summary written in third person (string)
- skills: list of technical and soft skills (array of strings)
- work_history: list of jobs, each with role, company, duration, and 2-4 highlights (array of objects)
- education: degrees or certifications (array of strings)
- projects: notable projects with one-line descriptions (array of strings)

If a field is missing from the resume, use null or an empty array. Do not invent facts.
Return ONLY the JSON object, no prose, no markdown fences.

RESUME:
{resume_text}
"""

async def build_profile(resume_text: str) -> dict:
    """Convert resume text into a profile dict — ALWAYS succeeds.

    Strategy: try the LLM extractor, then backfill any missing fields from a
    deterministic regex heuristic. If the provider is unreachable or returns
    unparseable output, we return the heuristic profile alone. This is why the
    resume is now reliably "detected" even when the free-tier LLM rate-limits —
    the candidate's name/email/skills come from the text directly.
    """
    if not resume_text.strip():
        return _fallback_profile("")

    heuristic = _heuristic_profile(resume_text)

    messages = [
        {
            "role": "user",
            "content": fill(_EXTRACTION_PROMPT, resume_text=resume_text),
        },
    ]
    try:
        raw = await llm.chat_json(messages)
    except LLMError as exc:  # provider down / quota / timeout
        _log.info("profile LLM unavailable, using heuristic: %s", exc)
        return heuristic

    parsed = _parse_lenient(raw)
    if parsed is None:
        return heuristic

    # Keep only known keys, then backfill anything the LLM left empty from the
    # heuristic so a partial extraction never drops the basics.
    result = {k: parsed.get(k) for k in PROFILE_KEYS}
    return _merge(result, heuristic)

def _is_empty(v) -> bool:
    return v is None or v == "" or v == [] or v == {}

def _merge(primary: dict, fallback: dict) -> dict:
    """Fill empty values in `primary` with `fallback`'s."""
    out = dict(primary)
    for k in PROFILE_KEYS:
        if _is_empty(out.get(k)) and not _is_empty(fallback.get(k)):
            out[k] = fallback[k]
    return out

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d ()\-.]{7,}\d)(?!\d)")
_SECTION_RE = re.compile(
    r"^\s*(skills?|technical skills|core competencies|technologies)\s*:?\s*$",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(
    r"^\s*(experience|education|projects?|summary|objective|work|employment|"
    r"certifications?|achievements?|contact|profile)\b",
    re.IGNORECASE,
)
_NON_NAME = re.compile(r"[@/\\|#0-9]")

def _looks_like_name(line: str) -> bool:
    """A line that plausibly is a person's name (1-4 capitalized words)."""
    s = line.strip().strip("•*-—– \t")
    if not s or len(s) > 48 or _NON_NAME.search(s):
        return False
    words = s.split()
    if not (1 <= len(words) <= 4):
        return False
    if s.upper() == s and len(s) > 6:  # ALL-CAPS banners like "CURRICULUM VITAE"
        # An all-caps name is fine, but reject obvious section banners.
        if any(w.lower() in {"resume", "curriculum", "vitae", "cv"} for w in words):
            return False
    alpha = [w for w in words if w.replace(".", "").replace("'", "").replace("-", "").isalpha()]
    return len(alpha) == len(words)

def _heuristic_profile(text: str) -> dict:
    """Best-effort structured profile pulled from the raw text with no LLM.

    Detects name (first name-shaped line near the top), email, phone, and a
    skills list (from a "Skills" section). Everything else stays empty — the
    point is to guarantee the candidate is *recognized*, not to be exhaustive.
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    nonempty = [ln for ln in lines if ln.strip()]

    email = (_EMAIL_RE.search(text or "") or [None])
    email = email.group(0) if email else None
    phone_m = _PHONE_RE.search(text or "")
    phone = phone_m.group(1).strip() if phone_m else None

    name = None
    for ln in nonempty[:12]:
        if _looks_like_name(ln):
            name = ln.strip().strip("•*-—– \t")
            break

    # Skills: collect lines under a Skills header until the next section.
    skills: list[str] = []
    for i, ln in enumerate(lines):
        if _SECTION_RE.match(ln):
            for nxt in lines[i + 1 : i + 12]:
                if not nxt.strip():
                    if skills:
                        break
                    continue
                if _HEADER_RE.match(nxt) or _SECTION_RE.match(nxt):
                    break
                parts = re.split(r"[,;|•·]|\s{2,}|\t", nxt)
                skills.extend(p.strip(" -–—•*\t") for p in parts if p.strip(" -–—•*\t"))
            break
    # De-dupe, keep it sane.
    seen: set = set()
    skills = [s for s in skills if len(s) <= 40 and not (s.lower() in seen or seen.add(s.lower()))][:30]

    headline = None
    if name:
        idx = next((i for i, ln in enumerate(nonempty) if ln.strip().startswith(name)), -1)
        if idx != -1 and idx + 1 < len(nonempty):
            cand = nonempty[idx + 1].strip()
            if cand and not _EMAIL_RE.search(cand) and len(cand) <= 80:
                headline = cand

    return {
        "name": name,
        "headline": headline,
        "years_experience": None,
        "current_role": headline,
        "summary": (_clean_summary(nonempty) or (text or "")[:600]) or None,
        "skills": skills,
        "work_history": [],
        "education": [],
        "projects": [],
        "contact": {"email": email, "phone": phone} if (email or phone) else None,
    }

def _clean_summary(nonempty: list[str]) -> str:
    """A short summary: text under a Summary/Objective header, else the top."""
    for i, ln in enumerate(nonempty):
        if re.match(r"^\s*(summary|objective|profile)\b", ln, re.IGNORECASE):
            body = " ".join(nonempty[i + 1 : i + 4]).strip()
            if body:
                return body[:600]
    return ""

def _parse_lenient(text: str) -> dict | None:
    """Try several strategies to pull a JSON object out of LLM output.

    Local models sometimes wrap JSON in ```json fences or prefix it with a
    polite sentence even when asked not to. This handles both, plus the
    last-resort first-`{` / last-`}` slice.
    """
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None

def _fallback_profile(resume_text: str) -> dict:
    """Minimal profile used when LLM extraction fails — keeps the app usable."""
    return {
        "name": None,
        "headline": None,
        "years_experience": None,
        "current_role": None,
        "summary": resume_text[:1000] if resume_text else None,
        "skills": [],
        "work_history": [],
        "education": [],
        "projects": [],
    }
