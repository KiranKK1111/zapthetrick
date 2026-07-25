"""Requirement rubric for document generation (vNext §3.3).

Extracts a document request's EXPLICIT asks into a small checklist the visual-QA
pass (§3.3) then verifies the rendered output against:

    {type, pages, style, must_include[], ats}

Deterministic + cheap by design (the visual-QA VLM does the heavy reading) — a
few high-precision cues, not an LLM call, so it never adds latency or a failure
mode to the generation path. Absent cues stay None/empty (the checker only
asserts what the user actually asked for).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "single": 1}

_PAGES_RE = re.compile(
    r"\b(one|two|three|four|single|\d+)[\s-]*page\b", re.I)
_ATS_RE = re.compile(r"\bats\b|applicant[\s-]*tracking", re.I)
_STYLES = ("modern", "minimal", "minimalist", "classic", "professional",
           "creative", "elegant", "clean", "traditional", "bold", "simple")
_TYPE_NOUNS = ("resume", "cv", "curriculum vitae", "cover letter", "report",
               "invoice", "proposal", "presentation", "letter", "essay",
               "article", "white paper", "newsletter", "brochure")
# "include a skills section", "with an education section", "must have a summary"
_INCLUDE_VERB_RE = re.compile(
    r"\b(include|with|must\s+have|should\s+have|add|containing|featuring)\b", re.I)
# Every "a/an/the <phrase> section" — the article anchors the capture to the
# section noun (not back to the verb). Harvested only when an include-verb is
# present, so a multi-section ask ("include a skills section and a projects
# section") captures both, without a stray "section" mention on a plain answer.
_SECTION_RE = re.compile(
    r"\b(?:a|an|the)\s+([a-z][a-z&/-]+(?:\s+[a-z][a-z&/-]+){0,2})\s+section\b",
    re.I)


@dataclass
class Rubric:
    type: str | None = None
    pages: int | None = None
    style: str | None = None
    must_include: list[str] = field(default_factory=list)
    ats: bool | None = None

    def is_empty(self) -> bool:
        return not (self.type or self.pages or self.style
                    or self.must_include or self.ats)

    def as_dict(self) -> dict:
        return {"type": self.type, "pages": self.pages, "style": self.style,
                "must_include": list(self.must_include), "ats": self.ats}

    def checklist_text(self) -> str:
        """A compact, human-legible checklist for the visual-QA prompt."""
        parts = []
        if self.type:
            parts.append(f"type: {self.type}")
        if self.pages:
            parts.append(f"pages: exactly {self.pages}")
        if self.style:
            parts.append(f"style: {self.style}")
        if self.ats:
            parts.append("ATS-friendly (single column, no tables/graphics, "
                         "standard headings)")
        if self.must_include:
            parts.append("must include sections: " + ", ".join(self.must_include))
        return "; ".join(parts)


def extract_rubric(request: str) -> Rubric:
    """Parse the explicit requirements from a document request. Never raises."""
    t = (request or "").strip()
    if not t:
        return Rubric()
    low = t.lower()

    doc_type = None
    for noun in _TYPE_NOUNS:
        if noun in low:
            doc_type = noun
            break

    pages = None
    m = _PAGES_RE.search(low)
    if m:
        tok = m.group(1).lower()
        pages = _WORD_NUM.get(tok) or (int(tok) if tok.isdigit() else None)

    style = next((s for s in _STYLES if re.search(rf"\b{s}\b", low)), None)
    if style == "minimalist":
        style = "minimal"

    ats = True if _ATS_RE.search(low) else None

    must = []
    if _INCLUDE_VERB_RE.search(low):
        for mm in _SECTION_RE.finditer(t):
            sec = mm.group(1).strip().lower()
            if sec and sec not in must:
                must.append(sec)

    return Rubric(type=doc_type, pages=pages, style=style,
                  must_include=must, ats=ats)


__all__ = ["Rubric", "extract_rubric"]
