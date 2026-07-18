"""
Section-aware resume chunker.

Resumes have a near-universal section layout (Summary, Experience, Skills,
Projects, Education). Splitting by heading first, then by size, keeps
semantic units intact — far better recall than naive fixed-window chunks.

Returns a list of `Chunk` dataclasses with both the text and the inferred
section label, so the retriever can filter or boost specific sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Headings we recognise. Order matters — the longer phrases come first so
# "Work Experience" matches before "Experience" inside the same line.
_HEADINGS = [
    ("Work Experience", "experience"),
    ("Professional Experience", "experience"),
    ("Experience", "experience"),
    ("Employment History", "experience"),
    ("Career History", "experience"),
    ("Technical Skills", "skills"),
    ("Skills", "skills"),
    ("Core Competencies", "skills"),
    ("Projects", "projects"),
    ("Notable Projects", "projects"),
    ("Selected Projects", "projects"),
    ("Education", "education"),
    ("Academic Background", "education"),
    ("Certifications", "certifications"),
    ("Licenses", "certifications"),
    ("Publications", "publications"),
    ("Summary", "summary"),
    ("Professional Summary", "summary"),
    ("Profile", "summary"),
    ("Objective", "summary"),
    ("Contact", "contact"),
    ("Awards", "awards"),
    ("Languages", "languages"),
]


@dataclass
class Chunk:
    """A retrievable slice of resume text with provenance."""
    text: str
    section: str | None
    position: int


def chunk_resume(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Chunk]:
    """Split a resume into section-tagged, size-bounded chunks.

    Algorithm:
      1. Walk the document line-by-line, detecting headings.
      2. Accumulate lines under the current section.
      3. Within each section, split into windows of ~chunk_size characters
         with `chunk_overlap` characters of overlap at the boundaries.
    """
    sections = _split_into_sections(text)
    out: list[Chunk] = []
    pos = 0
    for section_label, section_text in sections:
        for window in _window_text(section_text, chunk_size, chunk_overlap):
            out.append(Chunk(text=window, section=section_label, position=pos))
            pos += 1
    return out


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Walk the text and group lines under their detected section label."""
    lines = text.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_label: str | None = None
    current_lines: list[str] = []

    for line in lines:
        label = _detect_heading(line)
        if label is not None:
            if current_lines:
                sections.append((current_label, current_lines))
            current_label = label
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_label, current_lines))

    return [(label, "\n".join(ls).strip()) for label, ls in sections if "\n".join(ls).strip()]


def _detect_heading(line: str) -> str | None:
    """Return the section label if `line` looks like a heading, else None."""
    stripped = line.strip().rstrip(":").rstrip()
    if not stripped or len(stripped) > 50:
        return None
    # Section headings are usually short, often Title Case or ALL CAPS,
    # and consist mostly of letters.
    letters_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / max(len(stripped), 1)
    if letters_ratio < 0.7:
        return None
    lower = stripped.lower()
    for heading, label in _HEADINGS:
        if lower == heading.lower():
            return label
    # Heuristic: SHORT ALL-CAPS lines are almost always headings.
    if stripped.isupper() and len(stripped.split()) <= 4:
        # Try fuzzy match against known headings before falling back to "other".
        for heading, label in _HEADINGS:
            if heading.lower() in lower:
                return label
    return None


def _window_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Yield overlapping fixed-size windows from `text`.

    Prefers paragraph boundaries when one is near the end of a window, so
    chunks tend to land on natural breaks. Falls back to hard cuts when no
    paragraph break is within `overlap` characters of the target end.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        if end < n:
            # Prefer a nearby blank-line break.
            window_tail = text[max(end - overlap, i) : end]
            br = window_tail.rfind("\n\n")
            if br != -1:
                end = max(end - overlap, i) + br
        chunk = text[i:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return out


# Used by retrievers/orchestrator that need a single piece of text from a
# list of Chunk objects (e.g. when stitching retrieved hits into a prompt).
def chunks_to_text(chunks: list[Chunk]) -> str:
    """Re-stitch chunks into a readable block, preserving document order."""
    ordered = sorted(chunks, key=lambda c: c.position)
    parts = []
    last_section: str | None = None
    for ch in ordered:
        if ch.section != last_section and ch.section:
            parts.append(f"\n## {ch.section.title()}\n")
            last_section = ch.section
        parts.append(ch.text)
    return "\n\n".join(p for p in parts if p.strip())
