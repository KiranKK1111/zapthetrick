"""Document design templates — resume / doc layouts (roadmap Phase 4 #21).

The generators can render *content*; this adds the *design* layer the roadmap
flags as missing: a set of named, ATS-friendly layouts (typography, spacing,
section order, heading style) that structured content is rendered into. Every
template is deliberately SINGLE-COLUMN and graphic-free so ATS parsers read it
cleanly — the design lives in typographic rhythm, not columns or tables.

`render_resume(sections, template)` produces clean, consistent Markdown that the
existing `generators.py` renders to docx/PDF. Pure + deterministic + tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Template:
    id: str
    name: str
    description: str
    # Heading style: "upper" (SKILLS) | "title" (Skills) | "rule" (Skills + ---)
    heading: str = "title"
    # Section order (missing sections are skipped).
    order: list[str] = field(default_factory=lambda: [
        "summary", "experience", "projects", "skills", "education"])
    bullet: str = "-"
    ats_safe: bool = True


TEMPLATES: dict[str, Template] = {
    "classic": Template(
        id="classic", name="Classic",
        description="Serif-friendly, title-case headings with rules — timeless.",
        heading="rule",
        order=["summary", "experience", "education", "skills", "projects"]),
    "modern": Template(
        id="modern", name="Modern",
        description="Clean uppercase section labels, tight rhythm — contemporary.",
        heading="upper",
        order=["summary", "skills", "experience", "projects", "education"]),
    "compact": Template(
        id="compact", name="Compact",
        description="One-page density: title-case headings, no rules.",
        heading="title",
        order=["summary", "experience", "skills", "projects", "education"]),
}

_SECTION_LABELS = {
    "summary": "Summary", "experience": "Experience", "education": "Education",
    "skills": "Skills", "projects": "Projects",
}


def list_templates() -> list[dict]:
    return [{"id": t.id, "name": t.name, "description": t.description,
             "ats_safe": t.ats_safe} for t in TEMPLATES.values()]


def _heading(label: str, style: str) -> str:
    if style == "upper":
        return f"## {label.upper()}"
    if style == "rule":
        return f"## {label}\n"  # generators draw a rule under an h2
    return f"## {label}"


def _render_entries(section: str, items: list, bullet: str) -> list[str]:
    """Render a list section. Items may be strings or dicts with common keys."""
    out: list[str] = []
    for it in items or []:
        if isinstance(it, str):
            out.append(f"{bullet} {it}")
            continue
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or it.get("role") or it.get("name")
                 or it.get("degree") or "").strip()
        org = (it.get("company") or it.get("org") or it.get("school")
               or it.get("institution") or "").strip()
        when = (it.get("dates") or it.get("date") or it.get("period") or "").strip()
        head = " · ".join(p for p in (title, org) if p)
        if head and when:
            out.append(f"**{head}** — {when}")
        elif head:
            out.append(f"**{head}**")
        for b in (it.get("bullets") or it.get("highlights") or []):
            if str(b).strip():
                out.append(f"{bullet} {str(b).strip()}")
        desc = (it.get("description") or "").strip()
        if desc:
            out.append(desc)
    return out


def render_resume(sections: dict, template: str = "classic") -> str:
    """Render structured resume [sections] into ATS-friendly Markdown.

    `sections` keys: name, contact, summary (str), experience/projects/education
    (list of dicts or strings), skills (list[str] or str). Unknown template → classic.
    """
    tpl = TEMPLATES.get((template or "").lower(), TEMPLATES["classic"])
    lines: list[str] = []

    name = str(sections.get("name") or "").strip()
    contact = sections.get("contact")
    if name:
        lines.append(f"# {name}")
    if contact:
        c = " · ".join(str(x).strip() for x in contact) if isinstance(contact, list) \
            else str(contact).strip()
        if c:
            lines.append(c)
    if lines:
        lines.append("")

    # The template's own sections first (in its order), then any EXTRA sections
    # the caller supplied ("Certifications", "Publications", …) in their given
    # order — a template re-lays-out a resume, it never drops its content.
    extras = [k for k in sections
              if k not in tpl.order and k not in _NON_SECTION_KEYS]
    for key in list(tpl.order) + extras:
        val = sections.get(key)
        if not val:
            continue
        lines.append(_heading(_SECTION_LABELS.get(key, key.title()), tpl.heading))
        if key == "summary":
            lines.append(str(val).strip())
        elif key == "skills":
            skills = val if isinstance(val, list) else [s for s in str(val).split(",")]
            skills = [str(s).strip() for s in skills if str(s).strip()]
            lines.append(f"{tpl.bullet} " + f"\n{tpl.bullet} ".join(skills)
                         if len(skills) > 6 else " · ".join(skills))
        elif isinstance(val, str):
            lines.append(val.strip())
        else:
            lines.extend(_render_entries(key, val, tpl.bullet))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ── Markdown → structured sections (so an EXISTING resume can be re-templated) ─
_NON_SECTION_KEYS = ("name", "contact")

# Heading cues → canonical section key. Substring match, so "Professional
# Summary" / "Work Experience" / "Technical Skills" all land correctly.
_HEADING_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("summary", ("summary", "objective", "profile", "about me")),
    ("experience", ("experience", "employment", "work history", "career")),
    ("education", ("education", "academic", "qualification")),
    ("skills", ("skill", "technolog", "competenc", "stack")),
    ("projects", ("project", "portfolio")),
)


def section_key(heading: str) -> str | None:
    """Canonical resume-section key for a heading, or None when it's not one of
    the known sections (an extra section — preserved verbatim)."""
    h = (heading or "").strip().lower()
    if not h:
        return None
    for key, cues in _HEADING_CUES:
        if any(c in h for c in cues):
            return key
    return None


def _entries_from_blocks(blocks: list) -> list:
    """Group a section's blocks into resume ENTRIES: a paragraph opens an entry
    (its title line — "Role · Company — 2019–2022") and the list that follows
    supplies that entry's bullets."""
    from app.documents.model import ListBlock, Paragraph

    entries: list[dict] = []
    cur: dict | None = None
    for b in blocks:
        if isinstance(b, Paragraph):
            cur = {"title": (b.text or "").strip(), "bullets": []}
            entries.append(cur)
        elif isinstance(b, ListBlock):
            if cur is None:
                cur = {"title": "", "bullets": []}
                entries.append(cur)
            cur["bullets"].extend(str(i).strip() for i in b.items if str(i).strip())
    return [e for e in entries if e["title"] or e["bullets"]]


def sections_from_markdown(content: str) -> dict:
    """Parse resume Markdown back into the structured ``sections`` dict
    :func:`render_resume` consumes.

    Returns ``{}`` when the content isn't a resume we can safely re-lay-out —
    no recognizable section, or a block kind a resume layout can't carry
    (table / code / image). The caller then leaves the content ALONE: a template
    must never silently drop a user's content.
    """
    from app.documents.model import (
        ListBlock, Paragraph, markdown_to_model,
    )

    model = markdown_to_model(content or "")
    out: dict = {}
    known = 0
    for sec in model.sections:
        if any(not isinstance(b, (Paragraph, ListBlock)) for b in sec.blocks):
            return {}                      # unsupported block → don't touch it
        heading = (sec.heading or "").strip()
        if sec.level == 1 and "name" not in out:
            # The title line: "# Jane Doe" + an optional contact paragraph.
            out["name"] = heading
            for b in sec.blocks:
                if isinstance(b, Paragraph) and (b.text or "").strip():
                    out["contact"] = [p.strip() for p in
                                      re.split(r"\s*[·|]\s*", b.text.strip())
                                      if p.strip()]
                    break
            continue
        if not heading:
            continue                       # lead prose before the name — drop
        key = section_key(heading)
        if key is None:
            key = heading                  # extra section, preserved by key
        else:
            known += 1
        if key == "summary":
            out[key] = "\n\n".join(b.text.strip() for b in sec.blocks
                                   if isinstance(b, Paragraph) and b.text.strip())
        elif key == "skills":
            items: list[str] = []
            for b in sec.blocks:
                if isinstance(b, ListBlock):
                    items.extend(str(i).strip() for i in b.items if str(i).strip())
                elif isinstance(b, Paragraph) and b.text.strip():
                    items.extend(p.strip() for p in
                                 re.split(r"\s*[,·|]\s*", b.text.strip())
                                 if p.strip())
            out[key] = items
        else:
            out[key] = _entries_from_blocks(sec.blocks)
    if not known:
        return {}                          # not a resume → caller keeps content
    return out


def apply_template(content: str, template: str | None) -> str:
    """Re-lay-out resume Markdown through a named design template.

    FAIL-OPEN and identity-preserving by default: no template, an unknown
    template, content that isn't a resume, or ANY error → ``content`` is
    returned byte-for-byte unchanged. Only a clean parse of a real resume into a
    known template produces re-templated Markdown.
    """
    try:
        name = (template or "").strip().lower()
        if not name or name not in TEMPLATES:
            return content
        sections = sections_from_markdown(content or "")
        if not sections:
            return content
        rendered = render_resume(sections, name)
        return rendered if rendered.strip() else content
    except Exception:  # noqa: BLE001 — a design nicety never breaks an export
        return content


__all__ = ["Template", "TEMPLATES", "list_templates", "render_resume",
           "section_key", "sections_from_markdown", "apply_template"]
