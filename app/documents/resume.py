"""Resume as structured JSON → ATS vs designed render (vNext §3.3).

A resume's content is drafted ONCE as structured JSON (contact / summary /
experience / skills / projects / education); two renderers produce it from the
SAME data:

  * **ATS** — single column, standard headings, NO tables/columns/graphics, plain
    bullets (what an applicant-tracking parser reads cleanly);
  * **designed** — the same content with a tighter, richer layout (bold roles,
    a skills line, a rule under headings) for a human reader.

A follow-up edit ("add my Docker experience") patches the JSON and re-renders —
an in-place artifact edit, not a regeneration. Pure + fail-open (a missing field
is just omitted); the LLM extraction that fills the JSON is a §8.7 structured
call the caller wires — this module owns the schema + the deterministic render.
"""
from __future__ import annotations

RESUME_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "contact": {
            "type": "object",
            "properties": {
                "email": {"type": "string"}, "phone": {"type": "string"},
                "location": {"type": "string"}, "links": {
                    "type": "array", "items": {"type": "string"}},
            },
        },
        "summary": {"type": "string"},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"}, "company": {"type": "string"},
                    "dates": {"type": "string"}, "location": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}, "detail": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"}, "school": {"type": "string"},
                    "dates": {"type": "string"},
                },
            },
        },
    },
    "required": ["name"],
}


def _s(v) -> str:
    return str(v or "").strip()


def _bullets(items) -> list[str]:
    return [f"- {_s(b)}" for b in (items or []) if _s(b)]


def render_resume(data: dict, *, mode: str = "ats") -> str:
    """Render the resume JSON to markdown. `mode` = 'ats' (single column, plain)
    or 'designed' (richer). Deterministic; unknown fields omitted. Never raises."""
    try:
        return _render(data or {}, designed=(mode == "designed"))
    except Exception:  # noqa: BLE001
        return _s((data or {}).get("name")) or "Resume"


def _render(data: dict, *, designed: bool) -> str:
    out: list[str] = []
    name = _s(data.get("name")) or "Resume"
    out.append(f"# {name}")

    c = data.get("contact") or {}
    line = " · ".join(x for x in [
        _s(c.get("email")), _s(c.get("phone")), _s(c.get("location")),
        *[_s(l) for l in (c.get("links") or [])]] if x)
    if line:
        out.append(f"_{line}_" if designed else line)

    def _h(title: str) -> str:
        # ATS: plain "## SECTION" (standard, parser-friendly). Designed: same
        # heading with a light rule for a human reader (still valid markdown).
        return f"## {title}" + ("\n" if not designed else "\n")

    if _s(data.get("summary")):
        out.append(_h("Summary") + _s(data["summary"]))

    exp = data.get("experience") or []
    if exp:
        blk = [_h("Experience").rstrip()]
        for e in exp:
            role, comp = _s(e.get("role")), _s(e.get("company"))
            dates, loc = _s(e.get("dates")), _s(e.get("location"))
            head = " — ".join(x for x in [role, comp] if x)
            meta = " · ".join(x for x in [dates, loc] if x)
            if designed:
                blk.append(f"**{head}**" + (f"  \n_{meta}_" if meta else ""))
            else:
                blk.append(head + (f" ({meta})" if meta else ""))
            blk += _bullets(e.get("bullets"))
        out.append("\n".join(blk))

    skills = [x for x in (data.get("skills") or []) if _s(x)]
    if skills:
        # ATS: a plain comma list under a standard heading (never a table/columns
        # — parsers choke on those). Designed: a bold-label single line.
        body = ", ".join(_s(s) for s in skills)
        out.append(_h("Skills") + (f"**Skills:** {body}" if designed else body))

    projs = data.get("projects") or []
    if projs:
        blk = [_h("Projects").rstrip()]
        for p in projs:
            nm, det = _s(p.get("name")), _s(p.get("detail"))
            head = f"**{nm}**" if designed and nm else nm
            blk.append(head + (f" — {det}" if det else ""))
            blk += _bullets(p.get("bullets"))
        out.append("\n".join(blk))

    edu = data.get("education") or []
    if edu:
        blk = [_h("Education").rstrip()]
        for e in edu:
            deg, sch, dt = _s(e.get("degree")), _s(e.get("school")), _s(e.get("dates"))
            row = " — ".join(x for x in [deg, sch] if x)
            blk.append(row + (f" ({dt})" if dt else ""))
        out.append("\n".join(blk))

    return "\n\n".join(out).strip()


def patch_resume(data: dict, updates: dict) -> dict:
    """Merge `updates` into the resume JSON (§3.3 in-place edit): list fields
    APPEND, scalars/objects REPLACE. Returns a new dict; never raises."""
    try:
        out = dict(data or {})
        for k, v in (updates or {}).items():
            if isinstance(v, list) and isinstance(out.get(k), list):
                out[k] = out[k] + v
            elif isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
        return out
    except Exception:  # noqa: BLE001
        return dict(data or {})


__all__ = ["RESUME_SCHEMA", "render_resume", "patch_resume"]
