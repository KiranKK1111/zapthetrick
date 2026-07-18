"""Document structure enhancers — Phase 4 of the Document Generation roadmap.

Model → model transforms that make a document navigable and self-explaining
(DocuementGeneration.md #4 TOC-from-structure, #15 auto-glossary, #16 reusable
blocks). These operate on the Phase-1 DocumentModel and only ADD sections, so
every existing exporter renders them for free — no renderer changes.

  * ``build_toc(model)``      — a Table of Contents from the section headings.
  * ``build_glossary(model)`` — defines the known technical terms that appear.
  * ``enrich(model, ...)``    — inserts the TOC (after the lead) + glossary (at
                                the end). Off by default at the render layer
                                (``cfg.documents.auto_structure``), so output is
                                unchanged until enabled.
"""
from __future__ import annotations

import copy
import re
from dataclasses import replace as _replace

from app.documents.model import (
    CodeBlock, Diagram, DocumentModel, Heading, Image, ListBlock, Paragraph,
    Section, Table,
)

# Curated glossary of common technical terms (title-cased key → definition).
# Kept small + high-signal; matched case-insensitively on a word boundary.
_GLOSSARY: dict[str, str] = {
    "Kafka": "A distributed event-streaming platform for high-throughput "
             "publish/subscribe messaging.",
    "Redis": "An in-memory data store used as a cache, message broker, and "
             "lightweight database.",
    "JWT": "JSON Web Token — a compact, signed token for stateless "
           "authentication.",
    "OAuth": "An open standard for delegated authorization.",
    "OAuth2": "The 2.0 revision of the OAuth delegated-authorization framework.",
    "RAG": "Retrieval-Augmented Generation — grounding an LLM's answer on "
           "retrieved documents.",
    "REST": "Representational State Transfer — an HTTP-based API style.",
    "gRPC": "A high-performance RPC framework built on HTTP/2 and protocol "
            "buffers.",
    "GraphQL": "A query language and runtime for APIs that lets clients request "
               "exactly the data they need.",
    "SQL": "Structured Query Language for relational databases.",
    "NoSQL": "A family of non-relational databases (document, key-value, "
             "column, graph).",
    "Docker": "A platform for packaging applications into portable containers.",
    "Kubernetes": "An orchestrator for deploying, scaling, and operating "
                  "containerized applications.",
    "CI/CD": "Continuous Integration / Continuous Delivery — automated build, "
             "test, and deploy pipelines.",
    "API": "Application Programming Interface — a contract for software to "
           "interact.",
    "SLA": "Service-Level Agreement — a committed target for availability or "
           "performance.",
    "TLS": "Transport Layer Security — encryption for data in transit.",
    "CDN": "Content Delivery Network — geographically distributed edge caches.",
    "ORM": "Object-Relational Mapper — maps database rows to objects.",
    "LLM": "Large Language Model.",
}
_MAX_TOC_ENTRIES = 60


def build_toc(model: DocumentModel,
              title: str = "Table of Contents") -> Section | None:
    """A TOC section listing the document's headings, indented by level. Returns
    None when there aren't enough headings to warrant one."""
    heads = model.headings()
    if len(heads) < 3:
        return None
    base = min(lvl for lvl, _ in heads) or 1
    items = [("  " * max(0, lvl - base)) + head
             for lvl, head in heads[:_MAX_TOC_ENTRIES]]
    return Section(heading=title, level=1,
                   blocks=[ListBlock(items=items, ordered=False)])


def _all_text(model: DocumentModel) -> str:
    parts: list[str] = []
    for sec in model.sections:
        if sec.heading:
            parts.append(sec.heading)
        for b in sec.blocks:
            if isinstance(b, Paragraph):
                parts.append(b.text)
            elif isinstance(b, ListBlock):
                parts.extend(b.items)
            elif isinstance(b, CodeBlock):
                parts.append(b.code)
    return "\n".join(parts)


def build_glossary(model: DocumentModel,
                   title: str = "Glossary") -> Section | None:
    """Define the known technical terms that actually appear in the document.
    Returns None if none are found."""
    text = _all_text(model)
    found: list[str] = []
    for term in _GLOSSARY:
        # Escape + word boundaries; '/' and '2' in CI/CD / OAuth2 are literal.
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.I):
            found.append(term)
    if not found:
        return None
    items = [f"{term}: {_GLOSSARY[term]}" for term in found]
    return Section(heading=title, level=1,
                   blocks=[ListBlock(items=items, ordered=False)])


def _lead_offset(sections: list[Section]) -> int:
    """Insert index for the TOC: after the lead/title section (so the TOC sits
    below the title + intro, above the first real section)."""
    if sections and (sections[0].heading == "" or sections[0].level <= 1):
        return 1
    return 0


# ── figure/table numbering (#9) ─────────────────────────────────────────────
_FIGURES = (Image, Diagram)


def number_exhibits(model: DocumentModel, *, table_label: str = "Table",
                    figure_label: str = "Figure") -> DocumentModel:
    """Assign 'Table N' / 'Figure N' captions to tables and figures (images +
    diagrams) in document order. Idempotent (already-numbered captions kept).
    ``table_label``/``figure_label`` localize the prefix. Returns a copy."""
    m = copy.deepcopy(model)
    t = f = 0
    tpfx, fpfx = f"{table_label} ", f"{figure_label} "
    for b in m.iter_blocks():
        if isinstance(b, Table):
            t += 1
            if not (b.caption or "").startswith(tpfx):
                base = (b.caption or "").strip()
                b.caption = f"{tpfx}{t}" + (f": {base}" if base else "")
        elif isinstance(b, _FIGURES):
            f += 1
            if not (b.caption or "").startswith(fpfx):
                base = (b.caption or (b.alt if isinstance(b, Image) else "")
                        ).strip()
                b.caption = f"{fpfx}{f}" + (f": {base}" if base else "")
    return m


def build_list_of_exhibits(
        model: DocumentModel,
        title: str = "List of Figures & Tables") -> Section | None:
    """A list of every numbered figure + table. None if there are < 2 exhibits
    (a list wouldn't earn its place)."""
    items = [b.caption for b in model.iter_blocks()
             if isinstance(b, (Table,) + _FIGURES) and getattr(b, "caption", "")]
    if len(items) < 2:
        return None
    return Section(heading=title, level=1,
                   blocks=[ListBlock(items=items, ordered=False)])


# ── auto-diagram detection (#7) ─────────────────────────────────────────────
_ARROW_RE = re.compile(r"\s*(?:-->|->|→|⟶|➜|⇒)\s*")
_NODE_RE = re.compile(r"^[\w][\w \-/().]{0,34}$")


def _is_flow(text: str) -> bool:
    parts = [p.strip() for p in _ARROW_RE.split(text) if p.strip()]
    return len(parts) >= 3 and all(_NODE_RE.match(p) for p in parts)


def _to_mermaid(text: str) -> str:
    parts = [p.strip().rstrip(".") for p in _ARROW_RE.split(text) if p.strip()]
    lines = ["flowchart LR"]
    for i in range(len(parts) - 1):
        lines.append(f"    n{i}[{parts[i]}] --> n{i + 1}[{parts[i + 1]}]")
    return "\n".join(lines)


def detect_diagrams(model: DocumentModel) -> DocumentModel:
    """Turn a paragraph that describes a linear flow with arrows
    ("User -> Gateway -> Service", "A → B → C") into a Mermaid flowchart Diagram.
    Conservative: only clean arrow chains of >= 3 short node-like segments, so
    ordinary prose is never mangled. Returns a copy."""
    m = copy.deepcopy(model)
    for sec in m.sections:
        out = []
        for b in sec.blocks:
            if (isinstance(b, Paragraph) and _ARROW_RE.search(b.text)
                    and _is_flow(b.text)):
                out.append(Diagram(source=_to_mermaid(b.text),
                                   diagram_kind="mermaid"))
            else:
                out.append(b)
        sec.blocks = out
    return m


# ── smart appendix (#10) ────────────────────────────────────────────────────
_APPENDIX_LANGS = {"yaml", "yml", "json", "sql", "log", "ini", "toml", "xml",
                   "env", "properties", "conf", "dotenv"}
_APPENDIX_MIN_LINES = 30


def _appendix_worthy(b) -> bool:
    return isinstance(b, CodeBlock) and (
        (b.language or "").lower() in _APPENDIX_LANGS
        or (b.code.count("\n") + 1) >= _APPENDIX_MIN_LINES)


def smart_appendix(model: DocumentModel,
                   title: str = "Appendix") -> DocumentModel:
    """Move raw config / log / very-long code blocks into an Appendix section,
    leaving a short reference where each was — keeps the main narrative readable.
    No-op if nothing qualifies or an Appendix already exists. Returns a copy."""
    if any(s.heading.strip().lower() == "appendix" for s in model.sections):
        return model
    m = copy.deepcopy(model)
    moved: list = []
    n = 0
    for sec in m.sections:
        out = []
        for b in sec.blocks:
            if _appendix_worthy(b):
                n += 1
                label = f"{(b.language or 'code').upper()} listing {n}"
                moved.append(Heading(text=label, level=3))
                moved.append(b)
                out.append(Paragraph(text=f"(See Appendix — {label}.)"))
            else:
                out.append(b)
        sec.blocks = out
    if not moved:
        return m
    m.sections.append(Section(heading=title, level=1, blocks=moved))
    return m


def _labels_for(lang: str) -> dict:
    """Localized furniture labels (Phase 7) for ``lang`` — English defaults on
    blank/unknown/error, so the default path is byte-identical."""
    try:
        from app.documents.localization import localize_labels
        return localize_labels(lang)
    except Exception:  # noqa: BLE001
        return {"toc": "Table of Contents", "glossary": "Glossary",
                "appendix": "Appendix", "table": "Table", "figure": "Figure",
                "exhibits": "List of Figures & Tables"}


def enrich(model: DocumentModel, *, toc: bool = True, glossary: bool = True,
           diagrams: bool = True, number: bool = True, appendix: bool = True,
           exhibit_list: bool = True, lang: str = "") -> DocumentModel:
    """Apply the full Phase-4 structure pipeline (each step config-gateable):
    auto-diagram → smart appendix → figure/table numbering, then add a list of
    exhibits, a glossary, and (last, so it sees everything) a TOC. ``lang``
    (Phase 7) localizes every furniture label; blank → English (unchanged).
    Every step returns a copy; the input model is never mutated."""
    labels = _labels_for(lang or getattr(model.metadata, "language", "") or "")
    m = model
    if diagrams:
        m = detect_diagrams(m)
    if appendix:
        m = smart_appendix(m, title=labels["appendix"])
    if number:
        m = number_exhibits(m, table_label=labels["table"],
                            figure_label=labels["figure"])

    existing = {s.heading.lower() for s in m.sections if s.heading}
    sections = list(m.sections)

    if exhibit_list and labels["exhibits"].lower() not in existing:
        lx = build_list_of_exhibits(m, title=labels["exhibits"])
        if lx is not None:
            sections.append(lx)
    if glossary and labels["glossary"].lower() not in existing:
        g = build_glossary(m, title=labels["glossary"])
        if g is not None:
            sections.append(g)
    if toc and labels["toc"].lower() not in existing:
        t = build_toc(DocumentModel(sections=sections),   # includes the above
                      title=labels["toc"])
        if t is not None:
            sections.insert(_lead_offset(sections), t)

    return DocumentModel(metadata=_replace(m.metadata), sections=sections)


def auto_structure_enabled() -> bool:
    """Config gate for applying `enrich` at render time (default OFF → output
    unchanged)."""
    try:
        from app.core.config_loader import get_config
        return bool(getattr(get_config().documents, "auto_structure", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "build_toc", "build_glossary", "number_exhibits", "build_list_of_exhibits",
    "detect_diagrams", "smart_appendix", "enrich", "auto_structure_enabled",
]
