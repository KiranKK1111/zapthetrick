"""Structured Document Model (IR) — Phase 1 of the Document Generation roadmap.

The document's #1 / #20 recommendation: stop treating a document as "exported
Markdown". A document is a STRUCTURED MODEL — metadata + ordered sections of
typed blocks (headings, paragraphs, lists, tables, code, diagrams, images,
quotes) — and PDF/DOCX/HTML/Markdown are different RENDERINGS of that one model.

This module introduces the model plus:
  * ``markdown_to_model(md, title)`` — parse Markdown into the model (reuses the
    battle-tested ``generators.parse_blocks`` so today's parsing is preserved).
  * ``model_to_markdown(model)`` — canonical serialization (round-trippable at
    the STRUCTURE level; inline emphasis is normalized away, exactly as the
    existing docx/pdf renderers already do via ``parse_blocks``).
  * ``model_to_html(model)`` — a NEW, self-contained HTML renderer that consumes
    the model directly (the first exporter to prove the model-driven path).

Imports of ``generators`` are LAZY (function-local) to avoid an import cycle:
``generators.render_document`` calls into this module for the HTML/model path.
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass, field

# Fenced-code languages that are really DIAGRAMS, not source to run.
_DIAGRAM_LANGS = {"mermaid", "graphviz", "dot", "plantuml"}
_WORDS_PER_MIN = 200


# ── typed blocks ────────────────────────────────────────────────────────────
@dataclass
class Heading:
    text: str
    level: int = 1
    kind: str = "heading"


@dataclass
class Paragraph:
    text: str
    kind: str = "paragraph"


@dataclass
class ListBlock:
    items: list[str] = field(default_factory=list)
    ordered: bool = False
    kind: str = "list"


@dataclass
class CodeBlock:
    code: str
    language: str = ""
    kind: str = "code"


@dataclass
class Table:
    rows: list[list[str]] = field(default_factory=list)  # rows[0] = header
    caption: str = ""     # "Table 1", set by the numbering pass (Phase 4)
    kind: str = "table"


@dataclass
class Quote:
    text: str
    kind: str = "quote"


@dataclass
class Image:
    url: str
    alt: str = ""
    caption: str = ""     # "Figure 1"
    kind: str = "image"


@dataclass
class Diagram:
    source: str
    diagram_kind: str = "mermaid"
    caption: str = ""     # "Figure N"
    kind: str = "diagram"


@dataclass
class Section:
    """A heading and the blocks under it. The lead section (content before the
    first heading) has an empty ``heading`` and ``level`` 0."""
    heading: str = ""
    level: int = 0
    blocks: list = field(default_factory=list)


@dataclass
class Metadata:
    title: str = ""
    author: str = ""
    language: str = "en"
    doc_type: str = ""
    keywords: list[str] = field(default_factory=list)
    reading_time_min: int = 0


@dataclass
class ExportSettings:
    """Branding + presentation applied at render time (Phase 7 #14/#15), without
    touching document content. All optional → no branding when unset."""
    header: str = ""
    footer: str = ""
    logo_url: str = ""
    primary_color: str = ""        # e.g. "#4f46e5"
    confidentiality: str = ""      # e.g. "Confidential — Internal"
    author: str = ""


@dataclass
class DocumentModel:
    metadata: Metadata = field(default_factory=Metadata)
    sections: list[Section] = field(default_factory=list)
    export: ExportSettings = field(default_factory=ExportSettings)

    def iter_blocks(self):
        for sec in self.sections:
            yield from sec.blocks

    def headings(self) -> list[tuple[int, str]]:
        """(level, text) for every section heading — the basis for a TOC."""
        return [(s.level, s.heading) for s in self.sections if s.heading]

    def assets(self) -> list:
        """Every visual asset (images + diagrams) in order — the asset registry
        (Phase 4 #23)."""
        return [b for b in self.iter_blocks()
                if isinstance(b, (Image, Diagram))]


def slug(text: str) -> str:
    """A stable anchor slug for a heading — the id HTML links target."""
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


# ── Markdown → model ────────────────────────────────────────────────────────
def _raw_to_blocks(raw: list[tuple]) -> list:
    """Convert ``parse_blocks`` tuples into typed model blocks, grouping runs of
    list items and promoting diagram code fences to Diagram."""
    out: list = []
    i = 0
    n = len(raw)
    while i < n:
        b = raw[i]
        kind = b[0]
        if kind in ("bullet", "number"):
            ordered = kind == "number"
            items: list[str] = []
            while i < n and raw[i][0] == kind:
                items.append(raw[i][1])
                i += 1
            out.append(ListBlock(items=items, ordered=ordered))
            continue
        if kind == "h":
            out.append(Heading(text=b[2], level=b[1]))
        elif kind == "p":
            out.append(Paragraph(text=b[1]))
        elif kind == "code":
            lang = (b[2] if len(b) > 2 else "").lower()
            if lang in _DIAGRAM_LANGS:
                out.append(Diagram(source=b[1], diagram_kind=lang))
            else:
                out.append(CodeBlock(code=b[1], language=lang))
        elif kind == "quote":
            out.append(Quote(text=b[1]))
        elif kind == "table":
            out.append(Table(rows=b[1]))
        elif kind == "image":
            out.append(Image(url=b[1], alt=b[2]))
        i += 1
    return out


def _group_sections(blocks: list) -> list[Section]:
    sections: list[Section] = []
    cur = Section()  # lead section (no heading yet)
    for b in blocks:
        if isinstance(b, Heading):
            if cur.heading or cur.blocks:
                sections.append(cur)
            cur = Section(heading=b.text, level=b.level)
        else:
            cur.blocks.append(b)
    if cur.heading or cur.blocks:
        sections.append(cur)
    return sections


def _reading_time(blocks: list) -> int:
    words = 0
    for b in blocks:
        if isinstance(b, Paragraph):
            words += len(b.text.split())
        elif isinstance(b, ListBlock):
            words += sum(len(it.split()) for it in b.items)
        elif isinstance(b, Quote):
            words += len(b.text.split())
    return max(1, round(words / _WORDS_PER_MIN)) if words else 0


def _infer_title(sections: list[Section]) -> str:
    for s in sections:
        if s.heading:
            return s.heading
    return ""


def markdown_to_model(content: str, title: str = "") -> DocumentModel:
    """Parse Markdown into the structured DocumentModel."""
    from app.documents.generators import parse_blocks  # lazy: avoid import cycle

    blocks = _raw_to_blocks(parse_blocks(content or ""))
    sections = _group_sections(blocks)
    meta = Metadata(
        title=(title or "").strip() or _infer_title(sections),
        reading_time_min=_reading_time(blocks),
    )
    return DocumentModel(metadata=meta, sections=sections)


# ── model → Markdown (canonical) ────────────────────────────────────────────
def _block_to_md(b) -> list[str]:
    if isinstance(b, Paragraph):
        return [b.text]
    if isinstance(b, Heading):
        return [f"{'#' * max(1, b.level)} {b.text}"]
    if isinstance(b, ListBlock):
        return [f"{i + 1}. {it}" if b.ordered else f"- {it}"
                for i, it in enumerate(b.items)]
    if isinstance(b, CodeBlock):
        return [f"```{b.language}", b.code, "```"]
    if isinstance(b, Diagram):
        out = [f"```{b.diagram_kind}", b.source, "```"]
        if b.caption:
            out.append(f"*{b.caption}*")
        return out
    if isinstance(b, Quote):
        return [f"> {b.text}"]
    if isinstance(b, Image):
        out = [f"![{b.alt}]({b.url})"]
        if b.caption:
            out.append(f"*{b.caption}*")
        return out
    if isinstance(b, Table):
        rows = b.rows or []
        if not rows:
            return []
        width = len(rows[0])
        out = ["| " + " | ".join(rows[0]) + " |",
               "| " + " | ".join(["---"] * width) + " |"]
        for r in rows[1:]:
            out.append("| " + " | ".join(r) + " |")
        if b.caption:
            out.append(f"*{b.caption}*")
        return out
    return []


def model_to_markdown(model: DocumentModel) -> str:
    """Serialize the model back to canonical Markdown (structure-faithful)."""
    lines: list[str] = []
    for sec in model.sections:
        if sec.heading:
            lines.append(f"{'#' * max(1, sec.level)} {sec.heading}")
            lines.append("")
        for b in sec.blocks:
            lines.extend(_block_to_md(b))
            lines.append("")
    return "\n".join(lines).strip() + "\n"


# ── model → HTML (new exporter, model-driven) ───────────────────────────────
_HTML_CSS = (
    "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
    "line-height:1.6;color:#1a1a1a;max-width:52rem;margin:2rem auto;padding:0 "
    "1.25rem}h1,h2,h3,h4{line-height:1.25;margin:1.6em 0 .5em}h1{font-size:1.9rem}"
    "code{background:#f4f4f5;padding:.15em .35em;border-radius:4px;font-size:.9em}"
    "pre{background:#f4f4f5;padding:1rem;border-radius:8px;overflow:auto}"
    "pre code{background:none;padding:0}blockquote{border-left:3px solid #d4d4d8;"
    "margin:1em 0;padding:.2em 1em;color:#52525b}table{border-collapse:collapse;"
    "width:100%;margin:1em 0}th,td{border:1px solid #e4e4e7;padding:.5em .75em;"
    "text-align:left}th{background:#fafafa}img{max-width:100%}"
    ".meta{color:#71717a;font-size:.85rem;margin-bottom:1.5rem}"
    "figure{margin:1em 0}figcaption{color:#71717a;font-size:.85rem;"
    "text-align:center;margin-top:.35em}"
    # Semantic page breaks (#9): keep an exhibit + its caption whole in print.
    "@media print{figure,table,pre{page-break-inside:avoid}"
    "h1,h2,h3{page-break-after:avoid}}"
    "nav.toc a{text-decoration:none}nav.toc a:hover{text-decoration:underline}"
)


def _caption_html(text: str, esc) -> str:
    return f"<figcaption>{esc(text)}</figcaption>" if text else ""


def _block_to_html(b, esc) -> str:
    if isinstance(b, Paragraph):
        return f"<p>{esc(b.text)}</p>"
    if isinstance(b, Heading):
        lvl = min(6, max(1, b.level))
        return f'<h{lvl} id="{slug(b.text)}">{esc(b.text)}</h{lvl}>'
    if isinstance(b, ListBlock):
        tag = "ol" if b.ordered else "ul"
        items = "".join(f"<li>{esc(it)}</li>" for it in b.items)
        return f"<{tag}>{items}</{tag}>"
    if isinstance(b, CodeBlock):
        cls = f' class="language-{esc(b.language)}"' if b.language else ""
        return f"<pre><code{cls}>{esc(b.code)}</code></pre>"
    if isinstance(b, Diagram):
        # Rendered client-side / by a later diagram pass; keep the source visible.
        return (f'<figure><pre class="diagram diagram-{esc(b.diagram_kind)}">'
                f"<code>{esc(b.source)}</code></pre>"
                f"{_caption_html(b.caption, esc)}</figure>")
    if isinstance(b, Quote):
        return f"<blockquote>{esc(b.text)}</blockquote>"
    if isinstance(b, Image):
        return (f'<figure><img src="{esc(b.url)}" alt="{esc(b.alt)}">'
                f"{_caption_html(b.caption, esc)}</figure>")
    if isinstance(b, Table):
        rows = b.rows or []
        if not rows:
            return ""
        head = "".join(f"<th>{esc(c)}</th>" for c in rows[0])
        body = "".join(
            "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>"
            for r in rows[1:])
        table = (f"<table><thead><tr>{head}</tr></thead>"
                 f"<tbody>{body}</tbody></table>")
        if b.caption:
            return f"<figure>{table}{_caption_html(b.caption, esc)}</figure>"
        return table
    return ""


def _toc_html(sec, all_headings, esc) -> str:
    """Render a Table-of-Contents section as a clickable nav (its list items are
    heading titles → anchor links). Falls back to a plain list if it doesn't
    look like a TOC."""
    if not sec.blocks or not isinstance(sec.blocks[0], ListBlock):
        return ""
    known = {h.strip().lower(): slug(h) for _, h in all_headings}
    lis = []
    for item in sec.blocks[0].items:
        target = known.get(item.strip().lower())
        if target:
            lis.append(f'<li><a href="#{target}">{esc(item.strip())}</a></li>')
        else:
            lis.append(f"<li>{esc(item.strip())}</li>")
    return f'<nav class="toc"><ul>{"".join(lis)}</ul></nav>'


def model_to_html(model: DocumentModel) -> str:
    """Render the model as a self-contained HTML document (branding-aware)."""
    esc = _html.escape
    md = model.metadata
    br = getattr(model, "export", None) or ExportSettings()
    heads = model.headings()
    accent = (f"h1,h2,h3{{color:{esc(br.primary_color)}}}"
              f"a{{color:{esc(br.primary_color)}}}" if br.primary_color else "")
    parts = [
        "<!doctype html>",
        '<html lang="' + esc(md.language or "en") + '"><head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(md.title or 'Document')}</title>",
        f"<style>{_HTML_CSS}{accent}</style></head><body><article>",
    ]
    # Branding header (logo / header text / confidentiality banner).
    if br.confidentiality:
        parts.append(f'<div class="meta">{esc(br.confidentiality)}</div>')
    if br.logo_url:
        parts.append(f'<img class="logo" src="{esc(br.logo_url)}" alt="logo" '
                     'style="max-height:48px">')
    if br.header:
        parts.append(f'<header class="brand">{esc(br.header)}</header>')
    if md.title:
        parts.append(f'<h1 id="{slug(md.title)}">{esc(md.title)}</h1>')
    _byline = br.author or md.author
    if _byline:
        parts.append(f'<div class="meta">{esc(_byline)}</div>')
    if md.reading_time_min:
        parts.append(f'<div class="meta">~{md.reading_time_min} min read</div>')
    for sec in model.sections:
        if sec.heading:
            lvl = min(6, max(2, sec.level + 1))  # section headings start at h2
            parts.append(
                f'<h{lvl} id="{slug(sec.heading)}">{esc(sec.heading)}</h{lvl}>')
        # A TOC section renders as a clickable nav; others render normally.
        if sec.heading.strip().lower() == "table of contents":
            toc = _toc_html(sec, heads, esc)
            if toc:
                parts.append(toc)
                continue
        for b in sec.blocks:
            parts.append(_block_to_html(b, esc))
    if br.footer:
        parts.append(f'<footer class="brand meta">{esc(br.footer)}</footer>')
    parts.append("</article></body></html>")
    return "\n".join(p for p in parts if p)


__all__ = [
    "DocumentModel", "Metadata", "ExportSettings", "Section",
    "Heading", "Paragraph", "ListBlock", "CodeBlock", "Table", "Quote",
    "Image", "Diagram", "slug",
    "markdown_to_model", "model_to_markdown", "model_to_html",
]
