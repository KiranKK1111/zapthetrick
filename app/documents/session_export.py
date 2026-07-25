"""Session exports (vNext §3.12, Stage 8 Component E).

A whole conversation is itself a deliverable. §3.12 turns a session into a
`session-export` document:

  * a **cover** (title, date, scope summary);
  * a **hyperlinked TOC** — for a CHAT, one entry per TOPIC SEGMENT; for a LIVE
    session, one entry per QUESTION;
  * a **typographic body** — the turns rendered with code highlighting, embedded
    mermaid/tables preserved, and footnote CITATIONS;
  * a LIVE export is a REPORT (an exec summary + per-question breakdown).

Scope controls pick which turns are included; the theme comes from the design
system (Component D). This module owns the deterministic core — SEGMENTATION
(chat topic-segments / live per-question), the export ASSEMBLY (cover + TOC +
sections over the `DocumentModel` IR), and the MARKDOWN emit (hyperlinked TOC +
footnote citations). The binary emit (docx via the existing renderer, pdf via
Typst) reuses `DocumentModel` — the render itself is the on-pod seam. Pure +
fail-open. Flag-gated (`documents.session_export`, default OFF).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.documents.model import (
    CodeBlock, Diagram, DocumentModel, Heading, Metadata, Paragraph, Section,
    Table, slug,
)

CHAT = "chat"
LIVE = "live"


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.documents, "session_export", False))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Inputs (duck-typed) + scope
# --------------------------------------------------------------------------- #
@dataclass
class ExportScope:
    include_roles: tuple = ("user", "assistant")  # which roles to include
    topics: tuple = ()                            # only these topics (empty = all)
    from_index: int = 0
    to_index: int | None = None

    def selects(self, turn: dict, index: int) -> bool:
        if index < self.from_index:
            return False
        if self.to_index is not None and index > self.to_index:
            return False
        role = (turn.get("role") or "").lower()
        if self.include_roles and role not in self.include_roles:
            return False
        if self.topics:
            return (turn.get("topic") or "") in self.topics
        return True


def _turn(t) -> dict:
    """Duck-type a turn to a plain dict {role, content, topic, question, citations}."""
    if isinstance(t, dict):
        return t
    return {"role": getattr(t, "role", ""), "content": getattr(t, "content", ""),
            "topic": getattr(t, "topic", ""),
            "question": getattr(t, "question", ""),
            "citations": getattr(t, "citations", None)}


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    title: str
    turns: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"title": self.title, "turns": len(self.turns)}


def segment_chat(turns, *, scope: ExportScope | None = None,
                 window: int = 6) -> "list[Segment]":
    """Group chat turns into TOPIC segments: a new segment starts when the topic
    changes (or, absent topics, every `window` turns). Never raises → []."""
    try:
        sc = scope or ExportScope()
        picked = [_turn(t) for i, t in enumerate(turns or [])
                  if sc.selects(_turn(t), i)]
        if not picked:
            return []
        segments: list[Segment] = []
        cur_topic = object()
        for t in picked:
            topic = (t.get("topic") or "").strip()
            if topic:
                if not segments or topic != cur_topic:
                    segments.append(Segment(title=topic))
                    cur_topic = topic
                segments[-1].turns.append(t)
            else:
                # No topic → window grouping.
                if not segments or len(segments[-1].turns) >= window or cur_topic != "":
                    segments.append(Segment(title=f"Part {len(segments) + 1}"))
                    cur_topic = ""
                segments[-1].turns.append(t)
        return segments
    except Exception:  # noqa: BLE001
        return []


def segment_live(qa, *, scope: ExportScope | None = None) -> "list[Segment]":
    """One segment per QUESTION for a live session. The segment title is the
    question; its turns are the question + the answer. Never raises → []."""
    try:
        sc = scope or ExportScope(include_roles=())   # live keeps both sides
        segments: list[Segment] = []
        for i, raw in enumerate(qa or []):
            t = _turn(raw)
            if not sc.selects(t, i):
                continue
            q = (t.get("question") or "").strip()
            if q:
                seg = Segment(title=q)
                seg.turns.append(t)
                segments.append(seg)
            elif segments:
                segments[-1].turns.append(t)
        return segments
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
@dataclass
class SessionExport:
    title: str
    mode: str
    cover: dict = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)
    exec_summary: str = ""          # live report only

    def to_dict(self) -> dict:
        return {"title": self.title, "mode": self.mode, "cover": self.cover,
                "segments": [s.to_dict() for s in self.segments],
                "has_exec_summary": bool(self.exec_summary)}


def build_export(session, *, mode: str = CHAT, title: str = "",
                 scope: ExportScope | None = None, date: str = "",
                 exec_summary: str = "") -> SessionExport:
    """Assemble a SessionExport from a session's turns. `mode` = chat (topic
    segments) | live (per-question report + exec summary). Never raises → an
    empty export."""
    try:
        turns = session if isinstance(session, list) else getattr(session, "turns", [])
        if mode == LIVE:
            segments = segment_live(turns, scope=scope)
        else:
            segments = segment_chat(turns, scope=scope)
        cover = {"title": title or ("Live Session Report" if mode == LIVE
                                    else "Conversation Export"),
                 "date": date, "mode": mode, "segments": len(segments)}
        return SessionExport(
            title=cover["title"], mode=mode, cover=cover, segments=segments,
            exec_summary=exec_summary if mode == LIVE else "")
    except Exception:  # noqa: BLE001
        return SessionExport(title=title or "Export", mode=mode)


# --------------------------------------------------------------------------- #
# Markdown emit (hyperlinked TOC + footnote citations)
# --------------------------------------------------------------------------- #
def _collect_citations(segments) -> "list[str]":
    seen: list[str] = []
    for seg in segments:
        for t in seg.turns:
            for c in (t.get("citations") or []):
                c = str(c).strip()
                if c and c not in seen:
                    seen.append(c)
    return seen


def to_markdown(export: SessionExport) -> str:
    """Render a SessionExport to Markdown: cover, a hyperlinked TOC (anchor
    slugs), the typographic body, and a footnotes section for citations. Never
    raises → ''."""
    try:
        out: list[str] = []
        # Cover.
        out.append(f"# {export.title}")
        if export.cover.get("date"):
            out.append(f"*{export.cover['date']}*")
        out.append("")
        if export.exec_summary:
            out.append("## Executive summary")
            out.append(export.exec_summary.strip())
            out.append("")
        # Hyperlinked TOC.
        out.append("## Contents")
        for i, seg in enumerate(export.segments, 1):
            out.append(f"{i}. [{seg.title}](#{slug(seg.title)})")
        out.append("")
        # Footnote citation registry (assign numbers).
        citations = _collect_citations(export.segments)
        cite_no = {c: i + 1 for i, c in enumerate(citations)}
        # Body.
        for seg in export.segments:
            out.append(f"## {seg.title}")
            for t in seg.turns:
                role = (t.get("role") or "").strip()
                content = (t.get("content") or "").rstrip()
                if not content:
                    continue
                if role and role.lower() not in ("assistant", ""):
                    out.append(f"**{role.title()}:**")
                # Append footnote markers for this turn's citations.
                marks = "".join(f"[^{cite_no[str(c).strip()]}]"
                                for c in (t.get("citations") or [])
                                if str(c).strip() in cite_no)
                out.append(content + marks)
                out.append("")
        # Footnotes.
        if citations:
            out.append("---")
            for c in citations:
                out.append(f"[^{cite_no[c]}]: {c}")
        return "\n".join(out).strip() + "\n"
    except Exception:  # noqa: BLE001
        return ""


def to_document_model(export: SessionExport) -> DocumentModel:
    """Build a `DocumentModel` (the binary-render IR) from the export, so the
    existing docx/pdf renderer produces the file. Preserves code fences as
    CodeBlocks and mermaid fences as Diagrams. Never raises → an empty model."""
    try:
        sections: list[Section] = []
        # Cover / lead section.
        lead = Section(heading="", level=0, blocks=[
            Heading(text=export.title, level=1)])
        if export.cover.get("date"):
            lead.blocks.append(Paragraph(text=export.cover["date"]))
        sections.append(lead)
        if export.exec_summary:
            sections.append(Section(heading="Executive summary", level=2,
                                    blocks=[Paragraph(text=export.exec_summary)]))
        for seg in export.segments:
            blocks: list = []
            for t in seg.turns:
                blocks.extend(_content_to_blocks(t.get("content") or ""))
            sections.append(Section(heading=seg.title, level=2, blocks=blocks))
        return DocumentModel(
            metadata=Metadata(title=export.title, doc_type="session-export"),
            sections=sections)
    except Exception:  # noqa: BLE001
        return DocumentModel(metadata=Metadata(title=export.title))


def _content_to_blocks(content: str) -> list:
    """Split a turn's markdown into typed blocks — fenced code → CodeBlock (or
    Diagram for mermaid), everything else → Paragraphs. Lightweight (the full
    parser lives in model.py); this keeps session export dependency-light."""
    import re
    blocks: list = []
    parts = re.split(r"```(\w*)\n(.*?)```", content or "", flags=re.S)
    # re.split with 2 groups yields [text, lang, code, text, lang, code, …].
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            text = (parts[i] or "").strip()
            if text:
                blocks.append(Paragraph(text=text))
            i += 1
        else:
            lang = (parts[i] or "").lower()
            code = parts[i + 1] if i + 1 < len(parts) else ""
            if lang == "mermaid":
                blocks.append(Diagram(source=code.strip(), diagram_kind="mermaid"))
            else:
                blocks.append(CodeBlock(code=code.rstrip(), language=lang))
            i += 2
    return blocks or [Paragraph(text=(content or "").strip())]


__all__ = ["CHAT", "LIVE", "enabled", "ExportScope", "Segment", "segment_chat",
           "segment_live", "SessionExport", "build_export", "to_markdown",
           "to_document_model"]
