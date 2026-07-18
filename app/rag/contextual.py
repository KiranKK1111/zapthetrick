"""Contextual Retrieval — contextual-prefix chunking (roadmap Phase 3 #19).

Anthropic-style contextual retrieval: before embedding a chunk, prepend a short
document-level context (title / section / summary) so an otherwise-orphaned chunk
("it improved throughput by 40%") carries what it's about. Fixes the classic
"which resume/project does this line belong to?" ambiguity. Deterministic +
fail-open; the embedding itself reuses the existing embedder.
"""
from __future__ import annotations

_MAX_CTX = 240  # keep the prefix short so it doesn't dominate the chunk


def build_context_header(*, doc_title: str = "", section: str = "",
                         doc_summary: str = "") -> str:
    """Assemble a compact context header from available document metadata."""
    parts: list[str] = []
    if doc_title.strip():
        parts.append(doc_title.strip())
    if section.strip():
        parts.append(section.strip())
    if doc_summary.strip():
        parts.append(doc_summary.strip())
    header = " — ".join(parts)
    return header[:_MAX_CTX].rstrip()


def contextualize(chunk: str, *, doc_title: str = "", section: str = "",
                  doc_summary: str = "") -> str:
    """Prepend the context header to a chunk for embedding. Returns the chunk
    unchanged when there's no context to add."""
    try:
        header = build_context_header(doc_title=doc_title, section=section,
                                      doc_summary=doc_summary)
        if not header:
            return chunk or ""
        return f"[{header}]\n{chunk or ''}"
    except Exception:  # noqa: BLE001
        return chunk or ""


def contextualize_all(chunks: list[str], *, doc_title: str = "",
                      doc_summary: str = "") -> list[str]:
    """Contextualize a document's chunks (same doc-level context for each)."""
    try:
        return [contextualize(c, doc_title=doc_title, doc_summary=doc_summary)
                for c in (chunks or [])]
    except Exception:  # noqa: BLE001
        return list(chunks or [])


__all__ = ["build_context_header", "contextualize", "contextualize_all"]
