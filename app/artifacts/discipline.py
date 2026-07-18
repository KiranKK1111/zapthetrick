"""Artifact creation discipline (workspace-and-artifacts R4).

`should_create_artifact(answer, sources, explicit_format, min_chars) -> kind|None`
decides whether an answer is a *substantial structured output* worth saving as an
Artifact (Property 4). A normal conversational answer returns ``None`` (today's
behavior, R4.3). Deterministic + pure; never raises.
"""
from __future__ import annotations

import re

ARTIFACT_KINDS = ("document", "code", "markdown", "diagram", "sql", "html")

_MERMAID_RE = re.compile(r"```mermaid\b", re.I)
_CODE_FENCE_RE = re.compile(r"```([a-z0-9+#.\-]*)\n", re.I)
_SQL_RE = re.compile(r"\b(create\s+table|select\s+.+\s+from|insert\s+into|"
                     r"alter\s+table)\b", re.I)
_HTML_RE = re.compile(r"<!doctype html|<html[\s>]|<body[\s>]", re.I)
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.M)
_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$", re.M)

# Document formats that always warrant an artifact when explicitly requested.
_DOC_FORMATS = ("pdf", "docx", "word", "xlsx", "excel", "csv", "md",
                "markdown", "txt", "zip", "7z")


def should_create_artifact(answer: str, sources: dict | None = None,
                           explicit_format: str | None = None,
                           min_chars: int = 400) -> str | None:
    """Return an artifact kind, or None for a normal answer. Never raises."""
    try:
        return _decide(answer, sources, explicit_format, min_chars)
    except Exception:  # noqa: BLE001
        return None


def _decide(answer: str, sources, explicit_format, min_chars) -> str | None:
    text = answer or ""

    # 1) An explicit document/file request → an artifact regardless of length.
    fmt = (explicit_format or "").strip().lower()
    if fmt in _DOC_FORMATS:
        return "document"

    # 2) The doc-intent classifier already flagged this turn as a document.
    if isinstance(sources, dict) and sources.get("document"):
        return "document"

    # 3) Structured-content detection (substantial outputs only).
    if _MERMAID_RE.search(text):
        return "diagram"
    if _HTML_RE.search(text):
        return "html"

    # A dominant code block → a code artifact (a real file, not a snippet aside).
    code_blocks = _CODE_FENCE_RE.findall(text)
    if code_blocks and len(text) >= min_chars:
        # SQL-heavy code block → sql; otherwise a code artifact.
        if any(lang in ("sql",) for lang in code_blocks) or _SQL_RE.search(text):
            return "sql"
        return "code"

    # 4) Substantial structured prose: long, with headings or a table.
    if len(text) >= min_chars and (_HEADING_RE.search(text) or _TABLE_RE.search(text)):
        return "markdown"

    # 5) A plain conversational answer → no artifact (R4.3).
    return None


__all__ = ["should_create_artifact", "ARTIFACT_KINDS"]
