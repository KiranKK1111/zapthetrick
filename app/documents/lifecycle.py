"""Artifact lifecycle core — Phase 5 of the Document Generation roadmap.

The algorithmic heart of the document's lifecycle / incremental-update ideas
(#1 lifecycle, #9/#12 partial updates, #11 semantic anchors, #22 diff engine),
as pure operations on the Phase-1 DocumentModel — no persistence:

  * ``anchor_for(heading)`` — a stable semantic anchor (``architecture-security``)
    so "update the Security section" locates a section even when page numbers or
    order change (#11 semantic anchors).
  * ``find_section`` / ``replace_section`` — INCREMENTAL updates: swap or insert a
    single section, leaving the rest untouched ("update only the Database
    section" → don't regenerate the whole document, #9/#12).
  * ``diff_models`` — a version diff: sections added / removed / changed (#22).

Persisting versioned artifacts (a documents table + evolution timeline) is the
next increment — it needs a schema migration and is deliberately NOT here. These
operations are what that persistence layer will call.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace as _dc_replace

from app.documents.model import DocumentModel, Section, markdown_to_model


def anchor_for(heading: str) -> str:
    """A stable slug anchor for a heading. ``"Security & Auth"`` → ``security-auth``."""
    return re.sub(r"[^a-z0-9]+", "-", (heading or "").lower()).strip("-")


def _as_section(target: str, content) -> Section:
    """Coerce update content (Markdown str, block list, or Section) into a
    Section titled ``target``."""
    if isinstance(content, Section):
        return content
    if isinstance(content, str):
        blocks: list = []
        for sec in markdown_to_model(content).sections:
            blocks.extend(sec.blocks)
        return Section(heading=target, level=2, blocks=blocks)
    return Section(heading=target, level=2, blocks=list(content or []))


def _matches(section: Section, target: str) -> bool:
    if not section.heading:
        return False
    a = anchor_for(section.heading)
    ta = anchor_for(target)
    return a == ta or ta in a or target.strip().lower() in section.heading.lower()


def find_section(model: DocumentModel, target: str) -> Section | None:
    """Locate a section by semantic anchor or (fuzzy) title. None if absent."""
    for s in model.sections:
        if _matches(s, target):
            return s
    return None


def replace_section(model: DocumentModel, target: str, content) -> DocumentModel:
    """Return a copy of the model with the ``target`` section's body replaced by
    ``content`` (Markdown / blocks / Section). If no such section exists, the new
    section is APPENDED — so this handles both "update the X section" and "add an
    X section". Only the matched section changes; everything else is preserved."""
    new_section = _as_section(target, content)
    out: list[Section] = []
    replaced = False
    for s in model.sections:
        if not replaced and _matches(s, target):
            # Keep the original heading/level; swap only the body.
            out.append(Section(heading=s.heading, level=s.level,
                               blocks=new_section.blocks))
            replaced = True
        else:
            out.append(s)
    if not replaced:
        out.append(new_section)
    return DocumentModel(metadata=_dc_replace(model.metadata), sections=out)


def remove_section(model: DocumentModel, target: str) -> DocumentModel:
    """Return a copy with the ``target`` section removed (no-op if absent)."""
    out = [s for s in model.sections if not _matches(s, target)]
    return DocumentModel(metadata=_dc_replace(model.metadata), sections=out)


# ── version diff ────────────────────────────────────────────────────────────
@dataclass
class DocDiff:
    added: list[str] = field(default_factory=list)      # section headings
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)     # same anchor, new body
    unchanged: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def as_dict(self) -> dict:
        return {"added": self.added, "removed": self.removed,
                "changed": self.changed, "unchanged": self.unchanged}


def _section_fingerprint(sec: Section) -> str:
    from app.documents.model import model_to_markdown
    body = model_to_markdown(DocumentModel(sections=[sec]))
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def diff_models(old: DocumentModel, new: DocumentModel) -> DocDiff:
    """Section-level diff between two document versions, keyed by anchor."""
    old_map = {anchor_for(s.heading): s for s in old.sections if s.heading}
    new_map = {anchor_for(s.heading): s for s in new.sections if s.heading}
    diff = DocDiff()
    for a, s in new_map.items():
        if a not in old_map:
            diff.added.append(s.heading)
        elif _section_fingerprint(s) != _section_fingerprint(old_map[a]):
            diff.changed.append(s.heading)
        else:
            diff.unchanged.append(s.heading)
    for a, s in old_map.items():
        if a not in new_map:
            diff.removed.append(s.heading)
    return diff


def merge_update(prior_md: str, update_md: str) -> str:
    """Merge an incremental edit into a prior document (the heart of
    UPDATE_EXISTING). Every HEADED section in ``update_md`` replaces the
    same-anchor section in the prior document, or is appended when new — so
    "## Redis …" adds a Redis section and "## Database …" rewrites the existing
    one, leaving every other section untouched. Lead (heading-less) prose in the
    update is treated as conversational framing and dropped, not merged into the
    document body. Returns the merged Markdown.

    Fail-safe: if the update has no headed sections (nothing to merge on), the
    prior document is returned unchanged."""
    from app.documents.model import model_to_markdown

    prior = markdown_to_model(prior_md or "")
    update = markdown_to_model(update_md or "")
    result = prior
    edited: list[str] = []
    for sec in update.sections:
        if sec.heading:
            result = replace_section(result, sec.heading, sec)
            edited.append(sec.heading)
    # Cross-cutting metrics (roadmap): an UPDATE is a regeneration, and each
    # headed section it touches is an edited section. Fail-open — a metrics hiccup
    # never breaks a merge.
    if edited:
        try:
            from app.documents.metrics import (
                record_regeneration, record_section_edit)
            record_regeneration()
            for h in edited:
                record_section_edit(h)
        except Exception:  # noqa: BLE001
            pass
    return model_to_markdown(result)


__all__ = [
    "anchor_for", "find_section", "replace_section", "remove_section",
    "merge_update", "DocDiff", "diff_models",
]
