"""Artifact Intent Taxonomy — Phase 0 of the Document Generation roadmap.

Replaces the binary "does the user want a document?" question with a richer,
deterministic classification of the user's DESIRED OUTCOME (DocuementGeneration.md
Step 1 / Step 9):

    CHAT                 — an ordinary answer; NO artifact.
    ANSWER_AND_ARTIFACT  — produce NEW content AND package it as a file
                           ("create documentation for this", "write a report on X").
    ARTIFACT_ONLY        — package the EXISTING answer/context as a file, no new
                           reasoning ("generate a PDF", "export this project").
    UPDATE_EXISTING      — modify an artifact already produced ("convert the
                           above into Word", "add a Redis section to the doc").
    DOWNLOAD_EXISTING    — re-deliver an artifact already produced ("where's the
                           pdf", "resend the document") — no regeneration.
    UNKNOWN              — a deliverable is implied but under-specified.

DESIGN — this mirrors the discipline that keeps false documents from appearing:
it is DETERMINISTIC-first and never upgrades CHAT to an artifact on a fuzzy
signal (that split-brain caused unrequested PDFs). Triage remains the single
source of truth for WHETHER generation happens; this layer refines the intent
and drives the deterministic planner output `{intent, artifact_type, source,
reuse_response, requires_llm}` so the downstream pipeline needs no re-guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ArtifactIntent(str, Enum):
    CHAT = "CHAT"
    ANSWER_AND_ARTIFACT = "ANSWER_AND_ARTIFACT"
    ARTIFACT_ONLY = "ARTIFACT_ONLY"
    UPDATE_EXISTING = "UPDATE_EXISTING"
    DOWNLOAD_EXISTING = "DOWNLOAD_EXISTING"
    UNKNOWN = "UNKNOWN"


# Where the content to render comes from.
SOURCE_LAST_RESPONSE = "LAST_RESPONSE"   # render the answer already in the thread
SOURCE_NEW = "NEW"                       # generate new content, then render
SOURCE_EXISTING = "EXISTING"             # an artifact already produced


@dataclass
class PlannerDecision:
    """Deterministic planner output (DocuementGeneration.md Step 9). Consumed by
    the artifact pipeline so it never re-classifies."""
    intent: ArtifactIntent
    artifact_type: str | None = None     # 'pdf'|'docx'|'zip'|… or None (infer/ask)
    source: str = SOURCE_NEW
    reuse_response: bool = False          # render prior answer verbatim (no LLM)
    requires_llm: bool = True

    @property
    def wants_artifact(self) -> bool:
        return self.intent not in (ArtifactIntent.CHAT, ArtifactIntent.UNKNOWN)

    def as_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "artifact_type": self.artifact_type,
            "source": self.source,
            "reuse_response": self.reuse_response,
            "requires_llm": self.requires_llm,
        }


# Retrieval of an ALREADY-produced artifact → re-deliver, don't regenerate.
_DOWNLOAD_RE = re.compile(
    r"\b(?:where(?:'s| is| are)|show me|send me|resend|re-?send|"
    r"give me back|download)\b[^.?!\n]{0,24}?"
    r"\b(document|doc|file|pdf|docx?|word\s+doc(?:ument)?|excel|xlsx|"
    r"spreadsheet|powerpoint|pptx?|slides?|presentation|csv|zip|archive|"
    r"attachment|it|that|the\s+one)\b",
    re.I,
)
# Verbs that MODIFY an existing artifact (need a prior artifact + a reference).
_UPDATE_VERB_RE = re.compile(
    r"\b(convert|turn|change|update|modify|edit|revise|amend|adjust|add|"
    r"append|insert|remove|delete|drop|rename|regenerate|redo|rewrite|"
    r"reword|expand|shorten|translate|reformat|restyle)\b",
    re.I,
)
# A reference to the existing artifact / prior content ("the above", "this doc").
_REFERS_PRIOR_RE = re.compile(
    r"\b(the\s+above|above|this|that|it|the\s+doc(?:ument)?|the\s+pdf|"
    r"the\s+file|the\s+report|the\s+word\s+doc(?:ument)?|the\s+deck|"
    r"the\s+slides?|the\s+presentation|the\s+spreadsheet)\b",
    re.I,
)
# A STRONG export / archive intent (mirrors routes_agents `_wants_download`):
# a package verb + a target. Catches "export this project" / "zip the code" that
# the doc-FILE detector (`explicit_doc_request`) doesn't treat as a document.
_EXPORT_VERB_RE = re.compile(
    r"\b(export|download|zip|compress|archive|package|bundle)\b", re.I)
_EXPORT_TARGET_RE = re.compile(
    r"\b(this|that|it|the|project|projects|code|codebase|app|application|"
    r"source|files?|everything|repo|repository|conversation|chat|answer|"
    r"response|thread|deck|slides?)\b",
    re.I,
)
# Produce verb + a NEW subject to author ("a document ON kafka", "report ABOUT
# X") — distinguishes ANSWER_AND_ARTIFACT (author new content) from ARTIFACT_ONLY
# (package what's already here). Demonstratives ("on this/that") are NOT a new
# subject — they point back at existing content.
_PRODUCE_VERB_RE = re.compile(
    r"\b(generat\w*|creat\w*|make|produc\w*|build|draft|prepar\w*|writ\w*|"
    r"compos\w*)\b", re.I)
_NEW_SUBJECT_RE = re.compile(
    r"\b(on|about|regarding|covering|explaining|describing)\s+"
    r"(?!this\b|that\b|it\b|the\s+above\b)\w+",
    re.I,
)


def classify_artifact_intent(
    text: str,
    *,
    has_prior_artifact: bool = False,
    has_prior_content: bool = False,
) -> PlannerDecision:
    """Classify a turn's artifact intent, deterministically.

    ``has_prior_artifact`` — a downloadable file was already produced earlier in
    this conversation (enables UPDATE/DOWNLOAD of it). ``has_prior_content`` —
    there's a prior assistant answer to package (enables ARTIFACT_ONLY reuse).

    Returns a :class:`PlannerDecision`. CHAT when there's no deliverable signal —
    never guessed up from a fuzzy match.
    """
    from app.documents.detect import (
        explicit_doc_request, format_answer, mentions_format,
    )

    t = (text or "").strip()
    if not t:
        return PlannerDecision(ArtifactIntent.CHAT)
    low = t.lower()

    det, det_fmt = explicit_doc_request(t)

    # 1) DOWNLOAD_EXISTING — re-deliver an artifact already made. Requires a
    #    prior artifact so "download this data" on a fresh turn isn't misread.
    if has_prior_artifact and _DOWNLOAD_RE.search(low) \
            and not _UPDATE_VERB_RE.search(low):
        return PlannerDecision(
            ArtifactIntent.DOWNLOAD_EXISTING, det_fmt,
            source=SOURCE_EXISTING, reuse_response=True, requires_llm=False)

    # 2) UPDATE_EXISTING — modify a prior artifact ("convert the above to Word",
    #    "add a Redis section"). Needs a prior artifact AND an update verb AND a
    #    reference to it, so a first-time "convert this to excel" (no artifact
    #    yet) stays ARTIFACT_ONLY below.
    if has_prior_artifact and _UPDATE_VERB_RE.search(low) \
            and _REFERS_PRIOR_RE.search(low):
        _uf = det_fmt or (format_answer(t) if mentions_format(t) else None)
        return PlannerDecision(
            ArtifactIntent.UPDATE_EXISTING, _uf,
            source=SOURCE_EXISTING, reuse_response=False, requires_llm=True)

    # A strong export/archive ask that the doc-FILE detector misses ("export this
    # project" → a zip). Package verb + a target.
    _export = bool(_EXPORT_VERB_RE.search(low)
                   and _EXPORT_TARGET_RE.search(low))

    # No explicit deliverable signal at all → ordinary chat. This is the guard
    # that keeps "Can I have a solution", "give me more details", "create
    # documentation for this" (no file/format named) from becoming a file: the
    # deterministic detector said no, and we do NOT trust a fuzzy leg. NOTE: this
    # DEVIATES from DocuementGeneration.md's table (which maps "Create
    # documentation for this" → ANSWER_AND_ARTIFACT) — intentionally, to preserve
    # ZapTheTrick's explicit-only policy that a file requires a named format/
    # export verb (the fix for repeated unrequested-PDF reports).
    if not det and not _export:
        return PlannerDecision(ArtifactIntent.CHAT)

    fmt = det_fmt if det else "zip"     # an export-verb-only ask packages as zip
    src = SOURCE_LAST_RESPONSE if has_prior_content else SOURCE_NEW

    # 3) ANSWER_AND_ARTIFACT — author NEW content, then package it: a produce verb
    #    aimed at a NEW subject ("generate a document ON kafka", "write a report
    #    ABOUT X"). Always needs the LLM; never reuses the prior answer.
    if _PRODUCE_VERB_RE.search(low) and _NEW_SUBJECT_RE.search(low):
        return PlannerDecision(
            ArtifactIntent.ANSWER_AND_ARTIFACT, fmt,
            source=SOURCE_NEW, reuse_response=False, requires_llm=True)

    # 4) ARTIFACT_ONLY — package what's already here ("generate a PDF", "export
    #    this", "as a word doc", "zip the project"). Reuse the last response with
    #    no new LLM call when there's prior content to render.
    return PlannerDecision(
        ArtifactIntent.ARTIFACT_ONLY, fmt,
        source=src, reuse_response=has_prior_content,
        requires_llm=not has_prior_content)


__all__ = [
    "ArtifactIntent", "PlannerDecision", "classify_artifact_intent",
    "SOURCE_LAST_RESPONSE", "SOURCE_NEW", "SOURCE_EXISTING",
]
