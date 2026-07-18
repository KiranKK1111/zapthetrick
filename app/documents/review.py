"""Document quality analyzer + gate — Phase 3 of the Document Generation roadmap.

DocuementGeneration.md #6/#10/#21 (validation pipeline, quality gate) and #5/#13
(reviewers). Before a document ships, run structural checks on the Document
Model and produce a QualityReport (issues + a 0–100 score). This is the
DETERMINISTIC reviewer — heading hierarchy, empty sections, unresolved
placeholders, duplicate content, malformed tables, and completeness vs the
planner blueprint. An optional LLM reviewer panel is a flag-gated extension
(``llm_review`` hook) layered on top; the deterministic core ships on its own.

Non-blocking by default: the report is surfaced (X-Artifact-Validation), not a
hard gate — refusing a user's download over a style nit is worse than shipping
it with a note. A caller may choose to block on ``report.has_errors``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.documents.model import (
    DocumentModel, Heading, ListBlock, Paragraph, Quote, Table,
    markdown_to_model,
)

# Severity → score penalty.
_PENALTY = {"error": 15, "warning": 5, "info": 1}
_PLACEHOLDER_RE = re.compile(
    r"\b(TODO|TBD|FIXME|XXX|placeholder|lorem ipsum|coming soon|"
    r"to be (?:written|added|filled|determined))\b|\[\s*(?:insert|your|"
    r"placeholder|\.\.\.)\s*[^\]]*\]", re.I)
_MIN_DUP_LEN = 40         # only flag duplication of substantial paragraphs
_MIN_PARA_WORDS_EMPTY = 0


@dataclass
class QualityIssue:
    severity: str          # error | warning | info
    category: str          # e.g. heading_hierarchy, empty_section, placeholder
    message: str
    section: str = ""

    def as_dict(self) -> dict:
        return {"severity": self.severity, "category": self.category,
                "message": self.message, "section": self.section}


@dataclass
class QualityReport:
    score: int = 100
    issues: list[QualityIssue] = field(default_factory=list)
    # Phase 8 (tractable subset): per-section confidence (0..1) + an
    # accessibility pass/fail derived from the accessibility-category issues.
    confidence: dict = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def passed(self) -> bool:
        return not self.has_errors

    @property
    def accessible(self) -> bool:
        return not any(i.category == "accessibility" for i in self.issues)

    def as_dict(self) -> dict:
        return {"score": self.score, "passed": self.passed,
                "accessible": self.accessible, "confidence": self.confidence,
                "issues": [i.as_dict() for i in self.issues]}


# ── individual checks ───────────────────────────────────────────────────────
def _section_has_content(sec) -> bool:
    for b in sec.blocks:
        if isinstance(b, Paragraph) and b.text.strip():
            return True
        if isinstance(b, (ListBlock, Table, Quote)):
            return True
        if getattr(b, "kind", "") in ("code", "diagram", "image"):
            return True
    return False


def _check_headings(model: DocumentModel) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    prev = 0
    for sec in model.sections:
        if not sec.heading:
            continue
        if prev and sec.level > prev + 1:
            issues.append(QualityIssue(
                "warning", "heading_hierarchy",
                f"Heading level jumps from H{prev} to H{sec.level} "
                f"(skipped a level).", sec.heading))
        prev = sec.level
    return issues


def _check_empty_sections(model: DocumentModel) -> list[QualityIssue]:
    return [
        QualityIssue("error", "empty_section",
                     f"Section '{sec.heading}' has a heading but no content.",
                     sec.heading)
        for sec in model.sections
        if sec.heading and not _section_has_content(sec)
    ]


def _texts(model: DocumentModel) -> list[tuple[str, str]]:
    """(section_heading, text) for every text-bearing block."""
    out: list[tuple[str, str]] = []
    for sec in model.sections:
        for b in sec.blocks:
            if isinstance(b, (Paragraph, Quote)):
                out.append((sec.heading, b.text))
            elif isinstance(b, ListBlock):
                out.extend((sec.heading, it) for it in b.items)
    return out


def _check_placeholders(model: DocumentModel) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for head, txt in _texts(model):
        m = _PLACEHOLDER_RE.search(txt)
        if m:
            issues.append(QualityIssue(
                "warning", "placeholder",
                f"Unresolved placeholder: '{m.group(0)}'.", head))
    return issues


def _check_duplicates(model: DocumentModel) -> list[QualityIssue]:
    seen: dict[str, int] = {}
    for _head, txt in _texts(model):
        key = re.sub(r"\s+", " ", txt.strip().lower())
        if len(key) >= _MIN_DUP_LEN:
            seen[key] = seen.get(key, 0) + 1
    return [
        QualityIssue("warning", "duplicate_content",
                     f"Repeated paragraph ({n}×): '{key[:60]}…'.")
        for key, n in seen.items() if n > 1
    ]


def _check_tables(model: DocumentModel) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for sec in model.sections:
        for b in sec.blocks:
            if isinstance(b, Table) and b.rows:
                width = len(b.rows[0])
                if any(len(r) != width for r in b.rows[1:]):
                    issues.append(QualityIssue(
                        "warning", "malformed_table",
                        "Table has rows with inconsistent column counts.",
                        sec.heading))
                    break
    return issues


def _check_completeness(model: DocumentModel, blueprint) -> list[QualityIssue]:
    have = {h.lower().strip() for _, h in model.headings()}

    def _present(title: str) -> bool:
        t = title.lower().strip()
        return any(t in h or h in t for h in have)

    issues: list[QualityIssue] = []
    for s in getattr(blueprint, "sections", []):
        if s.required and not _present(s.title):
            issues.append(QualityIssue(
                "warning", "missing_section",
                f"Expected section '{s.title}' is missing "
                f"({blueprint.goal.value}).", s.title))
    return issues


def _score(issues: list[QualityIssue]) -> int:
    penalty = sum(_PENALTY.get(i.severity, 1) for i in issues)
    return max(0, min(100, 100 - penalty))


# ── Phase 8 (subset): accessibility + section confidence ────────────────────
def _check_accessibility(model: DocumentModel) -> list[QualityIssue]:
    """Deterministic accessibility checks (#16 accessibility): images need alt
    text, tables need a non-empty header row, and a multi-section document needs
    real headings for screen-reader navigation."""
    from app.documents.model import Image, Table

    issues: list[QualityIssue] = []
    for sec in model.sections:
        for b in sec.blocks:
            if isinstance(b, Image) and not (b.alt or "").strip():
                issues.append(QualityIssue(
                    "warning", "accessibility",
                    "Image has no alt text (inaccessible to screen readers).",
                    sec.heading))
            if isinstance(b, Table) and b.rows and not any(
                    c.strip() for c in b.rows[0]):
                issues.append(QualityIssue(
                    "warning", "accessibility",
                    "Table has an empty header row.", sec.heading))
    if len(list(model.iter_blocks())) > 6 and not model.headings():
        issues.append(QualityIssue(
            "warning", "accessibility",
            "Long document has no headings — hard to navigate."))
    return issues


_HEDGE_RE = re.compile(
    r"\b(?:might|maybe|perhaps|possibly|probably|i think|i believe|"
    r"not sure|unclear|unsure|roughly|approximately|to be confirmed|"
    r"presumably|it seems)\b", re.I)


def _section_confidence(sec) -> float:
    """0..1 heuristic confidence for a section: hedging language, unresolved
    placeholders, and very thin content lower it. Empty → 0."""
    text = " ".join(
        b.text for b in sec.blocks if isinstance(b, Paragraph)) + " " + \
        " ".join(it for b in sec.blocks if isinstance(b, ListBlock)
                 for it in b.items)
    words = text.split()
    if not words:
        return 0.0
    conf = 1.0
    conf -= 0.12 * len(_HEDGE_RE.findall(text))
    if _PLACEHOLDER_RE.search(text):
        conf -= 0.4
    if len(words) < 12:
        conf -= 0.25
    return max(0.0, min(1.0, conf))


def _check_confidence(model: DocumentModel) -> tuple[list[QualityIssue], dict]:
    """Score each section's confidence; flag the low ones (#3 confidence)."""
    issues: list[QualityIssue] = []
    scores: dict = {}
    for sec in model.sections:
        # Skip sections with no direct prose (titles / structural parents / empty
        # sections — those are handled by the empty-section check, not here).
        if not sec.heading or _section_confidence(sec) == 0.0:
            continue
        c = round(_section_confidence(sec), 2)
        scores[sec.heading] = c
        if c < 0.6:
            issues.append(QualityIssue(
                "info", "low_confidence",
                f"Section '{sec.heading}' reads low-confidence "
                f"({int(c * 100)}%) — hedging / thin / placeholders.",
                sec.heading))
    return issues, scores


def analyze_document(content, *, blueprint=None, title: str = "") -> QualityReport:
    """Run the deterministic quality checks. ``content`` may be Markdown text or
    an already-parsed :class:`DocumentModel`. ``blueprint`` (from the planner)
    enables completeness checking. Fail-open: any parse error → an empty pass."""
    try:
        model = (content if isinstance(content, DocumentModel)
                 else markdown_to_model(content or "", title))
    except Exception:  # noqa: BLE001
        return QualityReport(score=100, issues=[])
    issues: list[QualityIssue] = []
    issues += _check_headings(model)
    issues += _check_empty_sections(model)
    issues += _check_placeholders(model)
    issues += _check_duplicates(model)
    issues += _check_tables(model)
    issues += _check_accessibility(model)
    if blueprint is not None:
        issues += _check_completeness(model, blueprint)
    conf_issues, confidence = _check_confidence(model)
    issues += conf_issues
    return QualityReport(score=_score(issues), issues=issues,
                         confidence=confidence)


# ── flag-gated LLM reviewer panel (DocuementGeneration.md #5/#13) ────────────
_REVIEW_SYS = (
    "You are a meticulous document editor. Review the document below and report "
    "ONLY concrete, fixable problems: factual inconsistencies, contradictions, "
    "unclear or incomplete sentences, and structural gaps. Do NOT rewrite the "
    "document. Respond with a compact JSON array; each item is "
    '{"severity":"error|warning|info","category":"<slug>","message":"<one line>",'
    '"section":"<heading or empty>"}. Return [] when the document is clean.'
)
_MAX_REVIEW_CHARS = 12000


async def llm_review_document(content, *, title: str = "") -> list[QualityIssue]:
    """Optional LLM reviewer panel — layered on top of the deterministic checks
    when ``cfg.documents.llm_review`` is on. Fail-open: any error (no route,
    bad JSON, disabled) returns an empty list so a review never breaks a render.
    Cheap model, single call, no document rewrite."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.documents, "llm_review", False)):
            return []
    except Exception:  # noqa: BLE001
        return []
    text = content if isinstance(content, str) else model_to_markdown(content)
    text = (text or "").strip()
    if not text:
        return []
    try:
        import json

        from app.core import llm_client
        raw = await llm_client.llm.complete(
            [{"role": "system", "content": _REVIEW_SYS},
             {"role": "user", "content": text[:_MAX_REVIEW_CHARS]}],
            options={"temperature": 0.0, "max_tokens": 700})
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end < start:
            return []
        items = json.loads(raw[start:end + 1])
    except Exception:  # noqa: BLE001
        return []
    out: list[QualityIssue] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not it.get("message"):
            continue
        sev = str(it.get("severity", "info")).lower()
        out.append(QualityIssue(
            sev if sev in _PENALTY else "info",
            str(it.get("category", "llm_review"))[:40],
            str(it["message"])[:300], str(it.get("section", ""))[:120]))
    return out


# ── code-lint pass (Phase 3 — wire polyglot/linters into review) ─────────────
_LINT_LANGS = {"python", "javascript", "typescript", "js", "ts", "py"}
_LINT_CANON = {"py": "python", "js": "javascript", "ts": "typescript"}
_MAX_LINT_BLOCKS = 12


async def code_lint_document(content, *, title: str = "") -> list[QualityIssue]:
    """Lint the fenced CODE BLOCKS in the document (ruff for Python, eslint for
    JS/TS) and surface findings as quality issues — the review pipeline's code
    check (roadmap Phase 3: "reuse polyglot/linters.py for code blocks").

    Gated by ``cfg.documents.code_lint_review`` (default ON). Entirely fail-open:
    a missing linter binary → the language returns no findings ("not linted",
    never an error), and any exception yields an empty list. A finding maps to a
    ``warning`` (eslint errors → ``error``) in the ``code_lint`` category."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.documents, "code_lint_review", True)):
            return []
    except Exception:  # noqa: BLE001
        pass
    from app.documents.model import CodeBlock
    try:
        model = (content if isinstance(content, DocumentModel)
                 else markdown_to_model(content or "", title))
    except Exception:  # noqa: BLE001
        return []
    # (section_heading, CodeBlock) for lintable languages, bounded.
    targets: list[tuple[str, object]] = []
    for sec in model.sections:
        for b in sec.blocks:
            if isinstance(b, CodeBlock) and (b.language or "").lower() in _LINT_LANGS:
                targets.append((sec.heading, b))
                if len(targets) >= _MAX_LINT_BLOCKS:
                    break
        if len(targets) >= _MAX_LINT_BLOCKS:
            break
    if not targets:
        return []
    try:
        from app.polyglot.linters import lint_code
    except Exception:  # noqa: BLE001
        return []
    issues: list[QualityIssue] = []
    for head, b in targets:
        lang = _LINT_CANON.get((b.language or "").lower(), (b.language or "").lower())
        try:
            findings = await lint_code(lang, b.code)
        except Exception:  # noqa: BLE001
            findings = []
        for f in findings[:10]:
            sev = f.severity if f.severity in _PENALTY else "warning"
            where = f" (line {f.line})" if f.line else ""
            code = f" [{f.code}]" if f.code else ""
            issues.append(QualityIssue(
                sev, "code_lint",
                f"{lang}{where}: {f.message}{code}".strip(), head))
    return issues


# ── multi-reviewer role panel (Phase 3 — real minimal, LLM, flag-gated) ───────
# Each reviewer role sees the same document through a different lens. Real, but
# it costs one cheap LLM call per role, so it is OFF by default (fail-open).
_REVIEWER_ROLES: dict[str, str] = {
    "technical": "a technical reviewer. Report factual errors, incorrect claims, "
                 "wrong commands/APIs, and logical gaps.",
    "grammar": "a copy editor. Report grammar, spelling, and awkward or "
               "unclear sentences.",
    "formatting": "a formatting reviewer. Report inconsistent headings, list "
                  "style, capitalization, and layout problems.",
    "consistency": "a consistency reviewer. Report terminology that changes "
                   "mid-document, contradictions, and mismatched naming.",
}


def _reviewer_sys(role_desc: str) -> str:
    return (
        f"You are {role_desc} Review the document and report ONLY concrete, "
        "fixable problems in your lens. Do NOT rewrite it. Respond with a compact "
        'JSON array of {"severity":"error|warning|info","message":"<one line>",'
        '"section":"<heading or empty>"}. Return [] when clean.')


async def multi_reviewer_document(content, *, title: str = "") -> list[QualityIssue]:
    """Role-split reviewer panel (technical / grammar / formatting / consistency).
    Each role is one cheap LLM call; issues are tagged with the role as category.
    Gated by ``cfg.documents.multi_reviewer`` (default OFF). Fail-open per role —
    a failing role contributes nothing, it never breaks the others or the render."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.documents, "multi_reviewer", False)):
            return []
    except Exception:  # noqa: BLE001
        return []
    text = content if isinstance(content, str) else model_to_markdown(content)
    text = (text or "").strip()
    if not text:
        return []
    import json

    async def _one_role(role: str, desc: str) -> list[QualityIssue]:
        try:
            from app.core import llm_client
            raw = await llm_client.llm.complete(
                [{"role": "system", "content": _reviewer_sys(desc)},
                 {"role": "user", "content": text[:_MAX_REVIEW_CHARS]}],
                options={"temperature": 0.0, "max_tokens": 500})
            start, end = raw.find("["), raw.rfind("]")
            if start < 0 or end < start:
                return []
            items = json.loads(raw[start:end + 1])
        except Exception:  # noqa: BLE001
            return []
        out: list[QualityIssue] = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict) or not it.get("message"):
                continue
            sev = str(it.get("severity", "info")).lower()
            out.append(QualityIssue(
                sev if sev in _PENALTY else "info", f"review_{role}",
                str(it["message"])[:300], str(it.get("section", ""))[:120]))
        return out

    import asyncio
    results = await asyncio.gather(
        *[_one_role(r, d) for r, d in _REVIEWER_ROLES.items()],
        return_exceptions=True)
    issues: list[QualityIssue] = []
    for r in results:
        if isinstance(r, list):
            issues.extend(r)
    return issues


# ── deterministic safe-fix pass (Phase 3 — bounded, no LLM) ──────────────────
def safe_fix(content, *, title: str = "") -> tuple[str, list[str]]:
    """Apply a BOUNDED set of always-safe, deterministic fixes to a document and
    return ``(fixed_markdown, applied)``. No LLM, no content invention — only
    mechanical clean-ups that cannot change meaning:

      * drop consecutive EXACT-duplicate paragraphs (a copy/paste artifact),
      * strip trailing whitespace from paragraphs and list items,
      * collapse internal runs of blank space in a paragraph to single spaces.

    Fail-open: any error returns the original content unchanged with no applied
    fixes, so a fixer bug never corrupts an export."""
    try:
        model = (content if isinstance(content, DocumentModel)
                 else markdown_to_model(content or "", title))
    except Exception:  # noqa: BLE001
        return (content if isinstance(content, str) else "", [])
    applied: list[str] = []
    try:
        import copy
        m = copy.deepcopy(model)
        for sec in m.sections:
            new_blocks: list = []
            last_para: str | None = None
            for b in sec.blocks:
                if isinstance(b, Paragraph):
                    cleaned = re.sub(r"[ \t]+", " ", b.text.strip())
                    if cleaned != b.text:
                        if cleaned.strip() == b.text.strip():
                            applied.append("collapsed whitespace")
                        else:
                            applied.append("trimmed whitespace")
                        b.text = cleaned
                    if last_para is not None and cleaned and cleaned == last_para:
                        applied.append(f"removed duplicate paragraph in "
                                       f"'{sec.heading or 'lead'}'")
                        continue
                    last_para = cleaned
                    new_blocks.append(b)
                elif isinstance(b, ListBlock):
                    stripped = [it.rstrip() for it in b.items]
                    if stripped != b.items:
                        applied.append("trimmed list items")
                        b.items = stripped
                    last_para = None
                    new_blocks.append(b)
                else:
                    last_para = None
                    new_blocks.append(b)
            sec.blocks = new_blocks
        if not applied:
            return (content if isinstance(content, str)
                    else model_to_markdown(model), [])
        return model_to_markdown(m), applied
    except Exception:  # noqa: BLE001
        return (content if isinstance(content, str) else "", [])


async def analyze_document_async(content, *, blueprint=None,
                                 title: str = "") -> QualityReport:
    """``analyze_document`` + the code-lint pass + the flag-gated LLM reviewer
    panels (single panel + role-split multi-reviewer) merged in. Used by the
    export path; the extra issues re-score the report. Fail-open."""
    report = analyze_document(content, blueprint=blueprint, title=title)
    extra: list[QualityIssue] = []
    for producer in (
            code_lint_document(content, title=title),
            llm_review_document(content, title=title),
            multi_reviewer_document(content, title=title)):
        try:
            extra += await producer
        except Exception:  # noqa: BLE001
            pass
    if extra:
        report.issues.extend(extra)
        report.score = _score(report.issues)
    return report


def model_to_markdown(model):  # local alias to avoid a top-level import cycle
    from app.documents.model import model_to_markdown as _m2m
    return _m2m(model)


__all__ = ["QualityIssue", "QualityReport", "analyze_document",
           "analyze_document_async", "llm_review_document",
           "code_lint_document", "multi_reviewer_document", "safe_fix"]
