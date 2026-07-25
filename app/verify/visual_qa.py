"""Visual-QA loop for generated documents (vNext §3.3).

The main §3.3 addition: after a document renders and passes its STRUCTURAL check
(`doc_verify` — re-open with its own parser), rasterize its pages and have the
resident VLM critique them against the requirement rubric ("2 pages, rubric says
1", "skills section missing", "heading overlaps the margin"). Findings feed back
to the generator for a bounded repair.

**No `vision` import here** (keeps `verify` free of a `verify → vision` edge):
the VLM call is INJECTED as `describe_fn` by the api-layer caller (which already
has `api → vision`). Validation of the VLM's JSON critique goes through the
`app.core.structured` facade (existing `verify → core` edge). Rasterization uses
PyMuPDF (already a dependency). Fail-open: any error → a passing, empty report so
a QA hiccup never blocks delivery.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

_CRITIQUE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "meets_rubric": {"type": "boolean"},
        "page_count": {"type": "number"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},        # pages|missing|overflow|style|other
                    "detail": {"type": "string"},
                    "severity": {"type": "string"},     # high|low
                },
            },
        },
    },
    "required": ["meets_rubric"],
}


@dataclass
class VisualIssue:
    kind: str
    detail: str
    severity: str = "low"


@dataclass
class VisualQAReport:
    ok: bool = True                       # meets the rubric (or QA was skipped)
    page_count: int | None = None
    issues: list[VisualIssue] = field(default_factory=list)
    ran: bool = False                     # False = skipped / errored (fail-open)

    @property
    def blocking_issues(self) -> list[VisualIssue]:
        return [i for i in self.issues if i.severity == "high"]

    def as_dict(self) -> dict:
        return {"ok": self.ok, "ran": self.ran, "page_count": self.page_count,
                "note": self.note(),
                "issues": [{"kind": i.kind, "detail": i.detail,
                            "severity": i.severity} for i in self.issues]}

    def note(self) -> str:
        """Honest one-line delivery note (§3.3)."""
        if not self.ran:
            return "rendered, re-opened"
        base = f"rendered, re-opened, visually checked"
        if self.page_count:
            base += f" · {self.page_count} page{'s' if self.page_count != 1 else ''}"
        base += " · all rubric items present" if self.ok else " · see notes"
        return base


def rasterize_pdf(pdf_bytes: bytes, *, dpi: int = 120,
                  max_pages: int = 4) -> list[str]:
    """PDF bytes → base64 PNG per page (capped). Fail-open → []."""
    try:
        import base64

        import fitz  # PyMuPDF
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                pix = page.get_pixmap(dpi=dpi)
                out.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        finally:
            doc.close()
    except Exception:  # noqa: BLE001
        return []
    return out


def _critique_prompt(rubric_text: str, page_count: int) -> str:
    return (
        "You are a meticulous document reviewer. You are shown the rendered "
        f"pages ({page_count} page(s)) of a generated document. Check it against "
        "this requirement checklist and report ONLY real, visible problems.\n\n"
        f"Checklist: {rubric_text or '(general quality)'}\n\n"
        "Return ONLY a JSON object: {\"meets_rubric\": bool, \"page_count\": "
        "number, \"issues\": [{\"kind\": \"pages|missing|overflow|style|other\", "
        "\"detail\": \"...\", \"severity\": \"high|low\"}]}. A page-count "
        "mismatch or a missing required section is severity \"high\". If it looks "
        "good, return meets_rubric true with an empty issues list. Do NOT invent "
        "problems."
    )


def _findings_text(report: "VisualQAReport") -> str:
    """One paragraph the rewriter can act on — the high issues first."""
    lines = []
    for i in sorted(report.issues, key=lambda x: 0 if x.severity == "high" else 1):
        mark = "MUST FIX" if i.severity == "high" else "consider"
        lines.append(f"- [{mark}] ({i.kind}) {i.detail}".rstrip())
    return "\n".join(lines)


_REPAIR_SYS = (
    "You revise a document's SOURCE (markdown) to fix specific visual/layout "
    "problems a reviewer found in its rendered pages. Keep ALL existing "
    "information — only restructure, trim, or add so the noted problems are "
    "resolved (e.g. a missing section is added, an overflow is shortened, a "
    "page-count mismatch is addressed by tightening or expanding). Output ONLY "
    "the full revised markdown, no commentary, no code fences."
)


async def _rewrite_source(content: str, findings: str, rubric_text: str) -> str:
    """LLM rewrite of the source addressing `findings`. Fail-open → '' (caller
    keeps the original). Uses the core llm_client facade (existing verify→core
    edge) so this module stays free of an llm import."""
    try:
        from app.core import llm_client
        user = (
            f"Requirement checklist:\n{rubric_text or '(general quality)'}\n\n"
            f"Reviewer findings on the rendered pages:\n{findings}\n\n"
            f"Current source markdown:\n{content}"
        )
        raw = await llm_client.llm.complete(
            [{"role": "system", "content": _REPAIR_SYS},
             {"role": "user", "content": user}],
            options={"temperature": 0.1, "max_tokens": 4000})
        out = (raw or "").strip()
        # Strip an accidental fenced wrapper.
        if out.startswith("```"):
            nl = out.find("\n")
            out = out[nl + 1:] if nl >= 0 else out
            if out.rstrip().endswith("```"):
                out = out.rstrip()[:-3]
        return out.strip()
    except Exception:  # noqa: BLE001
        return ""


def _severity_cost(report: "VisualQAReport") -> tuple[int, int]:
    """(high count, total count) — lower is better; the loop keeps the min."""
    highs = sum(1 for i in report.issues if i.severity == "high")
    return (highs, len(report.issues))


async def repair_visual_qa(
    content: str,
    pdf_bytes: bytes,
    report: "VisualQAReport",
    rubric_text: str,
    *,
    render_fn: "Callable[[str], object]",
    describe_fn: Callable[[list[str], str], Awaitable[str]],
    page_hint: int | None = None,
    max_rounds: int = 1,
) -> tuple[bytes, "VisualQAReport"]:
    """Bounded visual-QA repair loop (§3.3): while the report has HIGH issues,
    rewrite the source to address them, re-render (via the injected `render_fn`,
    which returns PDF bytes), and re-check. Keeps the fewest-issues bytes seen.

    Advisory + fail-open: any error (or no improvement) ships the best bytes so
    far. `render_fn(revised_content)` may be sync or async and must return the
    rendered PDF bytes (the api layer wraps its own render closure)."""
    best_bytes, best_report = pdf_bytes, report
    cur_content = content
    try:
        for _ in range(max(0, int(max_rounds))):
            if best_report.ok or not best_report.blocking_issues:
                break
            revised = await _rewrite_source(
                cur_content, _findings_text(best_report), rubric_text)
            if not revised or revised == cur_content:
                break
            try:
                res = render_fn(revised)
                if _is_awaitable(res):
                    res = await res
                new_bytes = res if isinstance(res, (bytes, bytearray)) else None
            except Exception:  # noqa: BLE001
                new_bytes = None
            if not new_bytes:
                break
            new_report = await visual_qa(
                bytes(new_bytes), rubric_text,
                describe_fn=describe_fn, page_hint=page_hint)
            # Only accept a strict improvement (fewer high, then fewer total);
            # never regress a passing render into a worse one.
            if _severity_cost(new_report) < _severity_cost(best_report):
                best_bytes, best_report, cur_content = (
                    bytes(new_bytes), new_report, revised)
            else:
                break
    except Exception:  # noqa: BLE001
        return best_bytes, best_report
    return best_bytes, best_report


def _is_awaitable(obj) -> bool:
    import asyncio
    return asyncio.iscoroutine(obj) or asyncio.isfuture(obj)


async def visual_qa(
    pdf_bytes: bytes,
    rubric_text: str,
    *,
    describe_fn: Callable[[list[str], str], Awaitable[str]],
    page_hint: int | None = None,
) -> VisualQAReport:
    """Rasterize the PDF, have the injected VLM (`describe_fn(images_b64, prompt)`)
    critique it against `rubric_text`, and return a structured report. Never
    raises — any failure yields a passing, un-ran report (fail-open)."""
    pages = rasterize_pdf(pdf_bytes)
    if not pages:
        return VisualQAReport(ok=True, ran=False)
    try:
        raw = await describe_fn(pages, _critique_prompt(rubric_text, len(pages)))
    except Exception as exc:  # noqa: BLE001
        log.info("visual_qa: VLM critique failed: %s", exc)
        return VisualQAReport(ok=True, ran=False, page_count=len(pages))
    if not (raw or "").strip():
        return VisualQAReport(ok=True, ran=False, page_count=len(pages))
    try:
        from app.core.structured import parse_with_repair
        obj, errs = parse_with_repair(raw, _CRITIQUE_SCHEMA)
    except Exception:  # noqa: BLE001
        obj, errs = None, ["parse error"]
    if obj is None or errs:
        return VisualQAReport(ok=True, ran=False, page_count=len(pages))

    issues = [
        VisualIssue(
            kind=str(i.get("kind") or "other"),
            detail=str(i.get("detail") or ""),
            severity=("high" if str(i.get("severity")) == "high" else "low"),
        )
        for i in (obj.get("issues") or []) if isinstance(i, dict)
    ]
    pc = obj.get("page_count")
    try:
        pc = int(pc) if pc is not None else len(pages)
    except (TypeError, ValueError):
        pc = len(pages)
    meets = bool(obj.get("meets_rubric", True))
    # A confident page-count mismatch is a high issue even if the model forgot.
    if page_hint and pc and pc != page_hint and not any(
            i.kind == "pages" for i in issues):
        issues.append(VisualIssue("pages",
                                   f"rendered {pc} pages, expected {page_hint}",
                                   "high"))
        meets = False
    ok = meets and not any(i.severity == "high" for i in issues)
    return VisualQAReport(ok=ok, page_count=pc, issues=issues, ran=True)


__all__ = ["VisualIssue", "VisualQAReport", "visual_qa", "rasterize_pdf",
           "repair_visual_qa"]
