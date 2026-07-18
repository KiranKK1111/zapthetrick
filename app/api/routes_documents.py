"""Document export — generate a downloadable file from Markdown content.

`POST /api/documents/export` takes the content the user wants to save plus a
target format (md / txt / csv / xlsx / docx / pdf) and streams back the
generated file with a `Content-Disposition` so the client can save it.

Stateless on purpose: the Flutter client already holds the assistant's answer
(or the chosen artifact), so it just sends that text — no DB lookup, no
persistence. This is the Claude-style "download this as …" action.
"""
from __future__ import annotations

import base64
import re
import time
import urllib.parse

import contextlib

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.documents.generators import (
    SUPPORTED_FORMATS,
    UnsupportedFormat,
    apply_resume_template,
    normalize_format,
    render_document,
)

router = APIRouter(prefix="/api")


class DocumentExportRequest(BaseModel):
    content: str = Field(..., description="Markdown / text to convert.")
    format: str = Field(..., description="md|txt|csv|xlsx|docx|pdf (+ aliases).")
    filename: str | None = Field(
        None, description="Base filename (no extension needed)."
    )
    title: str | None = Field(None, description="Optional document title.")
    template: str | None = Field(
        None, description="Optional design template id (see GET "
                          "/api/documents/templates). Omit for the default "
                          "layout — the content is then rendered as-is.")
    language: str | None = Field(
        None, description="Optional target language for the auto-generated "
                          "furniture (Table of Contents / Glossary / captions) — "
                          "code or name (see GET /api/documents/languages). "
                          "Omit or 'en' for English (default).")


def _check_template(template: str | None) -> str | None:
    """Validate an optional design-template id. Empty → None (default path,
    unchanged behavior); unknown → 400 with the choices."""
    name = (template or "").strip().lower()
    if not name:
        return None
    from app.documents.templates import TEMPLATES
    if name not in TEMPLATES:
        raise HTTPException(
            400, detail=f"Unknown template '{template}'. Use one of: "
            + ", ".join(TEMPLATES))
    return name


def _safe_stem(name: str | None, fallback: str = "document") -> str:
    stem = (name or "").strip()
    # Drop any extension the client tacked on; we add the canonical one.
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", stem)
    stem = re.sub(r"[^\w\- ]+", "", stem).strip().replace(" ", "_")
    return stem[:80] or fallback


@router.get("/documents/formats")
async def list_formats() -> dict:
    """The export formats the UI can offer."""
    return {"formats": list(SUPPORTED_FORMATS)}


@router.get("/documents/languages")
async def list_document_languages() -> dict:
    """Phase 7 — the languages the export furniture (TOC/glossary/captions) can be
    localized into. The code goes back as `language` on /documents/export. Fail-
    open to just English so the UI can still offer the default."""
    try:
        from app.documents.localization import supported_languages
        return {"languages": supported_languages()}
    except Exception:  # noqa: BLE001
        return {"languages": [{"code": "en", "name": "English",
                               "native": "English", "rtl": False}]}


@router.get("/documents/templates")
async def list_document_templates() -> dict:
    """Phase 4 (#21) — the ATS-safe design templates a resume can be rendered
    through. The id goes back as `template` on /documents/export (or
    /documents/preview); omitting it keeps the default layout. Fail-open to an
    empty list so the UI just hides the picker."""
    try:
        from app.documents.templates import list_templates
        return {"templates": list_templates()}
    except Exception:  # noqa: BLE001
        return {"templates": []}


def _artifact_summary(r) -> dict:
    return {
        "doc_key": str(r.doc_key), "version": r.version, "title": r.title,
        "format": r.doc_format, "goal": r.goal,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@router.get("/documents/artifacts")
async def list_document_artifacts(session_id: str) -> dict:
    """Phase 5 — the generated documents persisted for a conversation (newest
    first). Fail-open: an empty list when the store is unavailable."""
    try:
        from app.documents.store import list_for_session
        from storage.db import get_session_factory
        factory = get_session_factory()
        if factory is None:
            return {"artifacts": []}
        async with factory() as s:
            rows = await list_for_session(s, session_id)
            return {"artifacts": [_artifact_summary(r) for r in rows]}
    except Exception:  # noqa: BLE001
        return {"artifacts": []}


@router.get("/documents/artifacts/{doc_key}/versions")
async def list_artifact_versions(doc_key: str) -> dict:
    """Phase 5 — the evolution timeline of one document (all versions, oldest
    first) with the source content for each."""
    try:
        from app.documents.store import list_versions
        from storage.db import get_session_factory
        factory = get_session_factory()
        if factory is None:
            return {"versions": []}
        async with factory() as s:
            rows = await list_versions(s, doc_key)
            return {"versions": [
                {**_artifact_summary(r), "content_md": r.content_md}
                for r in rows]}
    except Exception:  # noqa: BLE001
        return {"versions": []}


@router.get("/documents/artifacts/{doc_key}/diff")
async def diff_artifact_versions(
    doc_key: str,
    from_version: int | None = Query(
        None, alias="from", ge=1,
        description="Base version. Default: the one before `to`."),
    to_version: int | None = Query(
        None, alias="to", ge=1,
        description="Target version. Default: the latest."),
) -> dict:
    """Phase 5 (#22) — what CHANGED between two versions of one document.

    Section-level diff (added / removed / changed / unchanged headings, keyed by
    semantic anchor) plus both versions' source Markdown, so the client can also
    render a line-level before→after. With no `from`/`to` it compares the two
    most recent versions ("what changed in the last edit?").

    Unlike the list endpoints this does NOT fail open to an empty body — a diff
    the caller explicitly asked for must not silently look like "no changes".
    """
    try:
        from app.documents.store import list_versions
        from storage.db import get_session_factory
        factory = get_session_factory()
        if factory is None:
            raise HTTPException(503, detail="Document store is unavailable.")
        async with factory() as s:
            rows = await list_versions(s, doc_key)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, detail=f"Could not load versions: {exc}")

    by_version = {int(r.version): r for r in rows or []}
    if not by_version:
        raise HTTPException(404, detail=f"No versions for document {doc_key}.")

    to_v = int(to_version) if to_version is not None else max(by_version)
    if from_version is not None:
        from_v = int(from_version)
    else:
        older = [v for v in by_version if v < to_v]
        from_v = max(older) if older else to_v   # v1 only → diffs against itself
    for v in (from_v, to_v):
        if v not in by_version:
            raise HTTPException(
                404, detail=f"Version {v} not found for document {doc_key}.")

    old_row, new_row = by_version[from_v], by_version[to_v]
    try:
        from app.documents.lifecycle import diff_models
        from app.documents.model import markdown_to_model
        d = diff_models(markdown_to_model(old_row.content_md or ""),
                        markdown_to_model(new_row.content_md or ""))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, detail=f"Could not diff document: {exc}")

    return {
        "doc_key": str(doc_key),
        "title": new_row.title,
        "from_version": from_v,
        "to_version": to_v,
        "is_empty": d.is_empty,
        "diff": d.as_dict(),
        "from_content_md": old_row.content_md or "",
        "to_content_md": new_row.content_md or "",
    }


@router.get("/documents/artifacts/{doc_key}/staleness")
async def artifact_staleness(doc_key: str) -> dict:
    """Phase 5 — multi-format staleness. Which previously-exported formats of this
    document are now OUT OF DATE because a newer version of the source exists
    (and should be regenerated). Fail-open to a non-stale report."""
    try:
        from app.documents.staleness import staleness_for_document
        from storage.db import get_session_factory
        factory = get_session_factory()
        if factory is None:
            return {"doc_key": str(doc_key), "latest_version": 0,
                    "formats": [], "stale_formats": [], "any_stale": False}
        async with factory() as s:
            return await staleness_for_document(s, doc_key)
    except Exception:  # noqa: BLE001
        return {"doc_key": str(doc_key), "latest_version": 0,
                "formats": [], "stale_formats": [], "any_stale": False}


@router.get("/documents/search")
async def search_document_artifacts(q: str,
                                    session_id: str | None = None) -> dict:
    """Phase 6 — cross-artifact text search over generated documents (optionally
    scoped to one conversation). Fail-open to no results."""
    try:
        from app.documents.graph import search_documents
        from storage.db import get_session_factory
        factory = get_session_factory()
        if factory is None:
            return {"results": []}
        async with factory() as s:
            return {"results": await search_documents(
                s, q, session_id=session_id)}
    except Exception:  # noqa: BLE001
        return {"results": []}


@router.get("/documents/graph")
async def document_relationship_graph(session_id: str) -> dict:
    """Phase 6 — the artifact relationship graph for a conversation (documents +
    their version chains + sibling edges)."""
    try:
        from app.documents.graph import build_artifact_graph
        from storage.db import get_session_factory
        factory = get_session_factory()
        if factory is None:
            return {"documents": [], "edges": []}
        async with factory() as s:
            return await build_artifact_graph(s, session_id)
    except Exception:  # noqa: BLE001
        return {"documents": [], "edges": []}


class CompletionRequest(BaseModel):
    content: str = Field(..., description="Conversation / project text to assess.")


@router.post("/documents/completion")
async def document_completion(body: CompletionRequest) -> dict:
    """Phase 6 — goal-completion: which deliverables the project has, which are
    missing, and the concrete next-step suggestions."""
    try:
        from app.documents.completion import completion_report
        return completion_report(body.content or "")
    except Exception:  # noqa: BLE001
        return {"project_type": "generic", "present": [], "missing": [],
                "completion_pct": 0, "suggestions": []}


@router.get("/documents/preferences")
async def get_document_preferences() -> dict:
    """Phase 7 — the remembered document-generation preferences (persona,
    branding, defaults). Fail-open to empty."""
    try:
        from app.documents.preferences import get_preferences
        return {"preferences": await get_preferences()}
    except Exception:  # noqa: BLE001
        return {"preferences": {}}


class PreferencesRequest(BaseModel):
    preferences: dict = Field(default_factory=dict)


@router.put("/documents/preferences")
async def put_document_preferences(body: PreferencesRequest) -> dict:
    """Phase 7 — persist document-generation preferences."""
    try:
        from app.documents.preferences import save_preferences
        ok = await save_preferences(body.preferences or {})
        return {"saved": ok}
    except Exception:  # noqa: BLE001
        return {"saved": False}


@router.get("/documents/metrics")
async def document_metrics() -> dict:
    """Phase 8 (subset) — document-generation usage metrics (exports by format,
    failures, avg latency) to guide future improvements."""
    try:
        from app.documents.metrics import snapshot
        return snapshot()
    except Exception:  # noqa: BLE001
        return {"exports": 0, "failures": 0, "by_format": {},
                "avg_latency_ms": 0.0}


# Repair/beautify supported types → (media type, output extension).
_REPAIR_MIME = {
    "xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    "pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"),
    "pdf": ("application/pdf", "pdf"),
    "md": ("text/markdown; charset=utf-8", "md"),
    "code": ("text/plain; charset=utf-8", ""),
    "text": ("text/plain; charset=utf-8", "txt"),
}


@router.post("/documents/repair")
async def repair_document_endpoint(
    file: UploadFile = File(...),
    automate: bool = Form(False),
) -> Response:
    """Repair/beautify an UPLOADED document in its native format and stream the
    fixed file back. In-place for binary Office/PDF (openpyxl/python-docx/
    python-pptx/PyMuPDF); deterministic beautify for markdown; format+lint for
    code. `automate=true` also adds Excel SUM totals. The repair report rides
    the `X-Repair-Report` header."""
    orig = (file.filename or "file").strip()
    data = await file.read()
    if not data:
        raise HTTPException(400, detail="Empty file — nothing to repair.")
    from app.obs.jobs import jobs as _jobs
    _jid = _jobs().start(f"Repair · {orig}", kind="repair")
    try:
        from app.documents.repair import repair_document
        res = await repair_document(data, orig, automate=automate)
    except Exception as exc:  # noqa: BLE001
        _jobs().finish(_jid, ok=False, detail=str(exc)[:80])
        raise HTTPException(422, detail=f"Could not repair {orig}: {exc}")
    if not res.ok:
        _jobs().finish(_jid, ok=False, detail=res.reason[:80])
        raise HTTPException(422, detail=res.reason or "Could not repair the file.")

    mime, out_ext = _REPAIR_MIME.get(res.kind, ("application/octet-stream", ""))
    body = res.data if res.data else res.text.encode("utf-8")
    stem = _safe_stem(orig)
    # Keep the source extension for code (language matters); else the mapped one.
    ext = out_ext or (orig.rsplit(".", 1)[-1].lower() if "." in orig else "txt")
    filename = f"{stem}-repaired.{ext}"
    disposition = (
        f"attachment; filename=\"{filename}\"; "
        f"filename*=UTF-8''{urllib.parse.quote(filename)}"
    )
    headers = {"Content-Disposition": disposition,
               "Content-Length": str(len(body))}
    try:
        import json as _json
        headers["X-Repair-Report"] = _json.dumps(res.to_dict(),
                                                 separators=(",", ":"))[:4000]
    except Exception:  # noqa: BLE001
        pass
    _jobs().finish(_jid, ok=True)
    return Response(content=body, media_type=mime, headers=headers)


class DocumentPreviewRequest(BaseModel):
    content: str = Field(..., description="Markdown / text to preview.")
    title: str | None = Field(None, description="Optional document title.")
    dpi: int = Field(120, ge=72, le=200, description="Raster DPI per page.")
    template: str | None = Field(
        None, description="Optional design template id (see GET "
                          "/api/documents/templates) — preview the resume in "
                          "that layout. Omit for the default.")


# Extensions the sandbox can't resolve via `canonical()` alone.
_EXT_LANG_FALLBACK = {
    "ex": "elixir", "exs": "elixir", "mjs": "javascript", "cjs": "javascript",
    "h": "c", "hpp": "cpp", "cc": "cpp", "cxx": "cpp", "bash": "bash",
    "zsh": "bash", "pl": "perl", "r": "r", "jl": "julia",
}


class CodeVerifyRequest(BaseModel):
    content: str = Field(..., description="The raw source code to verify.")
    language: str | None = Field(
        None, description="Language id or file extension (py, js, cpp, …).")


_CODE_REPAIR_SYS = (
    "You fix a single self-contained source file so it COMPILES and RUNS "
    "cleanly. Output ONLY the corrected file in one fenced code block — no prose, "
    "no explanation. Keep the program's intent and public names; change only what "
    "is needed to make it build and run.")


def _looks_like_compile_error(stderr: str) -> bool:
    """Heuristic: does the failure look like a COMPILE/syntax error (a real
    defect) rather than a runtime error from missing stdin/args (expected for an
    input-driven file we ran with no data)?"""
    s = (stderr or "").lower()
    # Runtime/no-input signatures take PRECEDENCE — several of them literally
    # contain "error:" (EOFError:, IndexError:), so checking compile first would
    # misread a missing-stdin crash as a compile defect.
    _runtime_input = ("eoferror", "end of file", "eof when reading",
                      "index out of range", "list index", "brokenpipe",
                      "no input", "keyboardinterrupt", "stdin")
    if any(k in s for k in _runtime_input):
        return False
    _compile = ("syntaxerror", "indentationerror", "parse error",
                "cannot find symbol", "undeclared", "expected ",
                "unexpected token", "cannot compile", "compilation",
                "no such module", "error:", "is not defined", "unresolved",
                "cannot find")
    return any(k in s for k in _compile)


async def _repair_code_once(code: str, lang: str, stderr: str) -> str | None:
    """One bounded LLM repair pass for a source file that failed to build/run.
    Returns the corrected code, or None on any failure."""
    try:
        from app.core.llm_client import llm
        from app.codeintel.solution_verify import _extract_code_block
        user = (f"Language: {lang}\nThis file FAILED with:\n{(stderr or '')[:600]}"
                f"\n\nFix it:\n```\n{code[:6000]}\n```")
        txt, _ = await llm.complete_routed(
            [{"role": "system", "content": _CODE_REPAIR_SYS},
             {"role": "user", "content": user}], None, {"difficulty": "hard"})
        fixed = _extract_code_block(txt or "") or (txt or "")
        fixed = fixed.strip()
        return fixed if fixed and fixed != code.strip() else None
    except Exception:  # noqa: BLE001
        return None


@router.post("/documents/verify-code")
async def verify_code_file(body: CodeVerifyRequest) -> dict:
    """Compile/run a standalone source file in the sandbox before it's offered
    for download — the one downloadable type that had NO sandbox check.

    Best-effort and NON-blocking by contract: the client still lets the user
    download on any verdict; this only attaches an honest status and, on a real
    build failure, returns an auto-repaired `code` the client can save instead.
    Input-driven files (that read stdin/args) are not falsely failed. Never
    raises."""
    import asyncio

    code = (body.content or "").strip()
    raw_lang = (body.language or "").strip().lower().lstrip(".")
    if not code:
        return {"verified": False, "status": "empty",
                "detail": "nothing to verify", "language": raw_lang}
    try:
        from app.sandbox.lang_registry import canonical, container_supports
        from app.sandbox.executor import run_code
        lang = canonical(raw_lang) or _EXT_LANG_FALLBACK.get(raw_lang)
        if not lang:
            return {"verified": False, "status": "unsupported",
                    "detail": f"no sandbox toolchain for '.{raw_lang}'",
                    "language": raw_lang}
        try:
            if not container_supports(lang):
                return {"verified": False, "status": "unsupported",
                        "detail": f"{lang} isn't available in the sandbox",
                        "language": lang}
        except Exception:  # noqa: BLE001 — local backend has no container gate
            pass

        try:
            from app.codeintel.solution_verify import _reads_stdin
            _input_driven = _reads_stdin(code)
        except Exception:  # noqa: BLE001
            _input_driven = False

        async def _run(_code: str):
            return await asyncio.wait_for(
                asyncio.to_thread(run_code, _code, lang), timeout=60.0)

        res = await _run(code)
        _status = getattr(res, "status", "error")
        if _status == "ok":
            return {"verified": True, "status": "ok", "language": lang,
                    "detail": "compiled & ran cleanly in the sandbox"}
        if _status in ("unavailable", "timeout"):
            return {"verified": False, "status": _status, "language": lang,
                    "detail": "sandbox is unavailable"}
        _why = (getattr(res, "stderr", "") or getattr(res, "reason", "")
                or "compile/run failed").strip()
        # Input-driven file that failed WITHOUT a compile/syntax error → the
        # failure is almost certainly "we ran it with no input", not a defect.
        # Don't false-alarm: report a soft, non-blocking status.
        if _input_driven and not _looks_like_compile_error(_why):
            return {"verified": False, "status": "needs_input", "language": lang,
                    "detail": "builds; reads input at runtime (not run with "
                              "test data)"}
        # Real build/run failure → one bounded auto-repair, re-verify, and hand
        # back the corrected file so the user downloads a WORKING version.
        fixed = await _repair_code_once(code, lang, _why)
        if fixed:
            res2 = await _run(fixed)
            if getattr(res2, "status", "error") == "ok":
                return {"verified": True, "status": "repaired", "language": lang,
                        "code": fixed,
                        "detail": "had an error — auto-corrected & verified in "
                                  "the sandbox"}
        return {"verified": False, "status": _status, "language": lang,
                "detail": _why[:400]}
    except asyncio.TimeoutError:
        return {"verified": False, "status": "timeout", "language": raw_lang,
                "detail": "sandbox verification timed out"}
    except Exception as exc:  # noqa: BLE001 — verification never blocks a download
        return {"verified": False, "status": "error", "language": raw_lang,
                "detail": f"{type(exc).__name__}: {exc}"[:300]}


@router.post("/documents/preview")
async def preview_document(body: DocumentPreviewRequest) -> dict:
    """Render the content to a PDF, then rasterize each page to a PNG.

    Returns `{title, count, pages:[base64 png]}` so the client can show a
    Claude-style paged preview with no native PDF renderer — it just displays
    the page images. Download still offers every format via `/export`.
    """
    if not (body.content or "").strip():
        raise HTTPException(400, detail="Nothing to preview — content is empty.")
    title = (body.title or "").strip()
    _template = _check_template(body.template)
    try:
        pdf_bytes, _, _ = render_document(body.content, "pdf", title=title,
                                          template=_template)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, detail=f"Could not render preview: {exc}")

    try:
        import fitz  # PyMuPDF — already a dependency for PDF parsing.

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            pix = page.get_pixmap(dpi=body.dpi)
            pages.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        doc.close()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, detail=f"Could not rasterize preview: {exc}")

    return {"title": title, "count": len(pages), "pages": pages}


@router.post("/documents/export")
async def export_document(body: DocumentExportRequest) -> Response:
    if not (body.content or "").strip():
        raise HTTPException(400, detail="Nothing to export — content is empty.")
    # Phase-4 (#21) design layer: an explicitly chosen ATS-safe template
    # re-lays-out resume content before ANY of the render pipeline sees it, so
    # every format picks it up. No template → `content` is body.content
    # byte-for-byte (the default path is untouched).
    _template = _check_template(body.template)
    content = apply_resume_template(body.content, _template)
    # Track the export in the Task Center — archive verify/repair + sandbox
    # checks can take a moment, so this is a real user-facing background job.
    from app.obs.jobs import jobs as _jobs
    _jid = _jobs().start(
        f"Export · {(body.format or 'document').upper()}",
        kind="export", detail=(body.title or "").strip())
    try:
        fmt = normalize_format(body.format)
        # Phase-2 capability negotiation: when this deployment can't render the
        # requested format (renderer lib missing), degrade to the closest
        # format we CAN produce instead of failing — the substitution is
        # surfaced via the X-Format-Substituted header. Fail-open: a probe
        # error keeps today's behavior (attempt the requested format).
        _substituted = None
        try:
            from app.capabilities import negotiate_format
            _ok, _alt, _why = negotiate_format(fmt)
            if not _ok and _alt:
                _substituted = f"{fmt}->{_alt}: {_why}"
                fmt = normalize_format(_alt)
        except Exception:  # noqa: BLE001
            _substituted = None
    except UnsupportedFormat as exc:
        _jobs().finish(_jid, ok=False, detail="unsupported format")
        raise HTTPException(415, detail=str(exc))

    # Render OFF the event loop through the Document Job Manager (Phase 1b):
    # heavy PDF/DOCX/XLSX renders must not block the async server, and the
    # manager bounds concurrency + enforces a timeout. The per-job render_fn is
    # the Phase-4 render→validate→repair→degrade closed loop (render_validated);
    # its validation meta is captured via `_cap` and rides X-Artifact-Validation.
    _val_meta = None
    _cap: dict = {}
    _title = (body.title or "").strip()
    # Phase 7 localization: normalize the requested language (blank/English/
    # unknown → "" = default English furniture, unchanged output).
    try:
        from app.documents.localization import normalize_language
        _language = normalize_language(body.language) or ""
    except Exception:  # noqa: BLE001
        _language = ""

    # Phase 7 branding: remembered logo/colors/header/footer applied to the HTML
    # render (fail-open — no prefs → no branding, unchanged output).
    _export_settings = None
    try:
        from app.documents.preferences import (
            export_settings_from_prefs, get_preferences)
        _export_settings = export_settings_from_prefs(await get_preferences())
    except Exception:  # noqa: BLE001
        _export_settings = None

    # Phase 1b: run the deterministic render in an isolated SUBPROCESS when
    # `cfg.documents.sandbox_render` is on (crash/resource isolation). Default
    # OFF → in-process. Read once here; the closure below consults it per render.
    try:
        from app.core.config_loader import cfg as _cfg
        _sandbox_render = bool(getattr(_cfg.documents, "sandbox_render", False))
        _render_timeout = float(getattr(_cfg.documents, "export_timeout_s", 120.0))
    except Exception:  # noqa: BLE001
        _sandbox_render, _render_timeout = False, 120.0

    def _render(content: str, _fmt: str, title: str):
        # The render→validate→repair→degrade loop, branding threaded through so
        # EVERY format (not just HTML) picks up header/footer/confidentiality +
        # Phase-4 structure enrichment.
        if _sandbox_render:
            # Subprocess isolation — fail-open: any pool/timeout/pickle error
            # drops through to the in-process render below so a download is
            # never lost to a sandbox hiccup.
            try:
                from app.documents.render_isolated import render_isolated
                res = render_isolated(content, _fmt, title=title,
                                      export_settings=_export_settings,
                                      timeout=_render_timeout,
                                      language=_language)
                _cap["val_meta"] = res.get("val_meta")
                return res["data"], res["mime"], res["ext"]
            except Exception:  # noqa: BLE001 — fall back to in-process
                pass
        try:
            from app.documents.validators import render_validated
            d, m, e, vm = render_validated(content, _fmt, title=title,
                                           export_settings=_export_settings,
                                           language=_language)
            _cap["val_meta"] = vm
            return d, m, e
        except Exception:  # noqa: BLE001 — guard failure → legacy plain render
            return render_document(content, _fmt, title=title,
                                   export_settings=_export_settings,
                                   language=_language)

    from app.documents.jobs import JobStatus, get_manager
    # Time the render so the Phase-8 latency metric is real (it measures the
    # whole queue+render leg the user actually waits on, retries included).
    _t0 = time.perf_counter()
    try:
        _mgr = await get_manager()
        job = await _mgr.submit_and_wait(content, fmt, _title,
                                         render_fn=_render)
    except Exception as exc:  # noqa: BLE001 — surface generation failures cleanly
        _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
        with contextlib.suppress(Exception):
            from app.documents.metrics import record_export, record_template
            record_export(fmt, ok=False, latency_ms=_elapsed_ms)
            if _template:
                record_template(_template, ok=False)
        _jobs().finish(_jid, ok=False, detail=str(exc)[:80])
        raise HTTPException(422, detail=f"Could not generate {body.format}: {exc}")
    _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
    if job.status in (JobStatus.TIMEOUT,) or (
            job.status != JobStatus.DONE or job.result is None):
        with contextlib.suppress(Exception):
            from app.documents.metrics import record_export, record_template
            record_export(fmt, ok=False, latency_ms=_elapsed_ms)
            if _template:
                record_template(_template, ok=False)
    if job.status == JobStatus.TIMEOUT:
        _jobs().finish(_jid, ok=False, detail="render timed out")
        raise HTTPException(504, detail=f"Rendering {body.format} timed out.")
    if job.status != JobStatus.DONE or job.result is None:
        _jobs().finish(_jid, ok=False, detail=(job.error or "render failed")[:80])
        raise HTTPException(
            422, detail=f"Could not generate {body.format}: "
            f"{job.error or job.status.value}")
    data, mime, ext = job.result, job.mime, job.ext
    _val_meta = _cap.get("val_meta")
    with contextlib.suppress(Exception):        # Phase 8 metrics (fail-open)
        from app.documents.metrics import record_export, record_template
        record_export(ext, ok=True, latency_ms=_elapsed_ms)
        if _template:                           # template-success signal
            record_template(_template, ok=True)

    # Phase 3 — structural quality report on the SOURCE content (heading
    # hierarchy, empty sections, placeholders, duplication, completeness). Rides
    # X-Artifact-Validation so the UI can surface it. NON-blocking: a style nit
    # never refuses the user's download. Fail-open.
    try:
        # Phase 2 — the STAGED multi-pass assembler produces the quality report
        # (outline → content → structure → format → validate). Its outline pass
        # gives the validator a blueprint, so the completeness check is finally
        # live on the export path. `enrich_structure=False` — render_validated
        # already enriched the bytes, so we don't double-enrich here. Fail-open to
        # the direct reviewer.
        _q = None
        try:
            from app.documents.assembler import assemble_document, multi_pass_enabled
            if multi_pass_enabled():
                _asm = await assemble_document(
                    content, request_text=_title, title=_title,
                    enrich_structure=False)
                _q = _asm.quality
                if _val_meta is None:
                    _val_meta = {}
                _val_meta["assembly"] = {
                    "passes": _asm.passes,
                    "blueprint": (_asm.blueprint.as_dict()
                                  if _asm.blueprint is not None else None)}
        except Exception:  # noqa: BLE001
            _q = None
        if _q is None:
            from app.documents.review import analyze_document_async
            _q = await analyze_document_async(content, title=_title)
        if _val_meta is None:
            _val_meta = {}
        _val_meta["quality"] = _q.as_dict()
        # Strict quality gate (Phase 3, config-gated, default OFF): refuse the
        # download only on a HARD error (e.g. an empty section) so a broken
        # artifact never ships. Default is warn-not-block — the report rides the
        # X-Artifact-Validation header and the UI surfaces it.
        try:
            from app.core.config_loader import cfg
            _strict = bool(getattr(cfg.documents, "quality_strict", False))
        except Exception:  # noqa: BLE001
            _strict = False
        if _strict and _q.has_errors:
            _jobs().finish(_jid, ok=False, detail="quality gate failed")
            _errs = "; ".join(
                i.message for i in _q.issues if i.severity == "error")[:200]
            raise HTTPException(
                422, detail=f"Quality gate failed (score {_q.score}): {_errs}")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        pass

    # Closed PROJECT loop (user ask: end-to-end plan/build/verify/test like
    # Claude): a downloaded project archive is syntax-checked and its tests
    # are RUN in the dedicated sandbox; a failing build gets one model repair
    # round; a VERIFICATION.txt report ships inside the archive either way.
    # Fail-open — any error ships the original bytes.
    if ext in ("zip", "7z"):
        try:
            from app.verify.project_loop import verify_and_repair_archive
            data, _proj_meta = await verify_and_repair_archive(data, fmt=ext)
            if _proj_meta is not None and _val_meta is not None:
                _val_meta["project"] = _proj_meta
        except Exception:  # noqa: BLE001
            pass
    else:
        # Single documents get the same treatment as archives: the rendered
        # bytes are RE-OPENED by their own parser inside the sandbox
        # (docx/xlsx/pptx/pdf), so a file its own reader can't open never
        # ships silently. Best-effort; result rides X-Artifact-Validation.
        try:
            from app.verify.doc_verify import verify_document_bytes
            # Pass the SOURCE content so the sandbox can check the render
            # actually CONTAINS it (functional check, not just "does it open").
            _sbx = await verify_document_bytes(data, ext, expect_text=body.content)
            if _sbx is not None and _val_meta is not None:
                _val_meta["sandbox"] = _sbx
        except Exception:  # noqa: BLE001
            pass

    # NOTE: we deliberately do NOT persist a blob here. Export is stateless —
    # the source markdown already lives in the message, and both the preview and
    # download re-render from it. An earlier version wrote `generated/{hash}.ext`
    # that was never served back nor cleaned up on conversation delete, so the
    # blobs accumulated unbounded (orphan leak) for no benefit. Removed.
    stem = _safe_stem(body.filename)
    filename = f"{stem}.{ext}"
    # RFC 5987 so non-ASCII names survive the header.
    disposition = (
        f"attachment; filename=\"{filename}\"; "
        f"filename*=UTF-8''{urllib.parse.quote(filename)}"
    )
    _extra_headers = (
        {"X-Format-Substituted": _substituted} if _substituted else {})
    if _val_meta is not None:
        try:
            import json as _json
            _extra_headers["X-Artifact-Validation"] = _json.dumps(
                _val_meta, separators=(",", ":"))
        except Exception:  # noqa: BLE001
            pass
    _jobs().finish(_jid, ok=True)
    return Response(
        content=data,
        media_type=mime,
        headers={
            **_extra_headers,
            "Content-Disposition": disposition,
            "Content-Length": str(len(data)),
        },
    )
