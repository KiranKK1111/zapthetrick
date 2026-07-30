"""Diagram artifact API — the IR pipeline exposed to the client.

`routes_mermaid.py` owns the ONE model-repairs-syntax endpoint that the existing
free-text path needs. This router owns everything the IR makes possible
(MermaidDiagramVisualizations.md):

    POST /api/diagram/compose    plan → IR → deterministic Mermaid → validate →
                                 score → layout → version   (#1, #4, #5, #6, #17)
    POST /api/diagram/validate   validators + quality score for any source  (#4, #6, #7)
    POST /api/diagram/critique   second-model review returning edit ops      (#15)
    POST /api/diagram/edit       natural-language edit → ops → AST → Mermaid  (#10)
    POST /api/diagram/export     one diagram → Mermaid/PlantUML/DOT/Draw.io/ELK/JSON (#16)
    POST /api/diagram/normalize  lift raw Mermaid into the IR and re-emit it
    GET  /api/diagram/stages     stage vocabulary + how the answer path is
                                 configured (IR lane on? planner mode?)
    GET  /api/diagram/versions/{id}      history                             (#9)
    POST /api/diagram/versions/restore   make an earlier version current     (#9)
    GET  /api/diagram/stages     the stage vocabulary the UI paints          (#17)

Every endpoint is fail-open: the model half can be unavailable and the
deterministic half still answers. Nothing here raises into a client — a failure
comes back as a normal response carrying `ok: false` and the reason, because a
diagram is an enhancement to an answer and must never break the turn that
produced it.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

import app.diagrams.edits as E
import app.diagrams.export as X
import app.diagrams.ir as IR
import app.diagrams.lane as LANE
import app.diagrams.layout as L
import app.diagrams.planner as P
import app.diagrams.quality as Q
import app.diagrams.stages as S
import app.diagrams.validators as V
from app.diagrams.critic import critique as run_critique
from app.diagrams.parse import from_mermaid
from app.diagrams.versions import diagram_id, versions

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/diagram", tags=["diagram"])

MAX_SOURCE = 40_000


# ---- shared shapes --------------------------------------------------------
class _Base(BaseModel):
    """Either a raw mermaid `source` or a structured `ir` identifies a diagram.

    `source` is what the FE has in hand (it holds Mermaid text, not IR), so every
    endpoint accepts it and lifts it via the parser; `ir` short-circuits the lift
    when a caller already has the structure.
    """
    source: str = Field(default="", max_length=MAX_SOURCE)
    ir: dict | None = None
    diagram_id: str = Field(default="", max_length=64)


def _resolve(payload: _Base) -> tuple[IR.DiagramIR, str]:
    """→ (ir, source). Prefers an explicit IR; else lifts the source."""
    if payload.ir:
        structure = IR.from_dict(payload.ir)
        return structure, (payload.source or IR.to_mermaid(structure))
    return from_mermaid(payload.source), payload.source


def _key(payload: _Base, source: str) -> str:
    return (payload.diagram_id or "").strip() or diagram_id(source)


def _report(structure: IR.DiagramIR, source: str) -> tuple[dict, Q.QualityScore]:
    quality, report = Q.score(structure, source=source)
    return report.to_dict(), quality


# ---- compose (#1) --------------------------------------------------------
class ComposeRequest(BaseModel):
    request: str = Field(..., max_length=8_000)
    kind: str = Field(default="", max_length=32)
    context: str = Field(default="", max_length=8_000)
    diagram_id: str = Field(default="", max_length=64)
    # Let the layout planner override an explicit direction when the geometry
    # clearly disagrees with it.
    auto_layout: bool = True


class ComposeResponse(BaseModel):
    ok: bool
    source: str = ""
    ir: dict | None = None
    validation: dict | None = None
    quality: dict | None = None
    layout: dict | None = None
    stages: dict | None = None
    diagram_id: str = ""
    version: int | None = None
    errors: list[str] = []


@router.post("/compose", response_model=ComposeResponse)
async def compose(req: ComposeRequest) -> ComposeResponse:
    """Plan a diagram as IR, then GENERATE the Mermaid deterministically.

    This is the doc's #1 priority end to end. The model never emits Mermaid, so
    the failure modes the repair loop exists for cannot occur on this path.
    """
    tracker = S.StageTracker()
    tracker.begin("planning")
    structure, errors = await P.plan(
        req.request, kind=req.kind, context=req.context)
    if structure is None:
        tracker.fail("planning", "; ".join(errors[:2]) or "no structure")
        return ComposeResponse(ok=False, stages=tracker.frame(),
                               errors=errors or ["planning failed"])
    tracker.complete("planning", f"{len(structure.nodes)} nodes, "
                                 f"{len(structure.edges)} edges")

    tracker.begin("generating")
    source, plan = L.render_with_layout(
        structure, respect_explicit=not req.auto_layout)
    laid_out = structure.copy()
    laid_out.direction = plan.direction
    laid_out.label_wrap = plan.label_wrap
    tracker.complete("generating", f"{plan.direction}, {plan.renderer}")

    tracker.begin("validating")
    validation, quality = _report(laid_out, "")
    tracker.complete("validating", quality.summary)
    # Compiling + rendering happen in the client's mermaid webview; the ladder is
    # handed over with those still pending so the FE finishes it truthfully.

    key = (req.diagram_id or "").strip() or diagram_id(source)
    entry = versions.push(key, source, origin="compose",
                          note=req.request[:120], score=quality.overall,
                          ir=laid_out.to_dict())
    return ComposeResponse(
        ok=True, source=source, ir=laid_out.to_dict(), validation=validation,
        quality=quality.to_dict(), layout=plan.to_dict(),
        stages=tracker.frame(), diagram_id=key, version=entry.version,
        errors=errors or [])


# ---- validate (#4, #6, #7) -----------------------------------------------
class ValidateRequest(_Base):
    pass


class ValidateResponse(BaseModel):
    ok: bool
    validation: dict
    quality: dict
    ir: dict
    layout: dict | None = None
    unparsed: list[str] = []


@router.post("/validate", response_model=ValidateResponse)
async def validate(req: ValidateRequest) -> ValidateResponse:
    """Run the four-validator stack and score the result. No model involved, so
    this is cheap enough to run on every render."""
    structure, source = _resolve(req)
    validation, quality = _report(structure, source)
    return ValidateResponse(
        ok=validation["ok"], validation=validation, quality=quality.to_dict(),
        ir=structure.to_dict(),
        layout=L.plan_layout(structure).to_dict(),
        unparsed=list(structure.meta.get("unparsed") or []))


# ---- critique (#15) ------------------------------------------------------
class CritiqueRequest(_Base):
    request: str = Field(default="", max_length=4_000)
    apply: bool = False          # apply the critic's ops and return new source


class CritiqueResponse(BaseModel):
    ok: bool
    critique: dict
    source: str = ""             # only when `apply` produced a change
    ir: dict | None = None
    quality: dict | None = None
    applied: list[str] = []
    rejected: list[dict] = []
    version: int | None = None
    diagram_id: str = ""


@router.post("/critique", response_model=CritiqueResponse)
async def critique(req: CritiqueRequest) -> CritiqueResponse:
    """A second model reviews the diagram and returns edit OPS (never a rewrite).

    With `apply: true` the ops are run through the deterministic applier and the
    improved diagram comes back with a fresh score, so the round trip is one call.
    """
    structure, source = _resolve(req)
    if not structure.nodes:
        return CritiqueResponse(ok=False, critique={"verdict": "rebuild",
                                                    "assessment": "empty diagram",
                                                    "issues": [], "ops": [],
                                                    "errors": ["no nodes"]})
    report = V.validate(structure, source=source)
    verdict = await run_critique(structure, request=req.request, source=source,
                                 findings=report.findings)
    response = CritiqueResponse(ok=not verdict.errors, critique=verdict.to_dict(),
                                diagram_id=_key(req, source))
    if not (req.apply and verdict.ops):
        return response

    result = E.apply_edits(structure, verdict.ops)
    if not result.changed:
        response.rejected = result.rejected
        return response
    new_source, _plan = L.render_with_layout(result.ir)
    quality, _report = Q.score(result.ir)
    entry = versions.push(response.diagram_id, new_source, origin="critic",
                          note=verdict.assessment[:120], score=quality.overall,
                          ir=result.ir.to_dict())
    response.source = new_source
    response.ir = result.ir.to_dict()
    response.quality = quality.to_dict()
    response.applied = E.describe_ops(result.applied)
    response.rejected = result.rejected
    response.version = entry.version
    return response


# ---- edit (#10) ----------------------------------------------------------
class EditRequest(_Base):
    command: str = Field(..., max_length=1_000)
    # Pre-computed ops skip the model entirely — a UI button ("switch to LR")
    # is a deterministic edit and shouldn't cost a round trip.
    ops: list[dict] | None = None


class EditResponse(BaseModel):
    ok: bool
    source: str = ""
    ir: dict | None = None
    applied: list[str] = []
    rejected: list[dict] = []
    note: str = ""
    quality: dict | None = None
    validation: dict | None = None
    diagram_id: str = ""
    version: int | None = None
    errors: list[str] = []


@router.post("/edit", response_model=EditResponse)
async def edit(req: EditRequest) -> EditResponse:
    """Targeted edit: NL command → ops → AST → deterministic Mermaid.

    The diagram the user liked is preserved; only what they asked for changes.
    """
    structure, source = _resolve(req)
    key = _key(req, source)
    if not structure.nodes:
        return EditResponse(ok=False, diagram_id=key,
                            errors=["the diagram has no nodes to edit"])

    ops = [op for op in (req.ops or []) if isinstance(op, dict)]
    note, errors = "", []
    if not ops:
        ops, note, errors = await P.plan_edit(structure, req.command)
    if not ops:
        return EditResponse(ok=False, diagram_id=key, note=note,
                            errors=errors or ["no applicable change was found"])

    result = E.apply_edits(structure, ops)
    if not result.changed:
        return EditResponse(ok=False, diagram_id=key, note=note,
                            rejected=result.rejected,
                            errors=errors + ["every operation was rejected"])
    new_source, _plan = L.render_with_layout(result.ir)
    quality, report = Q.score(result.ir)
    entry = versions.push(key, new_source, origin="edit",
                          note=req.command[:120], score=quality.overall,
                          ir=result.ir.to_dict())
    return EditResponse(
        ok=True, source=new_source, ir=result.ir.to_dict(),
        applied=E.describe_ops(result.applied), rejected=result.rejected,
        note=note, quality=quality.to_dict(), validation=report.to_dict(),
        diagram_id=key, version=entry.version, errors=errors)


# ---- export (#16) --------------------------------------------------------
class ExportRequest(_Base):
    format: str = Field(default=X.MERMAID, max_length=16)
    stem: str = Field(default="diagram", max_length=64)
    all_formats: bool = False


class ExportResponse(BaseModel):
    ok: bool
    format: str = ""
    content: str = ""
    filename: str = ""
    mime: str = ""
    formats: dict | None = None
    available: dict


@router.post("/export", response_model=ExportResponse)
async def export(req: ExportRequest) -> ExportResponse:
    """Export the diagram as any TEXT format. Pixels stay with the renderer —
    the FE's mermaid webview already produces PNG and embeds diagrams into
    generated documents."""
    structure, source = _resolve(req)
    available = {key: {"label": label, "ext": ext, "mime": mime}
                 for key, (label, ext, mime) in X.EXPORT_FORMATS.items()}
    if req.all_formats:
        return ExportResponse(
            ok=True, formats=X.export_all(structure, mermaid_src=source,
                                          stem=req.stem),
            available=available)
    result = X.export(structure, req.format, mermaid_src=source, stem=req.stem)
    return ExportResponse(ok=bool(result.content), format=result.format,
                          content=result.content, filename=result.filename,
                          mime=result.mime, available=available)


# ---- normalize -----------------------------------------------------------
class NormalizeResponse(BaseModel):
    ok: bool
    source: str
    ir: dict
    changed: bool
    layout: dict
    quality: dict
    unparsed: list[str] = []
    # The mermaid diagram type the IR cannot model (gantt, pie, journey, …), when
    # that is WHY the rebuild was refused. Surfaced so the client can say so
    # instead of showing a generic failure.
    unsupported_kind: str = ""


@router.post("/normalize", response_model=NormalizeResponse)
async def normalize(req: ValidateRequest) -> NormalizeResponse:
    """Lift raw Mermaid into the IR and re-emit it deterministically.

    This is how an EXISTING (model-written or hand-edited) diagram joins the IR
    pipeline: whatever the parser understood is re-generated with correct quoting,
    balanced blocks and a planned layout. `unparsed` is returned honestly so a
    caller can see what the lift did not understand rather than trusting a
    lossy round trip.
    """
    structure, source = _resolve(req)
    refused = str(structure.meta.get("unsupported_kind") or "")
    if refused or not structure.nodes:
        # Never re-emit a diagram type the IR does not model — that would destroy
        # it. Same gate the answer-path compile lane uses.
        quality, _report = Q.score(structure, source=source)
        return NormalizeResponse(
            ok=False, source=source, ir=structure.to_dict(), changed=False,
            layout=L.plan_layout(structure).to_dict(),
            quality=quality.to_dict(),
            unparsed=list(structure.meta.get("unparsed") or []),
            unsupported_kind=refused)
    new_source, plan = L.render_with_layout(structure)
    quality, _report = Q.score(structure, source=source)
    return NormalizeResponse(
        ok=True, source=new_source, ir=structure.to_dict(),
        changed=new_source.strip() != (source or "").strip(),
        layout=plan.to_dict(), quality=quality.to_dict(),
        unparsed=list(structure.meta.get("unparsed") or []))


# ---- versions (#9) -------------------------------------------------------
class VersionsResponse(BaseModel):
    diagram_id: str
    versions: list[dict]
    head: int | None = None
    stats: dict


@router.get("/versions/{key}", response_model=VersionsResponse)
async def list_versions(key: str) -> VersionsResponse:
    """History for a diagram. Summaries only — no source bodies — so listing is
    cheap. In-memory and process-local (see `diagrams/versions.py`)."""
    entries = versions.list(key)
    return VersionsResponse(
        diagram_id=key, versions=[entry.summary() for entry in entries],
        head=entries[-1].version if entries else None,
        stats=versions.stats())


class SaveVersionRequest(_Base):
    origin: str = Field(default="manual", max_length=24)
    note: str = Field(default="", max_length=200)


class VersionResponse(BaseModel):
    ok: bool
    diagram_id: str = ""
    version: int | None = None
    source: str = ""
    quality: dict | None = None


@router.post("/versions/save", response_model=VersionResponse)
async def save_version(req: SaveVersionRequest) -> VersionResponse:
    """Record the current source as a version (e.g. after a hand edit in the
    source view). A push identical to the head is a no-op."""
    structure, source = _resolve(req)
    if not (source or "").strip():
        return VersionResponse(ok=False)
    key = _key(req, source)
    quality, _report = Q.score(structure, source=source)
    entry = versions.push(key, source, origin=req.origin, note=req.note,
                          score=quality.overall, ir=structure.to_dict())
    return VersionResponse(ok=True, diagram_id=key, version=entry.version,
                           source=entry.source, quality=quality.to_dict())


class RestoreRequest(BaseModel):
    diagram_id: str = Field(..., max_length=64)
    version: int


@router.post("/versions/restore", response_model=VersionResponse)
async def restore_version(req: RestoreRequest) -> VersionResponse:
    """Make an earlier version current. Append-only: the restore is itself a new
    version, so it can be undone."""
    entry = versions.restore(req.diagram_id, req.version)
    if entry is None:
        return VersionResponse(ok=False, diagram_id=req.diagram_id)
    quality, _report = Q.score_source(entry.source)
    return VersionResponse(ok=True, diagram_id=req.diagram_id,
                           version=entry.version, source=entry.source,
                           quality=quality.to_dict())


# ---- stages (#17) --------------------------------------------------------
@router.get("/stages")
async def stage_vocabulary() -> dict:
    """The ordered stage ladder + the formats/ops the client can offer.

    One source of truth: the FE mirrors these ids so its progress ladder and the
    server's `stages` frames can never drift apart.
    """
    return {
        "stages": [{"id": sid, "label": label, "detail": detail,
                    "conditional": sid in S.CONDITIONAL}
                   for sid, label, detail in S.STAGES],
        "states": [S.PENDING, S.ACTIVE, S.DONE, S.FAILED, S.SKIPPED],
        "formats": {key: {"label": label, "ext": ext, "mime": mime}
                    for key, (label, ext, mime) in X.EXPORT_FORMATS.items()},
        "ops": list(E.OPS),
        "kinds": list(IR.KINDS),
        "elk_enabled": L.elk_available(),
        "pass_threshold": Q.PASS_THRESHOLD,
        # How the answer path treats diagrams right now, so a client can say
        # whether a diagram was generated from structure or recompiled from text.
        "ir_lane_enabled": LANE.enabled(),
        "planner_mode": LANE.planner_mode(),
    }
