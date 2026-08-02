"""Mermaid repair — the doc-prescribed compile/repair loop's LLM half.

The FE's webview is the VALIDATOR (mermaid.parse runs there and produces the
real parser error). When a diagram fails to parse even after the FE's
rule-based fixes, it posts the source + the exact parser diagnostics here; an
LLM performs a SYNTAX-ONLY repair (never redesigning the diagram) and the FE
compiles again — up to a small retry budget, exactly the
generate → compile → repair → compile pipeline from
MermaidDiagramVisualizations.md.
"""
from __future__ import annotations

import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

# The repair contract lives with the diagram domain logic; imported here
# (not the other way round) so `app/diagrams` never depends on a route.
from app.diagrams.repair_contract import _REPAIR_PROMPT, _strip_fences

router = APIRouter(prefix="/api/mermaid", tags=["mermaid"])

# The doc's repair contract, verbatim in spirit: fix ONLY the syntax, keep the
# architecture, return only mermaid.


class MermaidRepairRequest(BaseModel):
    source: str = Field(..., max_length=20_000)
    error: str = Field(default="", max_length=2_000)


class MermaidRepairResponse(BaseModel):
    source: str
    changed: bool




@router.post("/repair", response_model=MermaidRepairResponse)
async def repair(req: MermaidRepairRequest) -> MermaidRepairResponse:
    src = req.source.strip()
    if not src:
        return MermaidRepairResponse(source=req.source, changed=False)
    try:
        from app.core.llm_client import llm
        text = await llm.complete(
            [{
                "role": "user",
                "content": _REPAIR_PROMPT.format(
                    error=(req.error or "unknown parse failure")[:1500],
                    source=src[:12_000],
                ),
            }],
            options={"temperature": 0.0, "max_tokens": 2_000},
        )
    except Exception:  # noqa: BLE001 — repair is best-effort; FE keeps its error card
        return MermaidRepairResponse(source=req.source, changed=False)
    fixed = _strip_fences(text)
    # A usable repair must still look like a mermaid diagram.
    if not fixed or len(fixed) < 8:
        return MermaidRepairResponse(source=req.source, changed=False)
    return MermaidRepairResponse(source=fixed, changed=fixed != src)


class MermaidVerifyRequest(BaseModel):
    source: str = Field(..., max_length=20_000)
    # Off ⇒ report only. On ⇒ attempt an LLM syntax repair and re-verify.
    repair: bool = True


class MermaidVerifyResponse(BaseModel):
    ok: bool
    source: str
    errors: list[str] = []
    warnings: list[str] = []
    repairs: int = 0
    stages: list[str] = []
    sandbox_available: bool = True


@router.post("/verify", response_model=MermaidVerifyResponse)
async def verify_diagram(req: MermaidVerifyRequest) -> MermaidVerifyResponse:
    """Validate → verify in the sandbox → repair → re-verify, BEFORE rendering.

    The webview remains the authoritative parser; this exists so a user who
    pastes Mermaid does not have to watch it fail first. A diagram that cannot
    be fixed comes back unchanged with its diagnostics rather than replaced by
    something invented — a wrong diagram is worse than a broken one, because the
    user cannot tell it is wrong.
    """
    from app.diagrams.verify import verify as _verify

    report = await _verify(req.source, repair=bool(req.repair))
    return MermaidVerifyResponse(
        ok=report.ok, source=report.source, errors=report.errors,
        warnings=report.warnings, repairs=report.repairs,
        stages=report.stages, sandbox_available=report.sandbox_available,
    )
