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

router = APIRouter(prefix="/api/mermaid", tags=["mermaid"])

# The doc's repair contract, verbatim in spirit: fix ONLY the syntax, keep the
# architecture, return only mermaid.
_REPAIR_PROMPT = (
    "The following Mermaid diagram failed compilation.\n\n"
    "Parser error:\n{error}\n\n"
    "Mermaid source:\n```mermaid\n{source}\n```\n\n"
    # NOTE: this string goes through `str.format`, so every literal brace below
    # must be doubled — a stray `{` raises KeyError and silently costs the repair.
    "Fix ONLY the syntax. Do NOT change the architecture, the nodes, the "
    "labels' meaning, or the layout direction. Common REAL fixes: wrap a label "
    "containing `(`, `)`, `\"`, `|`, `{{` or `}}` in double quotes "
    "(A[\"Fetch (REST)\"]); close every `subgraph` with `end`; give every link "
    "an arrowhead or terminator (`A --> B` or `A --- B`, never `A -- B`); use at "
    "least two dashes (`-->`, not `->`); declare nodes as `ID[\"Label\"]`.\n"
    "Do NOT 'fix' these — they are VALID mermaid: `--->` and `----` (extra "
    "dashes just set the rank distance), and `:`, `#`, `<`, `>`, `;`, `&`, `%` "
    "inside an unquoted label.\n"
    "Return ONLY the corrected Mermaid code, with no fences and no commentary."
)


class MermaidRepairRequest(BaseModel):
    source: str = Field(..., max_length=20_000)
    error: str = Field(default="", max_length=2_000)


class MermaidRepairResponse(BaseModel):
    source: str
    changed: bool


def _strip_fences(text: str) -> str:
    """The model sometimes fences the reply anyway — unwrap it."""
    t = (text or "").strip()
    m = re.search(r"```(?:mermaid)?\s*\n(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    return t


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
