"""The AI critic (MermaidDiagramVisualizations.md #15).

    Diagram → Second LLM → Review → Suggestions → Improve

Separating generation from review matters because the two want different things: a
generator is rewarded for producing *something*, a reviewer for finding what's
wrong with it. The same model in the same call does both badly.

The critic is the **judgement** half of the validator stack, not a replacement for
it. The deterministic validators already catch everything that is mechanically
checkable — dangling edges, orphans, duplicate labels, missing `accDescr`,
backwards data-store flow. So the critic is explicitly told those findings and
asked for what they cannot see: is the abstraction level consistent, is anything
important missing, is the grouping the right grouping, does the diagram actually
answer the question that was asked.

It returns **edit ops**, not a rewritten diagram. That keeps the reliability
property of the whole package (only the deterministic applier touches structure),
makes each suggestion individually acceptable/rejectable in the UI, and means a
critic that hallucinates a node id gets a rejection instead of a broken diagram.
Fail-open: no model / bad JSON → an empty critique.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.diagrams import validators as V
from app.diagrams.edits import OPS, ops_schema
from app.diagrams.ir import DiagramIR, to_mermaid

_SEVERITIES = ("high", "medium", "low")

_SYSTEM = (
    "You are a diagram reviewer. You are given a diagram's STRUCTURE (JSON), its "
    "rendered source, the request it was built for, and the findings of automated "
    "validators.\n\n"
    "Your job is the judgement the validators cannot make:\n"
    "- Does the diagram actually answer the request?\n"
    "- Is anything important MISSING (a component, a failure path, a return "
    "message, a boundary)?\n"
    "- Is the abstraction level consistent, or does one box hide a whole system "
    "while another shows a single function?\n"
    "- Is the grouping right? Are the labels the terms a reader would use?\n"
    "- Is the direction of flow correct end to end?\n\n"
    "Do NOT repeat the validator findings — they are already handled. Do NOT "
    "comment on colours, fonts or spacing. Do NOT rewrite the diagram.\n\n"
    "Return ONLY a JSON object:\n"
    "{\"verdict\": \"ship\" | \"revise\" | \"rebuild\",\n"
    " \"assessment\": \"one or two sentences\",\n"
    " \"issues\": [{\"severity\": \"high|medium|low\", \"issue\": \"...\", "
    "\"suggestion\": \"...\"}],\n"
    " \"ops\": [ edit operations that would fix them ]}\n\n"
    "Every op must be one of: " + ", ".join(OPS) + ". Every id you reference must "
    "already exist in the structure (except in add_node/add_group). If the diagram "
    "is good, return verdict \"ship\", an empty issues list and no ops."
)


@dataclass
class Critique:
    verdict: str = "ship"              # ship | revise | rebuild
    assessment: str = ""
    issues: list[dict] = field(default_factory=list)
    ops: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        return bool(self.ops)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "assessment": self.assessment,
                "issues": list(self.issues), "ops": list(self.ops),
                "errors": list(self.errors)}


def critique_schema() -> dict:
    """The critic's JSON contract, reusing the edit-op schema for `ops` so a
    suggestion is always something the deterministic applier can execute."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "assessment"],
        "properties": {
            "verdict": {"type": "string", "enum": ["ship", "revise", "rebuild"]},
            "assessment": {"type": "string"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["issue"],
                    "properties": {
                        "severity": {"type": "string", "enum": list(_SEVERITIES)},
                        "issue": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                },
            },
            "ops": ops_schema()["properties"]["ops"],
        },
    }


def critic_prompt(ir: DiagramIR, *, request: str = "",
                  findings: list[V.Finding] | None = None,
                  source: str = "") -> list[dict]:
    """Build the critic messages. Pure — testable without a model."""
    known = json.dumps(ir.to_dict(), separators=(",", ":"))[:8000]
    body = [f"Request the diagram was built for:\n{(request or '(not recorded)').strip()[:1200]}",
            f"Diagram structure (JSON):\n{known}",
            f"Rendered source:\n{(source or to_mermaid(ir))[:4000]}"]
    if findings:
        lines = [f"- [{f.severity}/{f.category}] {f.message}" for f in findings[:15]]
        body.append("Automated validator findings (already known — do not repeat "
                    "them):\n" + "\n".join(lines))
    else:
        body.append("Automated validator findings: none.")
    body.append("Return the JSON object now.")
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "\n\n".join(body)}]


def parse_critique(obj) -> Critique:
    """Coerce a critic payload into a :class:`Critique`. Never raises."""
    result = Critique()
    if not isinstance(obj, dict):
        result.errors.append("critique was not an object")
        return result
    try:
        verdict = str(obj.get("verdict") or "ship").strip().lower()
        result.verdict = verdict if verdict in ("ship", "revise", "rebuild") else "revise"
        result.assessment = str(obj.get("assessment") or "").strip()[:600]
        for raw in obj.get("issues") or []:
            if not isinstance(raw, dict):
                continue
            issue = str(raw.get("issue") or "").strip()
            if not issue:
                continue
            severity = str(raw.get("severity") or "medium").strip().lower()
            result.issues.append({
                "severity": severity if severity in _SEVERITIES else "medium",
                "issue": issue[:400],
                "suggestion": str(raw.get("suggestion") or "").strip()[:400],
            })
        for raw in obj.get("ops") or []:
            if isinstance(raw, dict) and str(raw.get("op") or "") in OPS:
                result.ops.append(raw)
        # A verdict that claims problems but names none is not useful; and one
        # that says "ship" while listing high-severity issues contradicts itself.
        if result.verdict == "ship" and any(
                i["severity"] == "high" for i in result.issues):
            result.verdict = "revise"
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"parse error: {exc}")
    return result


async def critique(ir: DiagramIR, *, request: str = "", source: str = "",
                   findings: list[V.Finding] | None = None,
                   options: dict | None = None) -> Critique:
    """Run the critic. Fail-open → an empty `ship` critique with the reason."""
    try:
        from app.response_arch.structured import generate_structured
    except Exception as exc:  # noqa: BLE001
        return Critique(errors=[f"structured generation unavailable: {exc}"])
    if not ir.nodes:
        return Critique(errors=["nothing to review"])

    opts = {"temperature": 0.2, "max_tokens": 1_500}
    opts.update(options or {})
    try:
        obj, errors = await generate_structured(
            critic_prompt(ir, request=request, findings=findings, source=source),
            critique_schema(), options=opts)
    except Exception as exc:  # noqa: BLE001
        return Critique(errors=[f"critique failed: {exc}"])
    if obj is None:
        return Critique(errors=errors or ["the critic returned nothing usable"])
    result = parse_critique(obj)
    result.errors.extend(errors or [])
    return result


__all__ = ["Critique", "critique", "critic_prompt", "critique_schema",
           "parse_critique"]
