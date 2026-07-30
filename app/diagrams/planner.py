"""The diagram planner — the model's ONLY job in the IR pipeline.

MermaidDiagramVisualizations.md #1, restated as a contract: the model produces
**structure**, never syntax.

    Prompt → Intent → Planner → Diagram IR (JSON) → Mermaid Generator → Parser → SVG

So the planner prompt never mentions arrows, brackets, `subgraph`/`end`, quoting or
direction tokens — it asks for entities, relationships and grouping, constrained to
:func:`ir.json_schema`. Whatever comes back is coerced by
:func:`ir.from_dict` (illegal ids folded, unknown fields dropped) and rendered by
the deterministic emitter, so the class of failure the doc opens with — "missing
`end`", "`--->`", "`Broker 1` instead of `B1[\"Broker 1\"]`" — is structurally
impossible rather than repaired after the fact.

Model access goes through `response_arch.structured.generate_structured`, which
already does request → schema-validate → one schema-guided repair and works across
providers. Fail-open: no model, a refusal, or unparseable JSON returns
`(None, reasons)` and the caller falls back to the existing free-text
```mermaid``` path.
"""
from __future__ import annotations

import json

from app.diagrams.ir import DiagramIR, KINDS, from_dict, json_schema

# What each kind is FOR, in the planner's terms. Giving the model the semantics
# (not the syntax) is what makes it pick the right structure.
_KIND_GUIDE = {
    "flowchart": "processes, pipelines, system architecture, decision flows — "
                 "nodes are components/steps, edges are flow or calls",
    "sequence": "an ordered interaction between participants over time — nodes "
                "are participants, edges are messages IN ORDER",
    "state": "a state machine — nodes are states, edges are transitions; give "
             "the entry state role \"start\" and the terminal state role \"end\"",
    "er": "a data model — nodes are entities, `members` are their fields "
          "(\"type name\"), edges carry `cardinality` (1-1, 1-*, *-1, *-*)",
    "class": "types and their relationships — `members` are attributes/methods, "
             "edges carry `relation` (inheritance, composition, aggregation, "
             "association, dependency)",
    "mindmap": "a hierarchy or breakdown — edges go parent → child",
}

_ROLE_GUIDE = (
    "Set `role` on every node so shapes are chosen for you: "
    "user/actor/person, start, end, decision, datastore/database, queue/broker, "
    "external, service."
)

_SYSTEM = (
    "You are a diagram planner. You do NOT write diagram code — you describe a "
    "diagram's STRUCTURE as JSON and a deterministic generator renders it.\n\n"
    "Rules:\n"
    "- Return ONLY the JSON object for the schema. No prose, no code fence.\n"
    "- `id` is a short stable identifier (letters, digits, underscore). `label` "
    "is the human text; keep labels under 40 characters and put detail in `note`.\n"
    "- Every edge's `src` and `dst` MUST be the `id` of a node you declared.\n"
    "- Label the edges that carry meaning (what flows, which branch).\n"
    "- Use `groups` for logical boundaries (a service, a tier, a bounded context) "
    "and set each node's `group` to that group's id.\n"
    "- Prefer 6-14 nodes. A diagram with 40 boxes teaches nothing; if the subject "
    "is that big, show the top level and say so in `acc_descr`.\n"
    "- ALWAYS set `acc_title` (a short name) and `acc_descr` (one or two "
    "sentences describing the diagram for someone who cannot see it).\n"
    "- Direction: LR for long linear flows, TD for wide fan-outs and hierarchies.\n"
    f"- {_ROLE_GUIDE}"
)


def plan_prompt(request: str, *, kind: str = "", context: str = "") -> list[dict]:
    """The planner messages for `request`.

    `kind` pins the diagram type when the caller already decided (e.g. from
    `quality.diagram_gate`); otherwise the model chooses and is told what each
    kind is for.
    """
    wanted = (kind or "").strip().lower()
    if wanted in KINDS:
        kind_block = (f"Diagram kind: **{wanted}** — {_KIND_GUIDE[wanted]}.\n"
                      f"Set \"kind\": \"{wanted}\".")
    else:
        kind_block = "Choose the `kind` that fits best:\n" + "\n".join(
            f"- {name}: {why}" for name, why in _KIND_GUIDE.items())

    user = [f"Request:\n{request.strip()}"]
    if context.strip():
        user.append(f"Relevant context (use it, don't restate it):\n"
                    f"{context.strip()[:4000]}")
    user.append(kind_block)
    user.append("Return the JSON object now.")
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(user)},
    ]


async def plan(request: str, *, kind: str = "", context: str = "",
               options: dict | None = None) -> tuple[DiagramIR | None, list[str]]:
    """Ask the model for an IR. Returns `(ir, errors)`; `ir` is None on failure.

    An IR with no nodes counts as a failure — an empty structure is worse than
    falling back to the free-text path, because it renders as a valid but empty
    diagram and looks like a bug.
    """
    try:
        from app.response_arch.structured import generate_structured
    except Exception as exc:  # noqa: BLE001
        return None, [f"structured generation unavailable: {exc}"]

    if not (request or "").strip():
        return None, ["empty request"]

    schema = json_schema()
    opts = {"temperature": 0.1, "max_tokens": 2_500}
    opts.update(options or {})
    try:
        obj, errors = await generate_structured(
            plan_prompt(request, kind=kind, context=context), schema,
            options=opts)
    except Exception as exc:  # noqa: BLE001
        return None, [f"planning failed: {exc}"]

    if obj is None:
        return None, errors or ["the planner returned nothing usable"]
    ir = from_dict(obj if isinstance(obj, dict) else {})
    if not ir.nodes:
        return None, (errors or []) + ["the planned diagram has no nodes"]
    return ir, errors


def plan_from_json(text: str) -> tuple[DiagramIR | None, list[str]]:
    """Coerce raw planner text into an IR — the offline half of :func:`plan`,
    so the coercion path is testable without a model."""
    try:
        from app.llm.constrained import extract_json
        payload = extract_json(text or "")
        obj = json.loads(payload)
    except Exception as exc:  # noqa: BLE001
        return None, [f"not JSON: {exc}"]
    ir = from_dict(obj if isinstance(obj, dict) else {})
    if not ir.nodes:
        return None, ["no nodes in the payload"]
    return ir, []


__all__ = ["plan", "plan_prompt", "plan_from_json"]


# ---- natural-language edits (doc #10, the model half) ---------------------
_EDIT_SYSTEM = (
    "You translate a user's edit request into OPERATIONS on a diagram's "
    "structure. You do NOT rewrite the diagram and you do NOT emit diagram code.\n\n"
    "Rules:\n"
    "- Return ONLY the JSON object {\"ops\": [...], \"note\": \"...\"}.\n"
    "- Emit the SMALLEST set of ops that satisfies the request. Preserve "
    "everything the user did not ask to change — that is the entire point of "
    "editing instead of regenerating.\n"
    "- Every `id`, `src` and `dst` you reference must ALREADY EXIST in the given "
    "structure, except in `add_node` / `add_group`.\n"
    "- \"Move X under Y\" in a flow means an EDGE from Y to X (and removing the "
    "old parent edge), not a group change — unless Y is a group.\n"
    "- If the request is impossible or ambiguous, return an empty `ops` list and "
    "explain why in `note`.\n"
    "- `note` is one short sentence describing what you changed."
)


def edit_prompt(ir: DiagramIR, command: str) -> list[dict]:
    """Messages that turn "move Broker 2 under Broker 1" into edit ops."""
    from app.diagrams.edits import OPS
    structure = json.dumps(ir.to_dict(), separators=(",", ":"))[:8000]
    inventory = (
        "Nodes: " + ", ".join(f"{n.id} (\u201c{n.text}\u201d)" for n in ir.nodes[:60])
        + ("\nGroups: " + ", ".join(g.id for g in ir.groups) if ir.groups else "")
        + ("\nEdges: " + ", ".join(f"{e.src}->{e.dst}" for e in ir.edges[:80])
           if ir.edges else "")
    )
    return [
        {"role": "system", "content": _EDIT_SYSTEM + "\n\nAvailable ops: "
                                      + ", ".join(OPS)},
        {"role": "user", "content":
            f"Current structure (JSON):\n{structure}\n\n{inventory}\n\n"
            f"Edit request:\n{command.strip()[:800]}\n\nReturn the JSON now."},
    ]


async def plan_edit(ir: DiagramIR, command: str, *,
                    options: dict | None = None) -> tuple[list[dict], str, list[str]]:
    """Translate `command` into edit ops → `(ops, note, errors)`.

    The ops are NOT applied here — :func:`app.diagrams.edits.apply_edits` does
    that deterministically and rejects anything that doesn't fit the structure.
    """
    from app.diagrams.edits import ops_schema
    if not (command or "").strip():
        return [], "", ["empty edit command"]
    if not ir.nodes:
        return [], "", ["nothing to edit"]
    try:
        from app.response_arch.structured import generate_structured
    except Exception as exc:  # noqa: BLE001
        return [], "", [f"structured generation unavailable: {exc}"]

    opts = {"temperature": 0.0, "max_tokens": 1_200}
    opts.update(options or {})
    try:
        obj, errors = await generate_structured(
            edit_prompt(ir, command), ops_schema(), options=opts)
    except Exception as exc:  # noqa: BLE001
        return [], "", [f"edit planning failed: {exc}"]
    if not isinstance(obj, dict):
        return [], "", (errors or ["the editor returned nothing usable"])
    ops = [op for op in (obj.get("ops") or []) if isinstance(op, dict)]
    note = str(obj.get("note") or "").strip()[:300]
    return ops, note, errors or []


__all__ = ["plan", "plan_prompt", "plan_from_json", "plan_edit", "edit_prompt"]
