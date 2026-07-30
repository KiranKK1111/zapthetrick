"""Targeted diagram edits (MermaidDiagramVisualizations.md #10).

    User: "Move Kafka Broker 2 under Broker 1."
      → Edit Command → Modify AST → Regenerate Mermaid → Compile → Render

instead of regenerating the whole diagram, which is how iterative editing loses
work: a full regeneration rewrites labels the user liked, drops nodes the model
forgot on the second pass, and reshuffles the layout for no reason.

The model's only job here is to translate the sentence into **edit operations**
against the IR. Applying them is deterministic, so an edit can never break the
syntax, and every op is either applied or REJECTED with a reason — a "move X
under Y" that names a node which doesn't exist comes back as a rejection the UI
can show, not a silently mangled diagram.

Ops (all fail-safe, all order-independent except add-before-reference):

    add_node      {id, label?, shape?, role?, group?, note?}
    remove_node   {id}                     — also removes its edges
    rename_node   {id, label}
    set_shape     {id, shape|role}
    move_node     {id, group}              — "" moves it to the top level
    add_edge      {src, dst, label?, style?, arrow?}
    remove_edge   {src, dst}
    relabel_edge  {src, dst, label}
    reverse_edge  {src, dst}
    add_group     {id, label?, parent?}
    remove_group  {id}                     — its nodes move to the top level
    rename_group  {id, label}
    set_direction {direction}
    set_layout    {layout}                 — dagre | elk
    set_title     {title}
    set_acc       {acc_title?, acc_descr?}

Pure module: no I/O, no model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.diagrams.ir import (
    ARROWS, DIRECTIONS, DiagramIR, EDGE_STYLES, Edge, Group, Node, ROLE_SHAPES,
    SHAPES, safe_id,
)

OPS = (
    "add_node", "remove_node", "rename_node", "set_shape", "move_node",
    "add_edge", "remove_edge", "relabel_edge", "reverse_edge",
    "add_group", "remove_group", "rename_group",
    "set_direction", "set_layout", "set_title", "set_acc",
)


@dataclass
class EditResult:
    ir: DiagramIR
    applied: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    def to_dict(self) -> dict:
        return {"changed": self.changed, "applied": list(self.applied),
                "rejected": list(self.rejected)}


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _reject(out: EditResult, op: dict, reason: str) -> None:
    out.rejected.append({"op": op, "reason": reason})


def apply_edits(ir: DiagramIR, ops: list[dict] | None) -> EditResult:
    """Apply `ops` to a COPY of `ir`. Never raises; never partially corrupts.

    Each op is validated against the current structure before it lands, so the
    result is always a coherent IR — which is the whole point of editing the AST
    instead of the text.
    """
    working = ir.copy()
    result = EditResult(ir=working)
    if not ops:
        return result
    for raw in ops:
        if not isinstance(raw, dict):
            _reject(result, {"op": str(raw)[:80]}, "not an object")
            continue
        name = _text(raw.get("op")).lower()
        try:
            handler = _HANDLERS.get(name)
            if handler is None:
                _reject(result, raw, f"unknown op `{name or '(missing)'}`")
                continue
            error = handler(working, raw)
            if error:
                _reject(result, raw, error)
            else:
                result.applied.append(raw)
        except Exception as exc:  # noqa: BLE001 — one bad op can't kill the batch
            _reject(result, raw, f"error: {exc}")
    return result


# ---- handlers: return "" on success, a reason string on rejection ----------
def _add_node(ir: DiagramIR, op: dict) -> str:
    node_id = safe_id(_text(op.get("id")) or _text(op.get("label")))
    if not node_id:
        return "add_node needs an id or a label"
    if ir.node(node_id):
        return f"node `{node_id}` already exists"
    group = safe_id(_text(op.get("group")), fallback="") if op.get("group") else ""
    if group and not ir.group(group):
        return f"group `{group}` does not exist"
    shape = _text(op.get("shape")).lower()
    ir.nodes.append(Node(
        id=node_id, label=_text(op.get("label")) or node_id,
        shape=shape if shape in SHAPES else "",
        role=_text(op.get("role")), group=group, note=_text(op.get("note")),
    ))
    return ""


def _remove_node(ir: DiagramIR, op: dict) -> str:
    node_id = safe_id(_text(op.get("id")))
    node = ir.node(node_id)
    if not node:
        return f"no node `{node_id}`"
    ir.nodes.remove(node)
    # Its edges go with it — leaving them would create dangling endpoints.
    ir.edges = [e for e in ir.edges if e.src != node_id and e.dst != node_id]
    return ""


def _rename_node(ir: DiagramIR, op: dict) -> str:
    node = ir.node(safe_id(_text(op.get("id"))))
    if not node:
        return f"no node `{_text(op.get('id'))}`"
    label = _text(op.get("label"))
    if not label:
        return "rename_node needs a label"
    node.label = label
    return ""


def _set_shape(ir: DiagramIR, op: dict) -> str:
    node = ir.node(safe_id(_text(op.get("id"))))
    if not node:
        return f"no node `{_text(op.get('id'))}`"
    shape = _text(op.get("shape")).lower()
    role = _text(op.get("role")).lower().replace(" ", "")
    if shape in SHAPES:
        node.shape = shape
        return ""
    if role in ROLE_SHAPES:
        node.role, node.shape = role, ""
        return ""
    return f"unknown shape/role `{shape or role}`"


def _move_node(ir: DiagramIR, op: dict) -> str:
    node = ir.node(safe_id(_text(op.get("id"))))
    if not node:
        return f"no node `{_text(op.get('id'))}`"
    target = _text(op.get("group"))
    if not target:
        node.group = ""                    # promote to the top level
        return ""
    group_id = safe_id(target)
    if not ir.group(group_id):
        return f"group `{group_id}` does not exist"
    node.group = group_id
    return ""


def _add_edge(ir: DiagramIR, op: dict) -> str:
    src = safe_id(_text(op.get("src")) or _text(op.get("from")))
    dst = safe_id(_text(op.get("dst")) or _text(op.get("to")))
    if not src or not dst:
        return "add_edge needs src and dst"
    if not ir.node(src):
        return f"no node `{src}`"
    if not ir.node(dst):
        return f"no node `{dst}`"
    style = _text(op.get("style")).lower() or "solid"
    arrow = _text(op.get("arrow")).lower() or "arrow"
    if any(e.src == src and e.dst == dst and e.label == _text(op.get("label"))
           for e in ir.edges):
        return f"edge {src}→{dst} already exists"
    ir.edges.append(Edge(
        src=src, dst=dst, label=_text(op.get("label")),
        style=style if style in EDGE_STYLES else "solid",
        arrow=arrow if arrow in ARROWS else "arrow",
    ))
    return ""


def _find_edge(ir: DiagramIR, op: dict) -> Edge | None:
    src = safe_id(_text(op.get("src")) or _text(op.get("from")))
    dst = safe_id(_text(op.get("dst")) or _text(op.get("to")))
    for edge in ir.edges:
        if edge.src == src and edge.dst == dst:
            return edge
    return None


def _remove_edge(ir: DiagramIR, op: dict) -> str:
    edge = _find_edge(ir, op)
    if not edge:
        return f"no edge {_text(op.get('src'))}→{_text(op.get('dst'))}"
    ir.edges.remove(edge)
    return ""


def _relabel_edge(ir: DiagramIR, op: dict) -> str:
    edge = _find_edge(ir, op)
    if not edge:
        return f"no edge {_text(op.get('src'))}→{_text(op.get('dst'))}"
    edge.label = _text(op.get("label"))
    return ""


def _reverse_edge(ir: DiagramIR, op: dict) -> str:
    edge = _find_edge(ir, op)
    if not edge:
        return f"no edge {_text(op.get('src'))}→{_text(op.get('dst'))}"
    edge.src, edge.dst = edge.dst, edge.src
    return ""


def _add_group(ir: DiagramIR, op: dict) -> str:
    group_id = safe_id(_text(op.get("id")) or _text(op.get("label")), fallback="g")
    if not group_id:
        return "add_group needs an id or a label"
    if ir.group(group_id):
        return f"group `{group_id}` already exists"
    parent = safe_id(_text(op.get("parent")), fallback="") if op.get("parent") else ""
    if parent and not ir.group(parent):
        return f"parent group `{parent}` does not exist"
    ir.groups.append(Group(id=group_id,
                           label=_text(op.get("label")) or group_id,
                           parent=parent))
    return ""


def _remove_group(ir: DiagramIR, op: dict) -> str:
    group_id = safe_id(_text(op.get("id")))
    group = ir.group(group_id)
    if not group:
        return f"no group `{group_id}`"
    ir.groups.remove(group)
    # Its members survive at the level the group used to sit at.
    for node in ir.nodes:
        if node.group == group_id:
            node.group = group.parent
    for child in ir.groups:
        if child.parent == group_id:
            child.parent = group.parent
    return ""


def _rename_group(ir: DiagramIR, op: dict) -> str:
    group = ir.group(safe_id(_text(op.get("id"))))
    if not group:
        return f"no group `{_text(op.get('id'))}`"
    label = _text(op.get("label"))
    if not label:
        return "rename_group needs a label"
    group.label = label
    return ""


def _set_direction(ir: DiagramIR, op: dict) -> str:
    direction = _text(op.get("direction")).upper()
    if direction not in DIRECTIONS:
        return f"`{direction}` is not one of {', '.join(DIRECTIONS)}"
    ir.direction = direction
    return ""


def _set_layout(ir: DiagramIR, op: dict) -> str:
    layout = _text(op.get("layout")).lower()
    if layout not in ("dagre", "elk", ""):
        return f"unknown layout `{layout}`"
    ir.layout = layout
    return ""


def _set_title(ir: DiagramIR, op: dict) -> str:
    ir.title = _text(op.get("title"))
    return ""


def _set_acc(ir: DiagramIR, op: dict) -> str:
    acc_title = _text(op.get("acc_title"))
    acc_descr = _text(op.get("acc_descr"))
    if not acc_title and not acc_descr:
        return "set_acc needs acc_title and/or acc_descr"
    if acc_title:
        ir.acc_title = acc_title
    if acc_descr:
        ir.acc_descr = acc_descr
    return ""


_HANDLERS = {
    "add_node": _add_node, "remove_node": _remove_node,
    "rename_node": _rename_node, "set_shape": _set_shape,
    "move_node": _move_node,
    "add_edge": _add_edge, "remove_edge": _remove_edge,
    "relabel_edge": _relabel_edge, "reverse_edge": _reverse_edge,
    "add_group": _add_group, "remove_group": _remove_group,
    "rename_group": _rename_group,
    "set_direction": _set_direction, "set_layout": _set_layout,
    "set_title": _set_title, "set_acc": _set_acc,
}


def ops_schema() -> dict:
    """JSON Schema for an edit batch — the contract the model is constrained to,
    so "move Broker 2 under Broker 1" can only come back as ops we can apply."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["ops"],
        "properties": {
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["op"],
                    "properties": {
                        "op": {"type": "string", "enum": list(OPS)},
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "shape": {"type": "string", "enum": list(SHAPES)},
                        "role": {"type": "string"},
                        "group": {"type": "string"},
                        "parent": {"type": "string"},
                        "note": {"type": "string"},
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "style": {"type": "string", "enum": list(EDGE_STYLES)},
                        "arrow": {"type": "string", "enum": list(ARROWS)},
                        "direction": {"type": "string", "enum": list(DIRECTIONS)},
                        "layout": {"type": "string", "enum": ["dagre", "elk"]},
                        "title": {"type": "string"},
                        "acc_title": {"type": "string"},
                        "acc_descr": {"type": "string"},
                    },
                },
            },
            "note": {"type": "string"},
        },
    }


def describe_ops(ops: list[dict] | None) -> list[str]:
    """One human line per op, for a "what changed" list in the UI."""
    out: list[str] = []
    for raw in ops or []:
        if not isinstance(raw, dict):
            continue
        name = _text(raw.get("op"))
        if name == "add_node":
            out.append(f"added node “{_text(raw.get('label')) or _text(raw.get('id'))}”")
        elif name == "remove_node":
            out.append(f"removed node `{_text(raw.get('id'))}`")
        elif name == "rename_node":
            out.append(f"renamed `{_text(raw.get('id'))}` to “{_text(raw.get('label'))}”")
        elif name == "move_node":
            target = _text(raw.get("group")) or "top level"
            out.append(f"moved `{_text(raw.get('id'))}` into {target}")
        elif name == "set_shape":
            out.append(f"reshaped `{_text(raw.get('id'))}`")
        elif name == "add_edge":
            out.append(f"connected {_text(raw.get('src'))} → {_text(raw.get('dst'))}")
        elif name == "remove_edge":
            out.append(f"disconnected {_text(raw.get('src'))} → {_text(raw.get('dst'))}")
        elif name == "relabel_edge":
            out.append(f"labelled {_text(raw.get('src'))} → {_text(raw.get('dst'))} "
                       f"“{_text(raw.get('label'))}”")
        elif name == "reverse_edge":
            out.append(f"reversed {_text(raw.get('src'))} → {_text(raw.get('dst'))}")
        elif name == "add_group":
            out.append(f"added group “{_text(raw.get('label')) or _text(raw.get('id'))}”")
        elif name == "remove_group":
            out.append(f"removed group `{_text(raw.get('id'))}`")
        elif name == "rename_group":
            out.append(f"renamed group `{_text(raw.get('id'))}`")
        elif name == "set_direction":
            out.append(f"direction → {_text(raw.get('direction'))}")
        elif name == "set_layout":
            out.append(f"layout → {_text(raw.get('layout'))}")
        elif name == "set_title":
            out.append("set the title")
        elif name == "set_acc":
            out.append("updated the accessible title/description")
        elif name:
            out.append(name.replace("_", " "))
    return out


__all__ = ["OPS", "EditResult", "apply_edits", "ops_schema", "describe_ops"]
