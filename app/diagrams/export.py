"""Export everywhere (MermaidDiagramVisualizations.md #16, and the payoff for #1).

    One diagram → Mermaid · PlantUML · Graphviz DOT · Draw.io · ELK · JSON

This module is the argument for the IR made concrete. Going from Mermaid text to
PlantUML text would mean writing a Mermaid parser and a PlantUML printer and
keeping the two in sync forever. Going from the IR means one small printer per
target — each of the functions below is a few dozen lines, and every new diagram
kind we teach the IR gets all of them for free.

Text formats only, by design. SVG/PNG/PDF are *render* outputs and belong where a
renderer lives: the FE's shared mermaid webview already rasterizes to PNG and
embeds diagrams into generated DOCX/PDF documents. Duplicating a headless
Chromium server-side to produce the same bytes would be a second rendering path to
keep consistent, so the split is: **this module owns source formats, the renderer
owns pixels.**

Every exporter is pure and fail-open (an error returns an empty string, never an
exception into a turn).
"""
from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from app.diagrams.ir import CLASS_RELATIONS, DiagramIR, to_mermaid

MERMAID = "mermaid"
PLANTUML = "plantuml"
DOT = "dot"
DRAWIO = "drawio"
ELK = "elk"
JSON_IR = "json"

# format → (label, file extension, mime type)
EXPORT_FORMATS: dict[str, tuple[str, str, str]] = {
    MERMAID: ("Mermaid", "mmd", "text/vnd.mermaid"),
    PLANTUML: ("PlantUML", "puml", "text/plain"),
    DOT: ("Graphviz DOT", "dot", "text/vnd.graphviz"),
    DRAWIO: ("Draw.io", "drawio", "application/xml"),
    ELK: ("ELK graph (JSON)", "elkjson", "application/json"),
    JSON_IR: ("Diagram IR (JSON)", "json", "application/json"),
}


@dataclass
class ExportResult:
    format: str
    content: str
    filename: str
    mime: str

    def to_dict(self) -> dict:
        return {"format": self.format, "content": self.content,
                "filename": self.filename, "mime": self.mime}


# ---- PlantUML -------------------------------------------------------------
_PUML_SHAPES = {
    "cylinder": "database", "circle": "circle", "doublecircle": "circle",
    "rhombus": "diamond", "hexagon": "hexagon", "stadium": "rectangle",
    "subroutine": "queue", "round": "actor",
}


def to_plantuml(ir: DiagramIR) -> str:
    """PlantUML source. Uses the activity/component syntax for flows, and
    PlantUML's native sequence / state / class / ER forms for the rest."""
    lines = ["@startuml"]
    if ir.title:
        lines.append(f"title {ir.title}")
    if ir.acc_descr:
        lines.append(f"' {ir.acc_descr}")

    if ir.kind == "sequence":
        for node in ir.nodes:
            keyword = "actor" if node.resolved_shape() == "round" else "participant"
            lines.append(f'{keyword} "{node.text}" as {node.id}')
        for edge in ir.edges:
            arrow = "-->" if edge.style == "dotted" else "->"
            label = f" : {edge.label}" if edge.label else ""
            lines.append(f"{edge.src} {arrow} {edge.dst}{label}")
    elif ir.kind == "state":
        lines.append("hide empty description")
        starts = {n.id for n in ir.nodes if (n.role or "").lower() in ("start", "initial")}
        ends = {n.id for n in ir.nodes if (n.role or "").lower() in ("end", "final")}
        for node in ir.nodes:
            if node.id in starts | ends:
                continue
            lines.append(f'state "{node.text}" as {node.id}')
        for edge in ir.edges:
            src = "[*]" if edge.src in starts else edge.src
            dst = "[*]" if edge.dst in ends else edge.dst
            label = f" : {edge.label}" if edge.label else ""
            lines.append(f"{src} --> {dst}{label}")
    elif ir.kind == "class":
        for node in ir.nodes:
            if node.members:
                lines.append(f"class {node.id} {{")
                lines.extend(f"  {m}" for m in node.members)
                lines.append("}")
            else:
                lines.append(f"class {node.id}")
        puml_relations = {"inheritance": "<|--", "extends": "<|--",
                          "implements": "<|..", "composition": "*--",
                          "aggregation": "o--", "association": "-->",
                          "dependency": "..>", "link": "--"}
        for edge in ir.edges:
            op = puml_relations.get(edge.relation or "association", "-->")
            label = f" : {edge.label}" if edge.label else ""
            lines.append(f"{edge.src} {op} {edge.dst}{label}")
    elif ir.kind == "er":
        for node in ir.nodes:
            lines.append(f'entity "{node.text}" as {node.id} {{')
            for member in node.members or []:
                lines.append(f"  {member}")
            lines.append("}")
        for edge in ir.edges:
            label = f" : {edge.label}" if edge.label else ""
            lines.append(f"{edge.src} ||--o{{ {edge.dst}{label}")
    else:
        # flowchart / mindmap → component diagram (keeps groups as packages).
        lines.append("left to right direction" if ir.direction in ("LR", "RL")
                     else "top to bottom direction")

        def declare(node) -> str:
            keyword = _PUML_SHAPES.get(node.resolved_shape(), "rectangle")
            return f'{keyword} "{node.text}" as {node.id}'

        for node in ir.nodes:
            if not node.group:
                lines.append(declare(node))

        def emit_group(group, indent: str) -> None:
            lines.append(f'{indent}package "{group.label or group.id}" {{')
            for child in ir.nodes:
                if child.group == group.id:
                    lines.append(f"{indent}  {declare(child)}")
            for sub in ir.groups:
                if sub.parent == group.id and sub.id != group.id:
                    emit_group(sub, indent + "  ")
            lines.append(indent + "}")

        for group in ir.groups:
            if not group.parent:
                emit_group(group, "")
        for edge in ir.edges:
            arrow = {"dotted": "..>", "thick": "==>", "invisible": "-[hidden]->"}\
                .get(edge.style, "-->")
            label = f" : {edge.label}" if edge.label else ""
            lines.append(f"{edge.src} {arrow} {edge.dst}{label}")

    lines.append("@enduml")
    return "\n".join(lines)


# ---- Graphviz DOT ---------------------------------------------------------
_DOT_SHAPES = {
    "rect": "box", "round": "ellipse", "stadium": "box", "subroutine": "box3d",
    "cylinder": "cylinder", "circle": "circle", "doublecircle": "doublecircle",
    "rhombus": "diamond", "hexagon": "hexagon", "parallelogram": "parallelogram",
    "trapezoid": "trapezium",
}


def _dot_escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')\
        .replace("<br/>", "\\n")


def to_dot(ir: DiagramIR) -> str:
    """Graphviz DOT — the lingua franca for graph tooling (and a second layout
    engine for free: `dot -Tsvg`)."""
    rankdir = {"TD": "TB", "TB": "TB", "BT": "BT", "LR": "LR", "RL": "RL"}\
        .get(ir.direction, "TB")
    lines = ["digraph G {",
             f'  rankdir={rankdir};',
             '  node [shape=box style="rounded,filled" fillcolor="#f6f8fa" '
             'fontname="Segoe UI"];',
             '  edge [fontname="Segoe UI" fontsize=10];']
    if ir.title or ir.acc_title:
        lines.append(f'  label="{_dot_escape(ir.title or ir.acc_title)}";')
        lines.append("  labelloc=t;")

    def declare(node, indent: str) -> str:
        shape = _DOT_SHAPES.get(node.resolved_shape(), "box")
        style = ' style="rounded,filled"' if shape == "box" else ""
        return (f'{indent}{node.id} [label="{_dot_escape(node.text)}" '
                f'shape={shape}{style}];')

    for node in ir.nodes:
        if not node.group:
            lines.append(declare(node, "  "))

    def emit_group(group, indent: str) -> None:
        lines.append(f"{indent}subgraph cluster_{group.id} {{")
        lines.append(f'{indent}  label="{_dot_escape(group.label or group.id)}";')
        lines.append(f'{indent}  style=rounded; color="#c9d1d9";')
        for child in ir.nodes:
            if child.group == group.id:
                lines.append(declare(child, indent + "  "))
        for sub in ir.groups:
            if sub.parent == group.id and sub.id != group.id:
                emit_group(sub, indent + "  ")
        lines.append(indent + "}")

    for group in ir.groups:
        if not group.parent:
            emit_group(group, "  ")

    for edge in ir.edges:
        attrs = []
        if edge.label:
            attrs.append(f'label="{_dot_escape(edge.label)}"')
        if edge.style == "dotted":
            attrs.append("style=dotted")
        elif edge.style == "thick":
            attrs.append("penwidth=2")
        elif edge.style == "invisible":
            attrs.append("style=invis")
        if edge.arrow == "bidirectional":
            attrs.append("dir=both")
        elif edge.arrow in ("open", "none"):
            attrs.append("arrowhead=none")
        suffix = f' [{" ".join(attrs)}]' if attrs else ""
        lines.append(f"  {edge.src} -> {edge.dst}{suffix};")
    lines.append("}")
    return "\n".join(lines)


# ---- Draw.io -------------------------------------------------------------
_DRAWIO_STYLES = {
    "rect": "rounded=1;whiteSpace=wrap;html=1;",
    "round": "ellipse;whiteSpace=wrap;html=1;",
    "stadium": "rounded=1;arcSize=50;whiteSpace=wrap;html=1;",
    "subroutine": "shape=process;whiteSpace=wrap;html=1;",
    "cylinder": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;",
    "circle": "ellipse;whiteSpace=wrap;html=1;aspect=fixed;",
    "doublecircle": "ellipse;shape=doubleEllipse;whiteSpace=wrap;html=1;",
    "rhombus": "rhombus;whiteSpace=wrap;html=1;",
    "hexagon": "shape=hexagon;whiteSpace=wrap;html=1;",
    "parallelogram": "shape=parallelogram;whiteSpace=wrap;html=1;",
    "trapezoid": "shape=trapezoid;whiteSpace=wrap;html=1;",
}


def to_drawio(ir: DiagramIR) -> str:
    """An uncompressed `.drawio` file (mxGraph XML).

    Draw.io accepts a plain, uncompressed `<mxfile>` — no deflate/base64 dance
    needed — so the output is diffable text. Geometry comes from the same
    longest-path layering the layout planner uses, which is enough for a file the
    user will immediately re-arrange by hand anyway.
    """
    from app.diagrams.layout import plan_layout

    plan = plan_layout(ir)
    horizontal = plan.direction in ("LR", "RL")
    ranks = _rank_map(ir)
    per_rank: dict[int, int] = {}

    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net",
                                   "type": "device", "agent": "zapthetrick"})
    diagram = ET.SubElement(mxfile, "diagram",
                            {"name": ir.title or ir.acc_title or "Diagram",
                             "id": "d0"})
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": "1100", "dy": "800", "grid": "1", "gridSize": "10",
        "page": "1", "pageWidth": "1100", "pageHeight": "850",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    # Groups become container cells so nesting survives the export.
    group_cells: dict[str, str] = {}
    for index, group in enumerate(ir.groups):
        cell_id = f"g{index}"
        group_cells[group.id] = cell_id
        cell = ET.SubElement(root, "mxCell", {
            "id": cell_id, "value": group.label or group.id,
            "style": "rounded=0;whiteSpace=wrap;html=1;dashed=1;fillColor=none;"
                     "verticalAlign=top;",
            "vertex": "1",
            "parent": group_cells.get(group.parent, "1"),
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(40 + index * 30), "y": str(40 + index * 30),
            "width": "460", "height": "260", "as": "geometry"})

    for node in ir.nodes:
        rank = ranks.get(node.id, 0)
        slot = per_rank.get(rank, 0)
        per_rank[rank] = slot + 1
        x = 60 + (rank if horizontal else slot) * 220
        y = 60 + (slot if horizontal else rank) * 110
        width, height = max(120, min(260, len(node.text) * 8 + 30)), 50
        cell = ET.SubElement(root, "mxCell", {
            "id": f"n_{node.id}", "value": node.text,
            "style": _DRAWIO_STYLES.get(node.resolved_shape(),
                                        _DRAWIO_STYLES["rect"]),
            "vertex": "1",
            "parent": group_cells.get(node.group, "1"),
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(x), "y": str(y), "width": str(width), "height": str(height),
            "as": "geometry"})

    for index, edge in enumerate(ir.edges):
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                 "endArrow=blockThin;")
        if edge.style == "dotted":
            style += "dashed=1;"
        if edge.arrow == "bidirectional":
            style += "startArrow=blockThin;"
        if edge.arrow in ("open", "none"):
            style = style.replace("endArrow=blockThin;", "endArrow=none;")
        cell = ET.SubElement(root, "mxCell", {
            "id": f"e{index}", "value": edge.label, "style": style,
            "edge": "1", "parent": "1",
            "source": f"n_{edge.src}", "target": f"n_{edge.dst}"})
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + \
        ET.tostring(mxfile, encoding="unicode")


def _rank_map(ir: DiagramIR) -> dict[str, int]:
    """Longest-path layering, shared with the layout planner's idea of shape."""
    incoming = {n.id: 0 for n in ir.nodes}
    adjacency: dict[str, list[str]] = {}
    for edge in ir.edges:
        if edge.src == edge.dst or edge.dst not in incoming or edge.src not in incoming:
            continue
        adjacency.setdefault(edge.src, []).append(edge.dst)
        incoming[edge.dst] += 1
    rank = {node_id: 0 for node_id in incoming}
    remaining = dict(incoming)
    queue = [node_id for node_id, count in incoming.items() if count == 0]
    guard = 0
    while queue and guard < 100_000:
        guard += 1
        node_id = queue.pop(0)
        for nxt in adjacency.get(node_id, []):
            rank[nxt] = max(rank[nxt], rank[node_id] + 1)
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                queue.append(nxt)
    return rank


# ---- HTML (a self-contained live preview) ---------------------------------
def to_html(ir: DiagramIR, *, mermaid_src: str = "") -> str:
    """A standalone HTML page that renders the diagram with mermaid from a CDN.

    Useful as a "share this diagram" artifact; kept out of [EXPORT_FORMATS] since
    it needs network access at open time, which a saved artifact shouldn't rely on.
    """
    source = mermaid_src or to_mermaid(ir)
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n"
        f"<title>{html.escape(ir.title or ir.acc_title or 'Diagram')}</title>\n"
        "<script src=\"https://cdn.jsdelivr.net/npm/mermaid@11/dist/"
        "mermaid.min.js\"></script>\n"
        "<style>body{font-family:'Segoe UI',Roboto,Arial,sans-serif;margin:2rem}"
        "</style></head>\n<body>\n"
        f"<pre class=\"mermaid\">{html.escape(source)}</pre>\n"
        "<script>mermaid.initialize({startOnLoad:true,securityLevel:'loose'});"
        "</script>\n</body></html>\n"
    )


# ---- dispatch -----------------------------------------------------------
def export(ir: DiagramIR, fmt: str, *, mermaid_src: str = "",
           stem: str = "diagram") -> ExportResult:
    """Export `ir` as `fmt`. Unknown format → Mermaid. Never raises."""
    key = (fmt or MERMAID).strip().lower()
    if key not in EXPORT_FORMATS:
        key = MERMAID
    label, extension, mime = EXPORT_FORMATS[key]
    try:
        if key == MERMAID:
            content = mermaid_src or to_mermaid(ir)
        elif key == PLANTUML:
            content = to_plantuml(ir)
        elif key == DOT:
            content = to_dot(ir)
        elif key == DRAWIO:
            content = to_drawio(ir)
        elif key == ELK:
            from app.diagrams.layout import to_elk_json
            content = json.dumps(to_elk_json(ir), indent=2)
        else:
            content = json.dumps(ir.to_dict(), indent=2)
    except Exception as exc:  # noqa: BLE001
        content = ""
        label = f"{label} (failed: {exc})"
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                        for ch in (stem or "diagram"))[:48] or "diagram"
    return ExportResult(format=key, content=content,
                        filename=f"{safe_stem}.{extension}", mime=mime)


def export_all(ir: DiagramIR, *, mermaid_src: str = "",
               stem: str = "diagram") -> dict[str, dict]:
    """Every format at once — the doc's "one diagram → generate everything"."""
    return {key: export(ir, key, mermaid_src=mermaid_src, stem=stem).to_dict()
            for key in EXPORT_FORMATS}


__all__ = [
    "EXPORT_FORMATS", "MERMAID", "PLANTUML", "DOT", "DRAWIO", "ELK", "JSON_IR",
    "ExportResult", "export", "export_all",
    "to_plantuml", "to_dot", "to_drawio", "to_html",
]
