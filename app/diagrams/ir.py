"""Diagram intermediate representation + the deterministic Mermaid emitter.

MermaidDiagramVisualizations.md's **#1 priority**: stop asking the model for
Mermaid. Ask it for a *structure* — nodes, edges, labels, subgraphs — and let a
code generator turn that structure into syntax:

    Prompt → Planner → Diagram IR (JSON) → Mermaid Generator → Parser → SVG

Everything that makes a model's Mermaid fail is a *syntax* concern the model no
longer touches here: identifier legality, label quoting, `<br/>` wrapping, arrow
spelling, `subgraph`/`end` balance, direction tokens, the fact that `sequenceDiagram`
must NOT carry a direction. The emitter owns all of it, so a well-formed IR
cannot produce a diagram that fails to parse.

The IR is one shape for every diagram kind we support (flowchart, sequence,
state, ER, class, mindmap). Per-kind emitters read the fields that are meaningful
to them and ignore the rest, so a planner never has to learn six schemas.

Pure module: no I/O, no config, no model. :func:`json_schema` is the contract the
planner is constrained to (see ``response_arch.structured.generate_structured``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---- vocabulary -----------------------------------------------------------
KINDS = ("flowchart", "sequence", "state", "er", "class", "mindmap")
DIRECTIONS = ("TD", "TB", "BT", "LR", "RL")

# Node shape → the mermaid delimiter pair that draws it. `rect` is the default.
SHAPES: dict[str, tuple[str, str]] = {
    "rect": ("[", "]"),
    "round": ("(", ")"),
    "stadium": ("([", "])"),
    "subroutine": ("[[", "]]"),
    "cylinder": ("[(", ")]"),
    "circle": ("((", "))"),
    "doublecircle": ("(((", ")))"),
    "rhombus": ("{", "}"),
    "hexagon": ("{{", "}}"),
    "parallelogram": ("[/", "/]"),
    "trapezoid": ("[/", "\\]"),
}
# A semantic role hints the shape when the planner didn't pick one — this is how
# a "datastore" reliably becomes a cylinder and a "decision" a rhombus without
# the model needing to know mermaid's delimiter table.
ROLE_SHAPES: dict[str, str] = {
    "start": "stadium", "end": "stadium", "terminal": "stadium",
    "decision": "rhombus", "choice": "rhombus", "gateway": "rhombus",
    "datastore": "cylinder", "database": "cylinder", "store": "cylinder",
    "queue": "subroutine", "broker": "subroutine",
    "external": "hexagon", "thirdparty": "hexagon",
    "actor": "round", "user": "round", "person": "round",
}
# Edge style → the mermaid link body (flowchart). `arrow` picks the head.
EDGE_STYLES = ("solid", "dotted", "thick", "invisible")
ARROWS = ("arrow", "open", "bidirectional", "none")
# Class-diagram relation → mermaid relation operator.
CLASS_RELATIONS: dict[str, str] = {
    "inheritance": "<|--", "extends": "<|--", "implements": "<|..",
    "composition": "*--", "aggregation": "o--",
    "association": "-->", "dependency": "..>", "link": "--",
}

_ID_BAD = re.compile(r"[^A-Za-z0-9_]")
_WORD = re.compile(r"\s+")


# ---- helpers (the emitter's whole reliability story) ----------------------
def safe_id(raw: str, *, fallback: str = "n") -> str:
    """A legal mermaid node identifier.

    Mermaid IDs may only carry letters, digits and `_`, and may not start with a
    digit. Anything a planner invents ("Kafka Broker 2", "api-gw", "café") is
    folded down to that alphabet — the *label* keeps the human text.
    """
    s = _ID_BAD.sub("_", (raw or "").strip())
    s = re.sub(r"_{2,}", "_", s).strip("_")
    if not s:
        return fallback
    if s[0].isdigit():
        s = f"{fallback}_{s}"
    return s[:64]


def wrap_label(label: str, width: int = 22) -> str:
    """Insert `<br/>` at word boundaries so a long label never widens a node
    past readability. Idempotent (a label that already breaks is left alone)."""
    text = (label or "").strip()
    if not text or "<br" in text or len(text) <= width:
        return text
    out: list[str] = []
    cur = ""
    for word in _WORD.split(text):
        if cur and len(cur) + 1 + len(word) > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return "<br/>".join(out)


def escape_label(label: str) -> str:
    """Make `label` safe INSIDE a mermaid double-quoted string.

    Mermaid resolves `#nnn;`/`#quot;` HTML entities inside quoted labels, so the
    two characters that would otherwise end the string (`"`) or start an entity
    (`#`) are escaped that way. Everything else — parentheses, colons, `<`, `>`,
    `&`, `|` — is safe once quoted, which is why the emitter ALWAYS quotes.

    A LITERAL newline is the one remaining parse-breaker: a quoted label may not
    span source lines, so a planner writing "Broker 1\\n:9092" (the natural way
    to ask for two lines) would emit source mermaid cannot parse. `<br/>` is
    mermaid's own line break, and is what `wrap_label` already inserts.
    """
    return (
        (label or "")
        .replace("#", "#35;")
        .replace('"', "#quot;")
        .replace("\r\n", "<br/>")
        .replace("\r", "<br/>")
        .replace("\n", "<br/>")
    )


def _quoted(label: str, *, width: int = 22) -> str:
    return f'"{escape_label(wrap_label(label, width))}"'


# ---- IR -------------------------------------------------------------------
@dataclass
class Node:
    """A vertex: box, participant, state, entity or class depending on `kind`."""
    id: str
    label: str = ""
    shape: str = ""                       # SHAPES key; "" → derive from role
    role: str = ""                        # semantic role (ROLE_SHAPES)
    group: str = ""                       # owning Group.id ("" = top level)
    note: str = ""                        # attached note (sequence/state/class)
    members: list[str] = field(default_factory=list)  # class attrs/methods
    css_class: str = ""                   # mermaid classDef name

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "shape": self.shape,
                "role": self.role, "group": self.group, "note": self.note,
                "members": list(self.members), "css_class": self.css_class}

    @property
    def text(self) -> str:
        """The display label, falling back to a de-slugged id."""
        return self.label.strip() or self.id.replace("_", " ").strip()

    def resolved_shape(self) -> str:
        if self.shape in SHAPES:
            return self.shape
        return ROLE_SHAPES.get((self.role or "").lower().replace(" ", ""), "rect")


@dataclass
class Edge:
    """A directed relation. `label` is the edge text; `relation` carries the
    class-diagram operator and `cardinality` the ER multiplicity."""
    src: str
    dst: str
    label: str = ""
    style: str = "solid"                  # EDGE_STYLES
    arrow: str = "arrow"                  # ARROWS
    relation: str = ""                    # CLASS_RELATIONS key (class diagrams)
    cardinality: str = ""                 # e.g. "1..*" (ER diagrams)

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "label": self.label,
                "style": self.style, "arrow": self.arrow,
                "relation": self.relation, "cardinality": self.cardinality}


@dataclass
class Group:
    """A subgraph / nested cluster. `parent` allows one level of nesting or more."""
    id: str
    label: str = ""
    parent: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "parent": self.parent}


@dataclass
class DiagramIR:
    kind: str = "flowchart"
    direction: str = "TD"
    title: str = ""
    acc_title: str = ""                   # mermaid accTitle (screen readers)
    acc_descr: str = ""                   # mermaid accDescr
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    layout: str = ""                      # "" | "dagre" | "elk"
    label_wrap: int = 22
    meta: dict = field(default_factory=dict)

    # -- lookups ----------------------------------------------------------
    def node(self, node_id: str) -> Node | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def group(self, group_id: str) -> Group | None:
        for g in self.groups:
            if g.id == group_id:
                return g
        return None

    @property
    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}

    def to_dict(self) -> dict:
        return {"kind": self.kind, "direction": self.direction,
                "title": self.title, "acc_title": self.acc_title,
                "acc_descr": self.acc_descr,
                "nodes": [n.to_dict() for n in self.nodes],
                "edges": [e.to_dict() for e in self.edges],
                "groups": [g.to_dict() for g in self.groups],
                "layout": self.layout, "label_wrap": self.label_wrap,
                "meta": dict(self.meta)}

    def copy(self) -> "DiagramIR":
        return from_dict(self.to_dict())


def _str(value, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def from_dict(data: dict | None) -> DiagramIR:
    """Build an IR from loose JSON — the planner's output, a stored version, an
    edit payload. Unknown keys are ignored, bad values fall back to defaults,
    node ids are made legal, and edges pointing at unknown nodes are KEPT (the
    validators report them as findings; silently dropping structure would hide a
    planning mistake). Never raises."""
    ir = DiagramIR()
    if not isinstance(data, dict):
        return ir
    try:
        kind = _str(data.get("kind"), "flowchart").lower()
        # "graph" is mermaid's older flowchart keyword; accept it as an alias.
        ir.kind = kind if kind in KINDS else ("flowchart" if kind == "graph" else "flowchart")
        direction = _str(data.get("direction"), "TD").upper()
        ir.direction = direction if direction in DIRECTIONS else "TD"
        ir.title = _str(data.get("title"))
        ir.acc_title = _str(data.get("acc_title")) or _str(data.get("accTitle"))
        ir.acc_descr = _str(data.get("acc_descr")) or _str(data.get("accDescr"))
        layout = _str(data.get("layout")).lower()
        ir.layout = layout if layout in ("dagre", "elk") else ""
        try:
            wrap = int(data.get("label_wrap") or 22)
            ir.label_wrap = wrap if 8 <= wrap <= 80 else 22
        except Exception:  # noqa: BLE001
            ir.label_wrap = 22
        if isinstance(data.get("meta"), dict):
            ir.meta = dict(data["meta"])

        # Groups first: node.group is validated against them.
        seen_groups: set[str] = set()
        for raw in data.get("groups") or []:
            if not isinstance(raw, dict):
                continue
            gid = safe_id(_str(raw.get("id")) or _str(raw.get("label")), fallback="g")
            if not gid or gid in seen_groups:
                continue
            seen_groups.add(gid)
            ir.groups.append(Group(
                id=gid, label=_str(raw.get("label")),
                parent=safe_id(_str(raw.get("parent")), fallback="") if raw.get("parent") else "",
            ))

        # Nodes: ids deduplicated, remembering the original → safe mapping so
        # edges written against the ORIGINAL names still resolve.
        alias: dict[str, str] = {}
        seen: set[str] = set()
        for raw in data.get("nodes") or []:
            if isinstance(raw, str):
                raw = {"id": raw, "label": raw}
            if not isinstance(raw, dict):
                continue
            original = _str(raw.get("id")) or _str(raw.get("label"))
            if not original:
                continue
            nid = safe_id(original)
            if nid in seen:
                continue
            seen.add(nid)
            alias[original] = nid
            alias.setdefault(nid, nid)
            shape = _str(raw.get("shape")).lower()
            members = [
                _str(m) for m in (raw.get("members") or []) if _str(m)
            ] if isinstance(raw.get("members"), list) else []
            grp = safe_id(_str(raw.get("group")), fallback="") if raw.get("group") else ""
            ir.nodes.append(Node(
                id=nid,
                label=_str(raw.get("label")) or original,
                shape=shape if shape in SHAPES else "",
                role=_str(raw.get("role")),
                group=grp if grp in seen_groups else "",
                note=_str(raw.get("note")),
                members=members,
                css_class=safe_id(_str(raw.get("css_class")), fallback="") if raw.get("css_class") else "",
            ))

        for raw in data.get("edges") or []:
            if not isinstance(raw, dict):
                continue
            src_raw = _str(raw.get("src")) or _str(raw.get("from")) or _str(raw.get("source"))
            dst_raw = _str(raw.get("dst")) or _str(raw.get("to")) or _str(raw.get("target"))
            if not src_raw or not dst_raw:
                continue
            style = _str(raw.get("style"), "solid").lower()
            arrow = _str(raw.get("arrow"), "arrow").lower()
            relation = _str(raw.get("relation")).lower()
            ir.edges.append(Edge(
                src=alias.get(src_raw, safe_id(src_raw)),
                dst=alias.get(dst_raw, safe_id(dst_raw)),
                label=_str(raw.get("label")),
                style=style if style in EDGE_STYLES else "solid",
                arrow=arrow if arrow in ARROWS else "arrow",
                relation=relation if relation in CLASS_RELATIONS else "",
                cardinality=_str(raw.get("cardinality")),
            ))
    except Exception:  # noqa: BLE001 — a malformed payload yields what parsed
        pass
    return ir


# ---- Mermaid emitters ----------------------------------------------------
# (style, arrow) → the flowchart link. Written out as a TABLE rather than
# assembled from pieces because mermaid's link spellings are irregular: an
# arrowless solid link needs THREE dashes (`---`, not `--`), a dotted link's tail
# differs from its head (`-. text .->`), and an invisible link takes no label at
# all. Every entry below is parsed by mermaid 11.15.0 in
# `tests/test_diagram_ir.py::test_every_link_spelling_is_valid`'s golden list.
_LINKS_PLAIN: dict[tuple[str, str], str] = {
    ("solid", "arrow"): "-->", ("solid", "open"): "---",
    ("solid", "none"): "---", ("solid", "bidirectional"): "<-->",
    ("dotted", "arrow"): "-.->", ("dotted", "open"): "-.-",
    ("dotted", "none"): "-.-", ("dotted", "bidirectional"): "<-.->",
    ("thick", "arrow"): "==>", ("thick", "open"): "===",
    ("thick", "none"): "===", ("thick", "bidirectional"): "<==>",
}
# The labelled form, as (prefix, suffix) around the quoted label.
_LINKS_LABELLED: dict[tuple[str, str], tuple[str, str]] = {
    ("solid", "arrow"): ("--", "-->"), ("solid", "open"): ("--", "---"),
    ("solid", "none"): ("--", "---"), ("solid", "bidirectional"): ("<--", "-->"),
    ("dotted", "arrow"): ("-.", ".->"), ("dotted", "open"): ("-.", ".-"),
    ("dotted", "none"): ("-.", ".-"), ("dotted", "bidirectional"): ("<-.", ".->"),
    ("thick", "arrow"): ("==", "==>"), ("thick", "open"): ("==", "==="),
    ("thick", "none"): ("==", "==="), ("thick", "bidirectional"): ("<==", "==>"),
}


def _flow_link(edge: Edge) -> str:
    """The flowchart link for an edge, label included.

    Only spellings mermaid actually accepts are ever produced — this is where
    `--->` (a classic model slip) becomes structurally impossible.
    """
    if edge.style == "invisible":
        return "~~~"                       # an invisible link carries no label
    key = (edge.style if edge.style in EDGE_STYLES else "solid",
           edge.arrow if edge.arrow in ARROWS else "arrow")
    if not edge.label:
        return _LINKS_PLAIN.get(key, "-->")
    prefix, suffix = _LINKS_LABELLED.get(key, ("--", "-->"))
    return f'{prefix} "{escape_label(edge.label)}" {suffix}'


def _emit_flowchart(ir: DiagramIR, lines: list[str]) -> None:
    lines.append(f"flowchart {ir.direction}")
    _emit_acc(ir, lines)

    # Declare nodes, grouped by subgraph so every id exists before it is linked.
    top = [n for n in ir.nodes if not n.group]
    for node in top:
        lines.append(f"  {_flow_node(node, ir.label_wrap)}")

    def emit_group(group: Group, indent: str) -> None:
        # A cluster TITLE is never wrapped. Mermaid lays a subgraph title out on
        # one line and reserves exactly one line of height for it, so a `<br/>`
        # here makes the title paint down INTO the cluster, over its first child
        # node. (Observed: a long subgraph name covering the node beneath it.)
        # Quote + escape only — no `wrap_label`.
        title = f'"{escape_label(group.label or group.id)}"'
        lines.append(f"{indent}subgraph {group.id}[{title}]")
        for child in ir.nodes:
            if child.group == group.id:
                lines.append(f"{indent}  {_flow_node(child, ir.label_wrap)}")
        for sub in ir.groups:
            if sub.parent == group.id and sub.id != group.id:
                emit_group(sub, indent + "  ")
        lines.append(f"{indent}end")            # balance is structural, not luck

    for group in ir.groups:
        if not group.parent:
            emit_group(group, "  ")

    for edge in ir.edges:
        lines.append(f"  {edge.src} {_flow_link(edge)} {edge.dst}")

    # classDef assignments last (mermaid allows them anywhere; keeping them at
    # the end keeps the graph readable in the source view).
    classes: dict[str, list[str]] = {}
    for node in ir.nodes:
        if node.css_class:
            classes.setdefault(node.css_class, []).append(node.id)
    for name, ids in classes.items():
        lines.append(f"  class {','.join(ids)} {name}")


def _flow_node(node: Node, wrap: int) -> str:
    open_ch, close_ch = SHAPES[node.resolved_shape()]
    return f"{node.id}{open_ch}{_quoted(node.text, width=wrap)}{close_ch}"


def _emit_sequence(ir: DiagramIR, lines: list[str]) -> None:
    # A sequenceDiagram takes NO direction token — emitting one is a parse error,
    # so the emitter simply never can.
    lines.append("sequenceDiagram")
    _emit_acc(ir, lines)
    lines.append("  autonumber")
    for node in ir.nodes:
        keyword = "actor" if node.resolved_shape() == "round" else "participant"
        lines.append(f"  {keyword} {node.id} as {escape_label(node.text)}")
    for edge in ir.edges:
        if edge.style == "dotted":
            arrow = "-->>" if edge.arrow != "open" else "-->"
        else:
            arrow = "->>" if edge.arrow != "open" else "->"
        label = escape_label(edge.label or "")
        lines.append(f"  {edge.src}{arrow}{edge.dst}: {label}")
    for node in ir.nodes:
        if node.note:
            lines.append(f"  Note over {node.id}: {escape_label(node.note)}")


def _emit_state(ir: DiagramIR, lines: list[str]) -> None:
    lines.append("stateDiagram-v2")
    _emit_acc(ir, lines)
    lines.append(f"  direction {ir.direction}")
    for node in ir.nodes:
        role = (node.role or "").lower()
        if role in ("start", "initial") or role in ("end", "final"):
            continue                       # emitted as [*] transitions below
        lines.append(f'  {node.id} : {escape_label(node.text)}')
    starts = {n.id for n in ir.nodes if (n.role or "").lower() in ("start", "initial")}
    ends = {n.id for n in ir.nodes if (n.role or "").lower() in ("end", "final")}
    for edge in ir.edges:
        src = "[*]" if edge.src in starts else edge.src
        dst = "[*]" if edge.dst in ends else edge.dst
        label = f" : {escape_label(edge.label)}" if edge.label else ""
        lines.append(f"  {src} --> {dst}{label}")
    for node in ir.nodes:
        if node.note and node.id not in starts | ends:
            lines.append(f"  note right of {node.id} : {escape_label(node.note)}")


def _emit_er(ir: DiagramIR, lines: list[str]) -> None:
    lines.append("erDiagram")
    _emit_acc(ir, lines)
    for edge in ir.edges:
        # Mermaid ER needs a relationship token; default to many-to-one
        # identifying, the safest generic shape.
        token = _er_token(edge.cardinality)
        label = escape_label(edge.label or "relates to").replace(" ", "_")
        lines.append(f"  {edge.src} {token} {edge.dst} : {label}")
    for node in ir.nodes:
        if not node.members:
            continue
        lines.append(f"  {node.id} {{")
        for member in node.members:
            parts = member.split()
            if len(parts) >= 2:
                lines.append(f"    {safe_id(parts[0])} {safe_id(parts[1])}")
            else:
                lines.append(f"    string {safe_id(member)}")
        lines.append("  }")


_ER_TOKENS = {
    "1-1": "||--||", "one-to-one": "||--||",
    "1-*": "||--o{", "one-to-many": "||--o{", "1..*": "||--o{",
    "*-1": "}o--||", "many-to-one": "}o--||",
    "*-*": "}o--o{", "many-to-many": "}o--o{",
    "0-1": "|o--o|", "zero-or-one": "|o--o|",
}


def _er_token(cardinality: str) -> str:
    return _ER_TOKENS.get((cardinality or "").strip().lower(), "||--o{")


def _emit_class(ir: DiagramIR, lines: list[str]) -> None:
    lines.append("classDiagram")
    _emit_acc(ir, lines)
    lines.append(f"  direction {ir.direction}")
    for node in ir.nodes:
        if node.members:
            lines.append(f"  class {node.id} {{")
            for member in node.members:
                lines.append(f"    {member.replace('{', '').replace('}', '')}")
            lines.append("  }")
        else:
            lines.append(f"  class {node.id}")
        if node.label and safe_id(node.label) != node.id:
            lines.append(f'  {node.id} : <<{escape_label(node.label)}>>')
    for edge in ir.edges:
        op = CLASS_RELATIONS.get(edge.relation or "association", "-->")
        label = f" : {escape_label(edge.label)}" if edge.label else ""
        lines.append(f"  {edge.src} {op} {edge.dst}{label}")


def _emit_mindmap(ir: DiagramIR, lines: list[str]) -> None:
    lines.append("mindmap")
    # Mindmap is indentation-driven: build a child map and walk from the roots.
    children: dict[str, list[str]] = {}
    has_parent: set[str] = set()
    for edge in ir.edges:
        children.setdefault(edge.src, []).append(edge.dst)
        has_parent.add(edge.dst)
    roots = [n.id for n in ir.nodes if n.id not in has_parent] or \
            [n.id for n in ir.nodes[:1]]

    seen: set[str] = set()

    def walk(node_id: str, depth: int) -> None:
        if node_id in seen or depth > 8:
            return
        seen.add(node_id)
        node = ir.node(node_id)
        text = escape_label(node.text if node else node_id)
        lines.append("  " * (depth + 1) + f"{node_id}[{text}]")
        for child in children.get(node_id, []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)


def _emit_acc(ir: DiagramIR, lines: list[str]) -> None:
    """Accessibility metadata — the doc's accessibility validator (#6) has
    something to check because the emitter can actually carry it."""
    if ir.title:
        lines.append(f"  title {ir.title}")
    if ir.acc_title:
        lines.append(f"  accTitle: {ir.acc_title}")
    if ir.acc_descr:
        # A multi-line description needs the block form.
        if "\n" in ir.acc_descr:
            lines.append("  accDescr {")
            for chunk in ir.acc_descr.splitlines():
                lines.append(f"    {chunk.strip()}")
            lines.append("  }")
        else:
            lines.append(f"  accDescr: {ir.acc_descr}")


_EMITTERS = {
    "flowchart": _emit_flowchart,
    "sequence": _emit_sequence,
    "state": _emit_state,
    "er": _emit_er,
    "class": _emit_class,
    "mindmap": _emit_mindmap,
}


def to_mermaid(ir: DiagramIR, *, init_directive: str = "") -> str:
    """Render `ir` to Mermaid source. Deterministic: same IR → same bytes.

    `init_directive` is an optional `%%{init: …}%%` line (see
    :mod:`app.diagrams.layout`) prepended verbatim. Fail-open: an emitter error
    degrades to a minimal but VALID diagram rather than raising into a turn.
    """
    try:
        lines: list[str] = []
        if init_directive:
            lines.append(init_directive)
        _EMITTERS.get(ir.kind, _emit_flowchart)(ir, lines)
        out = "\n".join(line.rstrip() for line in lines).strip()
        return out or "flowchart TD\n  empty[\"(empty diagram)\"]"
    except Exception:  # noqa: BLE001
        return 'flowchart TD\n  error["Diagram could not be generated"]'


# ---- the planner's contract ----------------------------------------------
def json_schema() -> dict:
    """JSON Schema for the IR — handed to constrained decoding so the planner is
    *structurally* prevented from inventing Mermaid syntax."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "nodes"],
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "direction": {"type": "string", "enum": list(DIRECTIONS)},
            "title": {"type": "string"},
            "acc_title": {"type": "string"},
            "acc_descr": {"type": "string"},
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "parent": {"type": "string"},
                    },
                },
            },
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "label"],
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "shape": {"type": "string", "enum": list(SHAPES)},
                        "role": {"type": "string"},
                        "group": {"type": "string"},
                        "note": {"type": "string"},
                        "members": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["src", "dst"],
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "label": {"type": "string"},
                        "style": {"type": "string", "enum": list(EDGE_STYLES)},
                        "arrow": {"type": "string", "enum": list(ARROWS)},
                        "relation": {"type": "string",
                                     "enum": list(CLASS_RELATIONS)},
                        "cardinality": {"type": "string"},
                    },
                },
            },
        },
    }


__all__ = [
    "KINDS", "DIRECTIONS", "SHAPES", "ROLE_SHAPES", "EDGE_STYLES", "ARROWS",
    "CLASS_RELATIONS", "Node", "Edge", "Group", "DiagramIR",
    "from_dict", "to_mermaid", "json_schema",
    "safe_id", "wrap_label", "escape_label",
]
