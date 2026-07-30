"""Mermaid → IR lift, so the IR pipeline works on diagrams it didn't generate.

The doc's targeted-edit and version-history items (#9/#10) both need an AST for a
diagram that already exists — one the model wrote as raw Mermaid, or one a user
hand-edited in the source view. Without this, IR features would only apply to
freshly planned diagrams and the old free-text path would stay a second-class
citizen forever.

Scope, stated honestly: this is a **pragmatic reader, not a full Mermaid
grammar**. It handles the constructs that actually appear in generated diagrams —
flowchart nodes in every shape, labelled/dotted/thick links, `subgraph`/`end`
nesting, sequence participants and messages, state transitions, ER relationships,
class relations, plus `title`/`accTitle`/`accDescr` and `%%` comments. Anything it
cannot interpret is recorded in ``ir.meta["unparsed"]`` so nothing is silently
lost and a round-trip is auditable. Pure, fail-open: a parse failure returns
whatever it understood.
"""
from __future__ import annotations

import re

from app.diagrams.ir import (
    CLASS_RELATIONS, DIRECTIONS, DiagramIR, Edge, Group, Node, SHAPES, safe_id,
)

_HEADER = re.compile(
    r"^\s*(flowchart(?:-elk)?|graph|sequenceDiagram|stateDiagram(?:-v2)?|"
    r"erDiagram|classDiagram(?:-v2)?|mindmap)\b\s*([A-Za-z]{2})?",
    re.I,
)
_HEADER_KIND = {
    "flowchart": "flowchart", "flowchart-elk": "flowchart", "graph": "flowchart",
    "sequencediagram": "sequence",
    "statediagram": "state", "statediagram-v2": "state",
    "erdiagram": "er",
    "classdiagram": "class", "classdiagram-v2": "class",
    "mindmap": "mindmap",
}

# Diagram types mermaid supports that the IR does NOT model. Recognising them is
# a SAFETY requirement, not a nicety: without it a gantt chart would be misread
# as a flowchart, and anything that re-emits from the IR (the answer-path compile
# lane, `/api/diagram/normalize`) would destroy it. Detected → the lift returns an
# IR with no nodes and `meta["unsupported_kind"]`, and every consumer leaves the
# source alone.
_UNSUPPORTED = (
    "gantt", "pie", "journey", "timeline", "gitgraph", "quadrantchart",
    "sankey", "sankey-beta", "xychart", "xychart-beta", "block", "block-beta",
    "packet", "packet-beta", "radar", "radar-beta", "treemap", "treemap-beta",
    "requirementdiagram", "kanban", "architecture", "architecture-beta",
    "zenuml", "c4context", "c4container", "c4component", "c4dynamic",
    "c4deployment",
)
# Longest name first, so `sankey-beta` is reported as itself rather than matching
# its own `sankey` prefix. (Either way it is refused; this makes the reason exact.)
_UNSUPPORTED_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(name) for name
                        in sorted(_UNSUPPORTED, key=len, reverse=True))
    + r")\b", re.I)


def unsupported_kind(source: str) -> str:
    """The mermaid diagram type in `source` that the IR cannot model, or "".

    Checked against the FIRST meaningful line (after `%%` comments and an
    `%%{init}%%` directive), which is where mermaid's own type detection looks.
    """
    try:
        for line in (_strip_fence(source) or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("%%"):
                continue
            match = _UNSUPPORTED_RE.match(stripped)
            return match.group(1).lower() if match else ""
    except Exception:  # noqa: BLE001
        return ""
    return ""
# Node shapes, longest delimiter first so `[[x]]` never matches as `[ [x] ]`.
_SHAPE_PAIRS: list[tuple[str, str, str]] = sorted(
    ((name, open_ch, close_ch) for name, (open_ch, close_ch) in SHAPES.items()),
    key=lambda item: len(item[1]) + len(item[2]),
    reverse=True,
)
_ID = r"[A-Za-z0-9_\-]+"
# Flowchart ids commonly contain `-` (`api-gw`), and `_LINK_*` relies on regex
# backtracking to tell `A-->B` apart from an id. The other diagram kinds use
# arrows made of dashes with NO surrounding whitespace requirement (`S-->>U`), so
# a dash-inclusive id class there eats the arrow's own dashes and silently
# mis-reads the link style. Those grammars use `_SID` instead.
_SID = r"[A-Za-z0-9_]+"
# Flowchart links, matched against a line whose node declarations have already
# been REDUCED to bare ids (see `_reduce_declarations`) — trying to match
# `A[One] -- t --> B[(Two)]` in one regex is how the previous version silently
# dropped edges. Three explicit forms, tried longest-first:
#
#   pipe      A -->|text| B          (also `A ~~~|text| B`)
#   middle    A -- text --> B        (also `-. t .->`, `== t ===`, `<-- t -->`)
#   plain     A --> B                (any dash count: `--->` is valid mermaid)
#
# `_PLAIN_OP` is ordered so a longer spelling wins over its own prefix.
_PLAIN_OP = (r"<-\.+->|<-{2,}>|<={2,}>|-\.+->|-\.+-|-{2,}>|-{3,}|={2,}>|={3,}|~{3,}")
_LINK_PIPE = re.compile(
    rf"^(?P<src>{_ID})\s*(?P<body>{_PLAIN_OP})\s*\|(?P<label>[^|]*)\|\s*"
    rf"(?P<dst>{_ID})\s*;?$")
_LINK_MIDDLE = re.compile(
    rf"^(?P<src>{_ID})\s+(?P<head><?(?:-{{2,}}|-\.+|={{2,}}))\s+"
    rf"(?P<label>\"[^\"]*\"|.+?)\s+"
    rf"(?P<tail>(?:-{{2,}}>?|\.+-{{1,2}}>?|={{2,}}>?))\s+(?P<dst>{_ID})\s*;?$")
_LINK_PLAIN = re.compile(
    rf"^(?P<src>{_ID})\s*(?P<body>{_PLAIN_OP})\s*(?P<dst>{_ID})\s*;?$")
# Sequence arrows, longest spelling first so `-->>` never reads as `->>`
# (solid vs. dotted — the difference between a call and a reply).
_SEQ_MSG = re.compile(
    rf"^\s*(?P<src>{_SID})\s*(?P<arrow>--x|--\)|-->>|-->|->>|->|-x|-\))\s*"
    rf"(?P<dst>{_SID})\s*:\s*(?P<label>.*)$"
)
_SEQ_PART = re.compile(
    rf"^\s*(?P<kw>participant|actor)\s+(?P<id>{_SID})(?:\s+as\s+(?P<label>.+))?$",
    re.I)
_STATE_TX = re.compile(
    rf"^\s*(?P<src>\[\*\]|{_SID})\s*-{{2,}}>\s*(?P<dst>\[\*\]|{_SID})"
    rf"\s*(?::\s*(?P<label>.*))?$")
_STATE_LABEL = re.compile(rf"^\s*(?P<id>{_SID})\s*:\s*(?P<label>.+)$")
_ER_REL = re.compile(
    rf"^\s*(?P<src>{_SID})\s+(?P<token>[|}}o][|o{{}}\-.]{{2,}}[|{{o]?)\s+"
    rf"(?P<dst>{_SID})\s*:\s*(?P<label>.+)$")
_CLASS_REL = re.compile(
    rf"^\s*(?P<src>{_SID})\s*(?P<op><\|--|<\|\.\.|\*--|o--|-->|\.\.>|--)\s*"
    rf"(?P<dst>{_SID})\s*(?::\s*(?P<label>.*))?$")
# `Animal : +int age` — a class member (or a `<<stereotype>>`) on one line.
_CLASS_MEMBER = re.compile(rf"^\s*(?P<id>{_SID})\s*:\s*(?P<member>.+)$")
# `class Animal {` opens a member block closed by `}`.
_CLASS_BLOCK = re.compile(rf"^\s*class\s+(?P<id>{_SID})\s*\{{\s*$", re.I)
_SUBGRAPH = re.compile(
    rf"^\s*subgraph\s+(?P<id>{_ID})?\s*(?:\[(?P<label>[^\]]*)\])?\s*$", re.I)
_ACC_TITLE = re.compile(r"^\s*accTitle\s*:\s*(?P<v>.*)$", re.I)
_ACC_DESCR = re.compile(r"^\s*accDescr\s*:\s*(?P<v>.*)$", re.I)
_TITLE = re.compile(r"^\s*title\s+(?P<v>.+)$", re.I)
_DIRECTION = re.compile(r"^\s*direction\s+(?P<v>[A-Za-z]{2})\s*$", re.I)
_INIT = re.compile(r"^\s*%%\{.*\}%%\s*$")


def _unescape(label: str) -> str:
    """Undo the emitter's entity escaping so a round-trip is lossless."""
    return (label or "").strip().strip('"').replace("#quot;", '"').replace("#35;", "#")


def _strip_fence(text: str) -> str:
    match = re.search(r"```mermaid\b\s*\n(.*?)```", text or "", re.S | re.I)
    return (match.group(1) if match else (text or "")).strip()


_SHAPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (shape, re.compile(
        rf"({_ID})\s*{re.escape(open_ch)}\s*(\"[^\"]*\"|.*?)\s*{re.escape(close_ch)}"))
    for shape, open_ch, close_ch in _SHAPE_PAIRS
]


def _reduce_declarations(line: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Strip node declarations out of `line`, leaving bare ids behind.

    `A[API] --> B[(DB)]` → `A --> B` plus
    `[("A", "API", "rect"), ("B", "DB", "cylinder")]`.

    Shapes are tried longest-delimiter-first and each match is CONSUMED, so a
    line mixing shapes is fully read. The previous version stopped at the first
    shape family that matched anything, which is why `A[API] --> B[(DB)]` lost
    both the `A` node and the edge.
    """
    found: list[tuple[str, str, str]] = []
    reduced = line
    for shape, pattern in _SHAPE_PATTERNS:
        def replace(match: "re.Match") -> str:
            found.append((match.group(1), _unescape(match.group(2)), shape))
            return match.group(1)
        reduced = pattern.sub(replace, reduced)
    return reduced.strip(), found


def _link_style(body: str) -> str:
    if "~" in body:
        return "invisible"
    if "." in body:
        return "dotted"
    if "=" in body:
        return "thick"
    return "solid"


def _link_arrow(head: str, tail: str) -> str:
    """`(<)` on the head and `>` on the tail decide the arrowheads."""
    if head.startswith("<"):
        return "bidirectional"
    return "arrow" if tail.endswith(">") else "open"


def _match_link(reduced: str):
    """→ (src, dst, label, style, arrow) for a SINGLE link, or None."""
    match = _LINK_PIPE.match(reduced)
    if match:
        body = match.group("body")
        return (match.group("src"), match.group("dst"),
                _unescape(match.group("label")), _link_style(body),
                _link_arrow(body, body))
    match = _LINK_MIDDLE.match(reduced)
    if match:
        head, tail = match.group("head"), match.group("tail")
        return (match.group("src"), match.group("dst"),
                _unescape(match.group("label")), _link_style(head + tail),
                _link_arrow(head, tail))
    match = _LINK_PLAIN.match(reduced)
    if match:
        body = match.group("body")
        return (match.group("src"), match.group("dst"), "",
                _link_style(body), _link_arrow(body, body))
    return None


# A chain: `A --> B --> C`, or `A -->|x| B -.-> C`. Mermaid allows these and
# models write them, so a reader that only handles one link per line loses edges.
_CHAIN_TOKEN = re.compile(
    rf"(?P<op>{_PLAIN_OP})(?:\s*\|(?P<pipe>[^|]*)\|)?\s*(?P<node>{_ID})")
_CHAIN_HEAD = re.compile(rf"^(?P<node>{_ID})\s*")


def _match_links(reduced: str) -> list[tuple[str, str, str, str, str]]:
    """Every link on the (declaration-reduced) line. Falls back to the chain
    tokenizer when the single-link forms don't match the whole line."""
    single = _match_link(reduced)
    if single:
        return [single]
    head = _CHAIN_HEAD.match(reduced)
    if not head:
        return []
    out: list[tuple[str, str, str, str, str]] = []
    previous = head.group("node")
    position = head.end()
    while position < len(reduced):
        token = _CHAIN_TOKEN.match(reduced, position)
        if not token:
            return []                      # trailing junk → not a clean chain
        body = token.group("op")
        out.append((previous, token.group("node"),
                    _unescape(token.group("pipe") or ""), _link_style(body),
                    _link_arrow(body, body)))
        previous = token.group("node")
        position = token.end()
        while position < len(reduced) and reduced[position] in " \t;":
            position += 1
    return out


def from_mermaid(source: str) -> DiagramIR:
    """Lift Mermaid `source` into a :class:`DiagramIR`. Never raises."""
    ir = DiagramIR()
    unparsed: list[str] = []
    try:
        body = _strip_fence(source)
        if not body.strip():
            return ir
        # A diagram type the IR cannot model must never be guessed at — return an
        # empty IR carrying the reason so callers pass the source through.
        refused = unsupported_kind(body)
        if refused:
            ir.meta["unsupported_kind"] = refused
            return ir
        lines = body.splitlines()

        # Header: kind + direction.
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("%%"):
                continue
            match = _HEADER.match(stripped)
            if match:
                ir.kind = _HEADER_KIND.get(match.group(1).lower(), "flowchart")
                direction = (match.group(2) or "").upper()
                if direction in DIRECTIONS:
                    ir.direction = direction
            break

        node_ids: set[str] = set()
        group_stack: list[str] = []

        def add_node(node_id: str, label: str = "", shape: str = "") -> None:
            clean = safe_id(node_id)
            if not clean:
                return
            existing = ir.node(clean)
            if existing is None:
                ir.nodes.append(Node(
                    id=clean, label=label or "", shape=shape or "",
                    group=group_stack[-1] if group_stack else "",
                ))
                node_ids.add(clean)
                return
            if label and not existing.label:
                existing.label = label
            if shape and not existing.shape:
                existing.shape = shape
            if group_stack and not existing.group:
                existing.group = group_stack[-1]

        first_header_seen = False
        in_acc_block = False
        acc_lines: list[str] = []
        class_body = ""                    # the class whose `{ … }` block is open
        for raw in lines:
            line = raw.rstrip()
            stripped = line.strip()
            if in_acc_block:
                if stripped == "}":
                    in_acc_block = False
                    ir.acc_descr = "\n".join(acc_lines).strip()
                else:
                    acc_lines.append(stripped)
                continue
            if not stripped or _INIT.match(stripped):
                continue
            if stripped.startswith("%%"):
                continue
            if not first_header_seen and _HEADER.match(stripped):
                first_header_seen = True
                continue
            if stripped.lower().startswith("accdescr {"):
                in_acc_block, acc_lines = True, []
                continue
            match = _ACC_TITLE.match(stripped)
            if match:
                ir.acc_title = match.group("v").strip()
                continue
            match = _ACC_DESCR.match(stripped)
            if match:
                ir.acc_descr = match.group("v").strip()
                continue
            match = _DIRECTION.match(stripped)
            if match and match.group("v").upper() in DIRECTIONS:
                ir.direction = match.group("v").upper()
                continue
            match = _TITLE.match(stripped)
            if match and ir.kind != "flowchart":
                ir.title = match.group("v").strip()
                continue
            if stripped.lower() in ("autonumber", "end"):
                if stripped.lower() == "end" and group_stack:
                    group_stack.pop()
                continue
            match = _SUBGRAPH.match(stripped)
            if match:
                gid = safe_id(match.group("id") or match.group("label") or "group",
                              fallback="g")
                ir.groups.append(Group(
                    id=gid, label=_unescape(match.group("label") or "") or gid,
                    parent=group_stack[-1] if group_stack else "",
                ))
                group_stack.append(gid)
                continue

            if ir.kind == "sequence":
                match = _SEQ_PART.match(stripped)
                if match:
                    add_node(match.group("id"),
                             _unescape(match.group("label") or ""),
                             "round" if match.group("kw").lower() == "actor" else "")
                    continue
                match = _SEQ_MSG.match(stripped)
                if match:
                    add_node(match.group("src"))
                    add_node(match.group("dst"))
                    arrow = match.group("arrow")
                    ir.edges.append(Edge(
                        src=safe_id(match.group("src")),
                        dst=safe_id(match.group("dst")),
                        label=_unescape(match.group("label")),
                        # `-->>` / `-->` are mermaid's DOTTED (reply) arrows;
                        # `->>` / `->` are solid (call).
                        style="dotted" if arrow.startswith("--") else "solid",
                        arrow="arrow" if arrow.endswith(">") else "open",
                    ))
                    continue
                if stripped.lower().startswith("note "):
                    continue
                unparsed.append(stripped)
                continue

            if ir.kind == "state":
                match = _STATE_TX.match(stripped)
                if match:
                    src, dst = match.group("src"), match.group("dst")
                    if src == "[*]":
                        add_node("start", "start")
                        node = ir.node("start")
                        if node:
                            node.role = "start"
                        src = "start"
                    if dst == "[*]":
                        add_node("done", "end")
                        node = ir.node("done")
                        if node:
                            node.role = "end"
                        dst = "done"
                    add_node(src)
                    add_node(dst)
                    ir.edges.append(Edge(src=safe_id(src), dst=safe_id(dst),
                                         label=_unescape(match.group("label") or "")))
                    continue
                match = _STATE_LABEL.match(stripped)
                if match:
                    add_node(match.group("id"), _unescape(match.group("label")))
                    continue
                unparsed.append(stripped)
                continue

            if ir.kind == "er":
                match = _ER_REL.match(stripped)
                if match:
                    add_node(match.group("src"))
                    add_node(match.group("dst"))
                    ir.edges.append(Edge(
                        src=safe_id(match.group("src")),
                        dst=safe_id(match.group("dst")),
                        label=_unescape(match.group("label")).replace("_", " "),
                        cardinality=_er_cardinality(match.group("token")),
                    ))
                    continue
                unparsed.append(stripped)
                continue

            if ir.kind == "class":
                match = _CLASS_BLOCK.match(stripped)
                if match:
                    add_node(match.group("id"))
                    class_body = safe_id(match.group("id"))
                    continue
                if stripped == "}":
                    class_body = ""
                    continue
                match = _CLASS_REL.match(stripped)
                if match:
                    add_node(match.group("src"))
                    add_node(match.group("dst"))
                    ir.edges.append(Edge(
                        src=safe_id(match.group("src")),
                        dst=safe_id(match.group("dst")),
                        label=_unescape(match.group("label") or ""),
                        relation=_class_relation(match.group("op")),
                    ))
                    continue
                match = re.match(rf"^\s*class\s+({_ID})", stripped)
                if match:
                    add_node(match.group(1))
                    continue
                # `Animal : +int age` — the one-line member form. The block form
                # (`class Animal {` … `}`) is handled by `class_body` below.
                match = _CLASS_MEMBER.match(stripped)
                if match:
                    add_node(match.group("id"))
                    node = ir.node(safe_id(match.group("id")))
                    member = _unescape(match.group("member"))
                    if node is not None and member:
                        # A `<<stereotype>>` is a label, not a member.
                        stereotype = re.match(r"^<<(.+)>>$", member)
                        if stereotype:
                            node.label = stereotype.group(1).strip()
                        elif member not in node.members:
                            node.members.append(member)
                    continue
                if class_body:
                    member = _unescape(stripped)
                    node = ir.node(class_body)
                    if node is not None and member and member not in node.members:
                        node.members.append(member)
                    continue
                unparsed.append(stripped)
                continue

            # --- flowchart (and mindmap, which reads close enough) ---------
            reduced, declared = _reduce_declarations(stripped)
            for node_id, label, shape in declared:
                add_node(node_id, label, shape)
            links = _match_links(reduced)
            if links:
                for src, dst, label, style, arrow in links:
                    add_node(src)
                    add_node(dst)
                    ir.edges.append(Edge(src=safe_id(src), dst=safe_id(dst),
                                         label=label, style=style, arrow=arrow))
                continue
            if declared:
                continue
            bare = re.match(rf"^({_ID})$", reduced)
            if bare:
                add_node(bare.group(1))
                continue
            unparsed.append(stripped)
    except Exception as exc:  # noqa: BLE001 — return whatever was understood
        unparsed.append(f"<parser error: {exc}>")
    if unparsed:
        ir.meta["unparsed"] = unparsed[:40]
    return ir


_ER_CARDINALITY = {
    "||--||": "1-1", "||--o{": "1-*", "}o--||": "*-1", "}o--o{": "*-*",
    "|o--o|": "0-1",
}


def _er_cardinality(token: str) -> str:
    return _ER_CARDINALITY.get((token or "").strip(), "")


# Operator → relation name. `CLASS_RELATIONS` maps several names onto the same
# operator (inheritance/extends both emit `<|--`), so the reverse map is built
# from an explicit CANONICAL order rather than by inverting the dict — inverting
# it silently kept whichever alias happened to be declared last.
_CANONICAL_RELATIONS = ("inheritance", "implements", "composition",
                        "aggregation", "association", "dependency", "link")
_CLASS_OPS: dict[str, str] = {}
for _name in _CANONICAL_RELATIONS:
    _CLASS_OPS.setdefault(CLASS_RELATIONS[_name], _name)


def _class_relation(op: str) -> str:
    return _CLASS_OPS.get((op or "").strip(), "association")


def round_trip(source: str) -> tuple[DiagramIR, str]:
    """`source` → IR → deterministic Mermaid. Handy for tests and for the
    "normalize this hand-written diagram" path."""
    from app.diagrams.ir import to_mermaid
    ir = from_mermaid(source)
    return ir, to_mermaid(ir)


__all__ = ["from_mermaid", "round_trip", "unsupported_kind"]
