"""The validator stack — syntax, semantic, style, accessibility.

MermaidDiagramVisualizations.md #6 argues one validator is not enough:

    LLM → Syntax → Semantic → Style → Accessibility → Render

and #7 ("Diagram Intelligence") makes the point that carries the whole idea:
*many diagrams are syntactically correct but logically poor*. `Database → User`
parses fine and is still backwards. A parser can never catch that; a semantic
checker can.

Each validator is a pure function over the IR (or, for syntax, over raw source)
returning :class:`Finding`s. Severities are `error` (won't render or is wrong),
`warn` (renders but reads badly) and `info` (a nicety). Nothing here mutates the
diagram — :mod:`app.diagrams.quality` turns findings into a score and the caller
decides what to do. Fail-open: a validator that throws contributes no findings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.diagrams.ir import DiagramIR

ERROR = "error"
WARN = "warn"
INFO = "info"

SYNTAX = "syntax"
SEMANTIC = "semantic"
STYLE = "style"
ACCESSIBILITY = "accessibility"

# Thresholds. Generous enough that a normal diagram is clean; tight enough that a
# runaway auto-generated graph is flagged before a user squints at it.
MAX_NODES = 60
MAX_EDGES = 120
MAX_LABEL_CHARS = 60
MAX_GROUP_DEPTH = 3
MAX_FAN_OUT = 8              # edges leaving one node before it reads as a hub
MIN_LABEL_CHARS = 2


@dataclass
class Finding:
    code: str
    severity: str
    category: str
    message: str
    target: str = ""          # node/edge/group id the finding is about
    hint: str = ""            # what to do about it

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "category": self.category, "message": self.message,
                "target": self.target, "hint": self.hint}


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    def by_category(self, category: str) -> list[Finding]:
        return [f for f in self.findings if f.category == category]

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok,
                "findings": [f.to_dict() for f in self.findings],
                "counts": {
                    "error": len(self.errors),
                    "warn": len(self.warnings),
                    "info": len([f for f in self.findings if f.severity == INFO]),
                }}


# ---- 1. syntax ------------------------------------------------------------
# Every rule below was checked against mermaid 11.15.0 (the version the FE
# bundles) rather than assumed, because the folklore here is wrong in both
# directions:
#   * `A ---> B` is VALID — extra dashes set the rank distance, they are not a
#     typo. An earlier version of this file flagged it as an error, and the repair
#     prompt still "fixed" it, which churned correct diagrams.
#   * `A -> B` (one dash) and `A -- B` (a link body with no head) are NOT valid.
#   * inside an unquoted `[...]` label, the characters that actually break the
#     parser are `(`, `)`, `"`, `|`, `{`, `}`. Colon, `#`, `<`, `>`, `;`, `&` and
#     `%` are all fine unquoted — flagging them produced false errors.
# A single dash arrow: `->` that isn't the tail of `-->`, `-.->`, `==>` etc.
_SINGLE_DASH = re.compile(r"(?<![-.=<>])->")
# A link body with no head or terminator at all: `A -- B`, `A -. B`, `A == B`.
# Requires end-of-statement after the target so a LABELLED link (`A -- t --> B`,
# which is valid) is not caught.
_HEADLESS_LINK = re.compile(
    r"^[A-Za-z0-9_\-]+\s*(?:--|-\.|==)\s+[A-Za-z0-9_\-]+\s*;?$")
_UNQUOTED_SPECIAL = re.compile(r"\[(?!\s*\")[^\]\n]*[()\"|{}][^\]\n]*\]")


def validate_syntax_source(source: str) -> list[Finding]:
    """Structural checks a text-level parser can do WITHOUT a browser.

    The real compiler is still mermaid itself (in the FE webview) — this catches
    the failures cheaply and server-side, which is what makes a repair loop fast.
    """
    out: list[Finding] = []
    try:
        text = source or ""
        lines = [line for line in text.splitlines()]
        stripped = [line.strip() for line in lines]
        meaningful = [line for line in stripped
                      if line and not line.startswith("%%")]
        if not meaningful:
            out.append(Finding("empty", ERROR, SYNTAX, "the diagram source is empty",
                               hint="generate a diagram before rendering"))
            return out

        # subgraph / end balance — the single most common model slip.
        opens = sum(1 for line in meaningful if re.match(r"^subgraph\b", line, re.I))
        closes = sum(1 for line in meaningful if line.lower() == "end")
        if opens > closes:
            out.append(Finding(
                "unclosed_subgraph", ERROR, SYNTAX,
                f"{opens - closes} `subgraph` block(s) are never closed",
                hint="add a matching `end` for every `subgraph`"))
        elif closes > opens:
            out.append(Finding(
                "extra_end", ERROR, SYNTAX,
                f"{closes - opens} stray `end` keyword(s)",
                hint="remove the unmatched `end`"))

        for index, line in enumerate(meaningful, start=1):
            if _SINGLE_DASH.search(line):
                out.append(Finding(
                    "single_dash_arrow", ERROR, SYNTAX,
                    f"line {index}: `->` is not a mermaid link (it needs at least "
                    f"two dashes)", hint="use `-->`"))
            if _HEADLESS_LINK.match(line):
                out.append(Finding(
                    "headless_link", ERROR, SYNTAX,
                    f"line {index}: a link body with no arrowhead or terminator",
                    hint="`A --- B` for a plain line, `A --> B` for an arrow"))
            if _UNQUOTED_SPECIAL.search(line):
                out.append(Finding(
                    "unquoted_label", ERROR, SYNTAX,
                    f"line {index}: a label contains `()`, `\"`, `|` or `{{}}` but "
                    f"is not quoted", hint='wrap it: A["Fetch (REST)"]'))
        return out
    except Exception:  # noqa: BLE001
        return out


def validate_syntax(ir: DiagramIR) -> list[Finding]:
    """Syntax checks against the IR.

    Deliberately thin: a well-formed IR *cannot* produce bad syntax (the emitter
    owns quoting, arrows and `subgraph`/`end` balance). What remains is checking
    the IR is well-formed enough to emit at all.
    """
    out: list[Finding] = []
    try:
        if not ir.nodes:
            out.append(Finding("no_nodes", ERROR, SYNTAX,
                               "the diagram has no nodes",
                               hint="a diagram needs at least one node"))
        seen: set[str] = set()
        for node in ir.nodes:
            if node.id in seen:
                out.append(Finding("duplicate_node", ERROR, SYNTAX,
                                   f"node `{node.id}` is declared twice",
                                   target=node.id,
                                   hint="give each node a unique id"))
            seen.add(node.id)
        for group in ir.groups:
            if group.parent and group.parent == group.id:
                out.append(Finding("group_self_parent", ERROR, SYNTAX,
                                   f"subgraph `{group.id}` is its own parent",
                                   target=group.id))
        return out
    except Exception:  # noqa: BLE001
        return out


# ---- 2. semantic ----------------------------------------------------------
# Roles that should be SOURCES of flow (things that initiate) and roles that
# should be SINKS (things acted upon). `Database --> User` inverts both.
_SOURCE_ROLES = {"user", "actor", "person", "client", "start", "initial",
                 "external", "browser", "caller"}
_SINK_ROLES = {"datastore", "database", "store", "cache", "warehouse", "log",
               "end", "final", "sink"}
_SOURCE_WORDS = re.compile(r"\b(user|client|browser|caller|actor|customer|"
                           r"visitor|operator|request)\b", re.I)
_SINK_WORDS = re.compile(r"\b(database|db|datastore|data store|cache|redis|"
                         r"postgres|mysql|s3|bucket|warehouse|table|log)\b", re.I)


def _role_of(node) -> str:
    return (node.role or "").strip().lower().replace(" ", "")


def _looks_like_source(node) -> bool:
    return _role_of(node) in _SOURCE_ROLES or bool(_SOURCE_WORDS.search(node.text))


def _looks_like_sink(node) -> bool:
    return _role_of(node) in _SINK_ROLES or bool(_SINK_WORDS.search(node.text))


def validate_semantic(ir: DiagramIR) -> list[Finding]:
    """Is the diagram *right*, not just parseable? (doc #7)"""
    out: list[Finding] = []
    try:
        ids = ir.node_ids

        # Dangling endpoints: an edge to a node that was never declared. The
        # emitter would happily create a phantom box with a slug for a label.
        for edge in ir.edges:
            for end, label in ((edge.src, "source"), (edge.dst, "target")):
                if end not in ids:
                    out.append(Finding(
                        "dangling_edge", ERROR, SEMANTIC,
                        f"edge {edge.src}→{edge.dst} has an undeclared {label} "
                        f"`{end}`", target=end,
                        hint=f"declare `{end}` as a node or drop the edge"))

        # Orphans: declared but never connected. In a 1-node diagram that's fine.
        if len(ir.nodes) > 1 and ir.kind != "mindmap":
            connected = {e.src for e in ir.edges} | {e.dst for e in ir.edges}
            for node in ir.nodes:
                if node.id not in connected:
                    out.append(Finding(
                        "orphan_node", WARN, SEMANTIC,
                        f"`{node.text}` is not connected to anything",
                        target=node.id,
                        hint="connect it, or drop it — an island adds no meaning"))

        # Duplicate edges (same pair, same label) are pure visual noise.
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in ir.edges:
            key = (edge.src, edge.dst, edge.label)
            if key in seen_edges:
                out.append(Finding(
                    "duplicate_edge", WARN, SEMANTIC,
                    f"the edge {edge.src}→{edge.dst} is drawn more than once",
                    target=f"{edge.src}->{edge.dst}"))
            seen_edges.add(key)

        # Self-loops: legitimate in a state machine, almost always a mistake in
        # an architecture flowchart.
        for edge in ir.edges:
            if edge.src == edge.dst and ir.kind not in ("state", "class"):
                out.append(Finding(
                    "self_loop", WARN, SEMANTIC,
                    f"`{edge.src}` points at itself",
                    target=edge.src,
                    hint="a self-loop in a flow usually means a missing node"))

        # Direction sanity — the doc's `Database → User` example.
        for edge in ir.edges:
            src_node, dst_node = ir.node(edge.src), ir.node(edge.dst)
            if not src_node or not dst_node:
                continue
            if _looks_like_sink(src_node) and _looks_like_source(dst_node):
                out.append(Finding(
                    "reversed_flow", WARN, SEMANTIC,
                    f"`{src_node.text}` → `{dst_node.text}` looks backwards: a "
                    f"data store driving a user/client is an unusual direction",
                    target=f"{edge.src}->{edge.dst}",
                    hint="if this is a response, label it (e.g. \"returns rows\") "
                         "or reverse the arrow"))

        # A decision node whose branches are unlabelled is unreadable — which
        # branch is yes?
        for node in ir.nodes:
            if node.resolved_shape() != "rhombus":
                continue
            out_edges = [e for e in ir.edges if e.src == node.id]
            if len(out_edges) >= 2 and any(not e.label for e in out_edges):
                out.append(Finding(
                    "unlabelled_branch", WARN, SEMANTIC,
                    f"decision `{node.text}` has unlabelled branches",
                    target=node.id,
                    hint="label each branch (yes / no, ok / error)"))

        # A flow described as a pipeline shouldn't loop back on itself.
        if ir.kind == "flowchart":
            cycle = _find_cycle(ir)
            if cycle:
                out.append(Finding(
                    "cycle", INFO, SEMANTIC,
                    "the flow contains a cycle: " + " → ".join(cycle),
                    target=cycle[0],
                    hint="intentional for a retry/feedback loop; otherwise an "
                         "arrow points the wrong way"))

        # A "flow" with no edges is a bag of boxes, not a diagram.
        if len(ir.nodes) >= 3 and not ir.edges and ir.kind != "mindmap":
            out.append(Finding(
                "no_edges", ERROR, SEMANTIC,
                "several nodes but no relationships — nothing is being shown",
                hint="a diagram's value IS the edges; add them or use a list"))
        return out
    except Exception:  # noqa: BLE001
        return out


def _find_cycle(ir: DiagramIR) -> list[str]:
    """One cycle as a node-id path (empty when acyclic). Iterative DFS so a huge
    graph can't blow the stack."""
    adjacency: dict[str, list[str]] = {}
    for edge in ir.edges:
        if edge.src != edge.dst:
            adjacency.setdefault(edge.src, []).append(edge.dst)
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}
    for start in list(adjacency):
        if colour.get(start, WHITE) != WHITE:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        path: list[str] = [start]
        colour[start] = GREY
        while stack:
            node, index = stack[-1]
            neighbours = adjacency.get(node, [])
            if index >= len(neighbours):
                colour[node] = BLACK
                stack.pop()
                if path:
                    path.pop()
                continue
            stack[-1] = (node, index + 1)
            nxt = neighbours[index]
            state = colour.get(nxt, WHITE)
            if state == GREY:
                if nxt in path:
                    return path[path.index(nxt):] + [nxt]
                return [nxt, nxt]
            if state == WHITE:
                colour[nxt] = GREY
                stack.append((nxt, 0))
                path.append(nxt)
    return []


# ---- 3. style -------------------------------------------------------------
def validate_style(ir: DiagramIR) -> list[Finding]:
    """Will it *read* well once drawn?"""
    out: list[Finding] = []
    try:
        if len(ir.nodes) > MAX_NODES:
            out.append(Finding(
                "too_many_nodes", WARN, STYLE,
                f"{len(ir.nodes)} nodes (over {MAX_NODES}) — this will render dense",
                hint="split it into a high-level diagram plus per-area detail"))
        if len(ir.edges) > MAX_EDGES:
            out.append(Finding(
                "too_many_edges", WARN, STYLE,
                f"{len(ir.edges)} edges (over {MAX_EDGES}) — expect crossings",
                hint="group related nodes into subgraphs"))

        for node in ir.nodes:
            text = node.text
            if len(text) > MAX_LABEL_CHARS:
                out.append(Finding(
                    "label_too_long", WARN, STYLE,
                    f"`{text[:28]}…` is {len(text)} characters — the box will "
                    f"dominate the layout", target=node.id,
                    hint="shorten it and move the detail into a note"))
            if node.id.lower() == text.lower().replace(" ", "_") and "_" in node.id:
                out.append(Finding(
                    "slug_label", INFO, STYLE,
                    f"`{node.id}` is labelled with its own identifier",
                    target=node.id,
                    hint="give it a human label"))

        # Duplicate labels make two different boxes look like the same thing.
        by_label: dict[str, list[str]] = {}
        for node in ir.nodes:
            by_label.setdefault(node.text.strip().lower(), []).append(node.id)
        for label, node_ids in by_label.items():
            if label and len(node_ids) > 1:
                out.append(Finding(
                    "duplicate_label", WARN, STYLE,
                    f"{len(node_ids)} nodes share the label `{label}`",
                    target=node_ids[0],
                    hint="distinguish them, or merge them into one node"))

        # Hubs: a node with a huge fan-out is where layouts fall apart.
        fan_out: dict[str, int] = {}
        for edge in ir.edges:
            fan_out[edge.src] = fan_out.get(edge.src, 0) + 1
        for node_id, count in fan_out.items():
            if count > MAX_FAN_OUT:
                node = ir.node(node_id)
                out.append(Finding(
                    "hub_node", WARN, STYLE,
                    f"`{node.text if node else node_id}` has {count} outgoing "
                    f"edges — it will crowd the layout", target=node_id,
                    hint="wrap its targets in a subgraph, or switch direction to LR"))

        # Deep nesting is hard to follow and mermaid renders it poorly.
        for group in ir.groups:
            depth, cursor, guard = 1, group.parent, 0
            while cursor and guard < 12:
                depth += 1
                parent = ir.group(cursor)
                cursor = parent.parent if parent else ""
                guard += 1
            if depth > MAX_GROUP_DEPTH:
                out.append(Finding(
                    "deep_nesting", WARN, STYLE,
                    f"subgraph `{group.id}` is nested {depth} levels deep",
                    target=group.id,
                    hint=f"keep nesting to {MAX_GROUP_DEPTH} levels"))

        # A tall chain in TD (or a wide fan in LR) wastes the viewport.
        if ir.kind == "flowchart" and ir.nodes:
            chain = _longest_path_length(ir)
            breadth = _max_rank_width(ir)
            if ir.direction in ("TD", "TB") and chain >= 7 and breadth <= 2:
                out.append(Finding(
                    "prefer_lr", INFO, STYLE,
                    f"a {chain}-step chain drawn top-down will be very tall",
                    hint="use `direction: LR` for long linear flows"))
            if ir.direction == "LR" and breadth >= 6 and chain <= 3:
                out.append(Finding(
                    "prefer_td", INFO, STYLE,
                    f"{breadth} parallel branches drawn left-right will be very wide",
                    hint="use `direction: TD` for wide fan-outs"))
        return out
    except Exception:  # noqa: BLE001
        return out


def _ranks(ir: DiagramIR) -> dict[str, int]:
    """Longest-path layering — the same idea a layout engine starts from."""
    incoming: dict[str, int] = {n.id: 0 for n in ir.nodes}
    adjacency: dict[str, list[str]] = {}
    for edge in ir.edges:
        if edge.src == edge.dst or edge.dst not in incoming:
            continue
        adjacency.setdefault(edge.src, []).append(edge.dst)
        incoming[edge.dst] += 1
    rank = {node_id: 0 for node_id in incoming}
    queue = [node_id for node_id, count in incoming.items() if count == 0]
    remaining = dict(incoming)
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


def _longest_path_length(ir: DiagramIR) -> int:
    ranks = _ranks(ir)
    return (max(ranks.values()) + 1) if ranks else 0


def _max_rank_width(ir: DiagramIR) -> int:
    ranks = _ranks(ir)
    if not ranks:
        return 0
    widths: dict[int, int] = {}
    for rank in ranks.values():
        widths[rank] = widths.get(rank, 0) + 1
    return max(widths.values())


# ---- 4. accessibility ----------------------------------------------------
_COLOUR_ONLY = re.compile(r"\b(red|green|blue|amber|yellow|orange|grey|gray)\b"
                          r"\s*(?:=|means|indicates|:)?", re.I)


def validate_accessibility(ir: DiagramIR) -> list[Finding]:
    """Can someone who cannot see the picture still get the information?

    Mermaid supports `accTitle` / `accDescr`, which land in the SVG as
    `<title>`/`<desc>` — a screen reader reads them. A diagram without them is a
    blank to assistive tech no matter how good the layout is.
    """
    out: list[Finding] = []
    try:
        if not ir.acc_title:
            out.append(Finding(
                "missing_acc_title", WARN, ACCESSIBILITY,
                "no `accTitle` — screen readers announce nothing for this diagram",
                hint="set acc_title to a short name, e.g. \"Checkout request flow\""))
        if not ir.acc_descr:
            out.append(Finding(
                "missing_acc_descr", WARN, ACCESSIBILITY,
                "no `accDescr` — there is no text alternative to the picture",
                hint="describe the flow in one or two sentences"))
        elif len(ir.acc_descr) < 25:
            out.append(Finding(
                "thin_acc_descr", INFO, ACCESSIBILITY,
                "the accessible description is very short to stand in for the diagram",
                hint="name the main actors and the direction of flow"))

        for node in ir.nodes:
            text = node.text.strip()
            if len(text) < MIN_LABEL_CHARS:
                out.append(Finding(
                    "unlabelled_node", ERROR, ACCESSIBILITY,
                    f"node `{node.id}` has no meaningful label",
                    target=node.id, hint="every node needs readable text"))
            elif re.fullmatch(r"[A-Za-z]?\d+", text):
                out.append(Finding(
                    "opaque_label", WARN, ACCESSIBILITY,
                    f"`{text}` is not self-describing", target=node.id,
                    hint="use words, not a bare number"))

        # Colour must never be the only carrier of meaning (WCAG 1.4.1).
        for node in ir.nodes:
            if node.css_class and not node.text:
                out.append(Finding(
                    "colour_only", WARN, ACCESSIBILITY,
                    f"node `{node.id}` is styled but unlabelled — colour alone "
                    f"carries the meaning", target=node.id,
                    hint="add a label or a shape difference as well as colour"))
        if _COLOUR_ONLY.search(ir.acc_descr or "") and not any(
                node.role for node in ir.nodes):
            out.append(Finding(
                "colour_legend", INFO, ACCESSIBILITY,
                "the description refers to colours; make sure shape or text "
                "carries the same distinction",
                hint="set a `role`/shape per node type as well"))

        # Edge labels: an unlabelled arrow in a big graph is ambiguous.
        if ir.edges:
            unlabelled = sum(1 for edge in ir.edges if not edge.label)
            if len(ir.edges) >= 6 and unlabelled == len(ir.edges):
                out.append(Finding(
                    "no_edge_labels", INFO, ACCESSIBILITY,
                    "no edge is labelled — the relationships are implied only by "
                    "position", hint="name what flows along the important edges"))
        return out
    except Exception:  # noqa: BLE001
        return out


# ---- the stack ------------------------------------------------------------
def validate(ir: DiagramIR, *, source: str = "") -> ValidationReport:
    """Run every validator in the doc's order and collect the findings.

    `source` is optional raw Mermaid: when given, the text-level syntax checks run
    too (useful for a hand-written or model-written diagram that was lifted into
    the IR — the lift can silently normalise a mistake away).
    """
    report = ValidationReport()
    try:
        if source:
            report.findings.extend(validate_syntax_source(source))
        report.findings.extend(validate_syntax(ir))
        report.findings.extend(validate_semantic(ir))
        report.findings.extend(validate_style(ir))
        report.findings.extend(validate_accessibility(ir))
    except Exception:  # noqa: BLE001
        pass
    return report


def validate_source(source: str) -> ValidationReport:
    """Validate raw Mermaid by lifting it into the IR first."""
    from app.diagrams.parse import from_mermaid
    try:
        return validate(from_mermaid(source), source=source)
    except Exception:  # noqa: BLE001
        return ValidationReport()


__all__ = [
    "ERROR", "WARN", "INFO", "SYNTAX", "SEMANTIC", "STYLE", "ACCESSIBILITY",
    "Finding", "ValidationReport", "validate", "validate_source",
    "validate_syntax", "validate_syntax_source", "validate_semantic",
    "validate_style", "validate_accessibility",
]
