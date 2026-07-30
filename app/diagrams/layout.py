"""Auto-layout planning (MermaidDiagramVisualizations.md #5).

    LLM → Logical Graph → Graph Layout Engine → SVG

rather than asking a model for coordinates it cannot reason about. Two halves:

**1. The planner (always on, deterministic).** From the IR's graph shape alone we
can decide the things that actually make a diagram readable, and they are the
things a model gets wrong:
  * **direction** — a 9-step chain drawn `TD` is a tall ribbon nobody scrolls; a
    6-way fan drawn `LR` is a wide one. Chain depth vs. rank breadth decides it.
  * **spacing** — node/rank separation scaled to the graph's density, so a sparse
    diagram breathes and a dense one doesn't overflow.
  * **curve + wrap** — edge curve style and label wrap width from the same shape.
The planner emits a `%%{init: …}%%` directive carrying those, which mermaid honours
whichever renderer is active. This is real layout improvement with no new
dependency.

**2. ELK (opt-in).** Mermaid 11 can delegate layout to the Eclipse Layout Kernel,
but only when the host registers the `@mermaid-js/layout-elk` loader. Our bundled
`mermaid.min.js` (11.15.0) registers **only `dagre` and `cose-bilkent`** — I
checked the bundle — so asking for `layout: elk` today would fall back to dagre
silently. Rather than pretend, ELK is gated behind
`cfg.response_arch.mermaid_elk` (default OFF) and :func:`elk_available` documents
the requirement. What *is* available now is :func:`to_elk_json`, which emits a
standard ELK graph so an external ELK renderer (or a future bundled loader) can
lay the same IR out — the IR makes that a 30-line function instead of a rewrite.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.diagrams.ir import DiagramIR

# Node box estimates (CSS px) for ELK — mermaid measures text itself, but ELK
# needs sizes up front. Derived from the label so wide labels get wide boxes.
_CHAR_WIDTH = 8.0
_NODE_PADDING = 32.0
_MIN_NODE_WIDTH = 80.0
_NODE_HEIGHT = 44.0
_LINE_HEIGHT = 20.0


def elk_available() -> bool:
    """Whether the ELK renderer may be requested.

    Config-gated AND honest about the client requirement: the FE must ship
    `@mermaid-js/layout-elk` and call `mermaid.registerLayoutLoaders(elkLayouts)`
    before `layout: elk` does anything. Default OFF.
    """
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.response_arch, "mermaid_elk", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass
class LayoutPlan:
    direction: str = "TD"
    renderer: str = "dagre"          # dagre | elk
    node_spacing: int = 50
    rank_spacing: int = 60
    curve: str = "basis"             # basis | linear | cardinal
    label_wrap: int = 22
    chain: int = 0                   # longest path, in nodes
    breadth: int = 0                 # widest rank
    density: float = 0.0             # edges per node
    crossings: int = 0               # edge crossings after the ordering pass
    reasons: list[str] = field(default_factory=list)

    def init_directive(self) -> str:
        """The `%%{init}%%` line to prepend to the emitted Mermaid."""
        config: dict = {
            "flowchart": {
                "curve": self.curve,
                "nodeSpacing": self.node_spacing,
                "rankSpacing": self.rank_spacing,
                "useMaxWidth": True,
                "htmlLabels": False,
            },
        }
        if self.renderer == "elk":
            # Mermaid 11 reads a top-level `layout`; harmless when the loader
            # isn't registered (it falls back to dagre).
            config["layout"] = "elk"
            config["elk"] = {"mergeEdges": False,
                             "nodePlacementStrategy": "BRANDES_KOEPF"}
        return "%%{init: " + json.dumps(config, separators=(",", ":")) + "}%%"

    def to_dict(self) -> dict:
        return {"direction": self.direction, "renderer": self.renderer,
                "node_spacing": self.node_spacing,
                "rank_spacing": self.rank_spacing, "curve": self.curve,
                "label_wrap": self.label_wrap, "chain": self.chain,
                "breadth": self.breadth, "density": round(self.density, 2),
                "crossings": self.crossings, "reasons": list(self.reasons),
                "init_directive": self.init_directive()}


def ranks(ir: DiagramIR) -> dict[str, int]:
    """Longest-path layering — the first phase of every layered graph drawing
    algorithm (Sugiyama), and the basis of both the shape heuristics and the
    crossing-reduction pass below."""
    incoming: dict[str, int] = {n.id: 0 for n in ir.nodes}
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


def _shape(ir: DiagramIR) -> tuple[int, int]:
    """(longest chain in nodes, widest rank)."""
    rank = ranks(ir)
    if not rank:
        return 0, 0
    widths: dict[int, int] = {}
    for value in rank.values():
        widths[value] = widths.get(value, 0) + 1
    return max(rank.values()) + 1, max(widths.values())


# ---- crossing reduction (the layout win ELK would have provided) ----------
def count_crossings(ir: DiagramIR, order: list[str] | None = None) -> int:
    """Edge crossings between adjacent ranks, for the given node `order`.

    Two edges (u₁→v₁) and (u₂→v₂) spanning the same rank pair cross when their
    endpoints are in opposite relative order. This is the standard bilayer
    crossing count, and it is what the ordering pass below minimises — and what
    makes the improvement MEASURABLE rather than asserted.
    """
    try:
        rank = ranks(ir)
        position = {node_id: index for index, node_id
                    in enumerate(order or [n.id for n in ir.nodes])}
        # Group edges by the rank pair they span.
        by_span: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for edge in ir.edges:
            if edge.src == edge.dst:
                continue
            if edge.src not in rank or edge.dst not in rank:
                continue
            span = (rank[edge.src], rank[edge.dst])
            by_span.setdefault(span, []).append(
                (position.get(edge.src, 0), position.get(edge.dst, 0)))
        total = 0
        for pairs in by_span.values():
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    (a1, b1), (a2, b2) = pairs[i], pairs[j]
                    if (a1 - a2) * (b1 - b2) < 0:
                        total += 1
        return total
    except Exception:  # noqa: BLE001
        return 0


def order_nodes(ir: DiagramIR, *, sweeps: int = 4) -> list[str]:
    """A node order that reduces edge crossings, as a list of node ids.

    Why this exists: MermaidDiagramVisualizations.md #5 wants a layout ENGINE
    rather than the model choosing positions. Mermaid already runs dagre, so the
    remaining lever we actually control is the ORDER we declare nodes in — dagre's
    ordering phase seeds from input order and uses it to break ties, so a
    pre-ordered graph comes out cleaner.

    The method is the median heuristic from layered graph drawing (the same phase
    ELK's `layered` algorithm runs): sweep down and up, placing each node at the
    median position of its neighbours in the adjacent rank, and keep the best
    order seen. Deterministic — ties fall back to the original order, so the same
    IR always yields the same bytes.

    Nodes inside a subgraph are ordered WITHIN their group only, because the
    emitter has to declare them inside their `subgraph` block regardless; moving
    one across a group boundary would change the diagram's meaning, not its
    layout.

    **Flowcharts only.** Crossing reduction assumes a layered graph, and the other
    kinds do not have one: in a `sequenceDiagram` the participant DECLARATION order
    IS the left-to-right lane order a reader follows, so "optimising" it turns
    `User → API → Auth` into `User → Auth → API` — a worse diagram with fewer
    notional crossings. State/ER/class/mindmap declaration order likewise carries
    reading intent rather than geometry. Those are returned in authored order.
    """
    try:
        if ir.kind != "flowchart":
            return [node.id for node in ir.nodes]
        rank = ranks(ir)
        if len(ir.nodes) < 3 or not ir.edges:
            return [node.id for node in ir.nodes]

        original_index = {node.id: index
                          for index, node in enumerate(ir.nodes)}
        # Layers, seeded with the authored order.
        layers: dict[int, list[str]] = {}
        for node in ir.nodes:
            layers.setdefault(rank.get(node.id, 0), []).append(node.id)

        predecessors: dict[str, list[str]] = {}
        successors: dict[str, list[str]] = {}
        for edge in ir.edges:
            if edge.src == edge.dst:
                continue
            successors.setdefault(edge.src, []).append(edge.dst)
            predecessors.setdefault(edge.dst, []).append(edge.src)

        def flatten(current: dict[int, list[str]]) -> list[str]:
            out: list[str] = []
            for layer_rank in sorted(current):
                out.extend(current[layer_rank])
            return out

        def positions(current: dict[int, list[str]]) -> dict[str, int]:
            return {node_id: index for index, node_id
                    in enumerate(flatten(current))}

        best = {r: list(nodes) for r, nodes in layers.items()}
        best_crossings = count_crossings(ir, flatten(best))

        working = {r: list(nodes) for r, nodes in layers.items()}
        for sweep in range(max(1, sweeps)):
            downward = sweep % 2 == 0
            order_of_ranks = sorted(working) if downward else sorted(
                working, reverse=True)
            pos = positions(working)
            for layer_rank in order_of_ranks:
                neighbours = predecessors if downward else successors

                def median(node_id: str) -> float:
                    adjacent = [pos[n] for n in neighbours.get(node_id, [])
                                if n in pos]
                    if not adjacent:
                        # No anchor → keep the authored position, so an isolated
                        # node never drifts.
                        return float(original_index.get(node_id, 0))
                    adjacent.sort()
                    middle = len(adjacent) // 2
                    if len(adjacent) % 2:
                        return float(adjacent[middle])
                    return (adjacent[middle - 1] + adjacent[middle]) / 2.0

                working[layer_rank] = sorted(
                    working[layer_rank],
                    # Stable, deterministic: median, then the authored order.
                    key=lambda node_id: (median(node_id),
                                         original_index.get(node_id, 0)))
                pos = positions(working)
            crossings = count_crossings(ir, flatten(working))
            if crossings < best_crossings:
                best_crossings = crossings
                best = {r: list(nodes) for r, nodes in working.items()}

        # Re-group: the emitter declares grouped nodes inside their subgraph, so
        # the returned order only has to be right WITHIN each group.
        return flatten(best)
    except Exception:  # noqa: BLE001 — ordering is an optimisation, never a risk
        return [node.id for node in ir.nodes]


def reorder(ir: DiagramIR) -> DiagramIR:
    """A COPY of `ir` with its nodes in crossing-reduced order."""
    try:
        order = order_nodes(ir)
        rank_of = {node_id: index for index, node_id in enumerate(order)}
        out = ir.copy()
        out.nodes.sort(key=lambda node: rank_of.get(node.id, 0))
        return out
    except Exception:  # noqa: BLE001
        return ir


def plan_layout(ir: DiagramIR, *, respect_explicit: bool = True) -> LayoutPlan:
    """Choose direction, spacing and renderer from the graph's shape.

    `respect_explicit=True` keeps a direction the planner/user set deliberately
    (it only *suggests* an alternative in `reasons`); pass False to let the
    geometry win. Never raises.
    """
    plan = LayoutPlan(direction=ir.direction, label_wrap=ir.label_wrap)
    try:
        node_count = len(ir.nodes)
        edge_count = len(ir.edges)
        chain, breadth = _shape(ir)
        plan.chain, plan.breadth = chain, breadth
        plan.density = (edge_count / node_count) if node_count else 0.0

        # --- direction -------------------------------------------------
        # Long-and-thin reads left-to-right; short-and-wide reads top-down.
        wants = ir.direction
        if chain >= 6 and breadth <= 2:
            wants = "LR"
            plan.reasons.append(
                f"{chain}-step chain, {breadth} wide → LR keeps it on screen")
        elif breadth >= 5 and chain <= 3:
            wants = "TD"
            plan.reasons.append(
                f"{breadth} parallel branches, {chain} deep → TD avoids a very "
                f"wide picture")
        elif node_count > 24:
            wants = "LR"
            plan.reasons.append(
                f"{node_count} nodes → LR uses the horizontal space a chat "
                f"column has")
        if ir.kind in ("sequence", "er"):
            plan.reasons.append(f"{ir.kind} diagrams lay themselves out; "
                                f"direction is advisory only")
        if wants != ir.direction:
            if respect_explicit:
                plan.reasons.append(
                    f"keeping the explicit direction {ir.direction} "
                    f"(suggested: {wants})")
            else:
                plan.direction = wants

        # --- spacing ---------------------------------------------------
        # Dense graphs need MORE room between ranks or edges pile up; sparse ones
        # look sterile spread out.
        if plan.density >= 1.6 or node_count > 30:
            plan.node_spacing, plan.rank_spacing = 40, 80
            plan.reasons.append("dense graph → tighter columns, taller ranks")
        elif node_count <= 6:
            plan.node_spacing, plan.rank_spacing = 60, 55
            plan.reasons.append("small graph → generous spacing")
        else:
            plan.node_spacing, plan.rank_spacing = 50, 65

        # --- curve + wrap ----------------------------------------------
        plan.curve = "linear" if plan.density < 0.9 else "basis"
        # Long labels in a wide graph must wrap harder or the graph overflows.
        longest = max((len(n.text) for n in ir.nodes), default=0)
        if longest > 40 and node_count > 8:
            plan.label_wrap = 18
            plan.reasons.append("long labels in a busy graph → wrap at 18")

        # --- renderer --------------------------------------------------
        requested = (ir.layout or "").lower()
        if requested == "elk" and elk_available():
            plan.renderer = "elk"
            plan.reasons.append("ELK requested and enabled")
        elif requested == "elk":
            plan.reasons.append(
                "ELK requested but not enabled (needs mermaid_elk=true and the "
                "@mermaid-js/layout-elk loader registered client-side) → dagre")
        elif node_count >= 20 and edge_count >= 28 and elk_available():
            plan.renderer = "elk"
            plan.reasons.append(
                f"{node_count} nodes / {edge_count} edges → ELK handles the "
                f"crossings better than dagre")
    except Exception:  # noqa: BLE001
        pass
    return plan


def _node_size(label: str) -> tuple[float, float]:
    lines = (label or "").split("<br/>")
    widest = max((len(line) for line in lines), default=0)
    width = max(_MIN_NODE_WIDTH, widest * _CHAR_WIDTH + _NODE_PADDING)
    height = max(_NODE_HEIGHT, len(lines) * _LINE_HEIGHT + 20.0)
    return width, height


def to_elk_json(ir: DiagramIR, *, plan: LayoutPlan | None = None) -> dict:
    """Emit a standard ELK graph for `ir`.

    This is the doc's "logical graph → layout engine" handoff made concrete: any
    ELK implementation (elkjs in a webview, elk-cli, a future bundled mermaid
    loader) can lay this out, and it doubles as an export format. Subgraphs become
    nested ELK children, which is exactly how ELK models clusters.
    """
    plan = plan or plan_layout(ir)
    direction = {"TD": "DOWN", "TB": "DOWN", "BT": "UP",
                 "LR": "RIGHT", "RL": "LEFT"}.get(plan.direction, "DOWN")

    def make_node(node) -> dict:
        width, height = _node_size(node.text)
        return {"id": node.id, "width": width, "height": height,
                "labels": [{"text": node.text}]}

    def children_of(group_id: str) -> list[dict]:
        out: list[dict] = [make_node(n) for n in ir.nodes if n.group == group_id]
        for group in ir.groups:
            if group.parent == group_id and group.id != group_id:
                out.append({
                    "id": group.id,
                    "labels": [{"text": group.label or group.id}],
                    "children": children_of(group.id),
                    "layoutOptions": {"elk.padding":
                                      "[top=32,left=16,bottom=16,right=16]"},
                })
        return out

    return {
        "id": "root",
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": direction,
            "elk.spacing.nodeNode": str(plan.node_spacing),
            "elk.layered.spacing.nodeNodeBetweenLayers": str(plan.rank_spacing),
            "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
            "elk.edgeRouting": "ORTHOGONAL" if plan.curve == "linear" else "SPLINES",
            "elk.hierarchyHandling": "INCLUDE_CHILDREN",
        },
        "children": children_of(""),
        "edges": [
            {"id": f"e{index}", "sources": [edge.src], "targets": [edge.dst],
             "labels": ([{"text": edge.label}] if edge.label else [])}
            for index, edge in enumerate(ir.edges)
        ],
    }


def render_with_layout(ir: DiagramIR, *, respect_explicit: bool = True,
                       reduce_crossings: bool = True) -> tuple[str, LayoutPlan]:
    """Plan the layout, apply it to a COPY of the IR, and emit Mermaid.

    The IR is copied so a caller's structure is never mutated by a layout
    decision — layout is presentation, structure is content. `reduce_crossings`
    reorders node DECLARATIONS (never their grouping or their edges), which is the
    one layout lever we hold over mermaid's own dagre pass.
    """
    from app.diagrams.ir import to_mermaid
    plan = plan_layout(ir, respect_explicit=respect_explicit)
    laid_out = reorder(ir) if reduce_crossings else ir.copy()
    laid_out.direction = plan.direction
    laid_out.label_wrap = plan.label_wrap
    plan.crossings = count_crossings(laid_out,
                                     [node.id for node in laid_out.nodes])
    return to_mermaid(laid_out, init_directive=plan.init_directive()), plan


__all__ = ["LayoutPlan", "plan_layout", "to_elk_json", "render_with_layout",
           "elk_available", "ranks", "order_nodes", "reorder",
           "count_crossings"]
