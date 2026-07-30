"""Multi-format export (doc #16), layout planning / ELK (doc #5), stages (#17).

Export is the payoff for the IR: one printer per target instead of a parser per
format pair. These tests check each printer emits the target's real syntax and
that structure (groups, labels, edge styles) survives the crossing.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import app.diagrams.export as X
import app.diagrams.layout as L
import app.diagrams.stages as S
from app.diagrams.ir import from_dict


def _ir(**kwargs):
    base = {
        "kind": "flowchart", "direction": "LR",
        "acc_title": "Request path",
        "acc_descr": "A user request reaches the API and then the store.",
        "groups": [{"id": "tier", "label": "Service tier"}],
        "nodes": [
            {"id": "U", "label": "User", "role": "user"},
            {"id": "API", "label": "API (REST)", "group": "tier"},
            {"id": "DB", "label": "Postgres", "role": "datastore", "group": "tier"},
        ],
        "edges": [
            {"src": "U", "dst": "API", "label": "asks"},
            {"src": "API", "dst": "DB", "label": "reads", "style": "dotted"},
        ],
    }
    base.update(kwargs)
    return from_dict(base)


# ---- PlantUML -----------------------------------------------------------
def test_plantuml_is_wrapped_and_carries_structure():
    out = X.to_plantuml(_ir())
    assert out.startswith("@startuml") and out.rstrip().endswith("@enduml")
    assert 'package "Service tier" {' in out
    assert "U --> API : asks" in out
    assert "API ..> DB : reads" in out          # dotted → `..>`
    assert "left to right direction" in out


def test_plantuml_sequence_uses_participants():
    out = X.to_plantuml(_ir(kind="sequence"))
    assert 'actor "User" as U' in out
    assert "U -> API : asks" in out


def test_plantuml_state_uses_star_for_start_and_end():
    ir = from_dict({"kind": "state",
                    "nodes": [{"id": "s", "label": "s", "role": "start"},
                              {"id": "Work", "label": "Work"},
                              {"id": "e", "label": "e", "role": "end"}],
                    "edges": [{"src": "s", "dst": "Work"},
                              {"src": "Work", "dst": "e"}]})
    out = X.to_plantuml(ir)
    assert "[*] --> Work" in out and "Work --> [*]" in out


def test_plantuml_class_emits_members_and_operators():
    ir = from_dict({"kind": "class",
                    "nodes": [{"id": "Base", "label": "Base",
                               "members": ["+area() float"]},
                              {"id": "Sub", "label": "Sub"}],
                    "edges": [{"src": "Base", "dst": "Sub",
                               "relation": "inheritance"}]})
    out = X.to_plantuml(ir)
    assert "class Base {" in out and "+area() float" in out
    assert "Base <|-- Sub" in out


# ---- Graphviz DOT -------------------------------------------------------
def test_dot_structure_and_clusters():
    out = X.to_dot(_ir())
    assert out.startswith("digraph G {") and out.rstrip().endswith("}")
    assert "rankdir=LR;" in out
    assert "subgraph cluster_tier {" in out
    assert "U -> API [label=\"asks\"];" in out
    assert "style=dotted" in out


def test_dot_maps_roles_to_shapes():
    out = X.to_dot(_ir())
    assert "shape=cylinder" in out            # datastore
    assert "shape=ellipse" in out             # user/actor


def test_dot_escapes_quotes_and_line_breaks():
    ir = _ir(nodes=[{"id": "A", "label": 'He said "hi"'}], edges=[], groups=[])
    out = X.to_dot(ir)
    assert '\\"hi\\"' in out


# ---- Draw.io ------------------------------------------------------------
def test_drawio_is_wellformed_uncompressed_mxgraph():
    out = X.to_drawio(_ir())
    root = ET.fromstring(out)
    assert root.tag == "mxfile"
    cells = root.findall(".//mxCell")
    assert any(cell.get("id") == "n_U" for cell in cells)
    assert any(cell.get("edge") == "1" for cell in cells)
    # Every vertex/edge needs geometry or draw.io renders nothing.
    for cell in cells:
        if cell.get("vertex") == "1" or cell.get("edge") == "1":
            assert cell.find("mxGeometry") is not None


def test_drawio_groups_become_container_parents():
    out = X.to_drawio(_ir())
    root = ET.fromstring(out)
    group = next(c for c in root.findall(".//mxCell")
                 if c.get("value") == "Service tier")
    child = next(c for c in root.findall(".//mxCell") if c.get("id") == "n_API")
    assert child.get("parent") == group.get("id")


# ---- ELK (doc #5) -------------------------------------------------------
def test_elk_json_is_a_layered_graph_with_nested_clusters():
    graph = L.to_elk_json(_ir())
    assert graph["layoutOptions"]["elk.algorithm"] == "layered"
    assert graph["layoutOptions"]["elk.direction"] == "RIGHT"
    top_ids = {child["id"] for child in graph["children"]}
    assert "U" in top_ids and "tier" in top_ids
    cluster = next(c for c in graph["children"] if c["id"] == "tier")
    assert {c["id"] for c in cluster["children"]} == {"API", "DB"}
    assert len(graph["edges"]) == 2


def test_elk_nodes_carry_sizes_so_elk_can_lay_them_out():
    graph = L.to_elk_json(_ir())
    node = next(c for c in graph["children"] if c["id"] == "U")
    assert node["width"] > 0 and node["height"] > 0


def test_elk_is_off_by_default_and_the_plan_says_why():
    # The bundled mermaid 11.15.0 registers only dagre + cose-bilkent, so a
    # request for ELK must degrade honestly rather than silently.
    assert L.elk_available() is False
    plan = L.plan_layout(_ir(layout="elk"))
    assert plan.renderer == "dagre"
    assert any("ELK requested but not enabled" in reason for reason in plan.reasons)


def test_elk_is_used_when_enabled(monkeypatch):
    monkeypatch.setattr(L, "elk_available", lambda: True)
    plan = L.plan_layout(_ir(layout="elk"))
    assert plan.renderer == "elk"
    assert '"layout":"elk"' in plan.init_directive()


# ---- layout planner ----------------------------------------------------
def test_planner_suggests_lr_for_a_long_chain():
    nodes = [{"id": f"N{i}", "label": f"Step {i}"} for i in range(9)]
    ir = from_dict({"direction": "TD", "nodes": nodes,
                    "edges": [{"src": f"N{i}", "dst": f"N{i+1}"}
                              for i in range(8)]})
    plan = L.plan_layout(ir, respect_explicit=False)
    assert plan.direction == "LR"
    assert plan.chain == 9 and plan.breadth == 1


def test_planner_respects_an_explicit_direction_but_records_the_suggestion():
    nodes = [{"id": f"N{i}", "label": f"Step {i}"} for i in range(9)]
    ir = from_dict({"direction": "TD", "nodes": nodes,
                    "edges": [{"src": f"N{i}", "dst": f"N{i+1}"}
                              for i in range(8)]})
    plan = L.plan_layout(ir, respect_explicit=True)
    assert plan.direction == "TD"
    assert any("suggested: LR" in reason for reason in plan.reasons)


def test_planner_prefers_td_for_a_wide_fan():
    targets = [{"id": f"T{i}", "label": f"T{i}"} for i in range(7)]
    ir = from_dict({"direction": "LR",
                    "nodes": [{"id": "R", "label": "Root"}] + targets,
                    "edges": [{"src": "R", "dst": t["id"]} for t in targets]})
    plan = L.plan_layout(ir, respect_explicit=False)
    assert plan.direction == "TD"


def test_dense_graphs_get_more_rank_room():
    nodes = [{"id": f"N{i}", "label": f"N{i}"} for i in range(34)]
    edges = [{"src": f"N{i}", "dst": f"N{(i + 1) % 34}"} for i in range(34)]
    edges += [{"src": f"N{i}", "dst": f"N{(i + 7) % 34}"} for i in range(34)]
    plan = L.plan_layout(from_dict({"nodes": nodes, "edges": edges}))
    assert plan.rank_spacing > plan.node_spacing


def test_init_directive_is_valid_json_and_disables_html_labels():
    plan = L.plan_layout(_ir())
    directive = plan.init_directive()
    assert directive.startswith("%%{init: ") and directive.endswith("}%%")
    config = json.loads(directive[len("%%{init: "):-len("}%%")])
    assert config["flowchart"]["htmlLabels"] is False
    assert config["flowchart"]["useMaxWidth"] is True


def test_render_with_layout_does_not_mutate_the_caller_ir():
    ir = _ir(direction="TD")
    nodes = [{"id": f"N{i}", "label": f"Step {i}"} for i in range(9)]
    ir = from_dict({"direction": "TD", "nodes": nodes,
                    "edges": [{"src": f"N{i}", "dst": f"N{i+1}"}
                              for i in range(8)]})
    source, plan = L.render_with_layout(ir, respect_explicit=False)
    assert plan.direction == "LR"
    assert ir.direction == "TD"                # layout is presentation, not content
    assert "flowchart LR" in source


def test_layout_planner_never_raises():
    for bad in (from_dict({}), from_dict({"nodes": [{"id": "A", "label": "A"}]})):
        assert L.plan_layout(bad) is not None


# ---- dispatch ----------------------------------------------------------
def test_export_every_registered_format_produces_content():
    ir = _ir()
    for fmt in X.EXPORT_FORMATS:
        result = X.export(ir, fmt, stem="my diagram")
        assert result.content, fmt
        assert result.format == fmt
        assert result.filename.startswith("my_diagram.")


def test_export_json_formats_are_parseable():
    ir = _ir()
    assert json.loads(X.export(ir, X.JSON_IR).content)["kind"] == "flowchart"
    assert json.loads(X.export(ir, X.ELK).content)["id"] == "root"


def test_unknown_format_falls_back_to_mermaid():
    result = X.export(_ir(), "visio")
    assert result.format == X.MERMAID
    assert "flowchart" in result.content


def test_export_mermaid_prefers_the_supplied_source():
    result = X.export(_ir(), X.MERMAID, mermaid_src="flowchart TD\n  custom --> x")
    assert result.content == "flowchart TD\n  custom --> x"


def test_export_all_covers_the_registry():
    everything = X.export_all(_ir())
    assert set(everything) == set(X.EXPORT_FORMATS)


def test_filenames_are_sanitised():
    result = X.export(_ir(), X.DOT, stem="../../etc/passwd")
    assert "/" not in result.filename and "\\" not in result.filename


def test_html_export_embeds_the_source_escaped():
    out = X.to_html(_ir(), mermaid_src='flowchart TD\n A["<b>"] --> B')
    assert "&lt;b&gt;" in out
    assert "mermaid.initialize" in out


def test_exporters_never_raise_on_an_empty_ir():
    empty = from_dict({})
    for fmt in X.EXPORT_FORMATS:
        assert X.export(empty, fmt) is not None


# ---- stages (doc #17) --------------------------------------------------
def test_ladder_starts_all_pending():
    ladder = S.stage_ladder()
    assert [stage.id for stage in ladder] == list(S.STAGE_IDS)
    assert all(stage.state == S.PENDING for stage in ladder)


def test_beginning_a_later_stage_completes_the_earlier_ones():
    tracker = S.StageTracker().begin("planning").complete("planning")
    tracker.begin("rendering")
    frame = tracker.frame()
    states = {stage["id"]: stage["state"] for stage in frame["stages"]}
    assert states["generating"] == S.DONE
    assert states["validating"] == S.DONE
    assert states["repairing"] == S.SKIPPED      # conditional, never needed
    assert states["rendering"] == S.ACTIVE


def test_a_failure_is_reported_with_where_and_why():
    frame = S.StageTracker().begin("compiling").fail(
        "compiling", "Unexpected token at line 12").frame()
    assert frame["ok"] is False
    assert frame["failed_at"] == "compiling"
    assert "line 12" in frame["summary"]


def test_finish_closes_the_ladder():
    frame = S.StageTracker().begin("planning").finish().frame()
    assert frame["ok"] is True
    assert all(stage["state"] in (S.DONE, S.SKIPPED) for stage in frame["stages"])
    assert frame["summary"] == "Done"


def test_active_stage_detail_drives_the_summary():
    frame = S.StageTracker().begin("validating").frame()
    assert "Checking syntax" in frame["summary"]


def test_unknown_stage_ids_are_ignored():
    tracker = S.StageTracker().begin("teleporting").complete("teleporting")
    assert tracker.frame()["ok"] is True


# ---- crossing reduction (doc #5, delivered without ELK) -------------------
def _bipartite_worst_case():
    """Two ranks wired so the AUTHORED order crosses as much as possible.

    Sources s0..s3 in order, targets t3..t0 declared in REVERSE, each si → ti.
    Every pair of edges crosses → 6 crossings for 4 edges.
    """
    sources = [{"id": f"s{i}", "label": f"Source {i}"} for i in range(4)]
    targets = [{"id": f"t{i}", "label": f"Target {i}"} for i in reversed(range(4))]
    return from_dict({
        "kind": "flowchart", "direction": "LR",
        "nodes": sources + targets,
        "edges": [{"src": f"s{i}", "dst": f"t{i}"} for i in range(4)],
    })


def test_count_crossings_measures_the_authored_order():
    ir = _bipartite_worst_case()
    authored = [node.id for node in ir.nodes]
    assert L.count_crossings(ir, authored) == 6


def test_ordering_reduces_crossings_on_that_case():
    # The measurable claim: the median-heuristic sweep untangles it completely.
    ir = _bipartite_worst_case()
    before = L.count_crossings(ir, [node.id for node in ir.nodes])
    after = L.count_crossings(ir, L.order_nodes(ir))
    assert after < before
    assert after == 0


def test_ordering_never_makes_crossings_worse():
    cases = [
        _bipartite_worst_case(),
        _ir(),
        from_dict({
            "nodes": [{"id": f"n{i}", "label": f"N{i}"} for i in range(8)],
            "edges": [{"src": "n0", "dst": "n3"}, {"src": "n1", "dst": "n2"},
                      {"src": "n2", "dst": "n5"}, {"src": "n3", "dst": "n4"},
                      {"src": "n4", "dst": "n7"}, {"src": "n5", "dst": "n6"},
                      {"src": "n1", "dst": "n6"}, {"src": "n0", "dst": "n7"}],
        }),
        from_dict({
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"},
                      {"id": "c", "label": "C"}],
            "edges": [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"},
                      {"src": "c", "dst": "a"}],   # a cycle
        }),
    ]
    for index, ir in enumerate(cases):
        before = L.count_crossings(ir, [node.id for node in ir.nodes])
        after = L.count_crossings(ir, L.order_nodes(ir))
        assert after <= before, f"case {index}: {before} → {after}"


def test_ordering_preserves_every_node_exactly_once():
    ir = _bipartite_worst_case()
    order = L.order_nodes(ir)
    assert sorted(order) == sorted(node.id for node in ir.nodes)
    assert len(order) == len(set(order))


def test_reorder_preserves_structure_and_grouping():
    ir = _ir()
    reordered = L.reorder(ir)
    assert reordered.node_ids == ir.node_ids
    assert len(reordered.edges) == len(ir.edges)
    # A node must never leave its subgraph — that would change meaning, not layout.
    for node in reordered.nodes:
        assert node.group == ir.node(node.id).group


def test_reorder_does_not_mutate_the_caller():
    ir = _bipartite_worst_case()
    before = [node.id for node in ir.nodes]
    L.reorder(ir)
    assert [node.id for node in ir.nodes] == before


def test_ordering_is_deterministic():
    ir = _bipartite_worst_case()
    assert L.order_nodes(ir) == L.order_nodes(ir)
    first, _ = L.render_with_layout(ir)
    second, _ = L.render_with_layout(ir)
    assert first == second


def test_tiny_and_edgeless_graphs_are_left_in_authored_order():
    for ir in (from_dict({"nodes": [{"id": "A", "label": "A"}]}),
               from_dict({"nodes": [{"id": "A", "label": "A"},
                                    {"id": "B", "label": "B"}]})):
        assert L.order_nodes(ir) == [node.id for node in ir.nodes]


def test_render_reports_the_resulting_crossings():
    ir = _bipartite_worst_case()
    _source, plan = L.render_with_layout(ir)
    assert plan.crossings == 0
    assert plan.to_dict()["crossings"] == 0


def test_render_can_opt_out_of_reordering():
    ir = _bipartite_worst_case()
    _source, plan = L.render_with_layout(ir, reduce_crossings=False)
    assert plan.crossings == 6


def test_ordering_never_raises():
    for ir in (from_dict({}), from_dict({"nodes": [{"id": "A", "label": "A"}],
                                         "edges": [{"src": "A", "dst": "Ghost"}]})):
        assert L.order_nodes(ir) is not None
        assert L.count_crossings(ir) >= 0


def test_non_flowchart_kinds_keep_their_authored_order():
    # A sequenceDiagram's participant DECLARATION order is the left-to-right lane
    # order a reader follows, so "optimising" it produces a worse diagram. Same
    # for state/ER/class/mindmap, where order carries reading intent, not geometry.
    for kind in ("sequence", "state", "er", "class", "mindmap"):
        ir = from_dict({
            "kind": kind,
            "nodes": [{"id": "U", "label": "User"}, {"id": "API", "label": "API"},
                      {"id": "Auth", "label": "Auth"}],
            "edges": [{"src": "U", "dst": "API"}, {"src": "API", "dst": "Auth"},
                      {"src": "Auth", "dst": "API"}],
        })
        assert L.order_nodes(ir) == ["U", "API", "Auth"], kind
        source, _plan = L.render_with_layout(ir)
        if kind == "sequence":
            # Participants must appear in the authored order in the output too.
            assert source.index("participant U") < source.index("participant API")
            assert source.index("participant API") < source.index("participant Auth")
