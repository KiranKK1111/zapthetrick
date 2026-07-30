"""The four-validator stack + the quality score (doc #4, #6, #7).

The syntax rules here are the ones I MEASURED against mermaid 11.15.0, not the
ones folklore says. Three tests below exist specifically to pin down false
positives an earlier version shipped: `--->` is valid, and `:`/`#`/`<`/`>` inside
an unquoted label are valid.
"""
from __future__ import annotations

import app.diagrams.quality as Q
import app.diagrams.validators as V
from app.diagrams.ir import from_dict


def _codes(findings) -> set[str]:
    return {f.code for f in findings}


def _flow(**kwargs):
    base = {"kind": "flowchart",
            "acc_title": "T", "acc_descr": "A description long enough to pass.",
            "nodes": [{"id": "A", "label": "Alpha"}, {"id": "B", "label": "Beta"}],
            "edges": [{"src": "A", "dst": "B"}]}
    base.update(kwargs)
    return from_dict(base)


# ---- syntax (source level) -----------------------------------------------
def test_unclosed_subgraph_is_an_error():
    findings = V.validate_syntax_source("flowchart TD\nsubgraph x\nA --> B")
    assert "unclosed_subgraph" in _codes(findings)


def test_stray_end_is_an_error():
    findings = V.validate_syntax_source("flowchart TD\nA --> B\nend")
    assert "extra_end" in _codes(findings)


def test_balanced_subgraph_is_clean():
    findings = V.validate_syntax_source(
        "flowchart TD\nsubgraph x\nA --> B\nend")
    assert not _codes(findings)


def test_long_arrows_are_NOT_flagged():
    # Measured against mermaid 11.15.0: extra dashes set the rank distance and
    # are perfectly valid. Flagging them churned correct diagrams.
    for source in ("flowchart LR\nA ---> B", "flowchart LR\nA ----> B",
                   "flowchart LR\nA ===> B"):
        assert not _codes(V.validate_syntax_source(source)), source


def test_single_dash_arrow_is_an_error():
    findings = V.validate_syntax_source("flowchart LR\nA -> B")
    assert "single_dash_arrow" in _codes(findings)


def test_headless_link_is_an_error():
    findings = V.validate_syntax_source("flowchart LR\nA -- B")
    assert "headless_link" in _codes(findings)


def test_labelled_link_is_not_a_headless_link():
    assert not _codes(V.validate_syntax_source("flowchart LR\nA -- text --> B"))


def test_label_specials_that_really_break_are_flagged():
    for bad in ("A[Fetch (REST)]", 'A[He said "hi"]', "A[a|b]", "A[{x}]"):
        findings = V.validate_syntax_source(f"flowchart LR\n{bad} --> B")
        assert "unquoted_label" in _codes(findings), bad


def test_label_specials_that_are_actually_fine_are_not_flagged():
    # Measured: these all parse unquoted.
    for ok in ("A[Kafka: events]", "A[Item #1]", "A[<billing>]", "A[a;b]",
               "A[a&b]", "A[100%]", 'A["Fetch (REST)"]'):
        findings = V.validate_syntax_source(f"flowchart LR\n{ok} --> B")
        assert "unquoted_label" not in _codes(findings), ok


def test_empty_source_is_an_error():
    assert "empty" in _codes(V.validate_syntax_source("   \n%% only a comment"))


# ---- syntax (IR level) ---------------------------------------------------
def test_ir_with_no_nodes_is_an_error():
    assert "no_nodes" in _codes(V.validate_syntax(from_dict({})))


def test_group_cannot_be_its_own_parent():
    from app.diagrams.ir import Group
    ir = _flow()
    ir.groups.append(Group(id="g", parent="g"))
    assert "group_self_parent" in _codes(V.validate_syntax(ir))


def test_duplicate_node_ids_are_an_error():
    from app.diagrams.ir import Node
    ir = _flow()
    ir.nodes.append(Node(id="A", label="again"))
    assert "duplicate_node" in _codes(V.validate_syntax(ir))


# ---- semantic (doc #7) --------------------------------------------------
def test_dangling_edge_endpoint_is_an_error():
    ir = _flow(edges=[{"src": "A", "dst": "Ghost"}])
    assert "dangling_edge" in _codes(V.validate_semantic(ir))


def test_orphan_node_is_a_warning():
    ir = _flow(nodes=[{"id": "A", "label": "A"}, {"id": "B", "label": "B"},
                      {"id": "C", "label": "Island"}])
    assert "orphan_node" in _codes(V.validate_semantic(ir))


def test_single_node_diagram_has_no_orphan_warning():
    ir = from_dict({"nodes": [{"id": "A", "label": "Only"}]})
    assert "orphan_node" not in _codes(V.validate_semantic(ir))


def test_duplicate_edge_is_a_warning():
    ir = _flow(edges=[{"src": "A", "dst": "B"}, {"src": "A", "dst": "B"}])
    assert "duplicate_edge" in _codes(V.validate_semantic(ir))


def test_self_loop_warns_in_a_flow_but_not_in_a_state_machine():
    flow = _flow(edges=[{"src": "A", "dst": "A"}])
    assert "self_loop" in _codes(V.validate_semantic(flow))
    state = from_dict({"kind": "state",
                      "nodes": [{"id": "A", "label": "A"}],
                      "edges": [{"src": "A", "dst": "A", "label": "retry"}]})
    assert "self_loop" not in _codes(V.validate_semantic(state))


def test_database_to_user_is_flagged_as_backwards():
    # The doc's own example (#7): syntactically fine, logically inverted.
    ir = from_dict({
        "nodes": [{"id": "DB", "label": "Database", "role": "datastore"},
                  {"id": "U", "label": "User", "role": "user"}],
        "edges": [{"src": "DB", "dst": "U"}]})
    assert "reversed_flow" in _codes(V.validate_semantic(ir))


def test_user_to_database_is_not_flagged():
    ir = from_dict({
        "nodes": [{"id": "U", "label": "User", "role": "user"},
                  {"id": "DB", "label": "Database", "role": "datastore"}],
        "edges": [{"src": "U", "dst": "DB"}]})
    assert "reversed_flow" not in _codes(V.validate_semantic(ir))


def test_unlabelled_decision_branches_warn():
    ir = from_dict({
        "nodes": [{"id": "D", "label": "Valid?", "role": "decision"},
                  {"id": "Y", "label": "Yes"}, {"id": "N", "label": "No"}],
        "edges": [{"src": "D", "dst": "Y", "label": "yes"},
                  {"src": "D", "dst": "N"}]})
    assert "unlabelled_branch" in _codes(V.validate_semantic(ir))


def test_cycle_is_reported_as_info():
    ir = from_dict({
        "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"},
                  {"id": "C", "label": "C"}],
        "edges": [{"src": "A", "dst": "B"}, {"src": "B", "dst": "C"},
                  {"src": "C", "dst": "A"}]})
    findings = V.validate_semantic(ir)
    assert "cycle" in _codes(findings)
    assert all(f.severity == V.INFO for f in findings if f.code == "cycle")


def test_acyclic_graph_reports_no_cycle():
    ir = from_dict({
        "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"},
                  {"id": "C", "label": "C"}],
        "edges": [{"src": "A", "dst": "B"}, {"src": "B", "dst": "C"},
                  {"src": "A", "dst": "C"}]})
    assert "cycle" not in _codes(V.validate_semantic(ir))


def test_nodes_with_no_edges_is_an_error():
    ir = _flow(nodes=[{"id": f"N{i}", "label": f"N{i}"} for i in range(4)],
               edges=[])
    assert "no_edges" in _codes(V.validate_semantic(ir))


# ---- style -------------------------------------------------------------
def test_too_many_nodes_warns():
    ir = _flow(nodes=[{"id": f"N{i}", "label": f"Node {i}"}
                      for i in range(V.MAX_NODES + 5)],
               edges=[{"src": f"N{i}", "dst": f"N{i+1}"}
                      for i in range(V.MAX_NODES + 4)])
    assert "too_many_nodes" in _codes(V.validate_style(ir))


def test_overlong_label_warns():
    ir = _flow(nodes=[{"id": "A", "label": "x" * (V.MAX_LABEL_CHARS + 10)},
                      {"id": "B", "label": "B"}])
    assert "label_too_long" in _codes(V.validate_style(ir))


def test_duplicate_labels_warn():
    ir = _flow(nodes=[{"id": "A", "label": "Same"}, {"id": "B", "label": "same"}])
    assert "duplicate_label" in _codes(V.validate_style(ir))


def test_hub_node_warns():
    targets = [{"id": f"T{i}", "label": f"T{i}"} for i in range(V.MAX_FAN_OUT + 2)]
    ir = _flow(nodes=[{"id": "H", "label": "Hub"}] + targets,
               edges=[{"src": "H", "dst": t["id"]} for t in targets])
    assert "hub_node" in _codes(V.validate_style(ir))


def test_long_chain_in_td_suggests_lr():
    nodes = [{"id": f"N{i}", "label": f"Step {i}"} for i in range(9)]
    ir = _flow(direction="TD", nodes=nodes,
               edges=[{"src": f"N{i}", "dst": f"N{i+1}"} for i in range(8)])
    assert "prefer_lr" in _codes(V.validate_style(ir))


def test_wide_fan_in_lr_suggests_td():
    targets = [{"id": f"T{i}", "label": f"T{i}"} for i in range(7)]
    ir = _flow(direction="LR", nodes=[{"id": "R", "label": "Root"}] + targets,
               edges=[{"src": "R", "dst": t["id"]} for t in targets])
    assert "prefer_td" in _codes(V.validate_style(ir))


def test_deep_nesting_warns():
    groups = [{"id": "g0", "label": "g0"}]
    for i in range(1, 5):
        groups.append({"id": f"g{i}", "label": f"g{i}", "parent": f"g{i-1}"})
    ir = _flow(groups=groups,
               nodes=[{"id": "A", "label": "A", "group": "g4"},
                      {"id": "B", "label": "B"}])
    assert "deep_nesting" in _codes(V.validate_style(ir))


# ---- accessibility -----------------------------------------------------
def test_missing_acc_metadata_warns():
    ir = from_dict({"nodes": [{"id": "A", "label": "Alpha"}]})
    codes = _codes(V.validate_accessibility(ir))
    assert {"missing_acc_title", "missing_acc_descr"} <= codes


def test_present_acc_metadata_is_clean():
    ir = _flow()
    codes = _codes(V.validate_accessibility(ir))
    assert "missing_acc_title" not in codes
    assert "missing_acc_descr" not in codes


def test_thin_acc_descr_is_info():
    ir = _flow(acc_descr="Short.")
    assert "thin_acc_descr" in _codes(V.validate_accessibility(ir))


def test_opaque_numeric_label_warns():
    ir = _flow(nodes=[{"id": "A", "label": "42"}, {"id": "B", "label": "Beta"}])
    assert "opaque_label" in _codes(V.validate_accessibility(ir))


def test_all_unlabelled_edges_is_info():
    nodes = [{"id": f"N{i}", "label": f"Node {i}"} for i in range(8)]
    ir = _flow(nodes=nodes,
               edges=[{"src": f"N{i}", "dst": f"N{i+1}"} for i in range(7)])
    assert "no_edge_labels" in _codes(V.validate_accessibility(ir))


# ---- the stack + report -------------------------------------------------
def test_validate_runs_every_category():
    ir = from_dict({"nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}],
                    "edges": [{"src": "A", "dst": "Ghost"}]})
    report = V.validate(ir, source="flowchart TD\nA -> B")
    categories = {f.category for f in report.findings}
    assert {V.SYNTAX, V.SEMANTIC, V.ACCESSIBILITY} <= categories
    assert not report.ok


def test_validate_source_lifts_then_validates():
    report = V.validate_source("flowchart LR\nA[Alpha] --> B[Beta]")
    assert isinstance(report, V.ValidationReport)
    assert "missing_acc_title" in _codes(report.findings)


def test_report_to_dict_shape():
    data = V.validate_source("flowchart LR\nA --> B").to_dict()
    assert set(data) == {"ok", "findings", "counts"}
    assert set(data["counts"]) == {"error", "warn", "info"}


def test_validators_never_raise_on_garbage():
    for bad in (None, "", "\x00", "flowchart"):
        assert V.validate_source(bad) is not None  # type: ignore[arg-type]


# ---- quality score (doc #4) --------------------------------------------
def test_clean_diagram_scores_high_and_passes():
    quality, _report = Q.score(_flow(edges=[{"src": "A", "dst": "B",
                                             "label": "sends"}]))
    assert quality.passed
    assert quality.overall >= 90
    assert quality.grade == "excellent"


def test_any_error_fails_regardless_of_score():
    ir = _flow(edges=[{"src": "A", "dst": "Ghost"}])
    quality, _report = Q.score(ir)
    assert not quality.passed
    assert quality.counts[V.ERROR] >= 1


def test_subscores_cover_every_category():
    quality, _report = Q.score(_flow())
    assert set(quality.subscores) == {V.SYNTAX, V.SEMANTIC, V.STYLE,
                                      V.ACCESSIBILITY}


def test_errors_are_listed_before_warnings_in_top_issues():
    ir = from_dict({"nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "A"},
                              {"id": "C", "label": "Island"}],
                    "edges": [{"src": "A", "dst": "Nope"}]})
    quality, _report = Q.score(ir)
    severities = [issue["severity"] for issue in quality.top_issues]
    assert severities == sorted(
        severities, key=lambda s: {"error": 0, "warn": 1, "info": 2}[s])


def test_score_is_deterministic():
    first, _ = Q.score(_flow())
    second, _ = Q.score(_flow())
    assert first.to_dict() == second.to_dict()


def test_missing_accessibility_only_dents_the_score():
    # Accessibility is weighted lowest: two warnings there must not fail a
    # diagram that is otherwise correct.
    ir = from_dict({"nodes": [{"id": "A", "label": "Alpha"},
                              {"id": "B", "label": "Beta"}],
                    "edges": [{"src": "A", "dst": "B", "label": "sends"}]})
    quality, _report = Q.score(ir)
    assert quality.passed
    assert quality.subscores[V.ACCESSIBILITY] < 100


def test_score_source_entry_point():
    quality, report = Q.score_source("flowchart LR\nA[Alpha] --> B[Beta]")
    assert 0 <= quality.overall <= 100
    assert isinstance(report, V.ValidationReport)


def test_summary_mentions_blocking_issues():
    quality, _report = Q.score(_flow(edges=[{"src": "A", "dst": "Ghost"}]))
    assert "blocking" in quality.summary
