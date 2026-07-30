"""Targeted AST edits (doc #10) and version history (doc #9).

The property that matters: an edit either lands cleanly or is REJECTED with a
reason. There is no third outcome where the diagram ends up half-changed or
syntactically broken, because the model only supplies ops and the applier is
deterministic.
"""
from __future__ import annotations

import app.diagrams.edits as E
from app.diagrams.ir import from_dict, to_mermaid
from app.diagrams.versions import DiagramVersionStore, diagram_id


def _ir():
    return from_dict({
        "kind": "flowchart", "direction": "TD",
        "groups": [{"id": "kafka", "label": "Kafka"}],
        "nodes": [{"id": "B1", "label": "Broker 1", "group": "kafka"},
                  {"id": "B2", "label": "Broker 2", "group": "kafka"},
                  {"id": "App", "label": "App"}],
        "edges": [{"src": "App", "dst": "B1", "label": "produces"}],
    })


# ---- the applier ---------------------------------------------------------
def test_the_source_ir_is_never_mutated():
    original = _ir()
    before = original.to_dict()
    E.apply_edits(original, [{"op": "remove_node", "id": "B2"}])
    assert original.to_dict() == before


def test_add_node_and_edge():
    result = E.apply_edits(_ir(), [
        {"op": "add_node", "id": "B3", "label": "Broker 3", "group": "kafka"},
        {"op": "add_edge", "src": "B1", "dst": "B3", "label": "replicates"},
    ])
    assert result.changed and not result.rejected
    assert result.ir.node("B3").group == "kafka"
    assert any(e.dst == "B3" for e in result.ir.edges)


def test_the_docs_own_example_move_broker_2_under_broker_1():
    result = E.apply_edits(_ir(), [
        {"op": "add_edge", "src": "B1", "dst": "B2", "label": "replicates"}])
    assert result.changed
    source = to_mermaid(result.ir)
    assert 'B1 -- "replicates" --> B2' in source
    # Everything the user did not ask about survived.
    assert result.ir.node("B2").label == "Broker 2"
    assert any(e.src == "App" and e.dst == "B1" for e in result.ir.edges)


def test_remove_node_takes_its_edges_with_it():
    result = E.apply_edits(_ir(), [{"op": "remove_node", "id": "B1"}])
    assert result.ir.node("B1") is None
    assert not [e for e in result.ir.edges if "B1" in (e.src, e.dst)]


def test_rename_and_reshape():
    result = E.apply_edits(_ir(), [
        {"op": "rename_node", "id": "B1", "label": "Leader"},
        {"op": "set_shape", "id": "App", "role": "datastore"},
    ])
    assert result.ir.node("B1").label == "Leader"
    assert result.ir.node("App").resolved_shape() == "cylinder"


def test_move_node_to_top_level_and_back():
    result = E.apply_edits(_ir(), [{"op": "move_node", "id": "B1", "group": ""}])
    assert result.ir.node("B1").group == ""
    again = E.apply_edits(result.ir, [{"op": "move_node", "id": "B1",
                                      "group": "kafka"}])
    assert again.ir.node("B1").group == "kafka"


def test_reverse_and_relabel_edge():
    result = E.apply_edits(_ir(), [
        {"op": "relabel_edge", "src": "App", "dst": "B1", "label": "writes"},
        {"op": "reverse_edge", "src": "App", "dst": "B1"},
    ])
    edge = result.ir.edges[0]
    assert (edge.src, edge.dst, edge.label) == ("B1", "App", "writes")


def test_remove_group_keeps_its_members():
    result = E.apply_edits(_ir(), [{"op": "remove_group", "id": "kafka"}])
    assert result.ir.group("kafka") is None
    assert result.ir.node("B1") is not None
    assert result.ir.node("B1").group == ""


def test_nested_group_reparents_when_its_parent_is_removed():
    ir = from_dict({
        "groups": [{"id": "outer", "label": "Outer"},
                   {"id": "inner", "label": "Inner", "parent": "outer"}],
        "nodes": [{"id": "A", "label": "A", "group": "inner"}]})
    result = E.apply_edits(ir, [{"op": "remove_group", "id": "outer"}])
    assert result.ir.group("inner").parent == ""


def test_direction_layout_title_and_accessibility_ops():
    result = E.apply_edits(_ir(), [
        {"op": "set_direction", "direction": "LR"},
        {"op": "set_layout", "layout": "elk"},
        {"op": "set_title", "title": "Kafka topology"},
        {"op": "set_acc", "acc_title": "Kafka", "acc_descr": "Brokers and app."},
    ])
    assert result.ir.direction == "LR"
    assert result.ir.layout == "elk"
    assert result.ir.acc_title == "Kafka"
    assert not result.rejected


# ---- rejections ---------------------------------------------------------
def test_edge_to_an_unknown_node_is_rejected_not_invented():
    result = E.apply_edits(_ir(), [{"op": "add_edge", "src": "B1", "dst": "Ghost"}])
    assert not result.changed
    assert "Ghost" in result.rejected[0]["reason"]


def test_editing_an_unknown_node_is_rejected():
    result = E.apply_edits(_ir(), [{"op": "rename_node", "id": "Nope",
                                    "label": "x"}])
    assert not result.changed and result.rejected


def test_duplicate_node_is_rejected():
    result = E.apply_edits(_ir(), [{"op": "add_node", "id": "B1", "label": "dup"}])
    assert not result.changed


def test_unknown_op_is_rejected_without_stopping_the_batch():
    result = E.apply_edits(_ir(), [
        {"op": "teleport", "id": "B1"},
        {"op": "rename_node", "id": "B1", "label": "Leader"},
    ])
    assert result.changed                      # the good op still landed
    assert len(result.rejected) == 1
    assert "unknown op" in result.rejected[0]["reason"]


def test_bad_direction_is_rejected():
    result = E.apply_edits(_ir(), [{"op": "set_direction", "direction": "SIDEWAYS"}])
    assert not result.changed


def test_node_added_into_an_unknown_group_is_rejected():
    result = E.apply_edits(_ir(), [{"op": "add_node", "id": "N", "label": "N",
                                    "group": "nope"}])
    assert not result.changed


def test_non_dict_ops_and_empty_batches_are_safe():
    assert not E.apply_edits(_ir(), None).changed
    assert not E.apply_edits(_ir(), []).changed
    result = E.apply_edits(_ir(), ["not an op", 42])  # type: ignore[list-item]
    assert not result.changed and len(result.rejected) == 2


def test_an_edited_diagram_still_emits_valid_looking_source():
    result = E.apply_edits(_ir(), [
        {"op": "add_node", "id": "X", "label": 'Odd (label) "here"'},
        {"op": "add_edge", "src": "X", "dst": "B1", "label": "why (not)?"}])
    source = to_mermaid(result.ir)
    assert source.count("subgraph ") == len(
        [line for line in source.splitlines() if line.strip() == "end"])
    assert '"Odd (label) #quot;here#quot;"' in source


def test_describe_ops_is_human_readable():
    lines = E.describe_ops([
        {"op": "add_node", "label": "Broker 3"},
        {"op": "add_edge", "src": "B1", "dst": "B3"},
        {"op": "set_direction", "direction": "LR"}])
    assert any("Broker 3" in line for line in lines)
    assert any("B1" in line and "B3" in line for line in lines)
    assert any("LR" in line for line in lines)


def test_ops_schema_lists_every_op():
    schema = E.ops_schema()
    enum = schema["properties"]["ops"]["items"]["properties"]["op"]["enum"]
    assert set(enum) == set(E.OPS)


# ---- version history (doc #9) ------------------------------------------
def test_versions_increment_and_list():
    store = DiagramVersionStore()
    store.push("d1", "v1", origin="compose")
    store.push("d1", "v2", origin="edit")
    entries = store.list("d1")
    assert [e.version for e in entries] == [1, 2]
    assert store.head("d1").source == "v2"


def test_pushing_an_identical_source_is_a_no_op():
    store = DiagramVersionStore()
    store.push("d1", "same")
    store.push("d1", "same")
    assert len(store.list("d1")) == 1


def test_restore_appends_rather_than_rewinding():
    store = DiagramVersionStore()
    store.push("d1", "v1")
    store.push("d1", "v2")
    restored = store.restore("d1", 1)
    assert restored is not None
    assert restored.version == 3            # append-only → the restore is undoable
    assert restored.origin == "restore"
    assert store.head("d1").source == "v1"


def test_restoring_an_unknown_version_returns_none():
    store = DiagramVersionStore()
    store.push("d1", "v1")
    assert store.restore("d1", 99) is None
    assert store.restore("nope", 1) is None


def test_history_is_capped_but_version_numbers_stay_monotonic():
    store = DiagramVersionStore(max_versions=3)
    for i in range(6):
        store.push("d1", f"v{i}")
    entries = store.list("d1")
    assert len(entries) == 3
    assert [e.version for e in entries] == [4, 5, 6]


def test_diagram_count_is_capped_evicting_the_oldest_touched():
    store = DiagramVersionStore(max_diagrams=2)
    store.push("a", "1")
    store.push("b", "1")
    store.push("a", "2")                    # refreshes `a`
    store.push("c", "1")                    # evicts `b`, the oldest touched
    assert store.list("b") == []
    assert store.list("a") and store.list("c")


def test_summary_omits_the_source_body():
    store = DiagramVersionStore()
    entry = store.push("d1", "flowchart TD\nA --> B", note="first")
    summary = entry.summary()
    assert "source" not in summary
    assert summary["chars"] == len("flowchart TD\nA --> B")


def test_diagram_id_is_stable_and_distinct():
    assert diagram_id("a") == diagram_id("a")
    assert diagram_id("a") != diagram_id("b")


def test_push_without_a_key_falls_back_to_the_content_hash():
    store = DiagramVersionStore()
    store.push("", "flowchart TD\nA --> B")
    assert store.list(diagram_id("flowchart TD\nA --> B"))


def test_clear_scoped_and_global():
    store = DiagramVersionStore()
    store.push("a", "1")
    store.push("b", "1")
    store.clear("a")
    assert not store.list("a") and store.list("b")
    store.clear()
    assert store.stats()["diagrams"] == 0
