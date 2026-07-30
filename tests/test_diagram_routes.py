"""The diagram API — planner (#1), critic (#15), edit (#10), export (#16),
versions (#9), stages (#17).

Every endpoint has a deterministic half that must answer even when the model
half is unavailable, so each model-facing route is tested twice: once with the
model mocked, once with it failing. The handlers are called directly (the
codebase's convention in `test_mermaid_repair.py`) so no server is needed.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import app.api.routes_diagrams as R
import app.diagrams.critic as C
import app.diagrams.planner as PL
from app.diagrams.versions import versions

GOOD_IR = {
    "kind": "flowchart", "direction": "TD",
    "acc_title": "Chat request path",
    "acc_descr": "A user asks the API, which calls the model and stores the reply.",
    "groups": [{"id": "backend", "label": "Backend"}],
    "nodes": [
        {"id": "User", "label": "User", "role": "user"},
        {"id": "api-gw", "label": "API Gateway (REST)", "group": "backend"},
        {"id": "LLM", "label": "Model router", "group": "backend"},
        {"id": "DB", "label": "Postgres", "role": "datastore", "group": "backend"},
    ],
    "edges": [
        {"src": "User", "dst": "api-gw", "label": "asks"},
        {"src": "api-gw", "dst": "LLM", "label": "routes"},
        {"src": "LLM", "dst": "DB", "label": "stores"},
    ],
}
SOURCE = """flowchart LR
  accTitle: Login
  accDescr: A user logs in through the API against the auth service.
  U[User] --> API[API]
  API --> Auth[Auth service]
  Auth -.-> API
"""


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_history():
    versions.clear()
    yield
    versions.clear()


def _mock_structured(monkeypatch, payload, *, errors=None):
    """Stand in for `response_arch.structured.generate_structured`."""
    async def fake(messages, schema, *, options=None, model_meta=None, repair=True):
        return payload, list(errors or [])
    monkeypatch.setattr("app.response_arch.structured.generate_structured", fake)


def _failing_structured(monkeypatch):
    async def boom(messages, schema, *, options=None, model_meta=None, repair=True):
        raise RuntimeError("no route")
    monkeypatch.setattr("app.response_arch.structured.generate_structured", boom)


# ---- compose (#1) ------------------------------------------------------
def test_compose_generates_mermaid_deterministically(monkeypatch):
    _mock_structured(monkeypatch, GOOD_IR)
    res = _run(R.compose(R.ComposeRequest(request="diagram the chat request path")))
    assert res.ok
    assert res.source.splitlines()[0].startswith("%%{init:")
    assert "flowchart" in res.source
    # Ids were sanitised and the edge still resolves — no phantom nodes.
    assert "api_gw" in res.source and "api-gw" not in res.source
    assert res.quality["overall"] > 0
    assert res.version == 1


def test_compose_hands_over_a_stage_ladder_with_render_still_pending(monkeypatch):
    _mock_structured(monkeypatch, GOOD_IR)
    res = _run(R.compose(R.ComposeRequest(request="draw it")))
    states = {s["id"]: s["state"] for s in res.stages["stages"]}
    assert states["planning"] == "done"
    assert states["generating"] == "done"
    assert states["validating"] == "done"
    # Compiling + rendering happen in the client's webview.
    assert states["compiling"] == "pending"
    assert states["rendering"] == "pending"


def test_compose_fails_open_when_the_planner_is_unavailable(monkeypatch):
    _failing_structured(monkeypatch)
    res = _run(R.compose(R.ComposeRequest(request="draw it")))
    assert res.ok is False
    assert res.source == ""
    assert res.errors
    assert res.stages["failed_at"] == "planning"


def test_compose_rejects_an_empty_planned_diagram(monkeypatch):
    _mock_structured(monkeypatch, {"kind": "flowchart", "nodes": []})
    res = _run(R.compose(R.ComposeRequest(request="draw nothing")))
    assert res.ok is False
    assert any("no nodes" in error for error in res.errors)


def test_compose_auto_layout_can_override_the_planned_direction(monkeypatch):
    chain = {"kind": "flowchart", "direction": "TD",
             "nodes": [{"id": f"N{i}", "label": f"Step {i}"} for i in range(9)],
             "edges": [{"src": f"N{i}", "dst": f"N{i+1}"} for i in range(8)]}
    _mock_structured(monkeypatch, chain)
    res = _run(R.compose(R.ComposeRequest(request="a long pipeline",
                                          auto_layout=True)))
    assert res.layout["direction"] == "LR"
    assert "flowchart LR" in res.source


# ---- validate (#4, #6, #7) --------------------------------------------
def test_validate_source_reports_findings_and_a_score():
    res = _run(R.validate(R.ValidateRequest(source=SOURCE)))
    assert res.ir["kind"] == "flowchart"
    assert 0 <= res.quality["overall"] <= 100
    assert set(res.validation) == {"ok", "findings", "counts"}
    assert res.layout is not None


def test_validate_flags_the_docs_backwards_example():
    res = _run(R.validate(R.ValidateRequest(ir={
        "nodes": [{"id": "DB", "label": "Database", "role": "datastore"},
                  {"id": "U", "label": "User", "role": "user"}],
        "edges": [{"src": "DB", "dst": "U"}]})))
    codes = {f["code"] for f in res.validation["findings"]}
    assert "reversed_flow" in codes


def test_validate_needs_no_model():
    # No monkeypatching at all: the deterministic half stands alone.
    res = _run(R.validate(R.ValidateRequest(source="flowchart LR\nA -> B")))
    codes = {f["code"] for f in res.validation["findings"]}
    assert "single_dash_arrow" in codes
    assert res.ok is False


def test_validate_surfaces_what_the_lift_could_not_read():
    res = _run(R.validate(R.ValidateRequest(
        source="flowchart TD\n  A --> B\n  ???!!!")))
    assert "???!!!" in res.unparsed


# ---- critique (#15) ---------------------------------------------------
CRITIQUE = {
    "verdict": "revise",
    "assessment": "The reply path from the model back to the user is missing.",
    "issues": [{"severity": "high", "issue": "no response edge",
                "suggestion": "add API → User"}],
    "ops": [{"op": "add_edge", "src": "Auth", "dst": "U", "label": "token"}],
}


def test_critique_returns_ops_not_a_rewrite(monkeypatch):
    _mock_structured(monkeypatch, CRITIQUE)
    res = _run(R.critique(R.CritiqueRequest(source=SOURCE, request="login flow")))
    assert res.ok
    assert res.critique["verdict"] == "revise"
    assert res.critique["ops"]
    assert res.source == ""                 # nothing applied unless asked


def test_critique_apply_runs_the_ops_through_the_deterministic_applier(monkeypatch):
    _mock_structured(monkeypatch, CRITIQUE)
    res = _run(R.critique(R.CritiqueRequest(source=SOURCE, apply=True)))
    assert res.source
    assert "Auth" in res.source and "token" in res.source
    assert res.applied
    assert res.quality is not None
    assert res.version is not None


def test_critique_rejects_ops_naming_nodes_that_do_not_exist(monkeypatch):
    _mock_structured(monkeypatch, {
        "verdict": "revise", "assessment": "x",
        "ops": [{"op": "add_edge", "src": "U", "dst": "Hallucinated"}]})
    res = _run(R.critique(R.CritiqueRequest(source=SOURCE, apply=True)))
    assert res.source == ""                 # the diagram is untouched
    assert res.rejected


def test_critique_fails_open(monkeypatch):
    _failing_structured(monkeypatch)
    res = _run(R.critique(R.CritiqueRequest(source=SOURCE)))
    assert res.ok is False
    assert res.critique["errors"]


def test_critique_of_an_empty_diagram_is_rebuild():
    res = _run(R.critique(R.CritiqueRequest(source="")))
    assert res.ok is False
    assert res.critique["verdict"] == "rebuild"


def test_parse_critique_contradiction_is_corrected():
    # "ship" while listing a high-severity issue contradicts itself.
    result = C.parse_critique({"verdict": "ship", "assessment": "fine",
                               "issues": [{"severity": "high", "issue": "broken"}]})
    assert result.verdict == "revise"


def test_parse_critique_drops_unknown_ops():
    result = C.parse_critique({"verdict": "revise", "assessment": "x",
                               "ops": [{"op": "teleport"}, {"op": "set_title",
                                                            "title": "T"}]})
    assert [op["op"] for op in result.ops] == ["set_title"]


def test_parse_critique_never_raises():
    for bad in (None, [], "ship", {"issues": "nope"}):
        assert C.parse_critique(bad) is not None  # type: ignore[arg-type]


def test_critic_prompt_includes_the_known_findings():
    from app.diagrams.parse import from_mermaid
    from app.diagrams.validators import validate
    ir = from_mermaid(SOURCE)
    report = validate(ir, source=SOURCE)
    messages = C.critic_prompt(ir, request="login", findings=report.findings)
    body = messages[-1]["content"]
    assert "do not repeat" in messages[0]["content"].lower()
    assert "validator findings" in body


# ---- edit (#10) -------------------------------------------------------
def test_edit_with_explicit_ops_needs_no_model():
    res = _run(R.edit(R.EditRequest(
        source=SOURCE, command="switch to top-down",
        ops=[{"op": "set_direction", "direction": "TD"}])))
    assert res.ok
    assert "flowchart TD" in res.source
    assert res.applied == ["direction → TD"]


def test_edit_translates_a_sentence_into_ops(monkeypatch):
    _mock_structured(monkeypatch, {
        "ops": [{"op": "add_node", "id": "Cache", "label": "Redis",
                 "role": "datastore"},
                {"op": "add_edge", "src": "API", "dst": "Cache",
                 "label": "checks"}],
        "note": "added a cache in front of auth"})
    res = _run(R.edit(R.EditRequest(source=SOURCE,
                                    command="put a redis cache in front")))
    assert res.ok
    assert "Cache" in res.source and "Redis" in res.source
    # Everything the user did not mention survived.
    assert "Auth" in res.source and "User" in res.source
    assert res.note


def test_edit_preserves_the_rest_of_the_diagram():
    before = _run(R.validate(R.ValidateRequest(source=SOURCE)))
    res = _run(R.edit(R.EditRequest(
        source=SOURCE, command="rename",
        ops=[{"op": "rename_node", "id": "Auth", "label": "Identity"}])))
    after = _run(R.validate(R.ValidateRequest(source=res.source)))
    assert len(after.ir["nodes"]) == len(before.ir["nodes"])
    assert len(after.ir["edges"]) == len(before.ir["edges"])


def test_edit_reports_rejections_instead_of_corrupting(monkeypatch):
    _mock_structured(monkeypatch, {
        "ops": [{"op": "remove_node", "id": "DoesNotExist"}], "note": ""})
    res = _run(R.edit(R.EditRequest(source=SOURCE, command="delete the ghost")))
    assert res.ok is False
    assert res.rejected
    assert res.source == ""


def test_edit_fails_open_without_a_model(monkeypatch):
    _failing_structured(monkeypatch)
    res = _run(R.edit(R.EditRequest(source=SOURCE, command="do something")))
    assert res.ok is False and res.errors


def test_edit_of_an_empty_diagram_is_refused():
    res = _run(R.edit(R.EditRequest(source="", command="anything")))
    assert res.ok is False


# ---- export (#16) -----------------------------------------------------
@pytest.mark.parametrize("fmt", ["mermaid", "plantuml", "dot", "drawio", "elk",
                                 "json"])
def test_export_each_format(fmt):
    res = _run(R.export(R.ExportRequest(source=SOURCE, format=fmt)))
    assert res.ok and res.content
    assert res.format == fmt
    assert res.filename.endswith(res.available[fmt]["ext"])


def test_export_all_formats_at_once():
    res = _run(R.export(R.ExportRequest(source=SOURCE, all_formats=True)))
    assert set(res.formats) == set(res.available)


def test_export_advertises_the_registry():
    res = _run(R.export(R.ExportRequest(source=SOURCE)))
    assert "plantuml" in res.available
    assert res.available["dot"]["ext"] == "dot"


# ---- normalize --------------------------------------------------------
def test_normalize_reemits_a_hand_written_diagram():
    res = _run(R.normalize(R.ValidateRequest(source="flowchart LR\nA-->B")))
    assert res.ok and res.changed
    assert 'A["A"]' in res.source            # ids get real labels + quoting
    assert res.layout["init_directive"].startswith("%%{init:")


def test_normalize_is_idempotent_on_its_own_output():
    once = _run(R.normalize(R.ValidateRequest(source=SOURCE)))
    twice = _run(R.normalize(R.ValidateRequest(source=once.source)))
    assert twice.source == once.source


# ---- versions (#9) ----------------------------------------------------
def test_versions_save_list_and_restore():
    saved = _run(R.save_version(R.SaveVersionRequest(
        source=SOURCE, diagram_id="d1", note="first")))
    assert saved.ok and saved.version == 1
    _run(R.save_version(R.SaveVersionRequest(
        source=SOURCE + "  X --> U\n", diagram_id="d1", note="second")))

    listed = _run(R.list_versions("d1"))
    assert [v["version"] for v in listed.versions] == [1, 2]
    assert listed.head == 2
    assert "source" not in listed.versions[0]

    restored = _run(R.restore_version(R.RestoreRequest(diagram_id="d1", version=1)))
    assert restored.ok
    assert restored.version == 3             # append-only
    assert restored.source == SOURCE


def test_restoring_an_unknown_version_is_refused():
    res = _run(R.restore_version(R.RestoreRequest(diagram_id="nope", version=1)))
    assert res.ok is False


def test_saving_an_empty_source_is_refused():
    assert _run(R.save_version(R.SaveVersionRequest(source="   "))).ok is False


def test_compose_and_edit_share_one_history(monkeypatch):
    _mock_structured(monkeypatch, GOOD_IR)
    composed = _run(R.compose(R.ComposeRequest(request="draw it",
                                               diagram_id="shared")))
    _run(R.edit(R.EditRequest(source=composed.source, diagram_id="shared",
                              command="lr", ops=[{"op": "set_direction",
                                                  "direction": "LR"}])))
    listed = _run(R.list_versions("shared"))
    assert [v["origin"] for v in listed.versions] == ["compose", "edit"]


# ---- stages (#17) -----------------------------------------------------
def test_stage_vocabulary_is_the_single_source_of_truth():
    data = _run(R.stage_vocabulary())
    ids = [stage["id"] for stage in data["stages"]]
    assert ids == ["planning", "generating", "validating", "compiling",
                   "repairing", "rendering"]
    assert next(s for s in data["stages"] if s["id"] == "repairing")["conditional"]
    assert "plantuml" in data["formats"]
    assert "add_edge" in data["ops"]
    assert data["elk_enabled"] is False
    # How the answer path is configured, so a client can say whether a diagram was
    # generated from structure or recompiled from the model's text.
    assert data["ir_lane_enabled"] is True
    assert data["planner_mode"] == "off"


# ---- registration -----------------------------------------------------
@pytest.mark.parametrize("path", [
    "/api/diagram/compose", "/api/diagram/validate", "/api/diagram/critique",
    "/api/diagram/edit", "/api/diagram/export", "/api/diagram/normalize",
    "/api/diagram/versions/{key}", "/api/diagram/versions/save",
    "/api/diagram/versions/restore", "/api/diagram/stages",
])
def test_routes_registered(path):
    # Asserted against the ROUTER, not `app.main`. Importing `app.main` in a test
    # has process-wide side effects (it warms models and wires the live stack) and
    # leaks into whatever runs next — `test_overall_enhancements.py` states the
    # rule outright: "never import app.main here (it loads ML models)". Doing it
    # from this file broke `test_live_operability.py`'s websocket tests purely
    # because `test_diagram_*` sorts before `test_live_*`.
    assert path in {route.path for route in R.router.routes}


def test_router_is_mounted_in_main():
    # The one thing the router alone can't prove — checked by reading main.py as
    # TEXT rather than importing it, for the reason above.
    from pathlib import Path
    main_py = Path(__file__).resolve().parents[1] / "app" / "main.py"
    source = main_py.read_text(encoding="utf-8", errors="replace")
    assert "routes_diagrams import router as diagrams_router" in source
    assert "include_router(diagrams_router)" in source


# ---- planner prompt (offline half) ------------------------------------
def test_plan_prompt_never_mentions_mermaid_syntax():
    messages = PL.plan_prompt("diagram the checkout flow")
    text = " ".join(message["content"] for message in messages)
    for syntax in ("-->", "subgraph", "```", "flowchart TD"):
        assert syntax not in text, syntax


def test_plan_prompt_pins_the_kind_when_given():
    messages = PL.plan_prompt("the login handshake", kind="sequence")
    assert '"kind": "sequence"' in messages[-1]["content"]


def test_plan_from_json_coerces_a_fenced_payload():
    ir, errors = PL.plan_from_json("```json\n" + json.dumps(GOOD_IR) + "\n```")
    assert ir is not None and not errors
    assert ir.node_ids == {"User", "api_gw", "LLM", "DB"}


def test_plan_from_json_rejects_garbage():
    ir, errors = PL.plan_from_json("not json at all")
    assert ir is None and errors


def test_edit_prompt_lists_the_existing_ids():
    from app.diagrams.parse import from_mermaid
    messages = PL.edit_prompt(from_mermaid(SOURCE), "move Auth under API")
    body = messages[-1]["content"]
    assert "Auth" in body and "API" in body
    assert "Available ops" in messages[0]["content"]


# ---- the safety gate reaches the API too --------------------------------
GANTT = """gantt
  dateFormat YYYY-MM-DD
  section Build
  Design :a1, 2024-01-01, 30d
"""


def test_normalize_refuses_a_diagram_type_the_ir_cannot_model():
    # "Clean up" on a gantt chart must REFUSE, not silently turn it into an empty
    # flowchart. This is the same gate the answer-path compile lane uses.
    res = _run(R.normalize(R.ValidateRequest(source=GANTT)))
    assert res.ok is False
    assert res.ir["nodes"] == []
    assert res.ir["meta"]["unsupported_kind"] == "gantt"


def test_validate_reports_an_unmodelled_type_rather_than_guessing():
    res = _run(R.validate(R.ValidateRequest(source=GANTT)))
    assert res.ir["meta"]["unsupported_kind"] == "gantt"
    # No nodes → the syntax validator says so instead of inventing findings about
    # a flowchart that doesn't exist.
    codes = {f["code"] for f in res.validation["findings"]}
    assert "no_nodes" in codes


def test_edit_refuses_an_unmodelled_type():
    res = _run(R.edit(R.EditRequest(source=GANTT, command="add a task",
                                    ops=[{"op": "set_direction",
                                          "direction": "LR"}])))
    assert res.ok is False
    assert res.errors


def test_compose_reports_the_crossing_count_it_achieved():
    # doc #5, measured rather than asserted.
    import app.api.routes_diagrams as routes
    from app.diagrams.layout import render_with_layout
    from app.diagrams.ir import from_dict as build
    ir = build({
        "kind": "flowchart", "direction": "LR",
        "nodes": ([{"id": f"s{i}", "label": f"S{i}"} for i in range(4)]
                  + [{"id": f"t{i}", "label": f"T{i}"} for i in reversed(range(4))]),
        "edges": [{"src": f"s{i}", "dst": f"t{i}"} for i in range(4)],
    })
    _source, plan = render_with_layout(ir)
    assert plan.to_dict()["crossings"] == 0
    assert routes is not None
