"""The answer-path compile lane (MermaidDiagramVisualizations.md #1, wired live).

Two contracts, and the second matters more than the first:

  1. a diagram the lane FULLY understands is recompiled from its structure;
  2. everything else is returned byte-for-byte unchanged.

(2) is what makes it safe to run on every answer, so most of this file is about
the refusals: unmodelled diagram types, incomplete lifts, author-set styling, and
rebuilds that don't round-trip.
"""
from __future__ import annotations

import app.diagrams.lane as L
from app.diagrams.parse import unsupported_kind

FLOW = """```mermaid
flowchart LR
  A[Fetch data] --> B{Valid?}
  B -->|yes| C[Store]
  B -->|no| D[Reject]
```"""


def _answer(fence: str) -> str:
    return f"Here is the flow.\n\n{fence}\n\nThat's the shape of it.\n"


# ---- the happy path -----------------------------------------------------
def test_a_model_written_diagram_is_recompiled():
    result = L.compile_diagrams(_answer(FLOW))
    assert result.changed
    outcome = result.outcomes[0]
    assert outcome.rewritten and outcome.kind == "flowchart"
    assert outcome.nodes == 4 and outcome.edges == 3
    # The generator's fingerprints: a layout directive, quoted labels.
    assert "%%{init:" in result.text
    assert 'A["Fetch data"]' in result.text
    assert 'B{"Valid?"}' in result.text


def test_the_prose_around_the_diagram_is_untouched():
    text = _answer(FLOW)
    result = L.compile_diagrams(text)
    assert result.text.startswith("Here is the flow.\n\n")
    assert result.text.rstrip().endswith("That's the shape of it.")


def test_the_fence_stays_a_mermaid_fence():
    result = L.compile_diagrams(_answer(FLOW))
    assert result.text.count("```mermaid") == 1
    assert result.text.count("```") == 2


def test_multiple_diagrams_are_each_handled():
    sequence = """```mermaid
sequenceDiagram
  participant U as User
  participant S as Server
  U->>S: GET /items
  S-->>U: 200 OK
```"""
    result = L.compile_diagrams(f"{FLOW}\n\ntext between\n\n{sequence}")
    assert len(result.outcomes) == 2
    assert all(outcome.rewritten for outcome in result.outcomes)
    assert [o.kind for o in result.outcomes] == ["flowchart", "sequence"]
    assert "text between" in result.text


def test_a_syntactically_broken_diagram_is_repaired_by_the_rebuild():
    # An unquoted parenthesised label is a real mermaid parse error; the emitter
    # always quotes, so the rebuild fixes it without a model call.
    broken = """```mermaid
flowchart LR
  A[Fetch (REST)] --> B[Store]
```"""
    result = L.compile_diagrams(broken)
    assert result.changed
    assert 'A["Fetch (REST)"]' in result.text
    assert result.outcomes[0].errors_before > 0
    assert result.outcomes[0].errors_after == 0


def test_the_outcome_carries_a_quality_score():
    result = L.compile_diagrams(FLOW)
    assert result.outcomes[0].score is not None
    assert 0 <= result.outcomes[0].score <= 100


def test_compile_answer_is_the_one_liner():
    assert L.compile_answer(_answer(FLOW)) == L.compile_diagrams(_answer(FLOW)).text


# ---- the safety gate ----------------------------------------------------
UNMODELLED = {
    "gantt": "gantt\n  title A\n  section S\n  Task :a1, 2024-01-01, 30d",
    "pie": 'pie title Pets\n  "Dogs" : 386\n  "Cats" : 85',
    "journey": "journey\n  title My day\n  section Go to work\n    Make tea: 5: Me",
    "timeline": "timeline\n  title History\n  2002 : LinkedIn",
    "gitGraph": "gitGraph\n  commit\n  branch develop\n  commit",
    "quadrantChart": 'quadrantChart\n  title Reach\n  x-axis Low --> High',
    "C4Context": 'C4Context\n  title System\n  Person(a, "A")',
    "sankey-beta": "sankey-beta\n\nA,B,10",
    "xychart-beta": 'xychart-beta\n  title "Sales"\n  bar [5, 10]',
    "block-beta": "block-beta\n  columns 1\n  A",
    "requirementDiagram": "requirementDiagram\n  requirement test_req {\n  id: 1\n  }",
    "erDiagram-lookalike": None,   # placeholder removed below
}
UNMODELLED.pop("erDiagram-lookalike")


def test_every_unmodelled_diagram_type_passes_through_untouched():
    # This is the check that makes the lane safe to run on every answer. Without
    # it, a gantt chart would be misread as a flowchart and DESTROYED.
    for name, body in UNMODELLED.items():
        text = f"```mermaid\n{body}\n```"
        result = L.compile_diagrams(text)
        assert not result.changed, name
        assert result.text == text, name
        assert result.outcomes[0].reason, name
        assert "not modelled" in result.outcomes[0].reason, name


def test_unsupported_kind_detects_past_comments_and_init():
    assert unsupported_kind("%% a note\n%%{init: {}}%%\ngantt\n  title x") == "gantt"
    assert unsupported_kind("flowchart LR\n A --> B") == ""


def test_an_author_set_init_directive_is_respected():
    text = """```mermaid
%%{init: {"theme":"forest"}}%%
flowchart LR
  A[One] --> B[Two]
```"""
    result = L.compile_diagrams(text)
    assert not result.changed
    assert result.text == text
    assert "author-set" in result.outcomes[0].reason


def test_author_styling_is_respected():
    for directive in ("style A fill:#f9f",
                      "classDef hot fill:#f00",
                      "linkStyle 0 stroke:#333",
                      "click A href \"https://example.com\""):
        text = f"```mermaid\nflowchart LR\n  A[One] --> B[Two]\n  {directive}\n```"
        result = L.compile_diagrams(text)
        assert not result.changed, directive
        assert result.text == text, directive


def test_a_line_the_reader_cannot_interpret_blocks_the_rewrite():
    text = """```mermaid
flowchart LR
  A[One] --> B[Two]
  ???!!! nonsense
```"""
    result = L.compile_diagrams(text)
    assert not result.changed
    assert result.text == text
    assert "not understood" in result.outcomes[0].reason


def test_an_incomplete_fence_is_never_touched():
    # A half-streamed diagram has no closing fence.
    text = "```mermaid\nflowchart LR\n  A --> B"
    assert L.compile_diagrams(text).text == text


def test_text_with_no_diagram_is_returned_as_is():
    text = "Just prose, with a ```python\nprint(1)\n``` block."
    result = L.compile_diagrams(text)
    assert result.text == text
    assert result.diagrams == 0


def test_an_empty_diagram_is_left_alone():
    text = "```mermaid\n\n```"
    assert L.compile_diagrams(text).text == text


def test_the_flag_disables_the_lane(monkeypatch):
    monkeypatch.setattr(L, "enabled", lambda: False)
    text = _answer(FLOW)
    result = L.compile_diagrams(text)
    assert result.text == text
    assert not result.changed


def test_the_lane_is_on_by_default():
    assert L.enabled() is True


def test_the_lane_never_raises():
    for bad in (None, "", "```mermaid", "```mermaid\n\x00\n```",
                "```mermaid\n" + "A-->B\n" * 500 + "```"):
        assert L.compile_diagrams(bad) is not None  # type: ignore[arg-type]


def test_the_lane_is_idempotent():
    once = L.compile_diagrams(_answer(FLOW)).text
    twice = L.compile_diagrams(once)
    # Its own output must survive a second pass unchanged — otherwise a re-saved
    # message would drift every time it was touched.
    assert twice.text == once
    # And it must be idempotent for the RIGHT reason: the second pass actually
    # recompiles and lands on the same bytes (the emitter is deterministic). It
    # must NOT be skipping its own output — an earlier version refused to touch
    # anything carrying an `%%{init}%%`, including the one it had just written,
    # which quietly made the lane a one-shot.
    assert twice.changed
    assert twice.outcomes[0].rewritten


def test_the_lane_recognises_its_own_layout_directive():
    compiled = L.compile_diagrams(_answer(FLOW)).text
    assert "%%{init:" in compiled
    # Our directive: not "author-set", so the lane still owns the diagram.
    assert L._authored_presentation(compiled) == ""


def test_an_author_theme_in_an_init_directive_is_still_respected():
    # A theme (or anything else outside the layout planner's own key set) is a
    # deliberate choice the IR does not round-trip.
    for payload in ('{"theme":"forest"}',
                    '{"flowchart":{"curve":"basis"},"theme":"dark"}',
                    '{"flowchart":{"diagramPadding":20}}',
                    '{"themeVariables":{"primaryColor":"#f00"}}',
                    'not json at all'):
        source = f"%%{{init: {payload}}}%%\nflowchart LR\n  A[One] --> B[Two]"
        assert L._authored_presentation(source) == "%%{init}%%", payload


def test_the_layout_planners_own_directive_is_recognised_exactly():
    from app.diagrams.ir import from_dict
    from app.diagrams.layout import plan_layout
    ir = from_dict({"nodes": [{"id": "A", "label": "A one"},
                              {"id": "B", "label": "B two"}],
                    "edges": [{"src": "A", "dst": "B"}]})
    directive = plan_layout(ir).init_directive()
    assert L._authored_presentation(f"{directive}\nflowchart LR\n  A --> B") == ""


def test_result_to_dict_shape():
    data = L.compile_diagrams(_answer(FLOW)).to_dict()
    assert set(data) == {"changed", "diagrams", "outcomes"}
    assert set(data["outcomes"][0]) == {
        "index", "rewritten", "kind", "reason", "nodes", "edges", "score",
        "errors_before", "errors_after"}


# ---- wiring -------------------------------------------------------------
def test_finalize_runs_the_lane():
    # `response_arch.finalize` is the shared choke point for the main chat path
    # and /api/solve.
    from app.response_arch import finalize
    shaped = finalize(_answer(FLOW), question="draw the flow")
    assert 'A["Fetch data"]' in shaped.text
    assert any("rebuilt from structure" in w for w in shaped.warnings)


def test_finalize_leaves_an_unmodelled_diagram_alone():
    from app.response_arch import finalize
    text = "```mermaid\ngantt\n  title A\n  section S\n  Task :a1, 2024-01-01, 30d\n```"
    shaped = finalize(text, question="a gantt chart")
    assert "gantt" in shaped.text
    assert "flowchart" not in shaped.text


def test_the_paths_that_skip_finalize_call_the_lane_directly():
    # Upload turns and agent runs never run `finalize`, so they must call the lane
    # themselves — otherwise they'd be the one place a diagram skipped the
    # compiler. Checked by reading the sources (importing these routes pulls in
    # the whole app).
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "app" / "api"
    uploads = (root / "routes_attachments.py").read_text(
        encoding="utf-8", errors="replace")
    assert "from app.diagrams.lane import compile_answer" in uploads
    assert "compile_answer(full_text)" in uploads
    agent_run = (root / "routes_chat_agent.py").read_text(
        encoding="utf-8", errors="replace")
    assert "from app.diagrams.lane import compile_answer" in agent_run
    # Both `final` handlers must go through the helper, not raw text.
    assert agent_run.count("_compile_diagrams(") >= 3


# ==========================================================================
# The PLANNER lane — the model half of #1, behind `response_arch.diagram_planner`
# ==========================================================================
import asyncio  # noqa: E402

import pytest  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


GOOD_PLAN = {
    "kind": "flowchart", "direction": "LR",
    "acc_title": "Validation flow",
    "acc_descr": "Data is fetched, validated, then stored or rejected.",
    "nodes": [
        {"id": "A", "label": "Fetch data"},
        {"id": "B", "label": "Valid?", "role": "decision"},
        {"id": "C", "label": "Store", "role": "datastore"},
        {"id": "D", "label": "Reject"},
        {"id": "E", "label": "Notify"},
    ],
    "edges": [
        {"src": "A", "dst": "B", "label": "rows"},
        {"src": "B", "dst": "C", "label": "yes"},
        {"src": "B", "dst": "D", "label": "no"},
        {"src": "D", "dst": "E", "label": "alert"},
    ],
}
# A diagram the reader cannot fully lift → the deterministic rebuild is refused,
# which is precisely the case the planner exists for.
UNLIFTABLE = """```mermaid
flowchart LR
  A[Fetch data] --> B[Store]
  ???!!! nonsense
```"""


def _mock_planner(monkeypatch, payload, *, errors=None):
    async def fake(messages, schema, *, options=None, model_meta=None, repair=True):
        return payload, list(errors or [])
    monkeypatch.setattr("app.response_arch.structured.generate_structured", fake)


@pytest.fixture
def repair_mode(monkeypatch):
    monkeypatch.setattr(L, "planner_mode", lambda: L.PLANNER_REPAIR)


@pytest.fixture
def always_mode(monkeypatch):
    monkeypatch.setattr(L, "planner_mode", lambda: L.PLANNER_ALWAYS)


# ---- the flag -----------------------------------------------------------
def test_the_planner_is_off_by_default():
    assert L.planner_mode() == L.PLANNER_OFF


def test_off_mode_never_calls_the_planner(monkeypatch):
    async def boom(*args, **kwargs):
        raise AssertionError("the planner must not be called when off")
    monkeypatch.setattr("app.diagrams.planner.plan", boom)
    text = _answer(FLOW)
    assert _run(L.plan_answer(text, request="draw it")) == text


def test_an_unknown_mode_falls_back_to_off(monkeypatch):
    class _Section:
        diagram_planner = "sometimes"

    class _Cfg:
        response_arch = _Section()
    monkeypatch.setattr("app.core.config_loader.cfg", _Cfg())
    assert L.planner_mode() == L.PLANNER_OFF


def test_mode_names_are_the_documented_three():
    assert L.PLANNER_MODES == ("off", "repair", "always")


# ---- repair mode --------------------------------------------------------
def test_repair_mode_leaves_a_clean_diagram_alone(monkeypatch, repair_mode):
    async def boom(*args, **kwargs):
        raise AssertionError("a clean diagram must not cost a model call")
    monkeypatch.setattr("app.diagrams.planner.plan", boom)
    compiled = L.compile_answer(_answer(FLOW))
    result = _run(L.plan_diagrams(compiled, request="draw the flow"))
    assert not result.changed
    assert result.text == compiled
    assert result.outcomes[0].reason


def test_repair_mode_replans_a_diagram_the_reader_could_not_lift(
        monkeypatch, repair_mode):
    _mock_planner(monkeypatch, GOOD_PLAN)
    compiled = L.compile_answer(UNLIFTABLE)     # refused → unchanged
    result = _run(L.plan_diagrams(compiled, request="draw the validation flow"))
    assert result.changed
    outcome = result.outcomes[0]
    assert outcome.planned
    assert "refused" in outcome.trigger
    # The planned structure replaced the text the reader couldn't parse.
    assert "???!!!" not in result.text
    assert 'B{"Valid?"}' in result.text
    assert "accTitle: Validation flow" in result.text


def test_repair_mode_records_the_before_and_after_score(monkeypatch, repair_mode):
    _mock_planner(monkeypatch, GOOD_PLAN)
    result = _run(L.plan_diagrams(L.compile_answer(UNLIFTABLE), request="x"))
    outcome = result.outcomes[0]
    assert outcome.score_after is not None
    assert outcome.nodes_after == 5


# ---- always mode --------------------------------------------------------
def test_always_mode_replans_even_a_clean_diagram(monkeypatch, always_mode):
    _mock_planner(monkeypatch, GOOD_PLAN)
    result = _run(L.plan_diagrams(L.compile_answer(_answer(FLOW)), request="draw it"))
    assert result.changed
    assert result.outcomes[0].trigger == "planner mode is `always`"
    assert "Notify" in result.text          # the planned diagram, not the original


def test_always_mode_still_refuses_unmodelled_types(monkeypatch, always_mode):
    async def boom(*args, **kwargs):
        raise AssertionError("a gantt chart has no IR to plan into")
    monkeypatch.setattr("app.diagrams.planner.plan", boom)
    text = "```mermaid\ngantt\n  title A\n  section S\n  T :a1, 2024-01-01, 1d\n```"
    result = _run(L.plan_diagrams(text, request="a gantt"))
    assert result.text == text
    assert not result.changed


def test_always_mode_respects_author_styling(monkeypatch, always_mode):
    async def boom(*args, **kwargs):
        raise AssertionError("deliberate styling must not be re-planned away")
    monkeypatch.setattr("app.diagrams.planner.plan", boom)
    text = "```mermaid\nflowchart LR\n  A[One] --> B[Two]\n  style A fill:#f9f\n```"
    assert _run(L.plan_diagrams(text, request="x")).text == text


# ---- the acceptance gate ------------------------------------------------
def test_a_planned_diagram_that_loses_nodes_is_rejected(monkeypatch, always_mode):
    _mock_planner(monkeypatch, {
        "kind": "flowchart",
        "acc_title": "T", "acc_descr": "A description long enough to pass here.",
        "nodes": [{"id": "A", "label": "Fetch data"}, {"id": "B", "label": "Store"}],
        "edges": [{"src": "A", "dst": "B", "label": "rows"}]})
    compiled = L.compile_answer(_answer(FLOW))   # 4 nodes
    result = _run(L.plan_diagrams(compiled, request="draw it"))
    assert not result.changed
    assert result.text == compiled
    assert "lost content" in result.outcomes[0].reason


def test_a_planned_diagram_that_loses_edges_is_rejected(monkeypatch, always_mode):
    _mock_planner(monkeypatch, {
        "kind": "flowchart",
        "acc_title": "T", "acc_descr": "A description long enough to pass here.",
        "nodes": [{"id": n, "label": f"{n} step"}
                  for n in ("Alpha", "Beta", "Gamma", "Delta")],
        "edges": [{"src": "Alpha", "dst": "Beta", "label": "next"}]})
    result = _run(L.plan_diagrams(L.compile_answer(_answer(FLOW)), request="x"))
    assert not result.changed
    assert "lost relationships" in result.outcomes[0].reason


def test_a_planned_diagram_with_validator_errors_is_rejected(
        monkeypatch, always_mode):
    _mock_planner(monkeypatch, {
        "kind": "flowchart",
        "nodes": [{"id": n, "label": n} for n in ("A", "B", "C", "D", "E")],
        # A dangling endpoint is an ERROR finding.
        "edges": [{"src": "A", "dst": "Ghost"}, {"src": "A", "dst": "B"},
                  {"src": "B", "dst": "C"}, {"src": "C", "dst": "D"},
                  {"src": "D", "dst": "E"}]})
    result = _run(L.plan_diagrams(L.compile_answer(_answer(FLOW)), request="x"))
    assert not result.changed
    assert "validator error" in result.outcomes[0].reason


def test_a_lower_scoring_plan_is_rejected(monkeypatch, always_mode):
    # Same node and edge COUNT, no validator errors — but two nodes share a label,
    # which is a readability warning the original doesn't have. A plan that is
    # merely different must not displace one that scores better.
    _mock_planner(monkeypatch, {
        "kind": "flowchart",
        "nodes": [{"id": "A", "label": "Fetch data"}, {"id": "B", "label": "Store"},
                  {"id": "C", "label": "Store"}, {"id": "D", "label": "Reject"}],
        "edges": [{"src": "A", "dst": "B", "label": "rows"},
                  {"src": "A", "dst": "C", "label": "rows"},
                  {"src": "A", "dst": "D", "label": "bad"}]})
    compiled = L.compile_answer(_answer(FLOW))
    result = _run(L.plan_diagrams(compiled, request="x"))
    assert not result.changed
    assert "scores lower" in result.outcomes[0].reason


def test_a_planner_failure_leaves_the_diagram_alone(monkeypatch, always_mode):
    async def boom(messages, schema, *, options=None, model_meta=None, repair=True):
        raise RuntimeError("no route")
    monkeypatch.setattr("app.response_arch.structured.generate_structured", boom)
    compiled = L.compile_answer(_answer(FLOW))
    result = _run(L.plan_diagrams(compiled, request="x"))
    assert result.text == compiled
    assert not result.changed
    assert result.outcomes[0].reason


def test_an_empty_plan_leaves_the_diagram_alone(monkeypatch, always_mode):
    _mock_planner(monkeypatch, {"kind": "flowchart", "nodes": []})
    compiled = L.compile_answer(_answer(FLOW))
    assert _run(L.plan_diagrams(compiled, request="x")).text == compiled


# ---- shape + safety ----------------------------------------------------
def test_the_planner_lane_never_raises(always_mode):
    for bad in (None, "", "```mermaid", "```mermaid\nflowchart LR\nA-->B\n```"):
        assert _run(L.plan_diagrams(bad)) is not None  # type: ignore[arg-type]


def test_prose_around_a_replanned_diagram_survives(monkeypatch, repair_mode):
    _mock_planner(monkeypatch, GOOD_PLAN)
    text = f"Before.\n\n{UNLIFTABLE}\n\nAfter.\n"
    result = _run(L.plan_diagrams(text, request="x"))
    assert result.text.startswith("Before.")
    assert result.text.rstrip().endswith("After.")
    assert result.text.count("```mermaid") == 1


def test_planned_result_to_dict_shape(monkeypatch, repair_mode):
    _mock_planner(monkeypatch, GOOD_PLAN)
    data = _run(L.plan_diagrams(L.compile_answer(UNLIFTABLE), request="x")).to_dict()
    assert set(data) == {"changed", "outcomes"}
    assert set(data["outcomes"][0]) == {
        "index", "planned", "trigger", "reason", "nodes_before", "nodes_after",
        "score_before", "score_after"}


def test_plan_answer_short_circuits_without_a_diagram(always_mode):
    assert _run(L.plan_answer("just prose")) == "just prose"


# ---- wiring ------------------------------------------------------------
def _route_source(name: str) -> str:
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "app" / "api"
    return (root / name).read_text(encoding="utf-8", errors="replace")


def test_every_answer_path_awaits_the_planner_lane():
    # Off by default, so these call sites cost nothing — but they must EXIST or
    # turning the flag on would only affect some turns.
    for name in ("routes_agents.py", "routes_solve.py", "routes_attachments.py",
                 "routes_chat_agent.py"):
        source = _route_source(name)
        assert "app.diagrams.lane import" in source, name
        assert "plan_answer" in source, name
        # Awaited under its own name or a local alias.
        assert ("await plan_answer(" in source
                or "await _plan_diagrams(" in source
                or "await _compile_diagrams(" in source), name


def test_the_upload_path_runs_the_lane_after_recovery_and_before_the_reveal():
    # Regression guard for a real placement bug: the lane originally sat directly
    # after the FIRST `full_text` assembly, so an answer rescued by the
    # empty-answer retry / escalating-tier fallback reassembled `full_text`
    # afterwards and its diagrams bypassed both lanes. It must run after the
    # recovery cascade and before the buffered reveal, or the text that is shown
    # and saved is not the compiled text.
    source = _route_source("routes_attachments.py")
    recovered = source.index("UPLOAD-DIAG answer recovered")
    lane = source.index("from app.diagrams.lane import compile_answer")
    reveal = source.index("Verify-before-reveal payoff")
    save = source.index("_save(full_text, incomplete=False, bump=True)")
    assert recovered < lane < reveal < save

    # And it must appear exactly once — a second call site would double-compile.
    assert source.count("from app.diagrams.lane import") == 1


# ---- budgets ------------------------------------------------------------
# The planner lane runs between the last streamed token and the `done` frame, so
# an unbounded model call would turn a slow route into a turn that looks hung.
def test_a_slow_planner_is_abandoned_and_the_diagram_stands(
        monkeypatch, always_mode):
    monkeypatch.setattr(L, "PLANNER_TIMEOUT_S", 0.05)

    async def slow(messages, schema, *, options=None, model_meta=None, repair=True):
        await asyncio.sleep(5)
        return GOOD_PLAN, []
    monkeypatch.setattr("app.response_arch.structured.generate_structured", slow)

    compiled = L.compile_answer(_answer(FLOW))
    result = _run(L.plan_diagrams(compiled, request="draw it"))
    assert not result.changed
    assert result.text == compiled
    assert "took longer than" in result.outcomes[0].reason


def test_the_planner_budget_caps_calls_per_answer(monkeypatch, always_mode):
    monkeypatch.setattr(L, "PLANNER_MAX_DIAGRAMS", 2)
    calls = {"n": 0}

    async def counted(messages, schema, *, options=None, model_meta=None,
                      repair=True):
        calls["n"] += 1
        return GOOD_PLAN, []
    monkeypatch.setattr("app.response_arch.structured.generate_structured", counted)

    answer = "\n\n".join(FLOW for _ in range(4))
    result = _run(L.plan_diagrams(L.compile_answer(answer), request="draw them"))
    assert calls["n"] == 2
    planned = [o for o in result.outcomes if o.planned]
    assert len(planned) == 2
    # The rest keep their deterministically-compiled form, which is already valid
    # syntax — and they say why they were skipped.
    skipped = [o for o in result.outcomes if not o.planned]
    assert len(skipped) == 2
    assert all("budget spent" in o.reason for o in skipped)


def test_the_budget_defaults_are_conservative():
    assert L.PLANNER_MAX_DIAGRAMS == 3
    assert 5 <= L.PLANNER_TIMEOUT_S <= 60
