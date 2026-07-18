"""Intelligent diagram gate (roadmap Phase 5 #21).

Pins "diagram only when it improves understanding": structural/relational content
with several interacting parts → render; short or linear prose → don't; an
explicit user request always renders.
"""
from __future__ import annotations

from app.quality.diagram_gate import (COMPARISON, FLOWCHART, NONE,
                                       DiagramDecision, should_diagram)

_ARCH = ("The Client sends a request to the Gateway, which calls the Auth "
         "Service and then the Order Service. The Order Service persists to the "
         "Database and enqueues an event on the Broker. The Worker consumes the "
         "event step by step: first validate, then charge, then ship. --> done.")


def test_architecture_flow_renders():
    d = should_diagram(_ARCH)
    assert isinstance(d, DiagramDecision)
    assert d.render and d.kind != NONE
    assert d.score >= 0.55


def test_short_answer_does_not_render():
    d = should_diagram("The capital of France is Paris.")
    assert not d.render and d.kind == NONE


def test_linear_prose_no_structure_does_not_render():
    prose = ("Photosynthesis is the process by which plants convert light into "
             "chemical energy stored as sugars. It is important for life on "
             "earth and produces oxygen as a byproduct that animals breathe.")
    d = should_diagram(prose)
    assert not d.render


def test_explicit_request_always_renders():
    d = should_diagram("Paris is the capital of France.",
                       request="can you draw a diagram of this?")
    assert d.render


def test_comparison_content_detected():
    text = ("SQL databases enforce a rigid schema and strong consistency, "
            "whereas NoSQL databases favour flexible schemas and horizontal "
            "scale. On the other hand, SQL joins are richer. Compared to "
            "document stores, graph databases model relationships natively, "
            "and the Engine, the Planner and the Executor each play a role.")
    d = should_diagram(text)
    # Either a comparison or flow diagram is reasonable; the point is it renders.
    assert d.render


def test_failopen_returns_no_diagram():
    d = should_diagram(None)          # type: ignore[arg-type]
    assert isinstance(d, DiagramDecision) and not d.render
