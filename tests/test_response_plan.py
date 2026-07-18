"""Formal ResponsePlan — first meaningful paint / predictive artifacts (P6 #5/#11/#12)."""
from __future__ import annotations

from app.response_arch.content_router import Shape
from app.response_arch.plan import build_response_plan, predict_artifacts


def test_prose_plan_is_single_section():
    p = build_response_plan("what is a monad?")
    assert p.shape == Shape.PROSE
    assert [s.id for s in p.sections] == ["answer"]
    assert p.refinable is False


def test_comparison_plan_enumerates_sections_before_tokens():
    p = build_response_plan("compare Postgres vs MySQL")
    assert p.shape == Shape.COMPARISON
    assert p.outline() == ["Summary", "Comparison", "Recommendation"]
    assert p.refinable is True
    frame = p.as_frame()
    assert frame["shape"] == "comparison"
    assert frame["sections"][1]["kind"] == "table"


def test_steps_plan_shape():
    p = build_response_plan("how to set up nginx step by step")
    assert p.shape == Shape.STEPS
    assert "Steps" in p.outline()


def test_predicts_artifacts_from_cues():
    arts = predict_artifacts("give me a dockerfile and docker-compose", Shape.CODE)
    names = {a["filename"] for a in arts}
    assert "Dockerfile" in names and "docker-compose.yml" in names
    assert all(a["predicted"] for a in arts)


def test_code_shape_predicts_one_artifact():
    p = build_response_plan("implement quicksort in python")
    assert p.shape == Shape.CODE
    assert p.artifacts and p.artifacts[0]["kind"] == "code"


def test_explicit_shape_override():
    p = build_response_plan("anything", shape="diagram")
    assert p.shape == Shape.DIAGRAM
    assert "Diagram" in p.outline()


def test_build_never_raises_on_garbage():
    p = build_response_plan(None, shape="not-a-shape")  # type: ignore[arg-type]
    assert p.sections  # degraded to a minimal plan
