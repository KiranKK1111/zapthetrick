"""Phase 2 — Document Planner + Blueprint (goal + depth → section plan)."""
from __future__ import annotations

import pytest

from app.documents.planner import (
    Blueprint, Depth, DocGoal,
    detect_depth, detect_document_goal, plan_blueprint, plan_document,
)



@pytest.fixture(autouse=True)
def _deterministic_classifiers(monkeypatch):
    """Pin the DETERMINISTIC fallback. detect_audience / detect_document_goal /
    detect_project_type are SEMANTIC-first (gates.classify), which is warm in the
    full suite and returns valid-but-different classes than the regex these tests
    pin. The semantic mechanism is covered in test_semantic_gates; here we test
    the fallback, so disable the embedding classifier."""
    import app.semantics.gates as _g
    monkeypatch.setattr(_g, "classify", lambda *a, **k: None)

class TestGoalDetection:
    @pytest.mark.parametrize("text,goal", [
        ("prepare interview notes for the candidate", DocGoal.INTERVIEW_NOTES),
        ("write a design document for the payment service", DocGoal.TECHNICAL_DESIGN),
        ("create a system design for this", DocGoal.TECHNICAL_DESIGN),
        ("draft a client proposal", DocGoal.PROPOSAL),
        ("write a research paper on transformers", DocGoal.RESEARCH),
        ("a step-by-step guide to deploy this", DocGoal.HOW_TO),
        ("an executive summary for my manager", DocGoal.EXECUTIVE_REPORT),
        ("send this to the CTO", DocGoal.EXECUTIVE_REPORT),
        ("meeting minutes from today's sync", DocGoal.MEETING_MINUTES),
        ("explain how kafka works", DocGoal.GENERAL),
    ])
    def test_goal(self, text, goal):
        assert detect_document_goal(text) == goal

    def test_specific_beats_general(self):
        # "design document" (specific) must win over a generic phrasing.
        assert detect_document_goal(
            "please write a detailed design document") == DocGoal.TECHNICAL_DESIGN


class TestDepthDetection:
    @pytest.mark.parametrize("text,depth", [
        ("give me a quick overview", Depth.QUICK),
        ("a brief summary", Depth.QUICK),
        ("write a comprehensive guide", Depth.DETAILED),
        ("an in-depth analysis", Depth.DETAILED),
        ("generate a 60-page report", Depth.DETAILED),
        ("write a design doc", Depth.MEDIUM),
    ])
    def test_depth(self, text, depth):
        assert detect_depth(text) == depth

    def test_detailed_beats_quick(self):
        assert detect_depth("a detailed overview of the system") == Depth.DETAILED


class TestBlueprint:
    def test_technical_design_has_core_sections(self):
        bp = plan_blueprint(DocGoal.TECHNICAL_DESIGN, Depth.MEDIUM)
        titles = bp.titles()
        assert "Overview" in titles and "Architecture" in titles
        assert "Implementation" in titles

    def test_quick_drops_optional_sections(self):
        quick = plan_blueprint(DocGoal.TECHNICAL_DESIGN, Depth.QUICK)
        full = plan_blueprint(DocGoal.TECHNICAL_DESIGN, Depth.DETAILED)
        assert len(quick.sections) < len(full.sections)
        # QUICK keeps only required sections.
        assert all(s.required for s in quick.sections)
        # DETAILED keeps optional ones too.
        assert any(not s.required for s in full.sections)

    def test_detailed_estimates_more_pages(self):
        med = plan_blueprint(DocGoal.RESEARCH, Depth.MEDIUM)
        det = plan_blueprint(DocGoal.RESEARCH, Depth.DETAILED)
        assert det.est_pages > med.est_pages

    def test_general_goal_imposes_no_template(self):
        bp = plan_blueprint(DocGoal.GENERAL, Depth.MEDIUM)
        assert bp.sections == [] and bp.est_pages == 0

    def test_plan_document_end_to_end(self):
        bp = plan_document("write a comprehensive design document for the API")
        assert isinstance(bp, Blueprint)
        assert bp.goal == DocGoal.TECHNICAL_DESIGN and bp.depth == Depth.DETAILED
        assert "Security" in bp.titles()   # optional section kept at detailed

    def test_as_dict_shape(self):
        js = plan_document("prepare interview notes, briefly").as_dict()
        assert js["goal"] == "interview_notes" and js["depth"] == "quick"
        assert isinstance(js["sections"], list) and "est_pages" in js
