"""Phase 2 — staged multi-pass document assembler."""
from __future__ import annotations

import asyncio

from app.documents.assembler import AssemblyResult, assemble_document


_MD = ("# System Design Document\n\nIntro.\n\n"
       "## Overview\n\nThe service.\n\n"
       "## Architecture\n\nUser -> Gateway -> Service\n\n"
       "## Components\n\nUses Kafka.\n")


def _run(**kw):
    return asyncio.run(assemble_document(_MD, **kw))


class TestPipeline:
    def test_all_named_passes_run(self):
        res = _run(request_text="write a design document", title="Design")
        names = [p["pass"] for p in res.passes]
        assert names == ["outline", "content", "structure", "format", "validate"]
        assert all(p["ok"] for p in res.passes)

    def test_returns_assembly_result(self):
        res = _run(request_text="design document")
        assert isinstance(res, AssemblyResult)
        assert res.blueprint is not None
        assert res.quality is not None
        assert res.markdown.strip()

    def test_outline_detects_technical_design_goal(self):
        res = _run(request_text="a technical design document for the system")
        assert res.blueprint.goal.value == "technical_design"

    def test_structure_pass_enriches(self):
        # enrich_structure=True re-serializes an enriched model (TOC, glossary).
        res = _run(request_text="design doc")
        assert "Table of Contents" in res.markdown or "Glossary" in res.markdown

    def test_skip_structure_leaves_content(self):
        res = _run(request_text="design doc", enrich_structure=False)
        names = [p["pass"] for p in res.passes]
        assert "structure" not in names and "format" not in names
        assert res.markdown == _MD

    def test_blueprint_completeness_flows_into_quality(self):
        # A technical-design doc missing 'Implementation' → the blueprint-scored
        # validator flags a missing_section (only possible because the assembler
        # passes its blueprint to the reviewer).
        res = _run(request_text="a technical design document")
        cats = {i.category for i in res.quality.issues}
        assert "missing_section" in cats

    def test_as_dict_is_json_shaped(self):
        d = _run(request_text="a technical design document").as_dict()
        assert set(d) == {"passes", "blueprint", "quality"}
        assert d["blueprint"]["goal"] == "technical_design"


class TestFailOpen:
    def test_empty_content_does_not_raise(self):
        res = asyncio.run(assemble_document("", request_text=""))
        assert isinstance(res, AssemblyResult)
        # content pass still runs on an empty doc.
        assert any(p["pass"] == "validate" for p in res.passes)
