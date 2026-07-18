"""Contextual Retrieval (P3 #19): a doc/section header is prepended to a chunk
before embedding so orphaned lines carry what they're about."""
from __future__ import annotations

from app.rag import contextual


def test_context_header_assembles_from_parts():
    h = contextual.build_context_header(
        doc_title="Resume", section="Experience", doc_summary="")
    assert "Resume" in h and "Experience" in h


def test_contextualize_prepends_header():
    out = contextual.contextualize(
        "improved throughput by 40%", doc_title="Resume", section="Experience")
    assert out.startswith("[Resume — Experience]")
    assert "improved throughput by 40%" in out


def test_contextualize_noop_without_context():
    assert contextual.contextualize("bare chunk") == "bare chunk"


def test_contextualize_all_applies_doc_context():
    outs = contextual.contextualize_all(
        ["a", "b"], doc_title="ProjectX", doc_summary="a data pipeline")
    assert all(o.startswith("[ProjectX") for o in outs)
    assert len(outs) == 2
