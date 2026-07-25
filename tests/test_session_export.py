"""Tests for session exports (vNext §3.12, Stage 8 Component E)."""
from __future__ import annotations

import app.documents.session_export as SE

_CHAT = [
    {"role": "user", "content": "What is Kafka?", "topic": "Kafka"},
    {"role": "assistant", "content": "A distributed log.", "topic": "Kafka"},
    {"role": "user", "content": "Postgres indexes?", "topic": "Postgres"},
    {"role": "assistant", "content": "B-tree.", "topic": "Postgres",
     "citations": ["docs.pg/indexes"]},
]
_LIVE = [
    {"role": "assistant", "content": "Use a heap.", "question": "Top-K?"},
    {"role": "assistant", "content": "O(log n).", "question": "B-tree cost?"},
]


# ---- chat segmentation ----------------------------------------------------
def test_chat_segments_by_topic():
    segs = SE.segment_chat(_CHAT)
    assert [s.title for s in segs] == ["Kafka", "Postgres"]
    assert [len(s.turns) for s in segs] == [2, 2]


def test_chat_window_grouping_without_topics():
    turns = [{"role": "user", "content": f"m{i}"} for i in range(14)]
    segs = SE.segment_chat(turns, window=6)
    assert len(segs) == 3               # 6 + 6 + 2
    assert all(t in ("Part 1", "Part 2", "Part 3") for t in [s.title for s in segs])


def test_scope_filters_roles():
    segs = SE.segment_chat(_CHAT, scope=SE.ExportScope(include_roles=("assistant",)))
    assert sum(len(s.turns) for s in segs) == 2   # only the 2 assistant turns


def test_scope_filters_topics():
    segs = SE.segment_chat(_CHAT, scope=SE.ExportScope(topics=("Kafka",)))
    assert [s.title for s in segs] == ["Kafka"]


def test_scope_index_range():
    segs = SE.segment_chat(_CHAT, scope=SE.ExportScope(from_index=2))
    assert [s.title for s in segs] == ["Postgres"]


def test_segment_chat_empty():
    assert SE.segment_chat([]) == []


# ---- live segmentation ----------------------------------------------------
def test_live_segments_per_question():
    segs = SE.segment_live(_LIVE)
    assert [s.title for s in segs] == ["Top-K?", "B-tree cost?"]


def test_live_answer_without_question_attaches_to_prev():
    qa = [
        {"role": "assistant", "content": "Part 1", "question": "Q1"},
        {"role": "assistant", "content": "Part 2"},   # continuation, no question
    ]
    segs = SE.segment_live(qa)
    assert len(segs) == 1 and len(segs[0].turns) == 2


# ---- build + markdown -----------------------------------------------------
def test_build_chat_export():
    exp = SE.build_export(_CHAT, mode=SE.CHAT, title="My Chat", date="2026-07-23")
    assert exp.mode == SE.CHAT
    assert exp.cover["segments"] == 2
    assert exp.exec_summary == ""       # chat has no exec summary


def test_build_live_report_has_exec_summary():
    exp = SE.build_export(_LIVE, mode=SE.LIVE, title="Interview",
                          exec_summary="Strong on DSA.")
    assert exp.mode == SE.LIVE
    assert exp.exec_summary == "Strong on DSA."


def test_markdown_has_cover_toc_and_anchors():
    exp = SE.build_export(_CHAT, mode=SE.CHAT, title="My Chat", date="2026-07-23")
    md = SE.to_markdown(exp)
    assert md.startswith("# My Chat")
    assert "*2026-07-23*" in md
    assert "## Contents" in md
    assert "[Kafka](#kafka)" in md      # hyperlinked TOC → anchor slug
    assert "## Kafka" in md and "## Postgres" in md


def test_markdown_footnote_citations():
    exp = SE.build_export(_CHAT, mode=SE.CHAT, title="X")
    md = SE.to_markdown(exp)
    assert "[^1]" in md                 # a footnote marker on the cited turn
    assert "[^1]: docs.pg/indexes" in md  # the footnote definition


def test_markdown_live_has_exec_summary_section():
    exp = SE.build_export(_LIVE, mode=SE.LIVE, title="Interview",
                          exec_summary="Solid.")
    md = SE.to_markdown(exp)
    assert "## Executive summary" in md and "Solid." in md


def test_markdown_empty_is_safe():
    md = SE.to_markdown(SE.build_export([], mode=SE.CHAT))
    assert md.startswith("# Conversation Export")


# ---- DocumentModel emit ---------------------------------------------------
def test_document_model_doctype_and_sections():
    exp = SE.build_export(_CHAT, mode=SE.CHAT, title="T")
    dm = SE.to_document_model(exp)
    assert dm.metadata.doc_type == "session-export"
    assert dm.metadata.title == "T"
    headings = [s.heading for s in dm.sections if s.heading]
    assert "Kafka" in headings and "Postgres" in headings


def test_document_model_splits_code_and_mermaid():
    turns = [{"role": "assistant", "topic": "Code",
              "content": "Text.\n\n```python\nprint(1)\n```\n\n```mermaid\nflowchart TD\n A-->B\n```"}]
    dm = SE.to_document_model(SE.build_export(turns, mode=SE.CHAT, title="T"))
    kinds = [b.kind for s in dm.sections for b in s.blocks]
    assert "code" in kinds and "diagram" in kinds


def test_document_model_live_has_exec_section():
    dm = SE.to_document_model(SE.build_export(_LIVE, mode=SE.LIVE, title="I",
                                              exec_summary="Good."))
    assert any(s.heading == "Executive summary" for s in dm.sections)


# ---- robustness -----------------------------------------------------------
def test_duck_typed_object_turns():
    class T:
        role = "assistant"
        content = "hi"
        topic = "Greet"
        question = ""
        citations = None
    segs = SE.segment_chat([T()])
    assert segs and segs[0].title == "Greet"


def test_build_never_raises_on_garbage():
    exp = SE.build_export(None, mode=SE.CHAT)   # type: ignore[arg-type]
    assert isinstance(exp, SE.SessionExport)
