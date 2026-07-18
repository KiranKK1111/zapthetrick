"""Distillation-data exporter (roadmap Phase 5 #14).

Pins: teacher traces become a clean chat-fine-tune JSONL dataset; refusals /
errors / too-short / low-quality / duplicate prompts are filtered out.
"""
from __future__ import annotations

import json

from app.llm.distillation import DistillationExporter, Trace


def _good(prompt="Explain binary search in one paragraph with its complexity.",
          completion=None):
    return Trace(prompt=prompt,
                 completion=completion or ("Binary search repeatedly halves a "
                 "sorted range, giving O(log n) time by discarding half the "
                 "candidates each comparison."),
                 system="You are a helpful tutor.", quality=0.9, model="teacher-x")


def test_builds_chat_records():
    recs, stats = DistillationExporter().build([_good()])
    assert stats.kept == 1 and len(recs) == 1
    roles = [m["role"] for m in recs[0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert recs[0]["teacher"] == "teacher-x"


def test_filters_refusals_and_errors():
    traces = [
        _good(),
        Trace("q1", "I can't help with that.", quality=0.9),
        Trace("q2", "Traceback (most recent call last): boom", quality=0.9),
    ]
    recs, stats = DistillationExporter().build(traces)
    assert stats.kept == 1
    assert stats.dropped_refusal == 1 and stats.dropped_error == 1


def test_filters_short_and_low_quality():
    traces = [
        Trace("q", "too short", quality=0.9),                 # short
        Trace("q2", "A" * 100, quality=0.2),                  # low quality
    ]
    recs, stats = DistillationExporter().build(traces)
    assert stats.kept == 0
    assert stats.dropped_short == 1 and stats.dropped_lowq == 1


def test_dedupes_identical_prompts():
    recs, stats = DistillationExporter().build([_good(), _good()])
    assert stats.kept == 1 and stats.dropped_dup == 1


def test_jsonl_roundtrips():
    body, stats = DistillationExporter().to_jsonl([_good()])
    lines = [l for l in body.splitlines() if l.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["messages"][-1]["role"] == "assistant"


def test_write_jsonl(tmp_path):
    path = tmp_path / "distill.jsonl"
    stats = DistillationExporter().write_jsonl([_good()], str(path))
    assert stats.kept == 1
    assert path.read_text(encoding="utf-8").strip()


def test_failopen_on_bad_trace():
    class _Weird:
        prompt = "x"
        # missing completion attr access will raise inside try → skipped
        def __getattr__(self, k):
            raise RuntimeError("bad")
    recs, stats = DistillationExporter().build([_Weird(), _good()])
    assert stats.kept == 1               # good one survives
