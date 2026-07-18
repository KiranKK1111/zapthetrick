"""Per-turn trace (Architecture.md §14)."""
from __future__ import annotations

from app.obs.trace import build_trace


def test_minimal_trace_has_id_and_omits_empties():
    t = build_trace(trace_id="abc")
    assert t["id"] == "abc"
    for k in ("tools", "graphs", "suggestions", "degraded", "stages",
              "model", "difficulty", "latency_ms"):
        assert k not in t  # empty/None omitted for compactness


def test_full_trace_records_facts_and_source_counts():
    t = build_trace(
        trace_id="t1", model="m", difficulty="hard", latency_ms=1200,
        tools=["retriever", "grounder"],
        memory_recalled=3, kg_neighbors=2, episodes=1,
        suggestions=[{"text": "a", "source": "profile"},
                     {"text": "b", "source": "profile"},
                     {"text": "c", "source": "memory_graph"}],
        degraded=["web"], stages={"retrieve": 210.4, "first_token": 900.0},
    )
    assert t["model"] == "m" and t["difficulty"] == "hard"
    assert t["tools"] == ["retriever", "grounder"]
    assert t["graphs"] == {"memory_recalled": 3, "kg_neighbors": 2, "episodes": 1}
    assert t["suggestions"] == {"profile": 2, "memory_graph": 1}
    assert t["degraded"] == ["web"]
    assert t["stages"]["retrieve"] == 210.4


def test_build_trace_never_raises_on_bad_input():
    t = build_trace(trace_id="x", suggestions=[None, "oops", {"no_source": 1}],
                    stages={"s": "not-a-number"})  # type: ignore[dict-item]
    assert t["id"] == "x"  # degrades gracefully
