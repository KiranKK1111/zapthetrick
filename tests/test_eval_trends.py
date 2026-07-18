"""Self-benchmarking trend tracking (roadmap Phase 7 #10)."""
from __future__ import annotations

from app.eval import trends


def test_record_and_report_direction(tmp_path):
    p = tmp_path / "trends.jsonl"
    trends.record_point("m", 0.50, path=p)
    trends.record_point("m", 0.55, path=p)
    trends.record_point("m", 0.60, path=p)
    rep = trends.trend_report("m", path=p)
    assert rep["scored"] == 3
    assert rep["latest"] == 0.60
    assert rep["first"] == 0.50
    assert rep["delta_vs_previous"] == 0.05
    assert rep["delta_vs_first"] == 0.10
    assert rep["direction"] == "improving"


def test_regressing_direction(tmp_path):
    p = tmp_path / "t.jsonl"
    trends.record_point("m", 0.8, path=p)
    trends.record_point("m", 0.7, path=p)
    assert trends.trend_report("m", path=p)["direction"] == "regressing"


def test_empty_report_is_safe(tmp_path):
    rep = trends.trend_report("nope", path=tmp_path / "none.jsonl")
    assert rep["scored"] == 0


def test_log_is_bounded(tmp_path, monkeypatch):
    p = tmp_path / "t.jsonl"
    monkeypatch.setattr(trends, "_MAX_POINTS", 10)
    for i in range(30):
        trends.record_point("m", i / 100.0, path=p)
    rep = trends.trend_report("m", path=p)
    assert rep["points"] == 10          # ring bounded


def test_run_and_record_persists_a_point(tmp_path):
    """Runs the real leaderboard, records its overall score as a trend point."""
    p = tmp_path / "t.jsonl"
    rep = trends.run_and_record(path=p)
    assert rep["points"] >= 1
    # The point exists even if the leaderboard scored None (fail-open).
    assert "metric" in rep


def test_metric_filtering(tmp_path):
    p = tmp_path / "t.jsonl"
    trends.record_point("a", 0.5, path=p)
    trends.record_point("b", 0.9, path=p)
    assert trends.trend_report("a", path=p)["latest"] == 0.5
    assert trends.trend_report("b", path=p)["latest"] == 0.9
