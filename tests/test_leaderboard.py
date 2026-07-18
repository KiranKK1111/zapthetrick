"""Unified benchmark leaderboard (P1 #1) — aggregates the per-subsystem suites."""
from __future__ import annotations

from app.eval import leaderboard


def test_extract_score_from_shapes():
    assert leaderboard._extract_score({"accuracy": 0.9}) == 0.9
    assert leaderboard._extract_score({"overall": {"pass_rate": 0.75}}) == 0.75
    assert leaderboard._extract_score({"nope": 1}) is None


def test_run_all_returns_scorecard():
    board = leaderboard.run_all()
    assert "suites" in board and "overall" in board
    # The two pure suites (no external deps) always run and score.
    assert "live_scenarios" in board["suites"]
    assert "live_synth" in board["suites"]
    assert board["ran"] >= 2
    assert isinstance(board["scored"], int)
    # A scored board yields a 0..1 aggregate.
    if board["overall"] is not None:
        assert 0.0 <= board["overall"] <= 1.0


def test_one_bad_suite_does_not_sink_the_board():
    board = leaderboard.run_all(embed_fn=lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("boom")))
    # Even if intent errors, pure suites still produced scores.
    assert board["scored"] >= 2
