"""Task-match scoring term (intelligent-model-routing R3, task 3.2).

Pins Property 2: with the task-match weight at 0 the score equals today's; a
positive weight lowers (favors) a better-fitting model; difficulty weights +
floor constants are untouched.
"""
from __future__ import annotations

from app.llm import router as R


def test_zero_weight_equals_today(monkeypatch):
    # Force both additive weights to 0 → score identical regardless of fit.
    monkeypatch.setattr(R, "_task_weight", lambda: 0.0)
    monkeypatch.setattr(R, "_learn_weight", lambda: 0.0)
    base = R._candidate_score(0, 1.0, 5, 5, "standard")
    perfect = R._candidate_score(0, 1.0, 5, 5, "standard", task_match=1.0)
    poor = R._candidate_score(0, 1.0, 5, 5, "standard", task_match=0.0)
    assert base == perfect == poor


def test_positive_weight_favors_better_fit(monkeypatch):
    monkeypatch.setattr(R, "_task_weight", lambda: 10.0)
    monkeypatch.setattr(R, "_learn_weight", lambda: 0.0)
    good = R._candidate_score(0, 1.0, 5, 5, "standard", task_match=1.0)
    bad = R._candidate_score(0, 1.0, 5, 5, "standard", task_match=0.0)
    assert good < bad          # lower score = picked first


def test_learned_term_additive(monkeypatch):
    monkeypatch.setattr(R, "_task_weight", lambda: 0.0)
    monkeypatch.setattr(R, "_learn_weight", lambda: 8.0)
    proven = R._candidate_score(0, 1.0, 5, 5, "standard", learned=1.0)
    unproven = R._candidate_score(0, 1.0, 5, 5, "standard", learned=0.0)
    assert proven < unproven


def test_difficulty_weights_and_floor_preserved():
    # The difficulty table + capability floor are untouched by this spec.
    assert R._DIFFICULTY["trivial"] == (0.0, 4.0)
    assert R._DIFFICULTY["expert"] == (8.0, 0.0)
    assert R._DIFFICULTY_FLOOR == {"hard": 18, "expert": 10}


def test_difficulty_still_dominates_with_default_weights(monkeypatch):
    # Default (off) weights: an expert turn still ranks the stronger model first.
    monkeypatch.setattr(R, "_task_weight", lambda: 0.0)
    monkeypatch.setattr(R, "_learn_weight", lambda: 0.0)
    strong = R._candidate_score(0, 1.0, 2, 9, "expert")
    weak = R._candidate_score(0, 1.0, 20, 1, "expert")
    assert strong < weak
