"""Capability registry (intelligent-model-routing R1, task 1.3).

Pins Property 1: every model gets a usable profile (explicit or derived), the
derived scores track intelligence_rank + id specialty markers, task_match maps
to 0..1, and the existing rank/speed/vision fields keep their meaning.
"""
from __future__ import annotations

from app.llm.capabilities import (
    TASK_CATEGORIES, CapabilityProfile, profile_for, task_match,
)


class _Model:
    def __init__(self, model_id, intelligence_rank=None, speed_rank=None,
                 supports_vision=False, capability_json=None):
        self.model_id = model_id
        self.intelligence_rank = intelligence_rank
        self.speed_rank = speed_rank
        self.supports_vision = supports_vision
        self.capability_json = capability_json


def test_derived_profile_has_all_categories():
    p = profile_for(_Model("llama-3.3-70b", intelligence_rank=9, speed_rank=2))
    assert isinstance(p, CapabilityProfile) and p.derived
    for c in TASK_CATEGORIES:
        assert 0 <= p.score_for(c) <= 100


def test_stronger_rank_scores_higher():
    strong = profile_for(_Model("gemini-2.5-pro", intelligence_rank=1))
    weak = profile_for(_Model("tiny-7b", intelligence_rank=22))
    assert strong.score_for("general") > weak.score_for("general")


def test_coding_specialist_boosted():
    coder = profile_for(_Model("qwen3-coder:free", intelligence_rank=3))
    generalist = profile_for(_Model("llama-3.3-70b", intelligence_rank=3))
    assert coder.score_for("coding") >= generalist.score_for("coding")
    assert task_match(coder, "coding") >= task_match(generalist, "coding")


def test_explicit_capability_json_wins():
    p = profile_for(_Model(
        "custom-x", intelligence_rank=50,
        capability_json={"scores": {"coding": 95, "general": 60},
                         "supports_tools": True, "supports_json": True}))
    assert p.derived is False
    assert p.score_for("coding") == 95
    assert p.supports_tools and p.supports_json


def test_vision_flag_preserved_and_reflected():
    p = profile_for(_Model("qwen2.5-vl-72b", intelligence_rank=2,
                           supports_vision=True))
    assert p.supports_vision is True
    assert p.score_for("vision") >= 80


def test_task_match_bounds_and_unknown_category():
    p = profile_for(_Model("llama-3.3-70b", intelligence_rank=9))
    assert 0.0 <= task_match(p, "coding") <= 1.0
    assert task_match(p, "not_a_category") == 0.5     # neutral


def test_profile_for_failopen_on_garbage():
    p = profile_for(object())     # no attributes → neutral, never raises
    assert isinstance(p, CapabilityProfile)
    assert 0.0 <= task_match(p, "general") <= 1.0
