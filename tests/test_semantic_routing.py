"""Embedding-clustered learned routing (Phase 2)."""
from __future__ import annotations

from app.llm import semantic_routing as sr


def setup_function():
    sr._reset_for_test()


# orthogonal unit vectors → distinct clusters
_C1 = [1.0, 0.0, 0.0]
_C2 = [0.0, 1.0, 0.0]


def test_neutral_with_no_data():
    assert sr.success_for(_C1, "modelA") == 0.5


def test_records_and_scores_per_cluster_per_model():
    for _ in range(5):
        sr.record(_C1, "modelA", True)
    for _ in range(5):
        sr.record(_C1, "modelB", False)
    # Laplace: A 5/5 → 6/7 ≈ .857 ; B 0/5 → 1/7 ≈ .143
    assert sr.success_for(_C1, "modelA") > 0.8
    assert sr.success_for(_C1, "modelB") < 0.2


def test_separate_clusters_dont_mix():
    for _ in range(4):
        sr.record(_C1, "modelA", True)
    for _ in range(4):
        sr.record(_C2, "modelA", False)
    # modelA is good on C1, bad on C2 — the two live in different clusters
    assert sr.success_for(_C1, "modelA") > 0.7
    assert sr.success_for(_C2, "modelA") < 0.3
    assert sr.stats()["clusters"] == 2


def test_near_vectors_join_same_cluster():
    sr.record([1.0, 0.0, 0.0], "m", True)
    # a very close vector (cosine > join threshold) joins the same cluster
    sr.record([0.98, 0.02, 0.0], "m", True)
    assert sr.stats()["clusters"] == 1


def test_unseen_model_in_known_cluster_is_neutral():
    for _ in range(6):
        sr.record(_C1, "modelA", True)
    assert sr.success_for(_C1, "modelZ") == 0.5      # no bias for a new model


def test_cluster_count_is_bounded():
    import math
    # feed many near-orthogonal vectors; cluster count must cap
    for i in range(sr._MAX_CLUSTERS + 20):
        ang = i * 0.3
        v = [math.cos(ang), math.sin(ang), 0.0]
        sr.record(v, "m", True)
    assert sr.stats()["clusters"] <= sr._MAX_CLUSTERS


def test_record_fail_open_on_bad_input():
    sr.record(None, "m", True)          # no embedding
    sr.record([1, 0, 0], None, True)    # no model
    assert sr.stats()["observations"] == 0


def test_success_for_fail_open():
    assert sr.success_for(None, "m") == 0.5
    assert sr.success_for([1, 0, 0], None) == 0.5


def test_clear():
    sr.record(_C1, "m", True)
    sr.clear(persist=False)
    assert sr.stats()["clusters"] == 0


def test_feedback_folds_quality_into_cluster():
    # a turn answered by modelA on C1, cached; a later 👎 records a failure
    sr.remember_turn("ep1", "modelA", _C1)
    assert sr.record_feedback("ep1", positive=False) is True
    # one failure recorded → Laplace (0+1)/(1+2) = 0.333 (below neutral)
    assert sr.success_for(_C1, "modelA") < 0.5


def test_feedback_unknown_episode_is_false():
    assert sr.record_feedback("nope", positive=True) is False
