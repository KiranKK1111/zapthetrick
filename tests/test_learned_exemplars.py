"""Learned intent exemplars — self-improving intent (roadmap #12 / Phase A)."""
from __future__ import annotations

import numpy as np

from app.clarify import intent_semantic as si
from app.clarify import learned_exemplars as le


def _enable(monkeypatch, *, negative_penalty=0.15):
    from app.core import config_loader as cl
    _npen = negative_penalty

    class _SI:
        enabled = True
        primary_threshold = 0.50
        learn_exemplars = True
        negative_penalty = _npen
    monkeypatch.setattr(cl.cfg, "semantic_intent", _SI(), raising=False)
    le._reset_for_test()
    si.reset_cache()


def _embed(mapping):
    """Deterministic embedder: a phrase's vector is the mapped axis for the first
    substring it contains, else an 'other' axis. Unit-normalized."""
    def embed(texts):
        out = []
        for t in texts:
            vec = None
            for key, v in mapping.items():
                if key in t:
                    vec = v
                    break
            a = np.array(vec if vec is not None else [0, 0, 0, 1], dtype="float32")
            n = float(np.linalg.norm(a)) or 1.0
            out.append((a / n).tolist())
        return out
    return embed


# ---- store: add / dedupe / cap / gating ----------------------------------

def test_add_and_positives_gated_by_enabled(monkeypatch):
    _enable(monkeypatch)
    assert le.add("knowledge", "what is a widget", persist=False) is True
    assert le.positives().get("knowledge") == ["what is a widget"]
    # disable → positives() hides everything (classifier sees seed only)
    monkeypatch.setattr("app.core.config_loader.cfg.semantic_intent.learn_exemplars",
                        False, raising=False)
    assert le.positives() == {}


def test_add_dedupes_case_insensitive(monkeypatch):
    _enable(monkeypatch)
    assert le.add("knowledge", "Explain This", persist=False) is True
    assert le.add("knowledge", "explain this", persist=False) is False
    assert len(le.positives()["knowledge"]) == 1


def test_add_rejects_too_short(monkeypatch):
    _enable(monkeypatch)
    assert le.add("knowledge", "hi", persist=False) is False


def test_add_caps_per_intent(monkeypatch):
    _enable(monkeypatch)
    for i in range(le.MAX_PER_INTENT + 10):
        le.add("knowledge", f"phrase number {i}", persist=False)
    assert len(le.positives()["knowledge"]) == le.MAX_PER_INTENT
    # oldest dropped, newest kept
    assert any(f"number {le.MAX_PER_INTENT + 9}" in p
               for p in le.positives()["knowledge"])


def test_negatives_bucket(monkeypatch):
    _enable(monkeypatch)
    le.add("code_generation", "explain this thing", negative=True, persist=False)
    assert le.negatives().get("code_generation") == ["explain this thing"]
    assert le.positives() == {}          # negative didn't land in positives


def test_version_bumps_and_clear(monkeypatch):
    _enable(monkeypatch)
    v0 = le.version()
    le.add("knowledge", "a new phrasing here", persist=False)
    assert le.version() > v0
    le.clear(persist=False)
    assert le.positives() == {} and le.negatives() == {}


# ---- classifier integration ----------------------------------------------

def test_learned_positive_wins(monkeypatch):
    _enable(monkeypatch)
    embed = _embed({"ZZQ": [1, 0, 0, 0]})   # only ZZQ-tagged phrases share an axis
    # Without a learned exemplar, the query matches no seed on that axis.
    le.add("code_generation", "ZZQ special builder phrase", persist=False)
    out = si.classify("please ZZQ this", embed_fn=embed)
    assert out is not None
    intent, sim = out
    assert intent == "code_generation"
    assert sim > 0.99                     # exact axis match with the learned one


def test_negative_penalty_demotes_score(monkeypatch):
    _enable(monkeypatch, negative_penalty=0.2)
    embed = _embed({"ZZQ": [1, 0, 0, 0]})
    le.add("code_generation", "ZZQ special builder phrase", persist=False)
    base = si.classify("please ZZQ this", embed_fn=embed)[1]
    # Now mark a ZZQ phrasing as a NEGATIVE for code_generation → penalize.
    le.add("code_generation", "ZZQ was misread", negative=True, persist=False)
    penalized = si.classify("please ZZQ this", embed_fn=embed)[1]
    assert penalized < base
    assert abs((base - penalized) - 0.2) < 1e-5     # weight * nsim(=1.0)


def test_learn_from_feedback_up_adds_positive(monkeypatch):
    _enable(monkeypatch)
    res = le.learn_from_feedback("up", "what does this loop do", "knowledge")
    assert res["added"] == [{"intent": "knowledge", "negative": False}]
    assert "what does this loop do" in le.positives()["knowledge"]


def test_learn_from_feedback_down_adds_negative(monkeypatch):
    _enable(monkeypatch)
    res = le.learn_from_feedback("down", "zip the project", "knowledge")
    assert res["added"] == [{"intent": "knowledge", "negative": True}]
    assert "zip the project" in le.negatives()["knowledge"]


def test_learn_from_feedback_down_with_correction(monkeypatch):
    _enable(monkeypatch)
    res = le.learn_from_feedback(
        "down", "zip the project", "knowledge", corrected_intent="archive")
    kinds = {(a["intent"], a["negative"]) for a in res["added"]}
    assert ("knowledge", True) in kinds        # demote the wrong intent
    assert ("archive", False) in kinds         # reinforce the right one
    assert "zip the project" in le.positives()["archive"]


def test_learn_from_feedback_noop_when_disabled(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr("app.core.config_loader.cfg.semantic_intent.learn_exemplars",
                        False, raising=False)
    assert le.learn_from_feedback("up", "q", "knowledge") == {"added": []}


def test_spaces_are_isolated_from_intent(monkeypatch):
    _enable(monkeypatch)
    le.add("hard", "prove this theorem", space="difficulty", persist=False)
    le.add("coding", "fix the bug", space="task", persist=False)
    assert le.positives(space="difficulty") == {"hard": ["prove this theorem"]}
    assert le.positives(space="task") == {"coding": ["fix the bug"]}
    assert le.positives() == {}              # intent space untouched


def test_learn_from_feedback_up_reinforces_all_three(monkeypatch):
    _enable(monkeypatch)
    res = le.learn_from_feedback(
        "up", "design a rate limiter", "design",
        difficulty="hard", task="architecture")
    kinds = {tuple(sorted(a.items())) for a in res["added"]}
    assert any(a.get("intent") == "design" for a in res["added"])
    assert "design a rate limiter" in le.positives(space="difficulty")["hard"]
    assert "design a rate limiter" in le.positives(space="task")["architecture"]


def test_learn_from_feedback_down_leaves_difficulty_task(monkeypatch):
    _enable(monkeypatch)
    # a 👎 is about answer quality, not classification → don't touch diff/task
    le.learn_from_feedback("down", "q here", "knowledge",
                           difficulty="hard", task="coding")
    assert le.positives(space="difficulty") == {}
    assert le.positives(space="task") == {}


def test_learn_from_feedback_noop_missing_fields(monkeypatch):
    _enable(monkeypatch)
    assert le.learn_from_feedback("up", None, "knowledge") == {"added": []}
    assert le.learn_from_feedback("up", "q", None) == {"added": []}


def test_disabled_learning_ignores_store(monkeypatch):
    _enable(monkeypatch)
    le.add("code_generation", "ZZQ special builder phrase", persist=False)
    # turn learning OFF: positives()/negatives() go empty, matrix = seed only
    monkeypatch.setattr("app.core.config_loader.cfg.semantic_intent.learn_exemplars",
                        False, raising=False)
    si.reset_cache()
    embed = _embed({"ZZQ": [1, 0, 0, 0]})
    intent, _ = si.classify("please ZZQ this", embed_fn=embed)
    assert intent != "code_generation"    # learned phrase not in the matrix
