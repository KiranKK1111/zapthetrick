"""Unified Understanding pass — the semantic brain (Phase 1a)."""
from __future__ import annotations

import numpy as np

from app.understanding import understanding_pass as up


def _embed(mapping, dims=6):
    """Deterministic embedder: axis per keyword group, unit-normalized."""
    def embed(texts):
        out = []
        for t in texts:
            v = np.zeros(dims)
            for axis, keys in mapping.items():
                if any(k in t for k in keys):
                    v[axis] = 1.0
            if v.sum() == 0:
                v[dims - 1] = 1.0
            n = float(np.linalg.norm(v)) or 1.0
            out.append((v / n).tolist())
        return out
    return embed


# task-category exemplars: coding=axis0, writing=axis1, others→fallback axis
_MAP = {
    0: ["function", "bug", "code", "refactor", "debug", "implement", "tests"],
    1: ["resume", "email", "persuasive", "tone", "cover letter", "write a",
        "professional"],
    2: ["prove", "distributed", "production", "end to end", "architect",
        "consensus", "fault-tolerant"],
}


def test_empty_text_returns_default():
    u = up.understand("", embed_fn=_embed(_MAP))
    assert u.intent == "general" and u.embedding is None


def test_coding_turn_gets_code_capability():
    u = up.understand("fix this bug in my code", embed_fn=_embed(_MAP))
    assert u.task_category == "coding"
    assert "code" in u.capabilities
    assert u.source == "semantic"
    assert u.embedding is not None


def test_writing_turn_gets_writing_capability():
    u = up.understand("write a persuasive professional resume",
                      embed_fn=_embed(_MAP))
    assert u.task_category == "writing"
    assert "writing" in u.capabilities


def test_implicit_topic_shift_without_keyword():
    prev = _embed(_MAP)(["fix this bug in my code"])[0]      # coding axis
    # a writing turn is orthogonal to the coding axis → cosine 0 < 0.35
    u = up.understand("write a persuasive resume", prev_embedding=prev,
                      embed_fn=_embed(_MAP))
    assert u.topic_shift is True         # detected with NO "new topic" phrase


def test_same_topic_no_shift():
    prev = _embed(_MAP)(["fix this bug in my code"])[0]
    u = up.understand("add tests for this function", prev_embedding=prev,
                      embed_fn=_embed(_MAP))
    assert u.topic_shift is False        # same axis → high similarity


def test_explicit_shift_phrase_always_flags():
    # even with no prev embedding, an explicit cue flags a shift
    u = up.understand("new topic: let's talk about something else",
                      embed_fn=_embed(_MAP))
    assert u.topic_shift is True


def test_difficulty_expert_exemplar():
    # Map an axis to terms that appear ONLY in the expert exemplars (optimal /
    # correctness / derive / consensus / multi-region), so the query lands on
    # expert, not the overlapping 'hard' set.
    m = {0: ["optimal", "correctness", "derive", "consensus", "multi-region",
             "fault-tolerant"]}
    u = up.understand("derive the optimal correctness proof for consensus",
                      embed_fn=_embed(m))
    assert u.difficulty == "expert"
    assert u.output_complexity == "large"    # expert → large


def test_vision_and_json_capabilities():
    u = up.understand("what is in this screenshot", has_image=True,
                      needs_json=True, embed_fn=_embed(_MAP))
    assert "vision" in u.capabilities and "json" in u.capabilities


def test_as_meta_is_json_safe_and_excludes_embedding():
    u = up.understand("fix this bug", embed_fn=_embed(_MAP))
    meta = u.as_meta()
    assert "embedding" not in meta
    assert meta["task_category"] == "coding"
    assert isinstance(meta["capabilities"], list)


def test_degrades_fail_open_on_embed_error():
    def boom(_texts):
        raise RuntimeError("embedder down")
    u = up.understand("new topic: rust", embed_fn=boom)
    assert u.source == "degraded"
    assert u.topic_shift is True          # explicit cue still works degraded
    assert u.embedding is None


def test_research_turn_sets_needs_fresh_and_web_capability():
    # a turn landing on the research axis → needs_fresh + web capability (G6)
    m = {0: ["latest", "recent", "current", "look up", "what happened",
             "news about"]}
    u = up.understand("what's the latest news about the release",
                      embed_fn=_embed(m))
    assert u.task_category == "research"
    assert u.needs_fresh is True
    assert "web" in u.capabilities
    assert u.as_meta()["needs_fresh"] is True


def test_non_research_turn_not_fresh():
    u = up.understand("fix this bug in my code", embed_fn=_embed(_MAP))
    assert u.needs_fresh is False
    assert "web" not in u.capabilities


def test_image_caption_folded_into_understanding():
    # G10: the caption carries the "code" signal; the text alone is vague
    seen: list = []

    def embed(texts):
        seen.extend(texts)
        return _embed(_MAP)(texts)

    u = up.understand("what's wrong here?", caption="a python function with a bug",
                      embed_fn=embed)
    # the query embed folds text + caption
    assert any("python function with a bug" in s for s in seen)
    assert u.task_category == "coding"                    # picture drove the read
    assert "vision" in u.capabilities                     # caption ⇒ has_image


def test_caption_only_turn_is_understood():
    u = up.understand("", caption="a bug in the code function",
                      embed_fn=_embed(_MAP))
    assert u.embedding is not None and u.task_category == "coding"


def test_ambiguity_higher_for_short_vague_turn():
    long = up.understand("write a persuasive professional resume for me",
                         embed_fn=_embed(_MAP))
    short = up.understand("do it", embed_fn=_embed(_MAP))
    assert short.ambiguity >= long.ambiguity
