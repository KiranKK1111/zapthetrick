"""Flags-on integration (gap G12): the composed brain path works end-to-end —
Understanding pass → semantic router signal → multi-model synthesis — with every
new flag enabled. Uses fakes for the model + embedder so it needs no provider.
"""
from __future__ import annotations

import asyncio

import numpy as np

from app.understanding import understanding_pass as up
from app.llm import semantic_routing as sr
from app.chat import synthesis as syn


def _run(coro):
    return asyncio.run(coro)


def _embed(mapping, dims=6):
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


# architecture-flavoured query lands on axis0 (its exemplars share these words)
_MAP = {0: ["design", "architecture", "schema", "structure", "system design",
            "high level"]}


def _all_flags_on(monkeypatch):
    from app.core import config_loader as cl

    class _SI:
        enabled = True
        primary_threshold = 0.5
        learn_exemplars = True
        negative_penalty = 0.15
        llm_disambiguation = False

    class _SYN:
        enabled = True
        max_sections = 5
        min_output_complexity = "large"
        self_eval = False
        max_concurrency = 3
        section_timeout_s = 90

    class _ROUT:
        semantic_learning = True
        learning_router = False
    monkeypatch.setattr(cl.cfg, "semantic_intent", _SI(), raising=False)
    monkeypatch.setattr(cl.cfg, "synthesis", _SYN(), raising=False)
    monkeypatch.setattr(cl.cfg, "routing", _ROUT(), raising=False)
    sr._reset_for_test()
    up.reset_cache()


def test_understanding_feeds_router_signal(monkeypatch):
    _all_flags_on(monkeypatch)
    embed = _embed(_MAP)
    # 1) the brain reads a composite design turn as hard + architecture + large
    u = up.understand("high level system design for a payments architecture",
                      embed_fn=embed)
    assert u.task_category == "architecture"
    assert u.embedding is not None

    # 2) its embedding keys the semantic router: record model outcomes, then the
    #    router's learned signal reflects which model worked on turns like this
    sr.record(u.embedding, "big-model", True)
    sr.record(u.embedding, "big-model", True)
    sr.record(u.embedding, "weak-model", False)
    assert sr.success_for(u.embedding, "big-model") > 0.6
    assert sr.success_for(u.embedding, "weak-model") < 0.4


def test_understanding_gates_and_drives_synthesis(monkeypatch):
    _all_flags_on(monkeypatch)
    embed = _embed(_MAP)
    u = up.understand("design the full architecture and write the summary",
                      embed_fn=embed)
    # a large/hard turn → synthesis applies
    u.output_complexity = "large"
    u.difficulty = "hard"
    assert syn.should_orchestrate(u.as_meta()) is True

    plan = ('{"sections": [{"title": "Arch", "prompt": "design", "task": '
            '"architecture"}, {"title": "Sum", "prompt": "summary", "task": '
            '"writing"}]}')
    routed: list = []

    async def complete(msgs, *, task_category=None, difficulty="hard"):
        routed.append(task_category)
        return plan if "lead author planning" in msgs[0]["content"] \
            else f"section[{task_category}]"

    sections = _run(syn.plan_and_run(
        "design the full architecture and write the summary",
        u.as_meta(), complete_fn=complete))
    assert sections and [s.task for s in sections] == ["architecture", "writing"]
    # each section was routed to its own task model
    assert "architecture" in routed and "writing" in routed

    async def stream_fn(msgs, *, task_category=None, difficulty="hard"):
        for tok in ["final ", "merged ", "doc"]:
            yield tok

    async def collect():
        return [c async for c in syn.synthesize_stream(
            "q", sections, stream_fn=stream_fn)]
    assert "".join(_run(collect())) == "final merged doc"


def test_simple_turn_skips_synthesis(monkeypatch):
    _all_flags_on(monkeypatch)
    u = up.understand("what is 2+2", embed_fn=_embed(_MAP))
    u.output_complexity = "small"
    u.difficulty = "standard"
    assert syn.should_orchestrate(u.as_meta()) is False
