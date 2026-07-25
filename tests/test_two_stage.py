"""Stage-5 §2.4 — two-stage selection (MODEL→PROVIDER) + same-model failover.

`_two_stage_order` is the pure reordering the router applies when
`routing.two_stage` is on: group by canonical identity, rank identities by
provider-independent quality (Stage 1), order each identity's providers by the
transient score (Stage 2). The property that matters: a model's OTHER providers
come BEFORE a different model, so a provider drop fails over to the same model.
"""
from __future__ import annotations

import types

import app.llm.router as R
from app.llm.identity import canonicalize


def _cand(model_id, platform, *, cid=None, quality=0.0, score=0.0, order=0,
          mid=None):
    """A synthetic scored candidate (the shape route_request builds)."""
    m = types.SimpleNamespace(id=(mid if mid is not None else id(model_id) % 100000),
                              model_id=model_id, platform=platform)
    return {"model": m, "cid": cid or canonicalize(platform, model_id).key(),
            "quality": quality, "score": score, "order": order}


def _seq(order):
    return [(c["model"].platform, c["model"].model_id) for c in order]


class TestGrouping:
    def test_same_identity_providers_are_consecutive(self):
        # Llama-3.3-70b on groq + cerebras is ONE identity; qwen is another.
        a1 = _cand("llama-3.3-70b-versatile", "groq", quality=10, score=5)
        a2 = _cand("Llama-3.3-70B-Instruct", "cerebras", quality=10, score=9)
        b = _cand("qwen-2.5-72b", "nvidia", quality=20, score=1)
        order = R._two_stage_order([b, a2, a1])
        seq = _seq(order)
        # Identity A (better quality) first, and BOTH its providers before B.
        assert seq[0][0] in ("groq", "cerebras")
        assert seq[1][0] in ("groq", "cerebras")
        assert seq[2] == ("nvidia", "qwen-2.5-72b")

    def test_stage1_ranks_by_provider_independent_quality(self):
        # B's single provider has a BETTER transient score, but A is the better
        # MODEL (lower quality value) → A wins Stage 1 despite the worse score.
        a = _cand("llama-3.3-70b", "groq", quality=5, score=100)
        b = _cand("weak-7b", "openrouter", quality=50, score=1)
        assert _seq(R._two_stage_order([b, a]))[0][1] == "llama-3.3-70b"

    def test_stage2_orders_providers_by_score(self):
        # Within one identity, the cheaper-to-serve (lower score) provider first.
        a1 = _cand("llama-3.3-70b-versatile", "groq", quality=10, score=8)
        a2 = _cand("Llama-3.3-70B-Instruct", "cerebras", quality=10, score=3)
        assert _seq(R._two_stage_order([a1, a2]))[0][0] == "cerebras"


class TestFailoverProperty:
    def test_next_candidate_is_same_model_elsewhere(self):
        # The core §2.4 guarantee: after the top provider, the NEXT try is the
        # SAME canonical identity on another provider — not a different model.
        a1 = _cand("llama-3.3-70b-versatile", "groq", quality=10, score=2)
        a2 = _cand("Llama-3.3-70B-Instruct", "cerebras", quality=10, score=6)
        other = _cand("qwen-2.5-72b", "nvidia", quality=11, score=1)
        order = R._two_stage_order([a1, a2, other])
        assert order[0]["cid"] == order[1]["cid"]      # same model, failover
        assert order[2]["cid"] != order[0]["cid"]      # then a different model

    def test_never_drops_a_candidate(self):
        cands = [_cand(f"m{i}", "groq", quality=i, score=i, mid=i)
                 for i in range(5)]
        assert len(R._two_stage_order(cands)) == 5     # never-empty preserved


class TestLruTiebreak:
    def test_equal_score_providers_rotate_by_lru(self, monkeypatch):
        a1 = _cand("llama-3.3-70b-versatile", "groq", quality=10, score=5, mid=1)
        a2 = _cand("Llama-3.3-70B-Instruct", "cerebras", quality=10, score=5,
                   mid=2)
        # Mark provider 1 as used more recently → provider 2 should come first.
        monkeypatch.setattr(R, "_last_used", {1: 1000.0, 2: 10.0})
        assert _seq(R._two_stage_order([a1, a2]))[0][0] == "cerebras"


class TestByteIdenticalWhenOff:
    def test_helper_only_runs_when_flag_on(self):
        # Sanity: the flag default is off, so route_request never calls the
        # reorder — the pool order (equiv+rest) is unchanged. This asserts the
        # config default that guarantees byte-identical behaviour.
        from app.core.config_loader import cfg
        assert bool(getattr(cfg.routing, "two_stage", False)) is False
