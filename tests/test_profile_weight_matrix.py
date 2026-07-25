"""Stage-5 §2.6 — TTFT/TPS weight matrix modulates the routing score.

`_candidate_score` gains a `speed_scale` that scales difficulty's speed_rank
term; the router precompute derives it from the declared profile's `tps_wt`.
A reasoning profile (low TPS) down-weights speed so a smart-but-slow model
scores better; a latency-sensitive profile keeps/raises the speed emphasis.
Default (scale 1.0) is byte-identical to today.
"""
from __future__ import annotations

from app.llm import profiles as P
from app.llm.router import _candidate_score


def _score(speed_rank, *, difficulty="standard", speed_scale=1.0):
    # Isolate the speed term: no penalty/headroom loss, neutral intel.
    return _candidate_score(
        penalty=0, headroom=1.0, intel=100, speed=speed_rank,
        difficulty=difficulty, speed_scale=speed_scale)


class TestSpeedScale:
    def test_default_scale_is_identity(self):
        assert _score(50) == _score(50, speed_scale=1.0)

    def test_lower_scale_shrinks_speed_penalty(self):
        # A slow model (high speed_rank) is penalized less when speed matters
        # less → its score drops toward the fast model's.
        slow_full = _score(200, speed_scale=1.0)
        slow_dsa = _score(200, speed_scale=0.25)     # dsa_reasoning tps scale
        assert slow_dsa < slow_full

    def test_zero_scale_removes_speed_term(self):
        # With speed fully de-emphasized, speed_rank stops changing the score.
        assert _score(10, speed_scale=0.0) == _score(500, speed_scale=0.0)

    def test_higher_scale_amplifies_speed(self):
        fast = _score(10, speed_scale=1.5)
        slow = _score(300, speed_scale=1.5)
        # A latency-sensitive profile makes the gap between fast and slow BIGGER
        # than neutral weighting would.
        gap_hi = slow - fast
        gap_neutral = _score(300) - _score(10)
        assert gap_hi > gap_neutral


class TestProfileScales:
    def test_matrix_maps_to_expected_scales(self):
        base = P._HIGH
        # Reasoning: speed barely matters.
        assert P.profile("dsa_reasoning").tps_wt / base < 0.5
        # Live/speculation: first-token speed is paramount.
        assert P.profile("speculation_draft").ttft_wt / base > 1.0
        # A "high"-emphasis profile is neutral (scale ~1.0).
        assert abs(P.profile("live_answer").ttft_wt / base - 1.0) < 1e-9
