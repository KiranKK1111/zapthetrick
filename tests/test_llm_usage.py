"""Real provider usage/finish_reason accounting (G6.1)."""
from __future__ import annotations

from app.llm import usage as U


def setup_function(_):
    U.reset()


def test_reads_recorded_usage_and_finish_reason():
    U.record({"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
             "stop")
    assert U.tokens() == (120, 45, 165)
    assert U.finish_reason() == "stop"


def test_missing_usage_returns_nones():
    U.reset()
    assert U.tokens() == (None, None, None)
    assert U.finish_reason() is None


def test_engine_prefers_real_total_over_estimate():
    from app.llm.engine import _real_total_tokens
    # No usage recorded → estimate (prompt est + chars//4).
    U.reset()
    assert _real_total_tokens(100, 400) == 100 + 100
    # Real total present → used verbatim.
    U.record({"total_tokens": 999}, "stop")
    assert _real_total_tokens(100, 400) == 999
    # Only prompt+completion present → summed.
    U.record({"prompt_tokens": 50, "completion_tokens": 20}, "length")
    assert _real_total_tokens(100, 400) == 70


def test_partial_usage_falls_back_per_field():
    U.record({"prompt_tokens": 80}, None)   # no completion / total
    from app.llm.engine import _real_total_tokens
    # prompt from usage (80) + completion estimated from chars (200//4=50)
    assert _real_total_tokens(999, 200) == 80 + 50
