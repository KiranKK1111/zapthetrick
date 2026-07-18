"""Tests for the streaming token coalescer (smooth-streaming-rendering R16)."""
from __future__ import annotations

import pytest

_mod = pytest.importorskip("app.chat.stream_coalesce")
TokenCoalescer = _mod.TokenCoalescer
coalesce_tokens = _mod.coalesce_tokens


def test_threshold_zero_is_passthrough():
    toks = ["a", "b", "cd", "e"]
    assert coalesce_tokens(toks, 0) == toks


def test_negative_threshold_is_passthrough():
    assert coalesce_tokens(["x", "y"], -5) == ["x", "y"]


def test_coalesces_until_threshold():
    # threshold 4: "ab"+"cd" = 4 → emit "abcd"; then "ef" buffered → flush "ef"
    assert coalesce_tokens(["ab", "cd", "ef"], 4) == ["abcd", "ef"]


def test_preserves_total_text():
    toks = ["The ", "quick ", "brown ", "fox ", "jumps"]
    for threshold in (0, 1, 3, 8, 100):
        assert "".join(coalesce_tokens(toks, threshold)) == "".join(toks)


def test_flush_returns_remainder_then_none():
    c = TokenCoalescer(10)
    assert c.push("hi") is None          # buffered (2 < 10)
    assert c.flush() == "hi"
    assert c.flush() is None             # nothing left


def test_empty_tokens_ignored():
    c = TokenCoalescer(0)
    assert c.push("") is None
    c2 = TokenCoalescer(5)
    assert c2.push("") is None
    assert c2.flush() is None


def test_single_token_over_threshold_emits_immediately():
    assert coalesce_tokens(["abcdefgh"], 4) == ["abcdefgh"]


# ---- adaptive chunk sizing (R46) -----------------------------------------

effective_threshold = _mod.effective_threshold


def test_effective_threshold_off_stays_off():
    assert effective_threshold(0, 0.9) == 0


def test_effective_threshold_no_load_is_base():
    assert effective_threshold(24, None) == 24
    assert effective_threshold(24, 0) == 24


def test_effective_threshold_scales_with_load():
    # full load → base * (1 + 1*3) = 4x
    assert effective_threshold(24, 1.0) == 96
    # half load → base * 2.5
    assert effective_threshold(24, 0.5) == 60


def test_effective_threshold_clamps_and_handles_bad_input():
    assert effective_threshold(10, 5.0) == 40   # clamped to 1.0
    assert effective_threshold(10, "x") == 10   # bad input → base
