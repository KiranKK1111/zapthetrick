"""Unit tests for the within-message size cap (app/chat/condense.py).

Pure-function tests — no DB, no LLM — so they run anywhere. They lock in the
contract the chat/agents/attachments routes rely on: an oversized message is
reduced (head + tail verbatim + an evenly-spaced, deduped, keyword-free sample of
the middle) under budget, while normal messages pass through untouched.
"""
from __future__ import annotations

import pytest

from app.chat.condense import MAX_MESSAGE_CHARS, condense_oversized


def _make_paste(n_lines: int) -> str:
    """A realistic noisy paste: mostly debug noise, some structural lines, a
    real instruction at the very end (mirrors the composer's paste-then-typed
    assembly order)."""
    out = []
    for i in range(n_lines):
        if i % 7 == 0:
            out.append("")
        elif i % 11 == 0:
            out.append(f"def handler_{i}(req): return process(req)")
        elif i % 13 == 0:
            out.append(f"ERROR: connection reset at node {i}")
        else:
            out.append(f'    log.debug("tick {i} ......................")')
    out.append("")
    out.append("Please find the bug causing the connection resets above.")
    return "\n".join(out)


# --- pass-through cases ---------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "hello", "a\nb\nc"])
def test_small_or_empty_passthrough(value):
    out, did = condense_oversized(value)
    assert out == value
    assert did is False


def test_exactly_at_cap_passthrough():
    text = "x" * MAX_MESSAGE_CHARS
    out, did = condense_oversized(text)
    assert did is False and out == text


# --- condensing cases -----------------------------------------------------

def test_oversized_is_condensed_under_budget():
    paste = _make_paste(200_000)
    assert len(paste) > MAX_MESSAGE_CHARS
    out, did = condense_oversized(paste)
    assert did is True
    # Soft cap: head + tail + middle + banner/markers. Allow a small banner
    # overhead but it must be a tiny fraction of the original.
    assert len(out) <= MAX_MESSAGE_CHARS + 2_000
    assert len(out) < len(paste) // 10


def test_head_and_tail_instruction_preserved():
    paste = _make_paste(50_000)
    out, _ = condense_oversized(paste)
    # The instruction sits at the very end — it must survive (tail kept).
    assert "Please find the bug causing the connection resets above." in out
    # The very first content line is in the head.
    assert "tick 1 " in out or "def handler_0" in out or "handler_11" in out


def test_exact_duplicates_are_collapsed():
    # Distinct head/tail (kept verbatim), a 200k-line repeated MIDDLE → the
    # keyword-free middle dedups the repeats to ~one sample.
    head = "\n".join(f"start line {i}" for i in range(70))
    tail = "\n".join(f"end line {i}" for i in range(70))
    body = head + "\n" + ("the same repeated middle line\n" * 200_000) + tail
    out, did = condense_oversized(body)
    assert did is True
    assert "start line 0" in out and "end line 69" in out
    # The repeated middle line survives only a handful of times — dedup
    # collapsed the 200k copies.
    assert out.count("the same repeated middle line") <= 3


def test_middle_sample_spans_the_body():
    # An evenly-spaced sample should include lines from both early and late in
    # the (deduped) middle, not just the front.
    out, _ = condense_oversized(_make_paste(100_000))
    import re
    nums = [int(m) for m in re.findall(r"tick (\d+)", out)]
    assert nums, "expected some sampled middle lines"
    assert max(nums) - min(nums) > 50_000  # coverage spans the body


def test_distinct_structures_surface_without_keywords():
    # Rare line STRUCTURES (a definition, an error) differ from the dominant
    # debug line only by structure, not by any keyword we look for — the
    # digit-masked template dedup should still surface one of each.
    out, _ = condense_oversized(_make_paste(100_000))
    assert "def handler_" in out
    assert "ERROR: connection reset at node" in out


def test_omission_markers_present():
    out, _ = condense_oversized(_make_paste(100_000))
    assert "omitted" in out


def test_banner_announces_condensation():
    out, _ = condense_oversized(_make_paste(20_000))
    assert out.startswith("[Note:")
    assert "condensed" in out.splitlines()[0]


def test_single_giant_line_is_truncated_not_dropped():
    out, did = condense_oversized("y" * 5_000_000)
    assert did is True
    assert len(out) < 10_000           # truncated hard
    assert "chars]" in out             # truncation marker present


def test_blank_lines_not_kept_in_middle():
    # A body that is head + many blanks + tail; the blank middle must collapse.
    body = "HEADER\n" + ("\n" * 200_000) + "FOOTER QUESTION?"
    out, did = condense_oversized(body)
    assert did is True
    assert "HEADER" in out and "FOOTER QUESTION?" in out
    # The middle blanks shouldn't dominate — output stays tiny.
    assert len(out) < 5_000


def test_idempotent_stays_under_budget():
    once, _ = condense_oversized(_make_paste(300_000))
    twice, did2 = condense_oversized(once)
    # Already within budget after the first pass → second pass is a no-op.
    assert did2 is False and twice == once


def test_custom_budget_respected():
    out, did = condense_oversized(_make_paste(50_000), max_chars=5_000)
    assert did is True
    assert len(out) <= 5_000 + 2_000
