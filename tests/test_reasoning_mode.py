"""User-controlled reasoning modes — Fast / Balanced / Thorough (P5 #26).

Pins that the mode is honoured by BOTH the router (via effective difficulty band
shift) and the governor (via the effective quality budget), and that BALANCED
(the default) is byte-identical to today.
"""
from __future__ import annotations

import pytest

from app.llm import reasoning_mode as rm
from app.quality.governor import Budgets, DEEP, FAST, select_pipeline


@pytest.fixture(autouse=True)
def _reset_mode():
    rm.reset()
    yield
    rm.reset()


# ── mode plumbing ────────────────────────────────────────────────────────────
def test_default_is_balanced():
    assert rm.current_mode() == rm.BALANCED
    assert rm.effective_difficulty("standard") == "standard"


def test_set_mode_normalises_invalid():
    assert rm.set_mode("nonsense") == rm.BALANCED
    assert rm.set_mode("THOROUGH") == rm.THOROUGH


def test_from_signals():
    assert rm.from_signals(depth="tldr") == rm.FAST
    assert rm.from_signals(depth="exhaustive") == rm.THOROUGH
    assert rm.from_signals(difficulty="expert") == rm.THOROUGH
    assert rm.from_signals(difficulty="standard") == rm.BALANCED


# ── router honours the mode (difficulty band shift) ──────────────────────────
def test_fast_shifts_difficulty_down():
    rm.set_mode(rm.FAST)
    assert rm.effective_difficulty("standard") == "trivial"
    assert rm.effective_difficulty("expert") == "hard"


def test_thorough_shifts_difficulty_up():
    rm.set_mode(rm.THOROUGH)
    assert rm.effective_difficulty("standard") == "hard"
    assert rm.effective_difficulty("trivial") == "standard"


def test_band_shift_clamps_at_extremes():
    rm.set_mode(rm.FAST)
    assert rm.effective_difficulty("trivial") == "trivial"      # can't go lower
    rm.set_mode(rm.THOROUGH)
    assert rm.effective_difficulty("expert") == "expert"        # can't go higher


def test_engine_helper_applies_mode():
    from app.llm.engine import _apply_reasoning_mode
    rm.set_mode(rm.THOROUGH)
    assert _apply_reasoning_mode("standard") == "hard"
    rm.reset()
    assert _apply_reasoning_mode("standard") == "standard"


# ── governor honours the mode via quality_budget → Budgets.quality ───────────
# The quality package must not import llm (import-boundary guardrail), so the
# mode reaches the governor as a Budgets.quality string built by the caller.
def test_quality_budget_maps_mode_to_governor_string():
    rm.set_mode(rm.THOROUGH)
    assert rm.quality_budget() == "thorough"
    rm.set_mode(rm.FAST)
    assert rm.quality_budget() == "fast"
    rm.set_mode(rm.BALANCED)
    assert rm.quality_budget() == "balanced"


def test_governor_honours_thorough_budget_from_mode():
    rm.set_mode(rm.THOROUGH)
    assert select_pipeline("trivial", Budgets(quality=rm.quality_budget())).kind == DEEP


def test_governor_honours_fast_budget_from_mode():
    rm.set_mode(rm.FAST)
    assert select_pipeline("standard", Budgets(quality=rm.quality_budget())).kind == FAST


def test_balanced_is_todays_behaviour():
    rm.set_mode(rm.BALANCED)
    q = rm.quality_budget()
    assert select_pipeline("standard", Budgets(quality=q)).kind == DEEP
    assert select_pipeline("trivial", Budgets(quality=q)).kind == FAST
