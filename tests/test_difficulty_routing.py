"""Tests for capability-aware routing — the difficulty score (router) and the
difficulty classifier's pure paths (app/chat/difficulty)."""
from __future__ import annotations

import pytest

from app.chat.difficulty import LEVELS, rigor_directive


# --- typed difficulty constants + validation (config-dynamic override) -----

def test_difficulty_constants_match_levels():
    from app.chat.difficulty import TRIVIAL, STANDARD, HARD, EXPERT, LEVELS as _L
    assert (TRIVIAL, STANDARD, HARD, EXPERT) == _L
    assert _L == ("trivial", "standard", "hard", "expert")


def test_is_level_validates_ui_override():
    from app.chat.difficulty import is_level
    for good in ("trivial", "standard", "hard", "expert", "EXPERT", " Hard "):
        assert is_level(good) is True, good
    for bad in ("", "supereasy", "medium", None, 3):
        assert is_level(bad) is False, bad


def test_request_schema_accepts_difficulty_override():
    from app.schemas import AgentsStreamRequest
    r = AgentsStreamRequest(message="hi", difficulty="expert")
    assert r.difficulty == "expert"
    assert AgentsStreamRequest(message="hi").difficulty is None  # auto by default


# --- vision capability detection (no hardcoded model list) -----------------

def test_detect_vision_by_id_and_metadata():
    from app.llm.catalog import detect_vision, is_vision_model_id
    # Multimodal families across providers → True.
    for mid in ("qwen/qwen2.5-vl-72b-instruct", "openai/gpt-4o",
                "google/gemini-2.0-flash", "meta-llama/llama-4-maverick",
                "mistralai/pixtral-12b", "anthropic/claude-sonnet-4"):
        assert is_vision_model_id(mid), mid
    # Text-only models → False.
    for mid in ("openai/gpt-oss-120b", "deepseek/deepseek-chat",
                "qwen/qwen3-coder", "moonshotai/kimi-k2"):
        assert not is_vision_model_id(mid), mid
    # Provider metadata wins even when the id is opaque.
    assert detect_vision("acme/x1", {"architecture": {"input_modalities":
                                                       ["text", "image"]}})
    assert not detect_vision("acme/x1", {"architecture":
                                         {"input_modalities": ["text"]}})


# --- routing score: difficulty must pull toward the right model -----------

def _score(order, intel, speed, difficulty, *, penalty=0, headroom=1.0):
    # `order` is accepted for call-site compatibility but no longer part of the
    # score — difficulty-weighted intelligence/speed decide; order is only a
    # stable tiebreak applied at sort time.
    from app.llm.router import _candidate_score
    return _candidate_score(penalty, headroom, intel, speed, difficulty)


def test_hard_task_prefers_stronger_model_over_manual_order():
    # Weak model (intel_rank 10) sits FIRST in the manual order; strong model
    # (intel_rank 1) sits 5th. On a HARD task the strong model must still win
    # (lower score), i.e. intelligence overrides the manual order.
    weak_first = _score(order=0, intel=10, speed=1, difficulty="hard")
    strong_fifth = _score(order=5, intel=1, speed=9, difficulty="hard")
    assert strong_fifth < weak_first


def test_expert_weights_intelligence_even_harder():
    weak = _score(order=0, intel=8, speed=1, difficulty="expert")
    strong = _score(order=3, intel=1, speed=9, difficulty="expert")
    assert strong < weak
    # and the gap is larger than at 'hard' (expert weights intelligence more)
    gap_expert = weak - strong
    gap_hard = (_score(order=0, intel=8, speed=1, difficulty="hard")
                - _score(order=3, intel=1, speed=9, difficulty="hard"))
    assert gap_expert > gap_hard


def test_trivial_task_prefers_faster_model():
    # On a trivial task, a faster model (speed_rank 1) beats a slower but
    # "smarter" one when manual order is equal.
    fast = _score(order=1, intel=10, speed=1, difficulty="trivial")
    slow_smart = _score(order=1, intel=1, speed=9, difficulty="trivial")
    assert fast < slow_smart


def test_standard_balances_capability_and_speed():
    # On a standard task we now weight BOTH capability and speed, so we don't
    # send normal Q&A to the slowest giant. Equal speed → stronger wins; but a
    # much faster model beats a marginally stronger, much slower one.
    strong = _score(order=0, intel=1, speed=5, difficulty="standard")
    weak = _score(order=0, intel=10, speed=5, difficulty="standard")
    assert strong < weak
    fast = _score(order=0, intel=6, speed=1, difficulty="standard")
    slow_strong = _score(order=0, intel=4, speed=12, difficulty="standard")
    assert fast < slow_strong


def test_trivial_ignores_capability_picks_fastest():
    # Trivial weights speed only — the fastest model wins even if much weaker.
    fast_weak = _score(order=0, intel=40, speed=1, difficulty="trivial")
    slow_strong = _score(order=0, intel=1, speed=10, difficulty="trivial")
    assert fast_weak < slow_strong


def test_availability_still_matters():
    # A strong model that's rate-limited (headroom 0) loses to an available
    # weaker one even on a hard task — we don't route into a wall.
    strong_throttled = _score(order=0, intel=1, speed=5, difficulty="hard",
                              headroom=0.0)   # +20 headroom +4 intel = 24
    decent_available = _score(order=0, intel=4, speed=5, difficulty="hard",
                              headroom=1.0)   # +16 intel = 16
    assert decent_available < strong_throttled  # don't route into a rate-limit wall


# --- classifier helpers ---------------------------------------------------

def test_rigor_directive_only_for_demanding():
    assert rigor_directive("hard")
    assert rigor_directive("expert")
    assert rigor_directive("standard") == ""
    assert rigor_directive("trivial") == ""


def test_classify_difficulty_empty_is_standard():
    # Empty input short-circuits to 'standard' with no LLM call → run inline.
    import asyncio

    from app.chat.difficulty import classify_difficulty
    assert asyncio.run(classify_difficulty("")) == "standard"
    assert asyncio.run(classify_difficulty("   ")) == "standard"


def test_levels_complete():
    assert LEVELS == ("trivial", "standard", "hard", "expert")


# --- build-request detection (clarification gate) --------------------------

class TestBuildRequestDetection:
    """`is_ambiguous_build_request` decides whether the route forces a build
    clarification. A self-contained code task that names its language must NOT
    be treated as an ambiguous project build (regression: 'reverse a string in
    Java using streams' was wrongly asked 'which programming language?')."""

    def test_specified_language_snippet_is_not_ambiguous(self):
        from app.chat.difficulty import is_ambiguous_build_request
        # The exact reported prompt: language (Java) + technique (streams) given.
        assert not is_ambiguous_build_request(
            "i want a program for reversing a given string in java using streams"
        )

    def test_snippet_without_language_is_still_not_a_project_build(self):
        from app.chat.difficulty import is_ambiguous_build_request
        # 'a program to reverse a string' is a single-file answer, not a project
        # — even with no language we don't force a build clarification here.
        assert not is_ambiguous_build_request(
            "write a program to reverse a string")
        assert not is_ambiguous_build_request("give me a script to sort a list")

    def test_open_ended_project_without_tech_is_ambiguous(self):
        from app.chat.difficulty import is_ambiguous_build_request
        # Genuine open-ended builds with no language → SHOULD clarify.
        assert is_ambiguous_build_request("build me a web app")
        assert is_ambiguous_build_request("create a todo application")
        assert is_ambiguous_build_request("build a website for my business")

    def test_project_with_named_tech_is_not_ambiguous(self):
        from app.chat.difficulty import is_ambiguous_build_request
        # Language/framework named anywhere → not ambiguous.
        assert not is_ambiguous_build_request(
            "build me a web app in React with a Node backend")
        assert not is_ambiguous_build_request("create a Flutter todo app")
        # Or named in the recent window (follow-up).
        assert not is_ambiguous_build_request(
            "build the dashboard", recent="we're using Django and Postgres")

    def test_java_and_other_languages_are_recognized(self):
        from app.chat.difficulty import _TECH_RE
        for lang in ("java", "scala", "dart", "kotlin", "swift", "go", "rust",
                     "sql", "bash", "c#", "c++"):
            assert _TECH_RE.search(lang), lang
        # 'javascript' must not be matched by a bare 'java' token rule.
        assert _TECH_RE.search("javascript")

    def test_is_build_request_scoped_to_projects(self):
        from app.chat.difficulty import is_build_request
        assert is_build_request("build a web app")
        assert is_build_request("create a REST api service")
        # A one-off snippet is not a whole-project build.
        assert not is_build_request("write a program to reverse a string")
        assert not is_build_request("give me a function to sort an array")

