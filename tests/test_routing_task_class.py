"""Prompt-type classifier (intelligent-model-routing R2, task 2.2).

Pins Property 3: intent + lexical cues map to a Task_Category, unknown → general,
deterministic, and no second LLM call (the function is plain + synchronous).
"""
from __future__ import annotations

import inspect

from app.llm.task_class import classify_task
from app.llm.capabilities import TASK_CATEGORIES


def test_intent_maps_to_category():
    assert classify_task("", intent="code_generation") == "coding"
    assert classify_task("", intent="design") == "architecture"
    assert classify_task("", intent="documentation") == "writing"
    assert classify_task("", intent="chitchat") == "conversation"
    assert classify_task("", intent="project_build") == "agentic"


def test_lexical_cues_when_intent_generic():
    assert classify_task("write a python function to sort a list") == "coding"
    assert classify_task("solve this integral and prove the theorem") == "math"
    assert classify_task("design a scalable microservice architecture") == "architecture"
    assert classify_task("write a blog article about cats") == "writing"


def test_unknown_defaults_to_general():
    assert classify_task("hello there") == "general"
    assert classify_task("") == "general"


def test_hard_no_cue_leans_reasoning():
    assert classify_task("ponder this deeply", difficulty="expert") == "reasoning"


def test_result_is_in_category_set_and_deterministic():
    text = "refactor this api endpoint and add unit tests"
    a = classify_task(text)
    b = classify_task(text)
    assert a == b and a in TASK_CATEGORIES


def test_classify_is_synchronous_no_llm():
    assert not inspect.iscoroutinefunction(classify_task)
