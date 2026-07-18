"""Tests for the prompt-evaluation framework (roadmap Phase 1 #7).

Deterministic and offline: a stub generator stands in for the model, so the
framework (render -> generate -> gate-score -> compare) is fully exercised with
no keys. The SAME API drives a real model-backed generator in production.
"""
from __future__ import annotations

import pytest

from app.eval import (
    PromptCase,
    PromptRegistry,
    PromptVariant,
    Verdict,
    compare_variants,
    evaluate_prompt,
)
from app.eval.scoring import contains_all, contains_none, min_words


def test_variant_render_and_missing_input():
    v = PromptVariant("answer", "v1", "Explain {topic} briefly.")
    assert v.render({"topic": "Kafka"}) == "Explain Kafka briefly."
    with pytest.raises(KeyError):
        v.render({})  # missing 'topic'


def test_registry_versioning():
    reg = PromptRegistry()
    reg.register(PromptVariant("answer", "v1", "A {q}"))
    reg.register(PromptVariant("answer", "v2", "B {q}"))
    assert reg.versions("answer") == ["v1", "v2"]
    assert reg.get("answer", "v2").template == "B {q}"


def test_scoring_is_gate_weighted():
    v = PromptVariant("t", "v1", "{q}")
    # generator echoes the prompt back, so we control the "output" via inputs.
    gen = lambda p: p
    cases = [
        PromptCase({"q": "kafka partitions replication"},
                   gates=[contains_all("kafka"), contains_all("partitions")],
                   name="good"),
        PromptCase({"q": "hello world"},
                   gates=[contains_all("kafka")],
                   name="bad"),
    ]
    r = evaluate_prompt(v, cases, gen)
    good = next(c for c in r.case_scores if c.name == "good")
    bad = next(c for c in r.case_scores if c.name == "bad")
    assert good.score == 1.0 and good.passed_gates == 2
    assert bad.score == 0.0 and bad.passed_gates == 0
    assert r.score == 0.5  # (1.0 + 0.0) / 2
    assert r.pass_rate == 0.5


def test_compare_detects_improvement_and_regression():
    # Two variants over the same case. The candidate produces output that also
    # satisfies a stricter gate the baseline missed -> BETTER.
    case_inputs = {"q": "answer about kafka"}
    gates = [contains_all("kafka"), min_words(3)]

    baseline_gen = lambda p: "kafka"               # 1 word -> fails min_words(3)
    candidate_gen = lambda p: "kafka is a broker"  # passes both

    v1 = PromptVariant("a", "v1", "{q}")
    v2 = PromptVariant("a", "v2", "{q}")
    cases = [PromptCase(case_inputs, gates, name="c0")]

    r1 = evaluate_prompt(v1, cases, baseline_gen)
    r2 = evaluate_prompt(v2, cases, candidate_gen)
    cmp = compare_variants(r1, r2)
    assert cmp.verdict == Verdict.BETTER
    assert cmp.candidate_score > cmp.baseline_score
    assert cmp.regressed_cases == []

    # Reverse -> WORSE, and the regressed case is reported.
    cmp2 = compare_variants(r2, r1)
    assert cmp2.verdict == Verdict.WORSE
    assert cmp2.regressed_cases == ["c0"]


def test_compare_same_is_a_tie():
    gen = lambda p: "kafka is a broker"
    v1 = PromptVariant("a", "v1", "{q}")
    v2 = PromptVariant("a", "v2", "{q}")
    cases = [PromptCase({"q": "x"}, [contains_all("kafka")], name="c0")]
    cmp = compare_variants(evaluate_prompt(v1, cases, gen),
                           evaluate_prompt(v2, cases, gen))
    assert cmp.verdict == Verdict.SAME
    assert cmp.delta == 0.0


def test_forbidden_term_gate():
    # A prompt whose output must AVOID a refusal phrase (relevant to the
    # Uncensored Coding Mode: no "I can't help with that" on legit coding).
    v = PromptVariant("code", "v1", "{q}")
    cases = [PromptCase({"q": "here is a port scanner in python"},
                        gates=[contains_none("i can't help", "i cannot help")],
                        name="no_refusal")]
    r = evaluate_prompt(v, cases, lambda p: p)
    assert r.score == 1.0
