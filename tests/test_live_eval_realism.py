"""Evaluation realism: synthetic scenarios + label-free proxies
(live-conversational-intelligence R27; task 21.2).

Pins Property 27: synthetic scenarios are tagged + auto-pass the decision fns,
metric proxies are label-free + reported, and the harness is dev-only with no
runtime effect.
"""
from __future__ import annotations

from app.eval import live_proxies, live_synth


# ---- synthetic scenarios -----------------------------------------------
def test_generate_scenarios_tagged_synthetic():
    scs = live_synth.generate_scenarios(50)
    assert scs
    assert all(s.name.startswith("synth:") for s in scs)


def test_generate_scenarios_respects_n():
    assert len(live_synth.generate_scenarios(5)) == 5


def test_synth_metrics_high_pass_and_zero_false_answer():
    m = live_synth.synth_metrics(50)
    assert m["synthetic"] is True
    assert m["overall"]["pass_rate"] >= 0.95
    assert m["false_answer_rate"] == 0.0
    assert "per_category" in m


# ---- label-free proxies -------------------------------------------------
def test_relevance_proxy_rewards_overlap():
    high = live_proxies.relevance_proxy(
        "How does Kafka handle partition rebalancing?",
        "Kafka handles partition rebalancing via the group coordinator.")
    low = live_proxies.relevance_proxy(
        "How does Kafka handle partition rebalancing?",
        "Pancakes are made with flour and eggs.")
    assert high > low
    assert 0.0 <= high <= 1.0


def test_hallucination_proxy_grounded_vs_ungrounded():
    grounded = live_proxies.hallucination_proxy(
        "Kafka uses partitions for parallelism.",
        "Kafka splits topics into partitions for parallelism and ordering.")
    ungrounded = live_proxies.hallucination_proxy(
        "Kafka was invented by Aristotle in ancient Greece.",
        "Kafka splits topics into partitions for parallelism.")
    assert ungrounded > grounded


def test_hallucination_proxy_unknown_without_context():
    assert live_proxies.hallucination_proxy("anything", None) == 0.5


def test_proxies_over_samples():
    out = live_proxies.proxies_over([
        {"question": "what is kafka", "answer": "kafka is a log", "context": "kafka is a distributed log"},
    ])
    assert out["count"] == 1
    assert 0.0 <= out["avg_relevance"] <= 1.0
    assert 0.0 <= out["avg_hallucination_risk"] <= 1.0


def test_proxies_over_empty():
    out = live_proxies.proxies_over([])
    assert out["count"] == 0
