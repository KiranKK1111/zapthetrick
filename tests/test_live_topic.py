"""Topic graph + drift detection (live-conversational-intelligence R5; task 4.2).

Pins Property 6: parent/child attach, follow-up attach, drift detection +
local reset preserving the graph, earlier-topic reference resolution, and
fail-open behavior (deterministic — no embedder needed).
"""
from __future__ import annotations

from app.live.topic_graph import TopicGraph


def test_subtopic_links_under_parent():
    g = TopicGraph()
    g.add_topic("Kafka")
    g.add_topic("partitions", parent="Kafka")
    node = g.node("partitions")
    assert node is not None
    assert node.parent == "kafka"
    assert g.current() == "partitions"


def test_observe_links_subtopic_when_related():
    g = TopicGraph()
    g.observe("kafka")
    drift = g.observe("kafka partitions")  # related -> sub-topic, no drift
    assert drift is False
    assert g.node("kafka partitions").parent == "kafka"


def test_drift_on_unrelated_topic():
    g = TopicGraph()
    g.observe("kafka")
    drift = g.observe("redis")  # unrelated -> drift
    assert drift is True
    assert g.current() == "redis"
    assert g.previous() == "kafka"


def test_drift_uses_similarity_when_supplied():
    g = TopicGraph()
    g.add_topic("kafka")
    # High similarity → not a drift even though strings differ.
    assert g.detect_drift("messaging", similarity=0.9) is False
    # Low similarity → drift.
    assert g.detect_drift("databases", similarity=0.1) is True


def test_graph_preserved_after_drift():
    g = TopicGraph()
    g.observe("kafka", turn_idx=0)
    g.observe("redis", turn_idx=1)
    # Both nodes survive the switch (graph preserved for later return).
    assert set(g.topics()) >= {"kafka", "redis"}


def test_resolve_back_to_reference():
    g = TopicGraph()
    g.observe("partitions")
    g.observe("consumer groups")
    assert g.resolve_reference("let's go back to partitions") == "partitions"


def test_resolve_earlier_topic_mention():
    g = TopicGraph()
    g.observe("kafka")
    g.observe("consumer groups")
    # "how is that different from kafka" references the earlier kafka node.
    assert g.resolve_reference("how is that different from kafka") == "kafka"


def test_empty_topic_is_noop():
    g = TopicGraph()
    assert g.add_topic("") is None
    assert g.detect_drift("") is False
    assert g.resolve_reference("") is None
