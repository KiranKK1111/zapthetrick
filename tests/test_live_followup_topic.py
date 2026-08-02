"""Follow-up predictions must belong to the topic actually being discussed.

From a live session: a question about Spring `@Controller` vs `@RestController`
produced

    predicted_followups: [Tell me about kubernetes pods., ... services.,
                          ... ingress., ... autoscaling., ... rolling updates.]

The matcher used `key in topic or topic in key` — a raw substring test — so any
short or garbled topic fragment inherited an unrelated topic's follow-ups.
`"net"` and `"ku"` both matched `"kubernetes"`, yielding "Tell me about net
pods."
"""
from __future__ import annotations

import pytest

from app.live.predict import predict_next


class WorldModel:
    def __init__(self, topic: str):
        self.topic = topic


def suggestions(topic: str, n: int = 5) -> list[str]:
    return predict_next(world_model=WorldModel(topic), max_n=n)


# ── The reported failure ────────────────────────────────────────────────────

@pytest.mark.parametrize("fragment", ["net", "ku", "be", "ing", "s"])
def test_a_topic_fragment_does_not_inherit_another_topics_followups(fragment):
    for s in suggestions(fragment):
        assert "pods" not in s and "ingress" not in s, \
            f"{fragment!r} inherited Kubernetes follow-ups: {s!r}"


def test_a_spring_question_never_suggests_kubernetes():
    """The exact shape of the live failure."""
    for topic in ("spring boot", "rest controller", "spring mvc"):
        joined = " ".join(suggestions(topic)).lower()
        assert "kubernetes" not in joined and "pods" not in joined, \
            f"{topic!r} -> {suggestions(topic)}"


# ── Real topics still get their specific follow-ups ─────────────────────────

@pytest.mark.parametrize("topic,expected", [
    ("kubernetes", "pods"),
    ("kafka", "partitions"),
    ("redis", "persistence"),
    ("postgres", "indexing"),
    ("react", "hooks"),
])
def test_a_known_topic_keeps_its_subtopics(topic, expected):
    """The fix must not simply stop matching."""
    assert any(expected in s for s in suggestions(topic)), suggestions(topic)


def test_a_multi_word_topic_containing_a_known_key_still_matches():
    assert any("pods" in s for s in suggestions("kubernetes networking"))


def test_an_unknown_topic_falls_back_to_generic_templates():
    out = suggestions("hexagonal architecture")
    assert out and all("hexagonal architecture" in s for s in out)


def test_an_empty_topic_predicts_nothing():
    assert suggestions("") == []
    assert predict_next() == []


def test_predictions_are_capped_and_unique():
    out = suggestions("kubernetes", n=4)
    assert len(out) == 4 and len(set(out)) == 4
