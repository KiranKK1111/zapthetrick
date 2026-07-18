"""Candidate experience & accessibility
(live-conversational-intelligence R19, R24; tasks 19.2).

Pins Properties 19, 24: talking-point distillation, candidate-awareness
suppression / build-on-candidate, language detection + code-switch + default
fallback.
"""
from __future__ import annotations

from app.live import language, surface


# ---- talking points ----------------------------------------------------
def test_talking_points_from_bullets():
    answer = "- First key point\n- Second key point\n- Third"
    pts = surface.talking_points(answer)
    assert pts == ["First key point", "Second key point", "Third"]


def test_talking_points_from_numbered():
    answer = "1. Define it\n2. Give an example\n3. Note the trade-off"
    pts = surface.talking_points(answer)
    assert len(pts) == 3
    assert pts[0] == "Define it"


def test_talking_points_fallback_to_first_sentences():
    answer = ("Kafka is a distributed log. It scales horizontally.\n\n"
              "Partitions enable parallelism. Each has an offset.")
    pts = surface.talking_points(answer)
    assert pts[0].startswith("Kafka is a distributed log")
    assert any("Partitions" in p for p in pts)


def test_talking_points_caps_count_and_length():
    answer = "\n".join(f"- point {i} " + "x" * 200 for i in range(10))
    pts = surface.talking_points(answer, max_points=4)
    assert len(pts) == 4
    assert all(len(p) <= 111 for p in pts)


def test_talking_points_empty():
    assert surface.talking_points("") == []


# ---- candidate awareness -----------------------------------------------
def test_candidate_awareness_suppresses_while_answering():
    c = surface.CandidateAwareness()
    c.observe_candidate("I would use a consistent hashing ring to shard the cache across nodes")
    assert c.is_answering_adequately() is True
    assert c.should_surface() is False


def test_candidate_awareness_surfaces_when_quiet_or_brief():
    c = surface.CandidateAwareness()
    assert c.should_surface() is True            # nothing said
    c.observe_candidate("um yeah")
    assert c.should_surface() is True            # too brief
    assert c.recent[-1] == "um yeah"


# ---- language detection ------------------------------------------------
def test_detect_english_default():
    assert language.detect_language("How does Kafka handle partitions?") == "en"


def test_detect_devanagari():
    assert language.detect_language("कैफ़्का क्या है") == "hi"


def test_detect_spanish_hints():
    assert language.detect_language("qué es el patrón porque la cómo") == "es"


def test_code_switch_favours_dominant_script():
    # Mixed English + Devanagari → Devanagari dominates.
    assert language.detect_language("explain कैफ़्का विभाजन कैसे") == "hi"


def test_answer_directive_for_nonenglish_only():
    assert language.answer_directive("en") == ""
    assert "Hindi" in language.answer_directive("hi")
    assert language.answer_directive("") == ""


def test_detect_failopen_empty():
    assert language.detect_language("") == "en"
