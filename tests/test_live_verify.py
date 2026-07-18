"""Live answer verifier: gibberish gate, verdict, and retry escalation."""
from __future__ import annotations

from app.live import verify as V


# --------------------------------------------------------------------------- #
# Deterministic gibberish / incoherence detection
# --------------------------------------------------------------------------- #
def test_empty_is_incoherent():
    assert V.looks_incoherent("") is True
    assert V.looks_incoherent("   ") is True


def test_replacement_and_unk_tokens():
    assert V.looks_incoherent("answer <unk> here <unk> broken") is True
    assert V.looks_incoherent("a � b � c � d") is True


def test_whitespace_free_mash():
    assert V.looks_incoherent("x" * 60) is True                      # one 60-char "word"
    assert V.looks_incoherent("a" * 500) is True                     # long, no spaces


def test_runaway_repetition():
    assert V.looks_incoherent(("spam " * 40).strip()) is True        # one word dominates


def test_clean_answer_is_coherent():
    good = ("A HashMap gives average O(1) lookup by hashing the key to a bucket; "
            "collisions are chained or probed. Use it when you need fast keyed access.")
    assert V.looks_incoherent(good) is False
    code = "def add(a, b):\n    return a + b\n\nThis returns the sum of two integers."
    assert V.looks_incoherent(code) is False


# --------------------------------------------------------------------------- #
# Reasoning / prompt / continuation leak detection (from the exported sessions)
# --------------------------------------------------------------------------- #
def test_leaked_reasoning_flagged():
    leaks = [
        "We need to answer in Portuguese, following the outline: Definition -> ...",
        "Thinking Process:\n1. Analyze the Request:\n * Role: Elite interview answer assistant.",
        "We need to continue from where the previous reply stopped. The previous reply ended with:",
        'The user is asking "What is Q proxy?" but based on the conversation history...',
        "INTERVIEWER QUESTION:\nWhat is Q proxy?",
        "We need to produce answer in Portuguese, following outline: Definition -> ...",
    ]
    for a in leaks:
        assert V.looks_like_leaked_reasoning(a) is True, a[:40]


def test_clean_answers_not_flagged_as_leak():
    good = [
        "A HashMap gives average O(1) lookup by hashing the key to a bucket.",
        "The CAP theorem states a distributed system can provide at most two of "
        "consistency, availability, and partition tolerance.",
        "Ingress is a Kubernetes API object that routes external HTTP/HTTPS traffic "
        "to internal Services by host and path.",
        'Use replaceAll("\\\\s+", "") to remove all whitespace from a Java string.',
    ]
    for a in good:
        assert V.looks_like_leaked_reasoning(a) is False, a[:40]


def test_leaked_verdict_is_not_ok_and_meta():
    v = V.Verdict(relevance=0.9, hallucination_risk=0.1, issue="leak", leaked=True)
    assert v.ok is False
    assert v.to_meta()["verdict"] == "weak"
    assert v.to_meta().get("leaked") is True


def test_critique_mentions_leak():
    v = V.Verdict(relevance=0.0, hallucination_risk=1.0, issue="leak", leaked=True)
    d = V.critique_directive(v, "what is a hashmap").lower()
    assert "only the final" in d or "thinking process" in d or "meta-commentary" in d


# --------------------------------------------------------------------------- #
# Verdict — gibberish forces a non-ok verdict
# --------------------------------------------------------------------------- #
def test_gibberish_verdict_is_not_ok():
    v = V.Verdict(relevance=0.95, hallucination_risk=0.05, issue="", gibberish=True)
    assert v.ok is False
    assert v.to_meta()["verdict"] == "weak"
    assert v.to_meta().get("gibberish") is True


def test_normal_good_verdict_ok():
    v = V.Verdict(relevance=0.9, hallucination_risk=0.1, issue="")
    assert v.ok is True
    assert "gibberish" not in v.to_meta()


def test_low_relevance_not_ok():
    v = V.Verdict(relevance=0.2, hallucination_risk=0.1, issue="off-topic")
    assert v.ok is False


def test_critique_mentions_garbled_for_gibberish():
    v = V.Verdict(relevance=0.0, hallucination_risk=1.0, issue="garbled", gibberish=True)
    d = V.critique_directive(v, "explain hashmaps")
    assert "garbled" in d.lower() or "incoherent" in d.lower()


# --------------------------------------------------------------------------- #
# Retry-difficulty escalation (routes_ws helper)
# --------------------------------------------------------------------------- #
def test_escalation_ladder():
    from app.api.routes_ws import _escalate_difficulty as esc
    # earlier retry bumps one tier
    assert esc("standard", 1, 2, False) == "hard"
    assert esc("trivial", 1, 2, False) == "standard"
    # final retry forces expert (→ different/stronger model)
    assert esc("standard", 2, 2, False) == "expert"
    assert esc("hard", 1, 2, False) == "expert"
    # a garbled answer jumps straight to expert regardless of stage
    assert esc("standard", 1, 2, True) == "expert"
    # never exceeds expert
    assert esc("expert", 2, 2, False) == "expert"
