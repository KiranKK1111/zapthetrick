"""Stage-5 §3.10 Component G — trivial-turn fast lane (semantic).

The trivial decision is the `trivial_turn` exemplar gate (AUTHORITY) with the
phrase list only as cold-start fallback — so paraphrased greetings the lexicon
misses are still caught. Per the semantic-gates convention, fallback tests PIN
`gates.matches` to None; authority tests pin it True/False.
"""
from __future__ import annotations

import asyncio

import pytest

from app.chat import difficulty, fast_lane


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def pin_none(monkeypatch):
    import app.semantics.gates as g
    monkeypatch.setattr(g, "matches", lambda *a, **k: None)
    yield


# --------------------------------------------------------------------------- #
class TestGate:
    def test_registered(self):
        from app.semantics.gates import GATES
        assert "trivial_turn" in GATES
        assert GATES["trivial_turn"]["positives"] and \
            GATES["trivial_turn"]["negatives"]


class TestIsTrivial:
    def test_ultra_short_is_trivial(self):
        assert fast_lane.is_trivial("k") is True   # structural, no embedder
        assert fast_lane.is_trivial("") is False

    def test_semantic_authority_true(self, monkeypatch):
        import app.semantics.gates as g
        monkeypatch.setattr(g, "matches", lambda name, t: True)
        # A phrase not in the lexicon is trivial when the gate says so.
        assert fast_lane.is_trivial("appreciate the help my friend") is True

    def test_semantic_authority_false(self, monkeypatch):
        import app.semantics.gates as g
        monkeypatch.setattr(g, "matches", lambda name, t: False)
        assert fast_lane.is_trivial("thanks") is False   # gate overrides

    def test_fallback_phrase_list(self, pin_none):
        # With the embedder cold, a real question is not trivial.
        assert fast_lane.is_trivial(
            "what is the difference between a list and a tuple") is False

    def test_enabled_default_off(self):
        assert fast_lane.enabled() is False


class TestDifficultySemanticTrivial:
    def test_paraphrased_greeting_is_trivial(self, monkeypatch):
        # The gate catches a greeting the hardcoded phrase list would miss →
        # difficulty short-circuits to "trivial" (no LLM classifier call).
        import app.semantics.gates as g
        monkeypatch.setattr(g, "matches", lambda name, t: True)
        assert _run(difficulty.classify_difficulty(
            "heya, hope you're having a lovely day")) == "trivial"

    def test_real_question_not_trivial_via_gate(self, monkeypatch):
        import app.semantics.gates as g
        # Gate says not-trivial; ensure the LLM classifier isn't reached in this
        # unit test by stubbing it to a deterministic value.
        monkeypatch.setattr(g, "matches", lambda name, t: False)

        async def _stub_llm_classify(*a, **k):
            return "standard"
        # classify_difficulty falls through to its own LLM path; we only assert
        # it did NOT short-circuit to trivial.
        monkeypatch.setattr(difficulty, "_is_heavy", lambda t: False)
        res = _run(difficulty.classify_difficulty("explain how kafka works"))
        assert res != "trivial"
