"""Stage-7 §4.8 — Live canonicalization + said-state: disfluency, multi-Q split, claims."""
from __future__ import annotations

import pytest

from app.live import canonical as C


@pytest.fixture(autouse=True)
def _fresh():
    C.reset_for_tests()
    yield
    C.reset_for_tests()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "canonicalize", True, raising=False)
    monkeypatch.setattr(cfg.live, "said_state", True, raising=False)


class TestCanonicalize:
    def test_strips_disfluencies(self):
        out = C.canonicalize("So um, what is, like, Kafka?")
        assert "um" not in out.lower() and " like" not in out.lower()
        assert "Kafka?" in out

    def test_strips_leading_filler(self):
        assert C.canonicalize("well, explain this") == "explain this"

    def test_clean_question_kept(self):
        assert C.canonicalize("Explain how Kafka works.") \
            == "Explain how Kafka works."

    def test_empty(self):
        assert C.canonicalize("") == "" and C.canonicalize(None) == ""


class TestSplitQuestions:
    def test_and_joins_two_questions(self):
        assert C.split_questions("What is Kafka and how does it scale?") == \
            ["What is Kafka?", "how does it scale?"]

    def test_also_across_a_sentence(self):
        parts = C.split_questions(
            "Tell me about your project. Also, why did you choose Go?")
        assert len(parts) == 2 and "why did you choose Go?" in parts[1]

    def test_two_full_questions(self):
        assert C.split_questions("What is a hash map? How does it resize?") == \
            ["What is a hash map?", "How does it resize?"]

    def test_then_joiner(self):
        parts = C.split_questions(
            "Walk me through the design and then explain the tradeoffs?")
        assert len(parts) == 2

    def test_single_question_not_split(self):
        assert C.split_questions("Explain how a hash map works.") == \
            ["Explain how a hash map works."]

    def test_and_between_nouns_not_split(self):
        # "a list and a tuple" — 'a' is not a WH word → one question.
        assert len(C.split_questions(
            "What is the difference between a list and a tuple?")) == 1

    def test_is_multi_question(self):
        assert C.is_multi_question("What is X and why does Y matter?") is True
        assert C.is_multi_question("Explain X.") is False

    def test_order_preserved(self):
        parts = C.split_questions("What is A and how does B work?")
        assert parts[0].startswith("What is A") and parts[1].startswith("how")

    def test_never_raises(self):
        C.split_questions(None)                    # type: ignore[arg-type]


class TestClaimsLedger:
    def test_record_new_then_dup(self, _on):
        assert C.record_claim("s1", "I built a Kafka pipeline") is True
        assert C.record_claim("s1", "I built a KAFKA pipeline!") is False  # dup
        assert C.claims("s1") == ["I built a Kafka pipeline"]

    def test_is_new_claim(self, _on):
        C.record_claim("s1", "I led a team")
        assert C.is_new_claim("s1", "I led a team") is False
        assert C.is_new_claim("s1", "I shipped a feature") is True

    def test_build_directive_names_claims(self, _on):
        C.record_claim("s1", "I built a Kafka pipeline")
        d = C.build_directive("s1")
        assert "Kafka pipeline" in d and "BUILD on these" in d

    def test_empty_directive_when_nothing_claimed(self, _on):
        assert C.build_directive("s1") == ""

    def test_per_session(self, _on):
        C.record_claim("s1", "I built X")
        assert C.claims("s2") == []

    def test_said_state_off_no_record(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "said_state", False, raising=False)
        assert C.record_claim("s1", "I built X") is False
        assert C.build_directive("s1") == ""

    def test_forget_session(self, _on):
        C.record_claim("s1", "I built X")
        C.forget_session("s1")
        assert C.claims("s1") == []
