"""Stage-6 §4.10 — voice contract + reasoning-leak guard (Layers 2 & 3)."""
from __future__ import annotations

import pytest

from app.live import voice as V


@pytest.fixture(autouse=True)
def _fresh():
    V.clear_dictatability()
    yield
    V.clear_dictatability()


@pytest.fixture
def _guard_on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "leak_guard", True, raising=False)


# --------------------------------------------------------------------------- #
class TestLeakedHead:
    def test_reasoning_leak(self):
        assert V.leaked_head("We need to answer the question about Kafka.") is True

    def test_meta_preamble(self):
        assert V.leaked_head("As an AI, here's my response to that.") is True
        assert V.leaked_head("Sure, here is how I would approach it.") is True

    def test_clean_answer(self):
        assert V.leaked_head(
            "I built a Kafka pipeline that cut latency 40%.") is False

    def test_empty_is_clean(self):
        assert V.leaked_head("") is False
        assert V.leaked_head(None) is False


class TestShouldHold:
    def test_off_never_holds(self):
        # Flag off → no hold even on an obvious leak.
        assert V.should_hold("We need to answer this.").hold is False

    def test_leak_holds_and_rotates(self, _guard_on):
        dec = V.should_hold("We need to produce the answer now.")
        assert dec.hold is True and dec.rotate_model is True
        assert dec.reason

    def test_clean_does_not_hold(self, _guard_on):
        assert V.should_hold(
            "I led the migration to Postgres over two quarters.").hold is False


class TestVoiceValidator:
    def test_good_spoken_prose_passes(self):
        v = V.validate_voice(
            "I built a Kafka pipeline. We cut end-to-end latency by 40 percent.")
        assert v.ok is True

    def test_markdown_scaffolding_fails_register(self):
        v = V.validate_voice("## Approach\n- point one\n- point two")
        assert v.spoken_register is False
        assert v.ok is False

    def test_code_fence_fails_register(self):
        v = V.validate_voice("Here's the code:\n```python\nprint(1)\n```")
        assert v.spoken_register is False

    def test_third_person_fails_first_person(self):
        v = V.validate_voice("The candidate should mention their Kafka work.")
        assert v.first_person is False
        assert v.ok is False

    def test_missing_first_person_fails(self):
        v = V.validate_voice("Kafka decouples producers from consumers reliably.")
        assert v.first_person is False

    def test_meta_opening_fails_scaffolding(self):
        v = V.validate_voice("As an AI, I would say I built a Kafka pipeline.")
        assert v.no_scaffolding is False

    def test_band_shape_via_contract(self):
        from app.live.contract import Contract
        # A tiny max length → a long answer is out of band shape.
        contract = Contract(style="professional", max_answer_seconds=1)
        long = "I " + "really " * 200 + "did that."
        v = V.validate_voice(long, contract=contract)
        assert v.band_shape_ok is False

    def test_empty_is_passing(self):
        assert V.validate_voice("").ok is True     # advisory, not a gate

    def test_never_raises(self):
        V.validate_voice(None)                     # type: ignore[arg-type]


class TestDictatabilityEwma:
    def test_unseen_is_optimistic(self):
        assert V.dictatability("groq:llama") == 1.0

    def test_failures_lower_the_rate(self):
        for _ in range(10):
            V.record_dictatability("groq:llama", ok=False)
        assert V.dictatability("groq:llama") < 0.2

    def test_successes_keep_it_high(self):
        for _ in range(5):
            V.record_dictatability("groq:llama", ok=True)
        assert V.dictatability("groq:llama") == 1.0

    def test_blank_key_is_noop(self):
        V.record_dictatability("", ok=False)
        assert V.dictatability("") == 1.0


class TestStyleClause:
    def test_never_restate_clause_present(self):
        c = V.never_restate_clause()
        assert "first person" in c.lower()
        assert "never restate" in c.lower()
