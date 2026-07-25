"""Stage-7 §3.9 — repeat-prompt variation engine: repeat→variation, approach ledger."""
from __future__ import annotations

import pytest

from app.chat import variation as V


@pytest.fixture(autouse=True)
def _fresh():
    V.reset_for_tests()
    yield
    V.reset_for_tests()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.chat, "variation_engine", True, raising=False)


_Q = "reverse a linked list"


class TestFingerprint:
    def test_stable_across_case_and_punctuation(self):
        assert V.fingerprint("Reverse a Linked List!") == V.fingerprint(_Q)

    def test_different_prompts_differ(self):
        assert V.fingerprint("reverse a list") != V.fingerprint("sort a list")

    def test_blank_is_empty(self):
        assert V.fingerprint("   ") == ""


class TestRepeatDetection:
    def test_first_ask_is_not_a_repeat(self, _on):
        assert V.is_repeat("c1", _Q) is False

    def test_second_ask_is_a_repeat(self, _on):
        V.record("c1", V.fingerprint(_Q), "iterative")
        assert V.is_repeat("c1", _Q) is True
        assert V.should_bypass_cache("c1", _Q) is True

    def test_repeat_is_per_conversation(self, _on):
        V.record("c1", V.fingerprint(_Q), "iterative")
        assert V.is_repeat("c2", _Q) is False        # different chat

    def test_disabled_never_repeats(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.chat, "variation_engine", False, raising=False)
        V.record("c1", V.fingerprint(_Q), "iterative")
        assert V.is_repeat("c1", _Q) is False and V.should_bypass_cache("c1", _Q) \
            is False


class TestApproachLedger:
    def test_records_and_counts(self, _on):
        fp = V.fingerprint(_Q)
        V.record("c1", fp, "iterative")
        V.record("c1", fp, "recursive")
        assert V.count("c1", fp) == 2
        assert V.approaches("c1", fp) == ["iterative", "recursive"]

    def test_blank_approach_gets_a_label(self, _on):
        fp = V.fingerprint(_Q)
        V.record("c1", fp, "")
        assert V.approaches("c1", fp)[0].startswith("approach")

    def test_forget_conversation(self, _on):
        fp = V.fingerprint(_Q)
        V.record("c1", fp, "x")
        V.forget_conversation("c1")
        assert V.count("c1", fp) == 0


class TestVariationParams:
    def test_temperature_widens_per_repeat(self, _on):
        fp = V.fingerprint(_Q)
        t0 = V.variation_params("c1", _Q).temperature
        V.record("c1", fp, "a")
        t1 = V.variation_params("c1", _Q).temperature
        V.record("c1", fp, "b")
        t2 = V.variation_params("c1", _Q).temperature
        assert t0 < t1 < t2

    def test_temperature_is_capped(self, _on):
        fp = V.fingerprint(_Q)
        for i in range(20):
            V.record("c1", fp, f"a{i}")
        assert V.variation_params("c1", _Q).temperature <= 1.1

    def test_rotates_model_from_first_repeat(self, _on):
        fp = V.fingerprint(_Q)
        assert V.variation_params("c1", _Q).rotate_model is False   # first ask
        V.record("c1", fp, "a")
        assert V.variation_params("c1", _Q).rotate_model is True

    def test_directive_names_prior_approaches(self, _on):
        fp = V.fingerprint(_Q)
        V.record("c1", fp, "iterative")
        V.record("c1", fp, "recursive")
        d = V.variation_params("c1", _Q).divergence
        assert "iterative" in d and "recursive" in d
        assert "different" in d.lower()

    def test_no_directive_on_first_ask(self, _on):
        assert V.variation_params("c1", _Q).divergence == ""

    def test_never_raises(self, _on):
        V.variation_params("c1", None)               # type: ignore[arg-type]
