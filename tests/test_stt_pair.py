"""Stage-6 §4.1 — STT streaming pair: domain-boost feed + partial/final reconcile."""
from __future__ import annotations

import types

import pytest

from app.stt import stt_pair as S
from app.stt import vocabulary_boost as VB


@pytest.fixture(autouse=True)
def _fresh():
    VB._session_terms.clear()
    yield
    VB._session_terms.clear()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "stt_pair", True, raising=False)


def _domain():
    return types.SimpleNamespace(
        vocab=["Kafka", "kubectl", "gRPC"],
        topics=["distributed systems"], role="backend engineer")


class TestRegisterDomain:
    def test_registers_vocab_topics_role(self, _on):
        n = S.register_domain(_domain())
        assert n == 5                                 # 3 vocab + 1 topic + 1 role
        terms = VB.build_boost_list()
        assert "Kafka" in terms and "kubectl" in terms
        assert "backend engineer" in terms

    def test_accepts_plain_list(self, _on):
        assert S.register_domain(["Redis", "Postgres"]) == 2
        assert "Redis" in VB.build_boost_list()

    def test_accepts_delimited_string(self, _on):
        assert S.register_domain("Go, Rust / Zig") == 3
        assert "Rust" in VB.build_boost_list()

    def test_domain_terms_rank_high(self, _on):
        VB.register_term("incidental", 0.5)
        S.register_domain(["Kafka"])                  # weight 2.0 > 0.5
        assert VB.build_boost_list()[0] == "Kafka"

    def test_disabled_is_noop(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "stt_pair", False, raising=False)
        assert S.register_domain(_domain()) == 0
        assert VB.build_boost_list() == []

    def test_none_and_error_fail_open(self, _on):
        assert S.register_domain(None) == 0
        assert S.register_domain(object()) == 0       # no .vocab → 0, no raise


class TestReconcileFinal:
    def test_final_wins_when_present(self):
        d = S.reconcile_final("i used cube control", "I used kubectl")
        assert d.source == "final" and d.text == "I used kubectl"
        assert d.changed is True
        assert 0.0 <= d.agreement <= 1.0

    def test_empty_final_keeps_partial(self):
        d = S.reconcile_final("i used kubectl", "")
        assert d.source == "partial" and d.text == "i used kubectl"
        assert d.changed is False

    def test_identical_final_not_changed(self):
        d = S.reconcile_final("hello world", "hello world")
        assert d.changed is False and d.agreement == 1.0

    def test_both_empty(self):
        d = S.reconcile_final("", "")
        assert d.text == "" and d.source == "partial"

    def test_agreement_low_on_big_rescore(self):
        d = S.reconcile_final("completely wrong words here",
                              "an entirely different final result")
        assert d.agreement < 0.5 and d.source == "final"

    def test_never_raises(self):
        S.reconcile_final(None, None)                 # type: ignore[arg-type]


class TestBoostTerms:
    def test_passthrough_ranked(self, _on):
        S.register_domain(["Kafka", "gRPC"])
        assert set(S.boost_terms()) >= {"Kafka", "gRPC"}

    def test_empty_when_nothing_registered(self):
        assert S.boost_terms() == []
