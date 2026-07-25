"""Stage-6 §4.11 — domain transcript repair v2: lexicon + session correction memory."""
from __future__ import annotations

import pytest

from app.live import repair_v2 as R


@pytest.fixture(autouse=True)
def _fresh():
    R.reset_for_tests()
    yield
    R.reset_for_tests()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "repair_v2", True, raising=False)


class TestLexicon:
    def test_seed_terms_present(self):
        terms = R.lexicon_terms()
        assert "kubectl" in terms and "postgresql" in terms and "grpc" in terms

    def test_lexicon_is_deduped_and_cached(self):
        a = R.lexicon_terms()
        b = R.lexicon_terms()
        assert a is b                       # cached
        assert len(a) == len(set(a))        # deduped


class TestCorrectionMemory:
    def test_remember_and_apply(self):
        R.remember("s1", "grpz", "gRPC")
        assert R.apply_memory("s1", "I used grpz for services") \
            == "I used gRPC for services"

    def test_preserves_leading_capitalization(self):
        R.remember("s1", "postgres", "PostgreSQL")
        assert R.apply_memory("s1", "Postgres is my database") \
            == "PostgreSQL is my database"

    def test_whole_word_only(self):
        R.remember("s1", "go", "Go")
        # "go" inside "goroutine" must NOT be touched.
        assert R.apply_memory("s1", "a goroutine and go") == "a goroutine and Go"

    def test_noop_and_overlong_ignored(self):
        R.remember("s1", "x", "x")          # no-op
        R.remember("s1", "y" * 61, "z")     # too long
        assert R.corrections("s1") == {}

    def test_per_session_isolation(self):
        R.remember("s1", "grpz", "gRPC")
        assert R.apply_memory("s2", "grpz here") == "grpz here"

    def test_forget_session(self):
        R.remember("s1", "grpz", "gRPC")
        R.forget_session("s1")
        assert R.corrections("s1") == {}


class TestRepairV2:
    def test_disabled_returns_unchanged(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "repair_v2", False, raising=False)
        R.remember("s1", "grpz", "gRPC")
        assert R.repair_v2("grpz here", session_id="s1") == "grpz here"

    def test_applies_session_memory_first(self, _on):
        R.remember("s1", "grpz", "gRPC")
        out = R.repair_v2("I used grpz today", session_id="s1")
        assert "gRPC" in out

    def test_learns_a_new_fix_into_memory(self, _on, monkeypatch):
        # Stub the underlying phonetic repair to change one token → it should be
        # learned so the next turn is instant.
        import app.live.repair as _rp
        monkeypatch.setattr(
            _rp, "repair",
            lambda text, vocab=None, topic_graph=None:
                text.replace("kubernetis", "kubernetes"))
        R.repair_v2("deploy on kubernetis now", session_id="s1")
        assert R.corrections("s1").get("kubernetis") == "kubernetes"
        # Next turn: the fix is applied straight from memory.
        assert "kubernetes" in R.apply_memory("s1", "kubernetis again")

    def test_clean_text_is_stable(self, _on):
        txt = "I deployed with kubectl and postgresql"
        assert R.repair_v2(txt, session_id="s1") == txt

    def test_empty_and_bad_input_fail_open(self, _on):
        assert R.repair_v2("", session_id="s1") == ""
        assert R.repair_v2(None, session_id="s1") == ""   # type: ignore[arg-type]
