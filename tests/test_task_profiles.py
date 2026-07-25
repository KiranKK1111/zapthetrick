"""Stage-5 §2.6 Component C — task-profile scoring + semantic profiles.

The profile is chosen SEMANTICALLY (injected gate authority, category fallback);
the router adds a MEASURED verify-pass penalty per (identity, profile). All
additive + flag-gated: weight 0 → byte-identical ranking.
"""
from __future__ import annotations

import pytest

from app.llm import identity, profiles, router
from app.llm import scorecards as sc


@pytest.fixture(autouse=True)
def _clean():
    sc.clear()
    yield
    sc.clear()


# --------------------------------------------------------------------------- #
class TestProfiles:
    def test_table_and_lookup(self):
        assert profiles.profile("chat_code").quality_basis == "verify_pass"
        assert profiles.profile("extraction_json").json_floor == 0.99
        assert profiles.profile("nope") is None
        assert "dsa_reasoning" in profiles.PROFILE_NAMES

    def test_classify_semantic_authority(self):
        def gate(t, classes, **kw):
            assert classes is profiles.PROFILE_EXEMPLARS
            return "dsa_reasoning"
        assert profiles.classify("optimize this", gate_classify=gate) \
            == "dsa_reasoning"

    def test_classify_gate_miss_uses_fallback(self):
        def gate(t, classes, **kw):
            return None                       # embedder cold / no class
        assert profiles.classify("write code", gate_classify=gate,
                                 fallback_category="code_generation") == "chat_code"

    def test_classify_fallback_map(self):
        assert profiles.classify("x", fallback_category="dsa") == "dsa_reasoning"
        assert profiles.classify("x", fallback_category="document") == "doc_script"
        assert profiles.classify("x", fallback_category="unknown_cat") is None

    def test_classify_empty(self):
        assert profiles.classify("") is None

    def test_enabled_default_off(self):
        assert profiles.enabled() is False


class TestScorecards:
    def test_ewma_lowers_on_failures(self):
        k = "id|70|-|-"
        for _ in range(3):
            sc.record_verify_outcome(k, "chat_code", passed=False)
        assert sc.verify_pass_rate(k, "chat_code") < 1.0

    def test_unmeasured_is_optimistic(self):
        assert sc.verify_pass_rate("never", "chat_code") == 1.0

    def test_no_profile_is_neutral(self):
        sc.record_verify_outcome("k", None, passed=False)   # no-op
        assert sc.verify_pass_rate("k", None) == 1.0

    def test_passes_raise_rate_back(self):
        k = "id2|-|-|-"
        sc.record_verify_outcome(k, "chat_code", passed=False)
        low = sc.verify_pass_rate(k, "chat_code")
        for _ in range(10):
            sc.record_verify_outcome(k, "chat_code", passed=True)
        assert sc.verify_pass_rate(k, "chat_code") > low

    def test_card_and_repair_schema(self):
        k = "id3|-|-|-"
        sc.record_verify_outcome(k, "chat_code", passed=True, repaired=True,
                                 schema_retried=True)
        c = sc.card(k, "chat_code")
        assert c["n"] == 1 and c["repair_rate"] > 0 and c["schema_retry"] > 0

    def test_weight_zero_when_flag_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "task_profiles", False, raising=False)
        assert sc.profile_verify_weight() == 0.0

    def test_weight_read_when_on(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "task_profiles", True, raising=False)
        monkeypatch.setattr(cfg.routing, "profile_verify_weight", 30.0,
                            raising=False)
        assert sc.profile_verify_weight() == 30.0


class TestRouterVerifyTerm:
    def test_failing_verify_scores_worse(self):
        base = router._candidate_score(0, 1.0, 10, 10, "standard")
        worse = router._candidate_score(0, 1.0, 10, 10, "standard",
                                        verify_pass=0.1, verify_w=50.0)
        assert worse > base                   # lower = picked first → worse

    def test_weight_zero_is_byte_identical(self):
        base = router._candidate_score(0, 1.0, 10, 10, "standard")
        same = router._candidate_score(0, 1.0, 10, 10, "standard",
                                       verify_pass=0.1, verify_w=0.0)
        assert same == base

    def test_perfect_verify_no_penalty(self):
        base = router._candidate_score(0, 1.0, 10, 10, "standard")
        good = router._candidate_score(0, 1.0, 10, 10, "standard",
                                       verify_pass=1.0, verify_w=50.0)
        assert good == base

    def test_end_to_end_identity_keyed(self, monkeypatch):
        # A model whose canonical identity has a bad verify record scores worse
        # than one with none — proving the (identity, profile) keying.
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "canonical_identity", True, raising=False)
        k = identity.identity_key("groq", "llama-3.3-70b")
        for _ in range(5):
            sc.record_verify_outcome(k, "chat_code", passed=False)
        bad = sc.verify_pass_rate(k, "chat_code")
        good = sc.verify_pass_rate(
            identity.identity_key("groq", "qwen2.5-32b"), "chat_code")
        assert bad < good == 1.0


class TestStructuredVerifySink:
    """The verify sink wired into structured() — a schema call feeds the
    extraction_json (identity, profile) scorecard self-contained."""

    def _route(self):
        import types
        return types.SimpleNamespace(platform="groq", model_id="llama-3.3-70b")

    def test_records_when_on(self, monkeypatch):
        from app.core.config_loader import cfg
        from app.llm import structured as st
        monkeypatch.setattr(cfg.routing, "task_profiles", True, raising=False)
        monkeypatch.setattr(cfg.routing, "canonical_identity", True, raising=False)
        st._record_schema_outcome(self._route(), valid=False,
                                  degraded=["schema_retry"])
        st._record_schema_outcome(self._route(), valid=False, degraded=[])
        k = identity.identity_key("groq", "llama-3.3-70b")
        assert sc.verify_pass_rate(k, "extraction_json") < 1.0

    def test_no_record_when_off(self, monkeypatch):
        from app.core.config_loader import cfg
        from app.llm import structured as st
        monkeypatch.setattr(cfg.routing, "task_profiles", False, raising=False)
        st._record_schema_outcome(self._route(), valid=False, degraded=[])
        assert sc.verify_pass_rate("groq:llama-3.3-70b", "extraction_json") == 1.0

    def test_none_route_is_safe(self, monkeypatch):
        from app.core.config_loader import cfg
        from app.llm import structured as st
        monkeypatch.setattr(cfg.routing, "task_profiles", True, raising=False)
        st._record_schema_outcome(None, valid=False, degraded=[])   # no raise
