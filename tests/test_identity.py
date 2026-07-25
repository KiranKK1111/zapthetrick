"""Stage-5 §2.4 Component A — canonical model identity.

The invariant that matters: the SAME weights served by different providers (and
spelled with different marketing/serving suffixes) normalize to ONE identity
key, while a different version or quantization is a DIFFERENT identity. Tests pin
the grouping equalities/inequalities, not the exact family spelling.
"""
from __future__ import annotations

import pytest

from app.llm import identity as I


def _k(model_id, platform="groq"):
    return I.canonicalize(platform, model_id).key()


# --------------------------------------------------------------------------- #
class TestGrouping:
    def test_same_model_across_providers_and_suffixes(self):
        keys = {
            _k("llama-3.3-70b-versatile", "groq"),
            _k("Llama-3.3-70B-Instruct", "cerebras"),
            _k("meta-llama/Llama-3.3-70B-Instruct", "openrouter"),
            _k("llama-3.3-70b-instruct:free", "openrouter"),
            _k("Llama-3.3-70B", "nvidia"),
        }
        assert len(keys) == 1              # all ONE identity

    def test_different_version_is_different_identity(self):
        assert _k("llama-3.3-70b-instruct") != _k("llama-3.1-70b-instruct")

    def test_quantization_is_part_of_identity(self):
        full = _k("qwen2.5-coder-32b-instruct")
        awq = _k("qwen2.5-coder-32b-instruct-awq")
        assert full != awq
        assert I.canonicalize("groq", "qwen2.5-coder-32b-instruct-awq").quant \
            == "awq"

    def test_variant_is_different_identity(self):
        # A "coder" variant is different weights from the base — not collapsed.
        assert _k("qwen2.5-coder-32b") != _k("qwen2.5-32b")

    def test_different_size_is_different_identity(self):
        assert _k("llama-3.3-70b-instruct") != _k("llama-3.1-8b-instruct")


class TestFields:
    def test_size_parsed(self):
        assert I.canonicalize("groq", "llama-3.3-70b").size_b == 70.0

    def test_moe_total_size(self):
        # 8x7B → ~56B total.
        assert I.canonicalize("groq", "mixtral-8x7b-instruct").size_b == 56.0

    def test_undisclosed_size_is_none(self):
        assert I.canonicalize("openai", "gpt-4o").size_b is None

    def test_default_quant_is_empty(self):
        assert I.canonicalize("groq", "llama-3.3-70b").quant == ""


class TestFailOpen:
    def test_unknown_id_is_stable_not_crash(self):
        a = I.canonicalize("x", "some-random-model")
        b = I.canonicalize("x", "some-random-model")
        assert a == b and a.key() == b.key()

    def test_empty_id(self):
        cid = I.canonicalize("x", "")
        assert cid.key() and cid.size_b is None

    def test_weird_input_never_raises(self):
        for m in ["", "///", ":::", "123", "b", "70b", "x" * 300]:
            I.canonicalize("p", m)         # must not raise


class TestProviderIndex:
    def test_build_index_and_providers_for(self):
        rows = [
            ("groq", "llama-3.3-70b-versatile", 1),
            ("cerebras", "Llama-3.3-70B-Instruct", 2),
            ("openrouter", "meta-llama/llama-3.1-70b-instruct", 3),
        ]
        idx = I.build_provider_index(rows)
        cid = I.canonicalize("groq", "llama-3.3-70b-versatile")
        serving = I.providers_for(cid, idx)
        # llama-3.3 has two providers; llama-3.1 is a separate identity.
        assert {p for p, _m, _r in serving} == {"groq", "cerebras"}
        assert len(idx) == 2

    def test_providers_for_miss(self):
        assert I.providers_for(I.canonicalize("x", "nope"), {}) == []


class TestFlagGate:
    def test_identity_key_off_is_raw(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "canonical_identity", False,
                            raising=False)
        assert I.identity_key("groq", "llama-3.3-70b") == "groq:llama-3.3-70b"
        assert I.enabled() is False

    def test_identity_key_on_is_canonical(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "canonical_identity", True,
                            raising=False)
        k = I.identity_key("groq", "llama-3.3-70b-versatile")
        assert k == I.canonicalize("groq", "llama-3.3-70b-versatile").key()
        # provider-independent: another provider's spelling → same key.
        assert k == I.identity_key("cerebras", "Llama-3.3-70B-Instruct")
