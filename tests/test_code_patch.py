"""Stage-4 §3.5 Component H — patch-based chat-code edits + toolchain prefetch.

`apply_str_replace` is pure string surgery; `is_edit_request` is SEMANTIC (the
`code_edit_request` gate is the authority, a cue regex only the cold-start
fallback). Per the semantic-gates test convention, the fallback tests PIN
`gates.matches` to None so the warm embedder in the full suite can't flake them;
the semantic mechanism itself is covered in test_semantic_gates.py.
"""
from __future__ import annotations

import pytest

from app.chat import code_patch as cp


@pytest.fixture
def pin_fallback(monkeypatch):
    """Force the deterministic cold-start path (embedder unavailable)."""
    import app.semantics.gates as g
    monkeypatch.setattr(g, "matches", lambda *a, **k: None)
    yield


# --------------------------------------------------------------------------- #
class TestApplyStrReplace:
    def test_single_patch(self):
        r = cp.apply_str_replace(
            "def f(x):\n    return x",
            [{"old_str": "return x", "new_str": "return x + 1"}])
        assert r.applied is True and r.applied_count == 1
        assert r.code == "def f(x):\n    return x + 1"

    def test_multiple_patches(self):
        r = cp.apply_str_replace(
            "a = 1\nb = 2",
            [{"old_str": "a = 1", "new_str": "a = 10"},
             {"old_str": "b = 2", "new_str": "b = 20"}])
        assert r.applied and r.applied_count == 2
        assert r.code == "a = 10\nb = 20"

    def test_missing_target_falls_back(self):
        original = "def f(): pass"
        r = cp.apply_str_replace(original, [{"old_str": "NOPE", "new_str": "x"}])
        assert r.applied is False
        assert r.code == original          # unchanged → caller regenerates
        assert r.failures

    def test_partial_then_miss_returns_original(self):
        original = "x = 1\ny = 2"
        r = cp.apply_str_replace(
            original,
            [{"old_str": "x = 1", "new_str": "x = 9"},
             {"old_str": "ZZZ", "new_str": "q"}])
        assert r.applied is False and r.code == original  # all-or-nothing

    def test_empty_patches(self):
        r = cp.apply_str_replace("code", [])
        assert r.applied is False

    def test_never_raises_on_malformed(self):
        r = cp.apply_str_replace("code", ["not a dict", {"new_str": "x"}])
        assert r.applied is False


class TestExtractLastCodeBlock:
    def test_returns_last_substantial_block(self):
        t = ("intro\n\n```python\na = 1\nb = 2\n```\n\n"
             "then\n\n```python\nc = 3\nd = 4\n```")
        got = cp.extract_last_code_block(t)
        assert got == ("c = 3\nd = 4", "python")

    def test_skips_one_liner(self):
        assert cp.extract_last_code_block("```python\nx = 1\n```") is None

    def test_none_without_fence(self):
        assert cp.extract_last_code_block("no code here") is None


class TestIsEditRequestFallback:
    def test_requires_prior_code(self, pin_fallback):
        assert cp.is_edit_request("add error handling", has_prior_code=False) \
            is False

    def test_edit_verbs_detected(self, pin_fallback):
        for t in ["now add error handling", "make it handle empty input",
                  "refactor this to use a loop", "fix the off-by-one"]:
            assert cp.is_edit_request(t, has_prior_code=True) is True

    def test_non_edit_rejected(self, pin_fallback):
        for t in ["write a new program to sort a list",
                  "what does this code do", "explain how this works"]:
            assert cp.is_edit_request(t, has_prior_code=True) is False

    def test_empty(self, pin_fallback):
        assert cp.is_edit_request("", has_prior_code=True) is False


class TestIsEditRequestSemanticAuthority:
    def test_gate_true_wins(self, monkeypatch):
        import app.semantics.gates as g
        monkeypatch.setattr(g, "matches", lambda name, t: True)
        # A phrase the cue regex would REJECT is accepted when the gate says yes.
        assert cp.is_edit_request("could you also cover the empty case",
                                  has_prior_code=True) is True

    def test_gate_false_wins(self, monkeypatch):
        import app.semantics.gates as g
        monkeypatch.setattr(g, "matches", lambda name, t: False)
        # A phrase the cue regex WOULD accept is rejected when the gate says no.
        assert cp.is_edit_request("add a new unrelated feature module",
                                  has_prior_code=True) is False


class TestGateAndSchema:
    def test_gate_registered(self):
        from app.semantics.gates import GATES
        assert "code_edit_request" in GATES
        g = GATES["code_edit_request"]
        assert g["positives"] and g["negatives"]

    def test_schema_shape(self):
        props = cp.CODE_PATCH_SCHEMA["properties"]["patches"]["items"]
        assert set(props["required"]) == {"old_str", "new_str"}

    def test_enabled_default_off(self):
        assert cp.enabled() is False


class TestPrefetch:
    def test_prefetch_non_blocking_and_safe(self):
        from app.sandbox import pool
        pool.prefetch_toolchain("python")     # returns immediately
        pool.prefetch_toolchain("brainfuck")  # unknown → no-op
        pool.prefetch_toolchain("")           # empty → no-op
