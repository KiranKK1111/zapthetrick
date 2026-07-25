"""Stage-4 §3.2 Component E — structured Solve extraction + language ladder +
fingerprint cache.

The extraction runs the VLM against a JSON schema and validates via §8.7; the
language resolves down a strict ladder; an already-solved problem returns from a
fingerprint cache. Additive + flag-gated (`code_solver.structured_extraction`).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.codeintel import solve_extract as sx
from app.codeintel import solve_fingerprint as sfp


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_fp():
    sfp.clear()
    yield
    sfp.clear()


_OBJ = {
    "platform": "leetcode",
    "title": "Two Sum",
    "statement": "Given an array nums and a target, return indices.",
    "constraints": ["2 <= nums.length <= 10^4"],
    "examples": [{"input": "[2,7], 9", "output": "[0,1]"}],
    "selected_language": "Python3",
    "starter_code": "class Solution:\n    def twoSum(self, nums, t):\n        pass",
    "ui_confidence": {"selected_language": 0.9, "examples": 1.0},
}


# --------------------------------------------------------------------------- #
class TestExtractedProblem:
    def test_from_obj_valid(self):
        p = sx.ExtractedProblem.from_obj(_OBJ)
        assert p is not None
        assert p.title == "Two Sum" and p.platform == "leetcode"
        assert p.selected_language == "Python3" and p.lang_confidence == 0.9
        assert len(p.examples) == 1

    def test_from_obj_requires_statement(self):
        assert sx.ExtractedProblem.from_obj({"title": "X", "statement": ""}) is None
        assert sx.ExtractedProblem.from_obj("nope") is None

    def test_to_delimited_roundtrips_sections(self):
        d = sx.ExtractedProblem.from_obj(_OBJ).to_delimited()
        assert "=== TITLE ===" in d and "Two Sum" in d
        assert "=== PROBLEM STATEMENT ===" in d
        assert "=== EXAMPLES ===" in d and "=== CONSTRAINTS ===" in d
        assert "twoSum" in d

    def test_summary(self):
        s = sx.ExtractedProblem.from_obj(_OBJ).summary()
        assert "Two Sum" in s and "Python3" in s and "1 example" in s


class TestLanguageLadder:
    def _p(self, **over):
        o = dict(_OBJ)
        o.update(over)
        return sx.ExtractedProblem.from_obj(o)

    def test_requested_wins(self):
        assert sx.resolve_language(self._p(), requested="go") == ("go", "requested")

    def test_selected_when_confident(self):
        assert sx.resolve_language(self._p(), threshold=0.7) == \
            ("Python3", "selected")

    def test_low_confidence_falls_to_starter(self):
        p = self._p(selected_language="", ui_confidence={"selected_language": 0.2})
        lang, src = sx.resolve_language(p, threshold=0.7)
        assert src == "starter" and lang  # inferred from the starter code

    def test_sticky_fallback(self):
        p = self._p(selected_language="", starter_code="",
                    ui_confidence={"selected_language": 0.0})
        assert sx.resolve_language(p, sticky="rust") == ("rust", "sticky")

    def test_unknown_fires_clarifier(self):
        p = self._p(selected_language="", starter_code="",
                    ui_confidence={"selected_language": 0.0})
        assert sx.resolve_language(p) == (None, "none")


class TestFingerprint:
    def test_normalize_collapses_noise(self):
        assert sfp.normalize_statement("  Given   Nums  ") == "given nums"

    def test_fingerprint_deterministic_and_language_scoped(self):
        a = sfp.fingerprint("Given nums", "python")
        b = sfp.fingerprint("given   nums", "Python")   # noise + case
        c = sfp.fingerprint("Given nums", "go")         # language differs
        assert a == b and a != c

    def test_cache_get_put(self):
        fp = sfp.fingerprint("problem X", "python")
        assert sfp.get(fp) is None
        sfp.put(fp, "the solution")
        assert sfp.get(fp) == "the solution"

    def test_empty_solution_not_stored(self):
        fp = sfp.fingerprint("p", "py")
        sfp.put(fp, "   ")
        assert sfp.get(fp) is None


class TestExtractStructured:
    def _stub_complete(self, monkeypatch, replies):
        from app.core.llm_client import llm
        calls = {"n": 0}

        async def fake(messages, model=None, options=None):
            i = min(calls["n"], len(replies) - 1)
            calls["n"] += 1
            return replies[i]
        monkeypatch.setattr(llm, "complete", fake)
        return calls

    def test_valid_json_extracts(self, monkeypatch):
        self._stub_complete(monkeypatch, [json.dumps(_OBJ)])
        p = _run(sx.extract_structured(b"imgbytes", vision_model="vlm"))
        assert p is not None and p.title == "Two Sum"

    def test_malformed_then_retry_then_none(self, monkeypatch):
        calls = self._stub_complete(monkeypatch, ["not json at all"])
        p = _run(sx.extract_structured(b"img", vision_model="vlm", retries=1))
        assert p is None
        assert calls["n"] == 2                      # retried once

    def test_retry_recovers(self, monkeypatch):
        self._stub_complete(monkeypatch, ["garbage", json.dumps(_OBJ)])
        p = _run(sx.extract_structured(b"img", vision_model="vlm", retries=1))
        assert p is not None and p.title == "Two Sum"

    def test_vlm_error_returns_none(self, monkeypatch):
        from app.core.llm_client import llm

        async def boom(*a, **k):
            raise RuntimeError("vlm down")
        monkeypatch.setattr(llm, "complete", boom)
        assert _run(sx.extract_structured(b"img", vision_model="vlm")) is None


def test_enabled_default_off():
    assert sx.enabled() is False
