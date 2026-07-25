"""Stage-4 §3.2 — problem-fingerprint solution cache."""
from __future__ import annotations

import asyncio

import pytest

from app.solve import fingerprint as F


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh():
    F.clear()
    yield
    F.clear()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.code_solver, "fingerprint_cache", True,
                        raising=False)


class TestFingerprint:
    def test_stable_across_cosmetic_edits(self):
        a = F.fingerprint("Two Sum: return indices.", "python")
        b = F.fingerprint("  two   sum:  return  indices!!  ", "python")
        assert a == b and a  # whitespace/case/punctuation-insensitive

    def test_language_is_part_of_the_key(self):
        assert (F.fingerprint("reverse a list", "python")
                != F.fingerprint("reverse a list", "java"))

    def test_digits_are_significant(self):
        # 10^5 vs 10^9 are genuinely different constraints → different problems.
        assert (F.fingerprint("n up to 10 5", "python")
                != F.fingerprint("n up to 10 9", "python"))

    def test_blank_statement_has_empty_fingerprint(self):
        assert F.fingerprint("   ", "python") == ""
        assert F.fingerprint("", "") == ""


class TestCache:
    def test_put_then_get_round_trips(self, _on):
        _run(F.put("Two Sum", "python", "def two_sum(): ..."))
        assert _run(F.get("Two Sum", "python")) == "def two_sum(): ..."

    def test_miss_when_language_differs(self, _on):
        _run(F.put("Two Sum", "python", "sol"))
        assert _run(F.get("Two Sum", "java")) is None

    def test_scope_isolates_users(self, _on):
        _run(F.put("Two Sum", "python", "A's solution", scope="userA"))
        assert _run(F.get("Two Sum", "python", scope="userB")) is None
        assert _run(F.get("Two Sum", "python", scope="userA")) == "A's solution"

    def test_disabled_is_a_noop(self):
        # Flag off → put stores nothing, get returns None.
        _run(F.put("Two Sum", "python", "sol"))
        assert _run(F.get("Two Sum", "python")) is None

    def test_error_marked_solution_not_cached(self, _on):
        _run(F.put("X", "python", "[LLM error: boom]"))
        assert _run(F.get("X", "python")) is None

    def test_blank_never_cached(self, _on):
        _run(F.put("   ", "python", "whatever"))
        assert _run(F.get("   ", "python")) is None
