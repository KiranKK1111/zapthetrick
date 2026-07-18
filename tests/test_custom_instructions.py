"""User-authored custom instructions (Architecture §17 / #11 Part A)."""
from __future__ import annotations

from app.personalization import instructions as ci


def test_load_missing_returns_empty():
    assert ci.load_custom_instructions(None) == ""
    assert ci.load_custom_instructions({}) == ""
    assert ci.load_custom_instructions({"other": "x"}) == ""


def test_load_trims_and_caps():
    assert ci.load_custom_instructions({"custom_instructions": "  hi  "}) == "hi"
    long = "x" * (ci.MAX_CHARS + 500)
    assert len(ci.load_custom_instructions({"custom_instructions": long})) == ci.MAX_CHARS


def test_load_ignores_non_string():
    assert ci.load_custom_instructions({"custom_instructions": 123}) == ""
    assert ci.load_custom_instructions({"custom_instructions": ["a"]}) == ""


def test_set_writes_without_mutating_input():
    orig = {"keep": 1}
    out = ci.set_custom_instructions(orig, "Be terse")
    assert out == {"keep": 1, "custom_instructions": "Be terse"}
    assert orig == {"keep": 1}          # input untouched


def test_set_blank_clears():
    out = ci.set_custom_instructions({"custom_instructions": "old", "keep": 1}, "  ")
    assert "custom_instructions" not in out
    assert out == {"keep": 1}


def test_set_caps_length():
    out = ci.set_custom_instructions({}, "y" * (ci.MAX_CHARS + 100))
    assert len(out["custom_instructions"]) == ci.MAX_CHARS


def test_frame_empty_is_blank():
    assert ci.frame_instructions("") == ""
    assert ci.frame_instructions(None) == ""
    assert ci.frame_instructions("   ") == ""


def test_frame_states_precedence_and_trust():
    block = ci.frame_instructions("Always answer in British English.")
    assert "Always answer in British English." in block
    # precedence: safety wins over user instructions
    assert "safety rules above" in block
    # framed as trusted (NOT the untrusted-data preamble)
    assert "trusted" in block.lower()
    assert "UNTRUSTED" not in block


def test_roundtrip_set_then_load():
    prefs = ci.set_custom_instructions(None, "Use metric units and cite sources.")
    assert ci.load_custom_instructions(prefs) == "Use metric units and cite sources."


def test_enabled_reads_config(monkeypatch):
    from app.core import config_loader as cl

    class _P:
        custom_instructions = False
    monkeypatch.setattr(cl.cfg, "personalization", _P(), raising=False)
    assert ci.enabled() is False

    class _P2:
        custom_instructions = True
    monkeypatch.setattr(cl.cfg, "personalization", _P2(), raising=False)
    assert ci.enabled() is True
