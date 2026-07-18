"""Intent Profile Registry (Architecture §4 / #5 Phase A)."""
from __future__ import annotations

from app.clarify import intent_profiles as ip


def test_every_taxonomy_intent_has_a_default_profile():
    from app.clarify.intent_pipeline import (
        INTENT_CHITCHAT, INTENT_KNOWLEDGE, INTENT_COMPARISON, INTENT_DEBUGGING,
        INTENT_TEST_GEN, INTENT_DOCS, INTENT_DESIGN, INTENT_CODE_GEN,
        INTENT_PROJECT_BUILD, INTENT_ARCHIVE)
    for label in (INTENT_CHITCHAT, INTENT_KNOWLEDGE, INTENT_COMPARISON,
                  INTENT_DEBUGGING, INTENT_TEST_GEN, INTENT_DOCS, INTENT_DESIGN,
                  INTENT_CODE_GEN, INTENT_PROJECT_BUILD, INTENT_ARCHIVE):
        assert label in ip.DEFAULTS, f"no default profile for {label}"


def test_resolve_known_intent_returns_its_profile():
    p = ip.resolve("code_generation")
    assert p.intent == "code_generation"
    assert p.response_shape == "code"
    assert p.doc_eligible is True
    assert ip.TOOL_COMPUTE in p.tools and ip.TOOL_CODE_SEARCH in p.tools
    assert p.consults(ip.GRAPH_CODE_KG) and p.consults(ip.GRAPH_MEMORY)
    assert p.suggestions == "iterate"


def test_resolve_is_case_insensitive_and_trims():
    assert ip.resolve("  KNOWLEDGE ").intent == "knowledge"


def test_resolve_unknown_intent_falls_back():
    p = ip.resolve("something_new")
    assert p is ip.FALLBACK
    assert p.consults(ip.GRAPH_KNOWLEDGE)


def test_resolve_none_falls_back():
    assert ip.resolve(None) is ip.FALLBACK


def test_knowledge_profile_is_not_doc_eligible():
    assert ip.resolve("knowledge").doc_eligible is False


def test_chitchat_consults_no_graphs_and_no_tools():
    p = ip.resolve("chitchat")
    assert p.graphs == ()
    assert p.tools == ()          # explicit "no tools", distinct from None
    assert p.consults(ip.GRAPH_MEMORY) is False


def test_default_frozen_profiles_are_immutable():
    p = ip.resolve("knowledge")
    try:
        p.doc_eligible = True  # type: ignore[misc]
        assert False, "profile should be frozen"
    except Exception:
        pass


# ---- config overlay (uses update_config + reload) ------------------------

def _with_profiles(monkeypatch, enabled, profiles):
    """Point cfg.intent_profiles at an ad-hoc config without touching YAML."""
    from app.core import config_loader as cl

    class _Fake:
        pass
    fake = _Fake()
    fake.enabled = enabled
    fake.profiles = profiles
    monkeypatch.setattr(cl.cfg, "intent_profiles", fake, raising=False)


def test_enabled_reads_config(monkeypatch):
    _with_profiles(monkeypatch, True, {})
    assert ip.enabled() is True
    _with_profiles(monkeypatch, False, {})
    assert ip.enabled() is False


def test_config_overlay_patches_fields(monkeypatch):
    _with_profiles(monkeypatch, True, {
        "knowledge": {"response_shape": "table", "doc_eligible": True,
                      "suggestions": "pivot"},
    })
    p = ip.resolve("knowledge")
    assert p.response_shape == "table"     # overridden
    assert p.doc_eligible is True          # overridden
    assert p.suggestions == "pivot"        # overridden
    assert p.consults(ip.GRAPH_MEMORY)     # untouched fields keep defaults


def test_config_overlay_coerces_sequences(monkeypatch):
    _with_profiles(monkeypatch, True, {
        "debugging": {"tools": ["code_solver", "web_search"],
                      "graphs": ["memory"]},
    })
    p = ip.resolve("debugging")
    assert p.tools == ("code_solver", "web_search")
    assert p.graphs == ("memory",)


def test_config_overlay_ignores_unknown_keys(monkeypatch):
    _with_profiles(monkeypatch, True, {
        "knowledge": {"bogus_key": 123, "response_shape": "steps"},
    })
    p = ip.resolve("knowledge")
    assert p.response_shape == "steps"
    assert not hasattr(p, "bogus_key")


def test_config_overlay_bad_max_tools_ignored(monkeypatch):
    _with_profiles(monkeypatch, True, {
        "knowledge": {"max_tools": "not-an-int"},
    })
    p = ip.resolve("knowledge")
    assert p.max_tools == ip.DEFAULTS["knowledge"].max_tools  # unchanged


def test_resolve_never_raises_on_bad_config(monkeypatch):
    _with_profiles(monkeypatch, True, {"knowledge": "not-a-dict"})
    # should fall back to the code default, not blow up
    assert ip.resolve("knowledge").response_shape == "prose"
