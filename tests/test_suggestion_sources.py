"""Graph-aware suggestion sources + blend (Architecture.md §6)."""
from __future__ import annotations

from app.response_arch.suggestion_sources import (
    blend, from_episodes, from_kg, from_memory)


def test_from_memory_tags_source_and_is_conservative():
    out = from_memory([
        "python",                                  # too short → skipped
        "Building a Flutter chat app with Firebase auth and offline sync",
    ])
    assert len(out) == 1
    assert out[0]["source"] == "memory_graph"
    assert out[0]["text"].startswith("Revisit:")


def test_from_memory_respects_limit_and_intent_hint():
    out = from_memory(
        ["a substantial prior thread about rust ownership and borrow checker",
         "another substantial thread about postgres indexing strategies"],
        limit=1, intent_of=lambda t: "knowledge")
    assert len(out) == 1
    assert out[0]["intent_hint"] == "knowledge"


def test_from_episodes_revisit_and_skips_current_thread():
    eps = [
        {"question": "how do I implement JWT auth in Flutter", "intent": "code_generation"},
        {"question": "explain how kafka partitions work", "intent": "knowledge"},
    ]
    # Current turn is about JWT → that episode is THIS thread, skipped; the
    # distinct kafka thread becomes an open-thread suggestion.
    out = from_episodes(eps, current_question="implement JWT auth in Flutter now",
                        limit=2)
    assert len(out) == 1
    assert out[0]["source"] == "memory_graph"
    assert "kafka" in out[0]["text"].lower()
    assert out[0]["text"].startswith("Revisit:")


def test_from_episodes_suppresses_meta_doc_requests():
    # A one-off doc/zip/download ask is not an "open thread" worth resuming —
    # it must never surface as a Revisit chip (the stale-chip bug).
    eps = [
        {"question": "can you get me a word document for this"},
        {"question": "zip the whole project and download it"},
        {"question": "export this as a pdf"},
    ]
    assert from_episodes(eps, current_question="explain hash maps", limit=3) == []


def test_from_episodes_skips_short_and_dedupes():
    out = from_episodes(
        [{"question": "hi"},                                   # too short
         {"question": "design a rate limiter for the api"},
         {"question": "design a rate limiter for the api"}],   # dup
        limit=5)
    assert len(out) == 1


def test_from_kg_related_concepts_and_dedupe():
    out = from_kg(["refresh token", "Refresh Token", "JWT"], limit=2)
    assert [s["source"] for s in out] == ["knowledge_graph", "knowledge_graph"]
    assert "refresh token" in out[0]["text"].lower()
    assert len(out) == 2  # deduped the case-variant


def test_from_kg_empty_scaffold_returns_nothing():
    assert from_kg([]) == []
    assert from_kg(None) == []


def test_blend_orders_profile_first_dedupes_and_caps():
    profile = [{"text": "Add tests", "source": "profile"},
               {"text": "Add tests", "source": "profile"}]          # dup
    memory = [{"text": "Revisit: the dashboard", "source": "memory_graph"}]
    kg = [{"text": "How does this relate to JWT?", "source": "knowledge_graph"}]
    out = blend(profile=profile, memory=memory, kg=kg, limit=3)
    assert [s["source"] for s in out] == ["profile", "memory_graph", "knowledge_graph"]
    assert len(out) == 3  # dup collapsed, one from each source


def test_blend_caps_total():
    profile = [{"text": f"s{i}", "source": "profile"} for i in range(5)]
    out = blend(profile=profile, limit=3)
    assert len(out) == 3
