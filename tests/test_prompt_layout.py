"""Stable-prefix PromptAssembler (vNext §8.1).

Pins: byte-stable prefix (deterministic, no leakage), layer-below invalidation,
the cache-breakpoint metadata, and byte-identity with the hand-built
[system]+history+user conversation the assembler replaces.
"""
from __future__ import annotations

from app.core.prompt_layout import PromptAssembler


def test_build_is_byte_identical_to_hand_built_conversation():
    sys = "You are preparing to answer.\n\nAvailable tools:\nTOOLS\n\nPROTOCOL"
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "x"},                 # no content → dropped
        {"role": "user", "content": ""},  # empty content → dropped
    ]
    question = "what now?"
    # Legacy construction (verbatim from app/chat/tool_loop.py).
    legacy = [{"role": "system", "content": sys}]
    for prior in (history or [])[-4:]:
        r, c = prior.get("role"), prior.get("content")
        if r and c:
            legacy.append({"role": r, "content": c})
    legacy.append({"role": "user", "content": question})

    got = PromptAssembler(persona=sys, recent=history, user=question).build()
    assert got == legacy


def test_prefix_hash_is_deterministic_no_leakage():
    a = PromptAssembler(persona="P", mode="M", memory="D")
    b = PromptAssembler(persona="P", mode="M", memory="D")
    assert a.prefix_hash() == b.prefix_hash()
    assert a.stable_prefix() == b.stable_prefix()


def test_layer_change_only_invalidates_below():
    a = PromptAssembler(persona="P", mode="M", project="J",
                        memory="D", history_summary="H")
    b = PromptAssembler(persona="P", mode="M", project="CHANGED",
                        memory="D", history_summary="H")
    la, lb = dict(a.layer_hashes()), dict(b.layer_hashes())
    # L0, L1 cumulative hashes unchanged...
    assert la["persona"] == lb["persona"]
    assert la["mode"] == lb["mode"]
    # ...L2 and everything below it changes.
    assert la["project"] != lb["project"]
    assert la["memory"] != lb["memory"]
    assert la["history_summary"] != lb["history_summary"]


def test_rag_sits_after_the_cached_prefix():
    a = PromptAssembler(persona="PERSONA", rag="THIS-TURN-DOCS")
    sc = a.system_content()
    assert sc == "PERSONA\n\nTHIS-TURN-DOCS"
    # The cached span is exactly the stable prefix (RAG excluded).
    _, meta = a.build_cached()
    assert meta["breakpoint_char"] == len("PERSONA")
    assert meta["breakpoint_index"] == 0
    assert meta["prefix_hash"] == a.prefix_hash()


def test_no_system_message_when_no_stable_or_rag():
    a = PromptAssembler(user="hi")
    msgs = a.build()
    assert msgs == [{"role": "user", "content": "hi"}]
    _, meta = a.build_cached()
    assert meta["breakpoint_index"] == -1


def test_anthropic_blocks_mark_only_the_prefix_cacheable():
    a = PromptAssembler(persona="STABLE", rag="VOLATILE")
    blocks = a.anthropic_system_blocks()
    assert blocks[0]["text"] == "STABLE"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == "VOLATILE"
    assert "cache_control" not in blocks[1]
