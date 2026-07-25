"""Tests for the TTS lane — spoken transform + pipelining (vNext §10.5, Stage 10 A)."""
from __future__ import annotations

import app.live.tts_lane as L


# ---- spoken transform -----------------------------------------------------
def test_code_block_becomes_on_screen():
    sp = L.to_spoken("Here is the solution:\n\n```python\nprint(1)\n```\n\nDone.")
    assert "code on screen" in sp
    assert "print(1)" not in sp                 # code is not read aloud


def test_mermaid_becomes_diagram_on_screen():
    sp = L.to_spoken("See:\n\n```mermaid\nflowchart TD\n A-->B\n```")
    assert "diagram on screen" in sp
    assert "flowchart" not in sp


def test_table_becomes_on_screen():
    sp = L.to_spoken("Results:\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "table on screen" in sp
    assert "|" not in sp


def test_image_announced():
    sp = L.to_spoken("Look: ![a cat](http://x/cat.png)")
    assert "image on screen" in sp and "cat" in sp


def test_markdown_markers_stripped():
    sp = L.to_spoken("# Heading\n\nThis is **bold** and *italic* and `code`.")
    assert "#" not in sp and "*" not in sp and "`" not in sp
    assert "bold" in sp and "italic" in sp


def test_abbreviations_expanded():
    sp = L.to_spoken("Use this, e.g. for speed, i.e. faster.")
    assert "for example" in sp and "that is" in sp


def test_to_spoken_never_raises():
    assert isinstance(L.to_spoken(None), str)   # type: ignore[arg-type]


# ---- sentence pipelining --------------------------------------------------
def test_splits_on_sentences():
    chunks = L.sentence_chunks("First sentence. Second sentence. Third one.")
    assert chunks == ["First sentence.", "Second sentence.", "Third one."]


def test_does_not_split_on_abbreviations():
    chunks = L.sentence_chunks("Use e.g. this approach. Done.")
    assert chunks == ["Use e.g. this approach.", "Done."]


def test_does_not_split_on_decimals():
    chunks = L.sentence_chunks("It handles 3.14 million events. Fast.")
    assert chunks == ["It handles 3.14 million events.", "Fast."]


def test_long_sentence_soft_wraps():
    long = "Clause one here, clause two here, clause three here, clause four here."
    chunks = L.sentence_chunks(long, max_chars=30)
    assert len(chunks) > 1
    assert all(len(c) <= 40 for c in chunks)    # wrapped near the cap


def test_first_chunk_drives_latency():
    assert L.first_chunk("Quick answer. Then detail.") == "Quick answer."


def test_empty_inputs():
    assert L.sentence_chunks("") == []
    assert L.first_chunk("") == ""


def test_single_sentence_one_chunk():
    assert L.sentence_chunks("Just one sentence here") == ["Just one sentence here"]


# ---- voice selection ------------------------------------------------------
def test_explicit_voice_pref_wins():
    assert L.resolve_voice("en_crisp").id == "en_crisp"


def test_language_default_when_no_pref():
    assert L.resolve_voice(language="hi").id == "hi_warm"
    assert L.resolve_voice(language="es").id == "es_warm"


def test_pref_mismatched_language_falls_to_language_default():
    # A user's EN voice pref but a HI session → the HI default, not the EN voice.
    assert L.resolve_voice("en_warm", language="hi").id == "hi_warm"


def test_unknown_pref_falls_to_language_default():
    assert L.resolve_voice("nonexistent", language="es").id == "es_warm"


def test_unknown_language_falls_to_english():
    assert L.resolve_voice(language="xx").id == "en_warm"


def test_voices_for_language():
    en = {v.id for v in L.voices_for_language("en")}
    assert en == {"en_warm", "en_crisp"}
    assert L.voices_for_language("hi") and L.voices_for_language("hi")[0].language == "hi"


def test_resolve_voice_never_raises():
    assert L.resolve_voice(None, language=None).id in L.VOICES  # type: ignore[arg-type]
