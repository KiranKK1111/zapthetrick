"""Multimodal input in the response envelope (Architecture §12 / #10)."""
from __future__ import annotations

from app.response_arch.envelope import (
    build_envelope, build_input, detect_input_modality, structure_artifacts)


# ---- modality detection --------------------------------------------------

def test_modality_text_when_no_attachments():
    assert detect_input_modality(text="hello") == "text"
    assert detect_input_modality() == "text"


def test_modality_single_kind():
    assert detect_input_modality(images=["a"]) == "image"
    assert detect_input_modality(files=["x.pdf"]) == "document"
    assert detect_input_modality(audio=b"...") == "audio"


def test_text_alongside_one_kind_keeps_that_kind():
    # text + an image is still primarily an image turn, not multimodal
    assert detect_input_modality(text="what is this?", images=["a"]) == "image"


def test_modality_multimodal_when_multiple_kinds():
    assert detect_input_modality(images=["a"], files=["x.pdf"]) == "multimodal"
    assert detect_input_modality(images=["a"], audio=b"x") == "multimodal"


# ---- input block ---------------------------------------------------------

def test_build_input_counts_and_flags():
    inp = build_input(text="hi", images=["a", "b"], files=["x.pdf"])
    assert inp == {"modality": "multimodal", "text": True,
                   "images": 2, "files": 1, "audio": False}


def test_build_input_pure_document():
    inp = build_input(files=["r.pdf"])
    assert inp["modality"] == "document"
    assert inp["text"] is False and inp["images"] == 0


# ---- envelope wiring -----------------------------------------------------

def test_envelope_carries_input_and_meta_modality():
    env = build_envelope(input=build_input(text="q", images=["a"]),
                         input_modality="image")
    assert env["input"]["modality"] == "image"
    assert env["input"]["images"] == 1
    assert env["meta"]["modality"] == "image"


def test_envelope_text_modality_has_no_input_block():
    # a plain text turn records meta.modality but no verbose input block
    env = build_envelope(input_modality="text")
    assert env["meta"]["modality"] == "text"
    assert "input" not in env


def test_envelope_modality_defaults_from_input_block():
    env = build_envelope(input=build_input(files=["x.pdf"]))
    assert env["meta"]["modality"] == "document"
    assert env["input"]["modality"] == "document"


def test_envelope_input_block_gets_modality_backfilled():
    # an input dict lacking 'modality' is backfilled from input_modality
    env = build_envelope(input={"images": 1}, input_modality="image")
    assert env["input"]["modality"] == "image"


# ---- output artifacts ----------------------------------------------------

def test_structure_artifacts_tags_kind_and_modality():
    out = structure_artifacts([
        {"kind": "image", "url": "u"},
        {"kind": "chart"},
        {"filename": "a.py"},          # no kind → defaults to code/text
    ])
    assert out[0] == {"kind": "image", "url": "u", "modality": "image"}
    assert out[1]["modality"] == "visual"
    assert out[2]["kind"] == "code" and out[2]["modality"] == "text"


def test_structure_artifacts_drops_non_dicts():
    assert structure_artifacts(["nope", None, 42, {"kind": "code"}]) == [
        {"kind": "code", "modality": "text"}]


def test_structure_artifacts_empty():
    assert structure_artifacts(None) == []
