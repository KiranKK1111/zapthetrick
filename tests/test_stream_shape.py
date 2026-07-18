"""Intent-aware stream mode — shaping drives streaming (P6 #4)."""
from __future__ import annotations

from app.response_arch.content_router import Shape
from app.response_arch.stream_shape import (
    ARTIFACT, BLOCK, TOKEN, stream_mode_for)


def test_prose_streams_token_mode():
    m = stream_mode_for(Shape.PROSE)
    assert m.name == TOKEN
    assert m.block_aware is False and m.emit_artifacts is False


def test_table_is_block_aware():
    m = stream_mode_for(Shape.TABLE)
    assert m.name == BLOCK
    assert m.block_aware is True and m.emit_artifacts is False


def test_code_emits_artifacts():
    m = stream_mode_for(Shape.CODE)
    assert m.name == ARTIFACT
    assert m.block_aware is True and m.emit_artifacts is True


def test_artifact_set_emits_artifacts():
    assert stream_mode_for(Shape.ARTIFACT_SET).emit_artifacts is True


def test_accepts_string_and_fails_open():
    assert stream_mode_for("comparison").block_aware is True
    assert stream_mode_for("bogus").name == TOKEN
    assert stream_mode_for(None).name == TOKEN


def test_as_frame():
    assert stream_mode_for(Shape.CODE).as_frame() == {
        "mode": "artifact", "block_aware": True, "emit_artifacts": True}
