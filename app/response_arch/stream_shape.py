"""Intent-aware STREAM MODE — shaping influences streaming (roadmap Phase 6 #4).

The shaping layer (`content_router` + `layer.finalize`) historically ran only as
a *post-stream* pass over the full text. That means the eight shapes never
affected how bytes reached the client — a table streamed the same
character-at-a-time as prose. This module makes the chosen shape drive a
``StreamMode`` the streaming loop can honour *while* generating:

* prose / plain answers  → smooth **token** streaming (lowest latency to paint);
* structured answers (table, comparison, trade-off, diagram, steps) → **block**
  streaming: hold a logical block until it is well-formed, then release it, so a
  half-written table never flashes on screen;
* code / artifact_set → **block + artifact**: buffer a fenced block until the
  closing fence, then emit it (and, when it is a file, an artifact) atomically.

Pure + fail-open: an unknown shape degrades to token mode.
"""
from __future__ import annotations

from dataclasses import dataclass

from .content_router import Shape

TOKEN = "token"    # stream raw token deltas as they arrive
BLOCK = "block"    # release logical blocks only when well-formed
ARTIFACT = "artifact"  # block mode + emit artifacts on fenced-block close


@dataclass(frozen=True)
class StreamMode:
    """How a turn should be streamed, derived from its shape."""

    name: str                    # token | block | artifact
    block_aware: bool            # assemble + validate logical blocks
    emit_artifacts: bool         # turn closed fenced blocks into artifact frames
    coalesce_hint: int           # suggested min chars per token frame (0 = as-is)

    def as_frame(self) -> dict:
        return {
            "mode": self.name,
            "block_aware": self.block_aware,
            "emit_artifacts": self.emit_artifacts,
        }


_TOKEN = StreamMode(TOKEN, block_aware=False, emit_artifacts=False,
                    coalesce_hint=0)
_BLOCK = StreamMode(BLOCK, block_aware=True, emit_artifacts=False,
                    coalesce_hint=0)
_ARTIFACT = StreamMode(ARTIFACT, block_aware=True, emit_artifacts=True,
                       coalesce_hint=0)

_BY_SHAPE: dict[Shape, StreamMode] = {
    Shape.PROSE: _TOKEN,
    Shape.STEPS: _TOKEN,          # ordered prose reads fine token-by-token
    Shape.TABLE: _BLOCK,
    Shape.COMPARISON: _BLOCK,
    Shape.TRADE_OFF: _BLOCK,
    Shape.DIAGRAM: _BLOCK,
    Shape.CODE: _ARTIFACT,
    Shape.ARTIFACT_SET: _ARTIFACT,
}


def stream_mode_for(shape: Shape | str | None) -> StreamMode:
    """Return the :class:`StreamMode` for a shape. Fail-open → token mode."""
    if shape is None:
        return _TOKEN
    try:
        s = shape if isinstance(shape, Shape) else Shape(shape)
    except Exception:  # noqa: BLE001
        return _TOKEN
    return _BY_SHAPE.get(s, _TOKEN)


__all__ = ["StreamMode", "stream_mode_for", "TOKEN", "BLOCK", "ARTIFACT"]
