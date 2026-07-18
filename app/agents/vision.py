"""Vision — extracts problems / diagrams from a screenshot.

P0 when an image attachment is present; otherwise idle. Calls into the
configured `vision_model` (Architecture.md §9).

TODO: implement the OCR + structured-extraction prompt against
google/gemini-2.5-flash or any vision-capable model.
"""
from __future__ import annotations

from ..blackboard.board import Blackboard
from ..blackboard.scheduler import P0
from .base import Agent


class VisionAgent(Agent):
    name = "vision"
    priority = P0
    expected_latency_ms = 1_500
    reads = frozenset({"image_bytes"})
    writes = frozenset({"vision_extract"})

    async def run(self, board: Blackboard) -> None:
        # TODO: vision model call.
        board.write("vision_extract", {"text": "", "diagrams": []}, agent=self.name)
