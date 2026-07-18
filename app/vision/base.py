"""Local Vision Intelligence Layer — the engine protocol (VisionAnalysis.md).

Mirrors `app/stt/base.py`. A vision engine takes one or more images plus a
task-agnostic PARSE prompt and returns a structured TEXT representation (OCR
text + layout + tables + a short description) — it never answers the user. That
text is then handed to whichever provider TEXT model reasons about the turn, so
no API/provider VISION model is ever used.

Adding an engine = write a module exposing `describe(images, prompt) -> str`
and register it in `app/vision/factory.py::_PROVIDERS`.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Vision(Protocol):
    def describe(self, images: Sequence[bytes], prompt: str) -> str:
        """Return a structured text description of `images` (raw decoded bytes,
        e.g. PNG/JPEG). Never answers the user's question — only reports what is
        visually present. Raises on load/inference failure so the factory can
        fall through to the next engine in the chain."""
        ...
