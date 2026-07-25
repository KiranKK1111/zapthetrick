"""Core-level facade for the stable-prefix PromptAssembler (vNext §8.1).

Call sites depend on ``app.core`` (the sanctioned LLM integration boundary)
rather than reaching into ``app.llm`` directly (architecture rule #13). Existing
edges ``chat/live/... -> core`` and ``core -> llm`` carry it — no new coupling.

    from app.core.prompt_layout import PromptAssembler
    convo = PromptAssembler(persona=sys, recent=history, user=question).build()
"""
from __future__ import annotations

from app.llm.prompt_layout import PromptAssembler

__all__ = ["PromptAssembler"]
