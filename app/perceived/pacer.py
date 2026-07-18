"""Stream pacing: concise-first + acknowledgment + adaptive pace (R7, R8).

Two latency/perception levers around a token stream:

- **Acknowledgment (R7.3):** if the first token would take longer than the TTFT
  budget, emit an immediate "starting…" acknowledgment frame so the UI shows
  motion within ~50ms even when the model is slow to start.
- **Concise-first (R7.1/R7.2):** when the chosen provider's measured latency is
  high, the prompt layer can request a concise initial answer and offer to
  elaborate. `should_concise_first` / `concise_directive` are the pure decisions
  the prompt builder consults (ProviderHealth feeds the latency in Phase 3;
  until then a measured first-byte timer does).

Adaptive byte-level coalescing of bursty streams is owned by the
`smooth-streaming-rendering` backend coalescer (`advanced_rag.stream_chunk_chars`)
and the client `FramePacer`; this module does NOT duplicate it — `pace` forwards
tokens unchanged (preserving total text) and only injects the acknowledgment.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator


def should_acknowledge(elapsed_to_first_s: float, budget_s: float) -> bool:
    """True when the first token took longer than the (positive) TTFT budget."""
    return budget_s > 0 and elapsed_to_first_s >= budget_s


def should_concise_first(latency_s: float, threshold_s: float) -> bool:
    """True when the provider is slow enough to prefer a concise-first answer."""
    return threshold_s > 0 and latency_s >= threshold_s


def concise_directive() -> str:
    """Prompt directive prepended on the slow-provider path (R7.1/R7.2)."""
    return ("Answer concisely first (a few sentences), then offer to elaborate "
            "if the user wants more detail.")


class StreamPacer:
    async def pace(
        self,
        source: AsyncIterator[str],
        *,
        ttft_budget_s: float = 0.0,
        ack_text: str = "Working on it…",
    ) -> AsyncIterator[dict]:
        """Wrap a token stream, emitting an acknowledgment frame BEFORE the
        first token when it is slower than `ttft_budget_s`. Yields dicts:
        ``{"kind": "ack"|"token", "text": ...}``. Total token text is preserved
        exactly (Property 14)."""
        it = source.__aiter__()
        if ttft_budget_s and ttft_budget_s > 0:
            first_task = asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({first_task}, timeout=ttft_budget_s)
            if not done:
                # First token is slow → acknowledge now (R7.3), then wait for it.
                yield {"kind": "ack", "text": ack_text}
            try:
                first = await first_task
            except StopAsyncIteration:
                return
            yield {"kind": "token", "text": first}
        async for chunk in it:
            yield {"kind": "token", "text": chunk}


__all__ = [
    "StreamPacer",
    "should_acknowledge",
    "should_concise_first",
    "concise_directive",
]
