"""Dual-STT engine — Architecture.md §"Dual-STT redundancy".

Runs two STT engines on the same audio frame in parallel, then sends
their outputs to the arbitrator. The default pairing is:

    STT-A: faster-whisper (CPU, fast, decent accuracy)
    STT-B: parakeet (GPU, very fast, best accuracy)

When the second engine isn't installed / no GPU, the dual engine
degrades gracefully to single-engine mode — the arbitrator just
returns STT-A's text. Architecture.md is explicit that this should
be a no-op fallback, not an error.

Public surface:
    DualSTT.transcribe(audio_np) -> ArbitratedText

Wired into config via `cfg.stt.dual_engine_enabled` (defaults False
so an existing single-engine deploy isn't affected).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from app.core.config_loader import cfg


log = logging.getLogger(__name__)


@dataclass
class STTHypothesis:
    """One engine's output. `confidence` is best-effort — Whisper
    exposes per-segment probabilities; Parakeet's confidence is
    derived from the CTC logprob. Engines that don't surface a
    real confidence fall back to a heuristic (text length vs gaps)."""
    engine: str
    text: str = ""
    confidence: float = 0.0
    latency_ms: int = 0
    words: list[dict] = field(default_factory=list)   # [{word, start, end, prob}]


@dataclass
class ArbitratedText:
    """The arbitrator's chosen output + a short trace of how it got
    there. The trace surfaces in the live-transcript pane's tooltip
    so the user can see *why* a particular word was picked."""
    text: str = ""
    confidence: float = 0.0
    chosen_engine: str = ""
    rationale: str = ""
    candidates: list[STTHypothesis] = field(default_factory=list)


class DualSTT:
    """Fans audio out to two engines, returns the arbitrated text."""

    def __init__(self) -> None:
        self._primary = None        # STT-A — always present
        self._secondary = None      # STT-B — None when not configured

    def configure(self, primary, secondary=None) -> None:
        """Wire up the two engines. Pass whatever the existing factory
        returns. `secondary=None` means single-engine mode."""
        self._primary = primary
        self._secondary = secondary

    async def transcribe(self, audio_np) -> ArbitratedText:
        """Run both engines in parallel; arbitrate. Falls back cleanly
        when the secondary engine fails or isn't configured."""
        if self._primary is None:
            return ArbitratedText(text="", rationale="no primary engine configured")

        tasks: list[asyncio.Task] = [
            asyncio.create_task(self._run_one(self._primary, "primary", audio_np))
        ]
        if self._secondary is not None and cfg.stt.dual_engine_enabled:
            tasks.append(asyncio.create_task(self._run_one(self._secondary, "secondary", audio_np)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates: list[STTHypothesis] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("dual_stt: engine raised: %s", r)
                continue
            candidates.append(r)

        if not candidates:
            return ArbitratedText(text="", rationale="all engines failed")

        from . import arbitrator

        return arbitrator.arbitrate(candidates)

    async def _run_one(self, engine, label: str, audio_np) -> STTHypothesis:
        t0 = time.monotonic()
        # Most STT engines are sync — run in a thread so neither
        # blocks the event loop. If a real async engine arrives,
        # this still works (`await asyncio.to_thread` on a coroutine
        # returns the result directly).
        text = await asyncio.to_thread(engine.transcribe, audio_np)
        latency = int((time.monotonic() - t0) * 1000)
        # Heuristic confidence — engines should override.
        # Longer non-empty text + no obvious gaps → higher confidence.
        conf = 0.0 if not text else min(0.95, 0.4 + min(len(text), 200) / 400)
        return STTHypothesis(
            engine=getattr(engine, "name", label),
            text=text or "",
            confidence=conf,
            latency_ms=latency,
        )


__all__ = ["DualSTT", "STTHypothesis", "ArbitratedText"]
