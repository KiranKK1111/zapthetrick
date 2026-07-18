"""Speculative multi-model drafting (perceived-speed R4).

Start more than one model generation at once and stream the FIRST one that
produces an opening, cancelling the losers (R4.1/R4.2/R4.4). The winner produces
the whole answer, so its continuation can never contradict its own opening
(R4.3/Property 5 — we deliberately do NOT switch models mid-answer, which is the
only way an "opening" could be contradicted).

Everything is gated by the SpeculationBudget: with `speculation_enabled=False`
or the per-draft concurrency cap reached, `race()` falls back to a single stream
so behavior is byte-for-byte today's (Property 1). All model calls still route +
account normally through the engine.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Callable

from app.perceived.budget import budget as _default_budget

log = logging.getLogger(__name__)

# A factory builds a fresh single-model token stream when called.
StreamFactory = Callable[[], AsyncIterator[str]]


def should_speculate() -> bool:
    """True only when speculative drafting is enabled AND speculation is on AND
    a draft slot is free."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.perceived, "speculative_drafting", False)):
            return False
    except Exception:  # noqa: BLE001
        return False
    return _default_budget.allow(kind="draft")


class SpeculativeDrafter:
    def __init__(self, budget=None) -> None:
        self._budget = budget or _default_budget

    async def race(self, factories: list[StreamFactory],
                   on_winner: Callable[[int], None] | None = None
                   ) -> AsyncIterator[str]:
        """Race the candidate streams; stream the winner's opening + the rest of
        the winner, cancelling losers. Falls back to a single stream when
        speculation is disabled, fewer than 2 candidates are given, or all
        candidates fail before producing a token. `on_winner(idx)` is invoked
        once with the winning candidate's index (0 for the single-stream
        fallbacks) so the caller can attribute the answer to the right model."""
        def _win(idx: int) -> None:
            if on_winner is not None:
                try:
                    on_winner(idx)
                except Exception:  # noqa: BLE001 — attribution must not break streaming
                    pass

        if not factories:
            return
        if len(factories) < 2 or not self._budget.allow(kind="draft"):
            _win(0)
            async for chunk in factories[0]():
                yield chunk
            return

        try:
            from app.core.config_loader import cfg
            cap = max(1, int(getattr(cfg.perceived, "max_concurrent_drafts", 2)))
        except Exception:  # noqa: BLE001
            cap = 2
        candidates = factories[:cap]

        with self._budget.cancel_scope(kind="draft"):
            self._budget.account(len(candidates))
            iters = [f() for f in candidates]

            async def _first(idx: int, it: AsyncIterator[str]):
                """Pull the first chunk; return (idx, chunk, it) or None on
                empty/failed stream."""
                try:
                    async for item in it:
                        return (idx, item, it)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — a failed draft just loses
                    return None
                return None

            tasks = [asyncio.ensure_future(_first(i, it))
                     for i, it in enumerate(iters)]
            winner = None
            pending = set(tasks)
            try:
                while pending and winner is None:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED)
                    for d in done:
                        res = d.result()
                        if res is not None:
                            winner = res
                            break
            finally:
                for t in tasks:
                    if not t.done():
                        t.cancel()

            if winner is None:
                # Every speculative draft failed → one clean single stream.
                _win(0)
                async for chunk in factories[0]():
                    yield chunk
                return

            win_idx, first_chunk, win_it = winner
            _win(win_idx)
            # Cancel/close the losing iterators so their model calls stop (R4.4).
            for i, other in enumerate(iters):
                if i != win_idx:
                    aclose = getattr(other, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception:  # noqa: BLE001
                            pass
            # Stream the winner: its opening, then its continuation (R4.2/R4.3).
            yield first_chunk
            async for chunk in win_it:
                yield chunk


async def speculative_auto_stream(
    messages: list[dict],
    options: dict,
    session_key: str | None,
    n: int = 2,
) -> AsyncIterator[str]:
    """Race `n` independent `engine.route_and_stream` generations. Each uses no
    sticky session so the router's load-spread picks different models; the
    fastest-opening one wins and the rest are cancelled."""
    from app.llm import engine

    # One sink per factory — the engine writes the chosen model into it when a
    # draft commits. Persistent per factory (so a fallback re-call reuses it).
    def _make_factory():
        sink: dict = {}

        def _factory() -> AsyncIterator[str]:
            # Fresh options dict per stream — route_and_stream mutates it (pops
            # difficulty/avoid). session_key=None so the candidates aren't all
            # pinned to the same sticky model. `_route_sink` collects the model.
            opts = dict(options)
            opts["_route_sink"] = sink
            return engine.route_and_stream(list(messages), opts,
                                           session_key=None)

        _factory._sink = sink  # type: ignore[attr-defined]
        return _factory

    factories = [_make_factory() for _ in range(max(2, n))]

    def _on_winner(idx: int) -> None:
        # Re-assert the winner's model as the authoritative last-model, on the
        # REAL session key — undoing any pollution from a losing draft.
        if 0 <= idx < len(factories):
            sink = getattr(factories[idx], "_sink", None)
            if isinstance(sink, dict) and sink.get("model_id"):
                engine.record_winner_model(session_key, sink.get("display_name"),
                                           sink.get("model_id"))

    drafter = SpeculativeDrafter()
    async for chunk in drafter.race(factories, on_winner=_on_winner):
        yield chunk


__all__ = ["SpeculativeDrafter", "speculative_auto_stream", "should_speculate"]
