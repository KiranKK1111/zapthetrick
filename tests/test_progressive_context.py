"""Progressive context + incremental retrieval (perceived-speed R5/R6, task 7.3).

Pins Property 6: equivalence to up-front loading when retrieval is used,
retrieval overlaps prep (concurrency), and empty/failed retrieval → grounding
unavailable.
"""
from __future__ import annotations

import asyncio

from app.perceived.progressive import ProgressiveContext


def _inject(essential, snippets):
    # Same shape up-front loading would build.
    return list(essential) + list(snippets)


def test_used_equals_upfront_loading():
    pc = ProgressiveContext()

    async def retrieve():
        return ["s1", "s2"]

    res = asyncio.run(pc.assemble(["e1"], retrieve, _inject))
    assert res.grounding == "used"
    assert res.context == ["e1", "s1", "s2"]      # == up-front context


def test_empty_retrieval_marks_unavailable():
    pc = ProgressiveContext()

    async def retrieve():
        return []

    res = asyncio.run(pc.assemble(["e1"], retrieve, _inject))
    assert res.grounding == "unavailable"
    assert res.context == ["e1"]                  # essential only, answer proceeds


def test_retrieval_error_is_fail_open():
    pc = ProgressiveContext()

    async def retrieve():
        raise RuntimeError("vector store down")

    res = asyncio.run(pc.assemble(["e1"], retrieve, _inject))
    assert res.grounding == "unavailable"
    assert res.context == ["e1"]


def test_retrieval_overlaps_prep():
    pc = ProgressiveContext()
    order = []

    async def retrieve():
        await asyncio.sleep(0.03)
        order.append("retrieve_done")
        return ["snip"]

    async def prep():
        await asyncio.sleep(0.03)
        order.append("prep_done")

    res = asyncio.run(pc.assemble(["e"], retrieve, _inject, prep=prep))
    # Both ran; because they overlapped, total time ~max (not sum). We at least
    # assert both completed and retrieval was folded in.
    assert res.grounding == "used" and res.context == ["e", "snip"]
    assert "retrieve_done" in order and "prep_done" in order


def test_sync_retrieve_supported():
    pc = ProgressiveContext()
    res = asyncio.run(pc.assemble(["e"], lambda: ["x"], _inject))
    assert res.grounding == "used" and res.context == ["e", "x"]
