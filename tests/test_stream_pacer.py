"""Stream pacing (perceived-speed R7/R8, task 8.3).

Pins Property 14 (partial): concise-first trigger, acknowledgment emission on a
slow first token, and pacing preserves total token text.
"""
from __future__ import annotations

import asyncio

from app.perceived.pacer import (
    StreamPacer,
    concise_directive,
    should_acknowledge,
    should_concise_first,
)


def test_should_acknowledge():
    assert should_acknowledge(0.6, 0.5) is True       # slow first token
    assert should_acknowledge(0.2, 0.5) is False      # fast enough
    assert should_acknowledge(5.0, 0.0) is False      # budget disabled


def test_should_concise_first():
    assert should_concise_first(2.0, 1.0) is True
    assert should_concise_first(0.3, 1.0) is False
    assert should_concise_first(9.9, 0.0) is False    # threshold disabled
    assert "concisely" in concise_directive().lower()


async def _src(chunks, *, first_delay=0.0):
    for i, c in enumerate(chunks):
        if i == 0 and first_delay:
            await asyncio.sleep(first_delay)
        yield c


def _run(agen):
    async def go():
        out = []
        async for ev in agen:
            out.append(ev)
        return out
    return asyncio.run(go())


def test_pace_preserves_total_text_fast_stream():
    pacer = StreamPacer()
    evs = _run(pacer.pace(_src(["Hel", "lo ", "world"]), ttft_budget_s=0.0))
    tokens = [e["text"] for e in evs if e["kind"] == "token"]
    assert "".join(tokens) == "Hello world"
    assert all(e["kind"] == "token" for e in evs)      # no ack when budget off


def test_pace_emits_ack_on_slow_first_token():
    pacer = StreamPacer()
    evs = _run(pacer.pace(_src(["A", "B", "C"], first_delay=0.05),
                          ttft_budget_s=0.01, ack_text="…"))
    assert evs[0]["kind"] == "ack"                     # acknowledged first (R7.3)
    tokens = [e["text"] for e in evs if e["kind"] == "token"]
    assert "".join(tokens) == "ABC"                    # text preserved


def test_pace_no_ack_when_first_token_fast():
    pacer = StreamPacer()
    evs = _run(pacer.pace(_src(["X", "Y"]), ttft_budget_s=0.5))
    assert all(e["kind"] == "token" for e in evs)
    assert "".join(e["text"] for e in evs) == "XY"
