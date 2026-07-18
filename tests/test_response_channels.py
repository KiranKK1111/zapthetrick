"""Multi-channel parallel streaming (P6 #8)."""
from __future__ import annotations

import asyncio

from app.response_arch.channels import (
    ARTIFACT, CHAT, PROGRESS, ChannelMultiplexer, merge)


def test_multiplexer_tags_channel_and_sequence():
    mux = ChannelMultiplexer()
    f1 = mux.frame(CHAT, "token", {"text": "a"})
    f2 = mux.frame(CHAT, "token", {"text": "b"})
    f3 = mux.frame(PROGRESS, "stage", {"name": "x"})
    assert '"_ch": "chat"' in f1 and '"_seq": 1' in f1
    assert '"_seq": 2' in f2
    assert '"_ch": "progress"' in f3 and '"_seq": 1' in f3   # per-channel seq
    assert mux.channels_seen() == {"chat": 2, "progress": 1}


def test_invalid_channel_defaults_to_chat():
    mux = ChannelMultiplexer()
    assert '"_ch": "chat"' in mux.frame("bogus", "token", {})


def test_merge_fans_in_all_producers():
    async def chat():
        for t in ("hi", "there"):
            yield ("token", {"text": t})

    async def progress():
        yield ("stage", {"name": "planning"})

    async def artifacts():
        yield ("artifact", {"filename": "a.py"})

    async def run():
        frames = []
        async for f in merge({CHAT: chat(), PROGRESS: progress(),
                              ARTIFACT: artifacts()}):
            frames.append(f)
        return frames

    frames = asyncio.run(run())
    joined = "".join(frames)
    assert joined.count("event: token") == 2
    assert "event: stage" in joined and "event: artifact" in joined
    # every frame is channel-tagged
    assert all("_ch" in f for f in frames)


def test_merge_survives_a_failing_producer():
    async def good():
        yield ("token", {"text": "ok"})

    async def bad():
        raise RuntimeError("boom")
        yield  # pragma: no cover

    async def run():
        return [f async for f in merge({CHAT: good(), PROGRESS: bad()})]

    frames = asyncio.run(run())
    assert any("event: token" in f for f in frames)
