"""SSE frame prioritization — answer > progress > telemetry (P6 #14)."""
from __future__ import annotations

from app.response_arch.priority import (
    P_ANSWER, P_PROGRESS, P_STRUCTURE, P_TELEMETRY, PriorityBuffer,
    frame_priority, sort_frames)


def test_priority_classes():
    assert frame_priority("token") == P_ANSWER
    assert frame_priority("plan") == P_STRUCTURE
    assert frame_priority("tool") == P_PROGRESS
    assert frame_priority("trace") == P_TELEMETRY
    assert frame_priority("unknown-event") == P_PROGRESS  # default


def test_sort_is_stable_within_class():
    frames = [("trace", {"a": 1}), ("token", {"t": "x"}),
              ("tool", {}), ("token", {"t": "y"}), ("plan", {})]
    out = sort_frames(frames)
    events = [e for e, _ in out]
    # answers first (in original order), then structure, progress, telemetry
    assert events == ["token", "token", "plan", "tool", "trace"]
    # FIFO preserved within the answer class
    assert out[0][1]["t"] == "x" and out[1][1]["t"] == "y"


def test_priority_buffer_drains_urgent_first():
    buf = PriorityBuffer()
    buf.push("trace", {})
    buf.push("token", {"n": 1})
    buf.push("tool", {})
    buf.push("token", {"n": 2})
    drained = buf.drain()
    assert [e for e, _ in drained] == ["token", "token", "tool", "trace"]
    assert len(buf) == 0                  # cleared after drain


def test_done_always_drains_last():
    buf = PriorityBuffer()
    buf.push("done", {})
    buf.push("token", {"n": 1})
    drained = buf.drain()
    assert [e for e, _ in drained] == ["token", "done"]
