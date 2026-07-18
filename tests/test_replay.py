"""SSE reconnect/replay buffer (Architecture §15 / #8 Phase B)."""
from __future__ import annotations

from app.api.replay import ReplayBuffer, new_stream_id


def test_new_stream_id_unique():
    assert new_stream_id() != new_stream_id()


def test_since_returns_frames_after_id():
    buf = ReplayBuffer()
    sid = new_stream_id()
    for i in range(1, 6):
        buf.append(sid, i, f"id: {i}\nevent: token\ndata: {{}}\n\n")
    # everything after id 2 → ids 3,4,5
    got = buf.since(sid, 2)
    assert [eid for eid, _ in got] == [3, 4, 5]
    # after the last id → empty (nothing missed)
    assert buf.since(sid, 5) == []
    # from the start
    assert [eid for eid, _ in buf.since(sid, 0)] == [1, 2, 3, 4, 5]


def test_since_unknown_stream_is_none():
    buf = ReplayBuffer()
    assert buf.since("nope", 0) is None


def test_ring_caps_frames_per_stream():
    buf = ReplayBuffer(max_frames_per_stream=3)
    sid = new_stream_id()
    for i in range(1, 6):
        buf.append(sid, i, f"frame{i}")
    got = buf.since(sid, 0)
    # only the most recent 3 are retained
    assert [eid for eid, _ in got] == [3, 4, 5]


def test_ttl_evicts_expired_streams():
    clock = {"t": 1000.0}
    buf = ReplayBuffer(ttl_seconds=10.0)
    buf._now = lambda: clock["t"]        # type: ignore[method-assign]
    old = new_stream_id()
    buf.append(old, 1, "x")
    clock["t"] = 1005.0                   # still fresh
    fresh = new_stream_id()
    buf.append(fresh, 1, "y")            # append triggers eviction pass
    assert buf.since(old, 0) is not None
    clock["t"] = 1020.0                   # old now > ttl past its last touch
    buf.append(fresh, 2, "z")            # eviction runs again
    assert buf.since(old, 0) is None      # evicted
    assert buf.since(fresh, 0) is not None


def test_max_streams_lru_eviction():
    buf = ReplayBuffer(max_streams=2)
    a, b, c = new_stream_id(), new_stream_id(), new_stream_id()
    buf.append(a, 1, "a")
    buf.append(b, 1, "b")
    buf.append(c, 1, "c")                 # evicts the oldest (a)
    assert buf.since(a, 0) is None
    assert buf.since(b, 0) is not None
    assert buf.since(c, 0) is not None


def test_append_refreshes_lru_position():
    buf = ReplayBuffer(max_streams=2)
    a, b, c = new_stream_id(), new_stream_id(), new_stream_id()
    buf.append(a, 1, "a")
    buf.append(b, 1, "b")
    buf.append(a, 2, "a2")               # touch a → b is now oldest
    buf.append(c, 1, "c")                 # evicts b, keeps a
    assert buf.since(a, 0) is not None
    assert buf.since(b, 0) is None
    assert buf.since(c, 0) is not None


def test_stats():
    buf = ReplayBuffer()
    sid = new_stream_id()
    buf.append(sid, 1, "x")
    buf.append(sid, 2, "y")
    s = buf.stats()
    assert s["streams"] == 1 and s["frames"] == 2
