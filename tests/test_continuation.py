"""Mid-stream continuation contract (Architecture §15 / #8 resilience).

Covers the pure seam/detection helpers in `app.llm.continuation` and the
`engine.stream_with_continuation` wrapper (with a fake `route_and_stream` +
fake `usage`, so no provider is touched).
"""
from __future__ import annotations

import asyncio

from app.llm import continuation as cont


# ---- pure helpers --------------------------------------------------------

def test_is_cutoff_detects_length_variants():
    assert cont.is_cutoff("length")
    assert cont.is_cutoff("LENGTH")
    assert cont.is_cutoff("max_tokens")
    assert cont.is_cutoff("model_length")
    assert not cont.is_cutoff("stop")
    assert not cont.is_cutoff("tool_calls")
    assert not cont.is_cutoff(None)
    assert not cont.is_cutoff("")


def test_build_continuation_messages_appends_partial_and_instruction():
    base = [{"role": "user", "content": "explain quicksort"}]
    msgs = cont.build_continuation_messages(base, "Quicksort picks a pivot and")
    assert msgs[0] == base[0]                       # original preserved
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-2]["content"] == "Quicksort picks a pivot and"
    assert msgs[-1]["role"] == "user"              # ends on a user turn
    assert "continu" in msgs[-1]["content"].lower()
    assert "Quicksort picks a pivot and" in msgs[-1]["content"]  # tail quoted


def test_build_continuation_trims_tail_but_keeps_full_assistant():
    long = "x" * 5000
    msgs = cont.build_continuation_messages(base_msgs(), long, tail_chars=100)
    assert msgs[-2]["content"] == long             # assistant keeps everything
    # only a 100-char tail is quoted in the instruction (not the whole 5000)
    assert "x" * 100 in msgs[-1]["content"]
    assert "x" * 101 not in msgs[-1]["content"]


def test_dedupe_seam_trims_overlap():
    # continuation repeats the last words already shown
    assert cont.dedupe_seam("the pivot and", " and partitions") == " partitions"
    assert cont.dedupe_seam("abcdef", "def ghi") == " ghi"


def test_dedupe_seam_no_overlap_passthrough():
    assert cont.dedupe_seam("hello world", "next section") == "next section"
    assert cont.dedupe_seam("", "anything") == "anything"
    assert cont.dedupe_seam("tail", "") == ""


def test_seam_deduper_inactive_passthrough():
    s = cont.SeamDeduper("prev tail", active=False)
    assert s.feed("anything") == "anything"
    assert s.feed("more") == "more"
    assert s.flush() == ""


def test_seam_deduper_buffers_then_dedupes_multichunk_overlap():
    # prior tail ended with "value"; the continuation restarts "value that is",
    # and the overlap is spread across several small chunks below the buffer.
    seam = cont.SeamDeduper("chosen pivot value", active=True, buffer=8)
    got = ""
    for ch in ["val", "ue ", "that ", "is"]:
        got += seam.feed(ch)
    got += seam.flush()
    assert got == " that is"       # duplicated "value" trimmed at the seam


def test_seam_deduper_flush_short_stream():
    s = cont.SeamDeduper("abc", active=True, buffer=100)
    assert s.feed("cde") == ""            # never reached buffer
    assert s.flush() == "de"             # trimmed 'c' overlap on flush


def base_msgs():
    return [{"role": "user", "content": "q"}]


# ---- engine wrapper ------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    out = []
    async for c in agen:
        out.append(c)
    return out


def test_wrapper_off_is_passthrough(monkeypatch):
    from app.llm import engine
    orig = engine.route_and_stream

    async def fake_stream(m, o, *, session_key=None, preferred_model_db_id=None):
        for c in ["a", "b", "c"]:
            yield c
    monkeypatch.setattr(engine, "route_and_stream", fake_stream)
    monkeypatch.setattr(engine, "_resilience_cfg", lambda: (False, 2, 8))

    out = _run(_collect(engine.stream_with_continuation([{"role": "user", "content": "q"}], {})))
    assert out == ["a", "b", "c"]


def test_override_forces_continuation_on_when_config_off(monkeypatch):
    # Config flag is OFF, but the caller opts in via options — a long answer that
    # hits a length cutoff must still be continued (the coding-solve path).
    from app.llm import engine
    import app.llm.usage as usage
    state = {"i": 0}
    scripts = [["Half "], ["and rest."]]

    async def fake_stream(m, o, *, session_key=None, preferred_model_db_id=None):
        i = state["i"]; state["i"] += 1
        for c in scripts[i]:
            yield c
    monkeypatch.setattr(engine, "route_and_stream", fake_stream)
    monkeypatch.setattr(usage, "finish_reason",
                        lambda: "length" if state["i"] == 1 else "stop")
    monkeypatch.setattr(engine, "_resilience_cfg", lambda: (False, 2, 8))  # OFF

    out = "".join(_run(_collect(engine.stream_with_continuation(
        [{"role": "user", "content": "q"}],
        {"mid_stream_continuation": True}))))   # caller forces ON
    assert out == "Half and rest."
    assert state["i"] == 2


def test_override_forces_continuation_off_when_config_on(monkeypatch):
    # Config flag is ON, but a live-style caller forces OFF → passthrough, no
    # continuation even on a length cutoff (avoids re-phrased echoes).
    from app.llm import engine
    import app.llm.usage as usage
    state = {"i": 0}

    async def fake_stream(m, o, *, session_key=None, preferred_model_db_id=None):
        state["i"] += 1
        yield "just this."
    monkeypatch.setattr(engine, "route_and_stream", fake_stream)
    monkeypatch.setattr(usage, "finish_reason", lambda: "length")
    monkeypatch.setattr(engine, "_resilience_cfg", lambda: (True, 2, 8))  # ON

    out = "".join(_run(_collect(engine.stream_with_continuation(
        [{"role": "user", "content": "q"}],
        {"mid_stream_continuation": False}))))  # caller forces OFF
    assert out == "just this."
    assert state["i"] == 1   # no continuation despite 'length'


def test_wrapper_continues_on_length_cutoff(monkeypatch):
    from app.llm import engine
    import app.llm.usage as usage

    state = {"i": 0}
    scripts = [["Quicksort picks a "], ["pivot and partitions."]]

    async def fake_stream(m, o, *, session_key=None, preferred_model_db_id=None):
        i = state["i"]; state["i"] += 1
        for c in scripts[i]:
            yield c
    # first attempt cut off (length), second finishes clean
    finishes = ["length", "stop"]
    monkeypatch.setattr(engine, "route_and_stream", fake_stream)
    monkeypatch.setattr(usage, "finish_reason",
                        lambda: finishes[min(state["i"] - 1, 1)])
    monkeypatch.setattr(engine, "_resilience_cfg", lambda: (True, 2, 8))

    out = "".join(_run(_collect(
        engine.stream_with_continuation([{"role": "user", "content": "q"}], {}))))
    assert out == "Quicksort picks a pivot and partitions."
    assert state["i"] == 2   # exactly one continuation


def test_wrapper_stops_at_max_continuations(monkeypatch):
    from app.llm import engine
    import app.llm.usage as usage
    state = {"i": 0}

    async def fake_stream(m, o, *, session_key=None, preferred_model_db_id=None):
        state["i"] += 1
        yield f"part{state['i']} "
    monkeypatch.setattr(engine, "route_and_stream", fake_stream)
    monkeypatch.setattr(usage, "finish_reason", lambda: "length")  # never clean
    monkeypatch.setattr(engine, "_resilience_cfg", lambda: (True, 2, 8))

    out = "".join(_run(_collect(
        engine.stream_with_continuation([{"role": "user", "content": "q"}], {}))))
    # 1 initial + 2 continuations = 3 attempts, then stop despite 'length'
    assert state["i"] == 3
    assert out == "part1 part2 part3 "


def test_wrapper_continues_on_midstream_error(monkeypatch):
    from app.llm import engine
    import app.llm.usage as usage
    state = {"i": 0}

    async def fake_stream(m, o, *, session_key=None, preferred_model_db_id=None):
        i = state["i"]; state["i"] += 1
        if i == 0:
            yield "Half an answer "
            raise RuntimeError("socket reset")   # drop AFTER emitting
        yield "and the rest."
    monkeypatch.setattr(engine, "route_and_stream", fake_stream)
    monkeypatch.setattr(usage, "finish_reason", lambda: "stop")
    monkeypatch.setattr(engine, "_resilience_cfg", lambda: (True, 2, 8))

    out = "".join(_run(_collect(
        engine.stream_with_continuation([{"role": "user", "content": "q"}], {}))))
    assert out == "Half an answer and the rest."
    assert state["i"] == 2


def test_wrapper_reraises_when_nothing_emitted(monkeypatch):
    from app.llm import engine
    import app.llm.usage as usage

    async def fake_stream(m, o, *, session_key=None, preferred_model_db_id=None):
        raise RuntimeError("failed before first token")
        yield  # pragma: no cover — make it a generator
    monkeypatch.setattr(engine, "route_and_stream", fake_stream)
    monkeypatch.setattr(usage, "finish_reason", lambda: None)
    monkeypatch.setattr(engine, "_resilience_cfg", lambda: (True, 2, 8))

    try:
        _run(_collect(engine.stream_with_continuation(
            [{"role": "user", "content": "q"}], {})))
        assert False, "expected the error to propagate"
    except RuntimeError as exc:
        assert "before first token" in str(exc)


def test_wrapper_dedupes_seam_across_attempts(monkeypatch):
    from app.llm import engine
    import app.llm.usage as usage
    state = {"i": 0}
    # continuation repeats "partitions" that was already shown
    scripts = [["...and it partitions"], [" partitions the array."]]

    async def fake_stream(m, o, *, session_key=None, preferred_model_db_id=None):
        i = state["i"]; state["i"] += 1
        for c in scripts[i]:
            yield c
    monkeypatch.setattr(engine, "route_and_stream", fake_stream)
    monkeypatch.setattr(usage, "finish_reason",
                        lambda: "length" if state["i"] == 1 else "stop")
    monkeypatch.setattr(engine, "_resilience_cfg", lambda: (True, 2, 4))

    out = "".join(_run(_collect(
        engine.stream_with_continuation([{"role": "user", "content": "q"}], {}))))
    # " partitions" overlap trimmed at the seam
    assert out == "...and it partitions the array."
