"""Iterative tool-use loop for chat (Architecture §13 / #6)."""
from __future__ import annotations

import asyncio

from app.chat import tool_loop as tl


def _run(coro):
    return asyncio.run(coro)


# ---- pure helpers --------------------------------------------------------

def test_gate_respects_min_difficulty():
    assert tl.gate("hard", "hard")
    assert tl.gate("expert", "hard")
    assert not tl.gate("standard", "hard")
    assert not tl.gate("trivial", "hard")
    assert tl.gate("standard", "standard")


def test_extract_action_plain_json():
    a = tl._extract_action('{"tool": "web_search", "args": {"query": "x"}}')
    assert a == {"tool": "web_search", "args": {"query": "x"}}


def test_extract_action_strips_fence_and_prose():
    txt = 'Sure, let me search.\n```json\n{"tool": "web_search", "args": {}}\n```'
    a = tl._extract_action(txt)
    assert a["tool"] == "web_search"


def test_extract_action_final():
    assert tl._extract_action('{"tool": "final"}') == {"tool": "final"}


def test_extract_action_none_when_no_tool_key():
    assert tl._extract_action('{"foo": 1}') is None
    assert tl._extract_action("just prose, no json") is None


def test_resolve_args_filters_and_merges_context():
    class _T:
        input_schema = {"properties": {"query": {}, "resume_id": {}}}
    out = tl._resolve_args(_T(), {"query": "q", "junk": 1},
                           {"resume_id": "r1", "conversation_id": "c1"})
    assert out == {"query": "q", "resume_id": "r1"}   # junk + conversation_id dropped


# ---- the loop (injected complete_fn + run_tool_fn) -----------------------

def _cfg(monkeypatch, *, enabled=True, max_rounds=3, min_diff="hard",
         tools=("web_search", "code_solver")):
    monkeypatch.setattr(tl, "_config",
                        lambda: (enabled, max_rounds, min_diff, list(tools)))
    # bypass the registry validation in _resolve_tool_names: pretend all
    # configured tools are registered and profiles are off.
    import app.tools.registry as reg

    class _Tool:
        def __init__(self, name):
            self.name = name
            self.description = f"{name} tool"
            self.input_schema = {"properties": {"query": {}}}
    monkeypatch.setattr(reg, "get", lambda n: _Tool(n) if n in tools else None)
    import app.clarify.intent_profiles as ip
    monkeypatch.setattr(ip, "enabled", lambda: False)


def test_loop_disabled_returns_empty(monkeypatch):
    _cfg(monkeypatch, enabled=False)
    res = _run(tl.run_tool_loop(question="q", difficulty="expert",
                                complete_fn=_never, run_tool_fn=_never2))
    assert res.evidence == [] and not res


def test_loop_gated_out_by_difficulty(monkeypatch):
    _cfg(monkeypatch, min_diff="hard")
    res = _run(tl.run_tool_loop(question="q", difficulty="standard",
                                complete_fn=_never, run_tool_fn=_never2))
    assert not res


def test_force_bypasses_difficulty_gate(monkeypatch):
    # G6: a freshness turn forces the loop even on standard difficulty
    _cfg(monkeypatch, min_diff="hard", tools=("web_search",))
    step = {"n": 0}

    async def complete(convo, difficulty):
        step["n"] += 1
        return ('{"tool": "web_search", "args": {"query": "latest"}}'
                if step["n"] == 1 else '{"tool": "final"}')

    async def run_tool(name, args):
        return {"results": ["fresh info"]}

    res = _run(tl.run_tool_loop(question="latest news", difficulty="standard",
                                force=True, complete_fn=complete,
                                run_tool_fn=run_tool))
    assert len(res.evidence) == 1 and "fresh info" in res.evidence[0]


def test_loop_runs_tool_then_finalizes(monkeypatch):
    _cfg(monkeypatch, tools=("web_search",))
    calls = {"n": 0}

    async def complete(convo, difficulty):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"tool": "web_search", "args": {"query": "capital of France"}}'
        return '{"tool": "final"}'

    async def run_tool(name, args):
        assert name == "web_search"
        assert args["query"] == "capital of France"
        return {"results": ["Paris is the capital of France"]}

    res = _run(tl.run_tool_loop(question="capital?", difficulty="hard",
                                complete_fn=complete, run_tool_fn=run_tool))
    assert len(res.evidence) == 1
    assert "Paris" in res.evidence[0]
    assert "UNTRUSTED" in res.evidence[0]           # framed via trust boundary
    assert res.calls == [{"tool": "web_search",
                          "args": {"query": "capital of France"}, "ok": True}]


def test_loop_respects_max_rounds(monkeypatch):
    _cfg(monkeypatch, max_rounds=2, tools=("web_search",))

    async def complete(convo, difficulty):
        return '{"tool": "web_search", "args": {"query": "x"}}'  # never finalizes

    async def run_tool(name, args):
        return "some result"

    res = _run(tl.run_tool_loop(question="q", difficulty="expert",
                                complete_fn=complete, run_tool_fn=run_tool))
    assert len(res.calls) == 2                        # capped at max_rounds


def test_loop_stops_on_unavailable_tool(monkeypatch):
    _cfg(monkeypatch, tools=("web_search",))

    async def complete(convo, difficulty):
        return '{"tool": "delete_everything", "args": {}}'   # not allowed

    res = _run(tl.run_tool_loop(question="q", difficulty="expert",
                                complete_fn=complete, run_tool_fn=_never2))
    assert res.calls == [] and not res


def test_loop_tool_failure_is_captured_not_fatal(monkeypatch):
    _cfg(monkeypatch, tools=("web_search",))
    step = {"n": 0}

    async def complete(convo, difficulty):
        step["n"] += 1
        return ('{"tool": "web_search", "args": {"query": "x"}}'
                if step["n"] == 1 else '{"tool": "final"}')

    async def run_tool(name, args):
        raise RuntimeError("network down")

    res = _run(tl.run_tool_loop(question="q", difficulty="hard",
                                complete_fn=complete, run_tool_fn=run_tool))
    assert res.calls[0]["ok"] is False
    assert "network down" in res.evidence[0]


def test_loop_profile_no_tools_returns_empty(monkeypatch):
    _cfg(monkeypatch, tools=("web_search",))
    import app.clarify.intent_profiles as ip

    class _Prof:
        tools = ()          # profile explicitly allows no tools
    monkeypatch.setattr(ip, "enabled", lambda: True)
    monkeypatch.setattr(ip, "resolve", lambda intent: _Prof())

    res = _run(tl.run_tool_loop(question="q", difficulty="expert", intent="chitchat",
                                complete_fn=_never, run_tool_fn=_never2))
    assert not res


async def _never(*a, **k):
    raise AssertionError("should not be called")


async def _never2(*a, **k):
    raise AssertionError("should not be called")
