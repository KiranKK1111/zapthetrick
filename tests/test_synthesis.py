"""Multi-model answer synthesis (Phase 3)."""
from __future__ import annotations

import asyncio

from app.chat import synthesis as syn


def _run(coro):
    return asyncio.run(coro)


def _enable(monkeypatch, *, enabled=True, self_eval=False,
            min_cx="large", max_sections=5):
    from app.core import config_loader as cl

    class _S:
        pass
    s = _S()
    s.enabled = enabled
    s.self_eval = self_eval
    s.min_output_complexity = min_cx
    s.max_sections = max_sections
    monkeypatch.setattr(cl.cfg, "synthesis", s, raising=False)


class _U:
    def __init__(self, cx="large", diff="expert"):
        self.output_complexity = cx
        self.difficulty = diff


# ---- gating --------------------------------------------------------------

def test_should_orchestrate_gates_on_complexity_and_difficulty(monkeypatch):
    _enable(monkeypatch)
    assert syn.should_orchestrate(_U("large", "expert"))
    assert not syn.should_orchestrate(_U("small", "expert"))
    assert not syn.should_orchestrate(_U("large", "standard"))
    assert not syn.should_orchestrate(None)


def test_should_orchestrate_accepts_meta_dict(monkeypatch):
    _enable(monkeypatch)
    assert syn.should_orchestrate(
        {"output_complexity": "large", "difficulty": "hard"})


# ---- plan parsing --------------------------------------------------------

def test_parse_plan_valid():
    raw = ('{"sections": [{"title": "Arch", "prompt": "design it", '
           '"task": "coding"}, {"title": "Summary", "prompt": "sum it", '
           '"task": "writing"}]}')
    secs = syn.parse_plan(raw, max_sections=5)
    assert [s.task for s in secs] == ["coding", "writing"]
    assert secs[0].title == "Arch"


def test_parse_plan_caps_and_defaults_task():
    raw = '{"sections": [{"prompt": "a", "task": "bogus"}, {"prompt": "b"}]}'
    secs = syn.parse_plan(raw, max_sections=1)
    assert len(secs) == 1                       # capped
    assert secs[0].task == "general"            # bogus/absent → general


def test_parse_plan_empty_or_garbage():
    assert syn.parse_plan('{"sections": []}', max_sections=5) == []
    assert syn.parse_plan("not json", max_sections=5) == []
    assert syn.parse_plan("", max_sections=5) == []


# ---- orchestrate (injected LLM) ------------------------------------------

def _fake_complete(plan, section_prefix="section", merged="MERGED",
                   verdict="OK", seen=None):
    async def complete(msgs, *, task_category=None, difficulty="hard"):
        if seen is not None:
            seen.append(task_category)
        c = msgs[0]["content"]
        if "lead author planning" in c:
            return plan
        if "Merge the drafted" in c:
            return merged
        if "Review this deliverable" in c:
            return verdict
        if "Revise this deliverable" in c:
            return "REVISED"
        return f"{section_prefix}[{task_category}]"
    return complete


_PLAN2 = ('{"sections": [{"title": "Arch", "prompt": "design", "task": '
          '"coding"}, {"title": "Summary", "prompt": "sum", "task": "writing"}]}')


def test_orchestrate_plans_routes_and_synthesizes(monkeypatch):
    _enable(monkeypatch)
    seen: list = []
    res = _run(syn.orchestrate("write a design doc", _U(),
                               complete_fn=_fake_complete(_PLAN2, seen=seen)))
    assert res is not None
    assert res.text == "MERGED"
    assert [s.task for s in res.sections] == ["coding", "writing"]
    # each section routed to its own task category (+ plan=reasoning, synth=writing)
    assert "coding" in seen and "writing" in seen


def test_orchestrate_none_when_disabled(monkeypatch):
    _enable(monkeypatch, enabled=False)
    assert _run(syn.orchestrate("x", _U(),
                                complete_fn=_fake_complete(_PLAN2))) is None


def test_orchestrate_none_when_not_complex(monkeypatch):
    _enable(monkeypatch)
    assert _run(syn.orchestrate("x", _U("small", "standard"),
                                complete_fn=_fake_complete(_PLAN2))) is None


def test_orchestrate_falls_back_when_atomic(monkeypatch):
    _enable(monkeypatch)
    atomic = '{"sections": []}'
    assert _run(syn.orchestrate("x", _U(),
                                complete_fn=_fake_complete(atomic))) is None


def test_orchestrate_falls_back_when_sections_fail(monkeypatch):
    _enable(monkeypatch)

    async def complete(msgs, *, task_category=None, difficulty="hard"):
        if "lead author planning" in msgs[0]["content"]:
            return _PLAN2
        raise RuntimeError("model down")     # every section fails
    assert _run(syn.orchestrate("x", _U(), complete_fn=complete)) is None


def test_self_eval_revises_on_gaps(monkeypatch):
    _enable(monkeypatch, self_eval=True)
    res = _run(syn.orchestrate(
        "write a design doc", _U(),
        complete_fn=_fake_complete(_PLAN2, merged="MERGED",
                                   verdict="Gap: missing the risk analysis")))
    assert res.text == "REVISED"            # self-eval found a gap → revised


def test_run_sections_bounds_concurrency():
    import asyncio
    live = {"now": 0, "max": 0}

    async def run(sec):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.01)
        live["now"] -= 1
        return f"done {sec.title}"

    secs = [syn.Section(title=str(i), prompt="p") for i in range(6)]
    out = _run(syn.run_sections(secs, run_fn=run, concurrency=2))
    assert all(s.text for s in out)
    assert live["max"] <= 2                  # never more than 2 at once


def test_run_sections_timeout_drops_hung_section():
    import asyncio

    async def run(sec):
        if sec.title == "slow":
            await asyncio.sleep(5)
        return "quick"

    secs = [syn.Section(title="slow", prompt="p"),
            syn.Section(title="fast", prompt="p")]
    out = _run(syn.run_sections(secs, run_fn=run, concurrency=2, timeout_s=0.05))
    by = {s.title: s.text for s in out}
    assert by["slow"] == "" and by["fast"] == "quick"   # hung section dropped


def test_plan_and_run_returns_sections(monkeypatch):
    _enable(monkeypatch)
    done = _run(syn.plan_and_run("write a design doc", _U(),
                                 complete_fn=_fake_complete(_PLAN2)))
    assert done is not None and len(done) == 2
    assert all(s.text for s in done)


def test_synthesize_stream_yields_merge_chunks():
    sections = [syn.Section(title="A", prompt="p", text="alpha"),
                syn.Section(title="B", prompt="p", text="beta")]

    async def stream_fn(messages, *, task_category=None, difficulty="hard"):
        assert "alpha" in messages[0]["content"]     # sections in the merge prompt
        for tok in ["merged ", "answer ", "here"]:
            yield tok

    async def collect():
        out = []
        async for c in syn.synthesize_stream("t", sections, stream_fn=stream_fn):
            out.append(c)
        return out

    import asyncio
    chunks = asyncio.run(collect())
    assert "".join(chunks) == "merged answer here"


def test_synthesize_stream_fail_open():
    async def boom(messages, **k):
        raise RuntimeError("stream down")
        yield  # pragma: no cover

    async def collect():
        out = []
        async for c in syn.synthesize_stream(
                "t", [syn.Section(title="A", prompt="p", text="x")],
                stream_fn=boom):
            out.append(c)
        return out

    import asyncio
    assert asyncio.run(collect()) == []              # ends cleanly, no raise


def test_self_eval_keeps_when_ok(monkeypatch):
    _enable(monkeypatch, self_eval=True)
    res = _run(syn.orchestrate(
        "write a design doc", _U(),
        complete_fn=_fake_complete(_PLAN2, merged="MERGED", verdict="OK")))
    assert res.text == "MERGED"             # 'OK' → unchanged
