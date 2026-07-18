"""Multi-model answer synthesis (Phase 3 — planner → sections → synthesize).

For a complex, *composite* answer (a technical design doc, a multi-part plan) no
single free model is best at every part. This:

  1. **plans** — one LLM call decomposes the task into 2–N independent sections,
     each tagged with the task type it needs (coding / writing / reasoning / …);
  2. **routes + runs** — each section runs on the free model best suited to that
     task type (the semantic router picks it), in parallel;
  3. **synthesizes** — one LLM call merges the sections into a single coherent
     deliverable (dedup, consistent voice), optionally self-evaluating once.

Distinct from `app/orchestration/` (that decomposes *code* tasks into sandboxed
agent roles). This is answer-shaped: prose sections merged into one reply.

Gated to genuinely composite/complex turns (`synthesis.enabled` + the
Understanding pass says large/expert). Fail-open: any failure returns None and
the caller streams the normal single-model answer. LLM + per-section runners are
injectable, so the whole flow is unit-testable without a provider.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

_VALID_TASKS = {"coding", "writing", "reasoning", "architecture", "research",
                "math", "general"}

_PLAN_PROMPT = (
    "You are a lead author planning a COMPOSITE deliverable. Break the user's "
    "request into 2 to {max_sections} INDEPENDENT sections that can be written "
    "separately and then merged into one coherent result. For each section give "
    "a short title, a self-contained instruction (enough to write it alone), and "
    "the single best task type it needs: "
    "coding|writing|reasoning|architecture|research|math|general.\n"
    "Reply with ONLY JSON:\n"
    '{{"sections": [{{"title": "...", "prompt": "...", "task": "..."}}]}}\n'
    "If the request is simple/atomic and should NOT be split, reply "
    '{{"sections": []}}.\n\nUser request:\n{task}'
)

_SYNTH_PROMPT = (
    "Merge the drafted sections below into ONE polished, coherent deliverable "
    "that fully answers the user's request. Keep all substantive content, remove "
    "repetition, ensure a consistent voice and smooth flow, and add brief "
    "connective text where needed. Do not mention that it was written in parts.\n\n"
    "User request:\n{task}\n\n{body}"
)


@dataclass
class Section:
    title: str
    prompt: str
    task: str = "general"
    text: str = ""                       # filled after the section runs
    model: str | None = None             # model that wrote it (obs)


@dataclass
class SynthesisResult:
    text: str
    sections: list[Section] = field(default_factory=list)

    def as_meta(self) -> dict:
        return {"sections": [{"title": s.title, "task": s.task,
                              "model": s.model} for s in self.sections]}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.synthesis, "enabled", False))
    except Exception:  # noqa: BLE001
        return False


def _cfg_int(name: str, default: int) -> int:
    try:
        from app.core.config_loader import cfg
        return int(getattr(cfg.synthesis, name, default))
    except Exception:  # noqa: BLE001
        return default


def _get(u, field: str, default):
    """Read a field from an Understanding object OR its meta dict."""
    if isinstance(u, dict):
        return u.get(field, default)
    return getattr(u, field, default)


def should_orchestrate(understanding) -> bool:
    """Only decompose genuinely composite/complex turns. `understanding` is an
    `app.understanding.Understanding` or its `as_meta()` dict (None → don't)."""
    if understanding is None:
        return False
    try:
        from app.core.config_loader import cfg
        min_cx = str(getattr(cfg.synthesis, "min_output_complexity", "large"))
    except Exception:  # noqa: BLE001
        min_cx = "large"
    order = {"small": 0, "medium": 1, "large": 2}
    cx_ok = order.get(_get(understanding, "output_complexity", "small"), 0) \
        >= order.get(min_cx, 2)
    hard = _get(understanding, "difficulty", "standard") in ("hard", "expert")
    return bool(cx_ok and hard)


def parse_plan(raw: str, *, max_sections: int) -> list[Section]:
    """Tolerant parse of the planner's JSON → validated sections (≤ max). Never
    raises; a bad/absent plan yields []."""
    if not raw:
        return []
    s = raw.strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return []
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return []
    out: list[Section] = []
    for item in (obj.get("sections") or [])[:max_sections]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        task = str(item.get("task") or "general").strip().lower()
        if task not in _VALID_TASKS:
            task = "general"
        out.append(Section(title=title or prompt[:40], prompt=prompt, task=task))
    return out


async def _default_complete(messages: list[dict], *, task_category=None,
                            difficulty="hard") -> str:
    from app.core.llm_client import llm
    opts = {"difficulty": difficulty, "temperature": 0.3}
    if task_category:
        opts["task_category"] = task_category
    return await llm.complete(messages, options=opts)


async def plan_sections(task: str, *, max_sections: int, complete_fn) -> list[Section]:
    prompt = _PLAN_PROMPT.format(max_sections=max_sections, task=task[:4000])
    raw = await complete_fn([{"role": "user", "content": prompt}],
                            task_category="reasoning", difficulty="hard")
    return parse_plan(raw, max_sections=max_sections)


async def run_sections(sections: list[Section], *, run_fn,
                       concurrency: int = 3,
                       timeout_s: float = 90.0) -> list[Section]:
    """Run each section (routed to its best model) in parallel — but BOUNDED: at
    most `concurrency` at once (so a 5-section plan can't fire 5 parallel calls
    and trip free-tier rate limits) and each section capped at `timeout_s` (so a
    hung section can't stall the turn). A section that fails/times out keeps empty
    text and is dropped by the caller (G8)."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run(sec: Section) -> Section:
        try:
            async with sem:
                sec.text = (await asyncio.wait_for(
                    run_fn(sec), timeout=timeout_s)) or ""
        except asyncio.TimeoutError:
            log.info("synthesis: section %r timed out (%.0fs)", sec.title, timeout_s)
            sec.text = ""
        except Exception as exc:  # noqa: BLE001
            log.info("synthesis: section %r failed: %s", sec.title, exc)
            sec.text = ""
        return sec
    return list(await asyncio.gather(*(_run(s) for s in sections)))


async def synthesize(task: str, sections: list[Section], *, complete_fn) -> str:
    body = "\n\n".join(f"## {s.title}\n{s.text}" for s in sections if s.text)
    prompt = _SYNTH_PROMPT.format(task=task[:2000], body=body[:16000])
    return (await complete_fn([{"role": "user", "content": prompt}],
                              task_category="writing", difficulty="hard")).strip()


async def plan_and_run(
    task: str,
    understanding,
    *,
    complete_fn: Callable[..., Awaitable[str]] | None = None,
    run_fn: Callable[[Section], Awaitable[str]] | None = None,
    board=None,
) -> list[Section] | None:
    """The plan + per-section-model phase (no synthesis). Returns the completed
    sections, or None when orchestration doesn't apply / too few succeeded.
    Never raises."""
    try:
        if not enabled() or not should_orchestrate(understanding):
            return None
        complete = complete_fn or _default_complete
        _mark(board, "planning")
        sections = await plan_sections(
            task, max_sections=_cfg_int("max_sections", 5), complete_fn=complete)
        if len(sections) < 2:
            return None
        run = run_fn or _make_section_runner(complete, board)
        sections = await run_sections(
            sections, run_fn=run,
            concurrency=_cfg_int("max_concurrency", 3),
            timeout_s=float(_cfg_int("section_timeout_s", 90)))
        done = [s for s in sections if s.text.strip()]
        return done if len(done) >= 2 else None
    except Exception as exc:  # noqa: BLE001
        log.info("synthesis plan_and_run aborted: %s", exc)
        return None


async def orchestrate(
    task: str,
    understanding,
    *,
    complete_fn: Callable[..., Awaitable[str]] | None = None,
    run_fn: Callable[[Section], Awaitable[str]] | None = None,
    board=None,
) -> SynthesisResult | None:
    """Plan → run sections on per-task models → synthesize (non-streaming; used
    by callers that want the whole result, e.g. with self-eval). Returns None
    when it doesn't apply, so the caller falls back to a single-model answer."""
    try:
        done = await plan_and_run(task, understanding, complete_fn=complete_fn,
                                  run_fn=run_fn, board=board)
        if not done:
            return None
        complete = complete_fn or _default_complete
        _mark(board, "synthesizing")
        merged = await synthesize(task, done, complete_fn=complete)
        if not merged:
            return None
        if _self_eval_on():
            _mark(board, "self-eval")
            merged = await self_eval(task, merged, complete_fn=complete) or merged
        return SynthesisResult(text=merged, sections=done)
    except Exception as exc:  # noqa: BLE001
        log.info("synthesis orchestrate aborted: %s", exc)
        return None


async def _default_stream(messages, *, task_category=None, difficulty="hard"):
    from app.core.llm_client import llm
    opts = {"difficulty": difficulty}
    if task_category:
        opts["task_category"] = task_category
    async for chunk in llm.stream_chat(messages, options=opts):
        yield chunk


async def synthesize_stream(task: str, sections: list[Section], *,
                            board=None, stream_fn=None):
    """Stream the final merge token-by-token (G3) — a Claude-like reveal instead
    of waiting behind progress chips. Self-eval is skipped in streaming mode (it
    needs the full text). Never raises: on error the generator simply ends."""
    body = "\n\n".join(f"## {s.title}\n{s.text}" for s in sections if s.text)
    prompt = _SYNTH_PROMPT.format(task=task[:2000], body=body[:16000])
    _mark(board, "synthesizing")
    stream = stream_fn or _default_stream
    try:
        async for chunk in stream([{"role": "user", "content": prompt}],
                                  task_category="writing", difficulty="hard"):
            yield chunk
    except Exception as exc:  # noqa: BLE001
        log.info("synthesis stream aborted: %s", exc)


def _self_eval_on() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.synthesis, "self_eval", False))
    except Exception:  # noqa: BLE001
        return False


async def self_eval(task: str, merged: str, *, complete_fn) -> str:
    """One critic pass over the merged deliverable → a single revision when gaps
    are found; returns the merged text unchanged when it's already solid. Never
    raises (returns the input on any failure)."""
    try:
        judge = (
            "Review this deliverable against the request. If it fully and "
            "coherently satisfies the request, reply with EXACTLY 'OK'. "
            "Otherwise list the concrete gaps to fix (one per line).\n\n"
            f"Request:\n{task[:1500]}\n\nDeliverable:\n{merged[:12000]}"
        )
        verdict = (await complete_fn(
            [{"role": "user", "content": judge}],
            task_category="reasoning", difficulty="hard")).strip()
        if verdict.upper().startswith("OK") or len(verdict) < 8:
            return merged
        fix = (
            "Revise this deliverable to fix the gaps below, keeping everything "
            f"that is already good.\n\nGaps:\n{verdict[:2000]}\n\n"
            f"Deliverable:\n{merged[:12000]}"
        )
        revised = (await complete_fn(
            [{"role": "user", "content": fix}],
            task_category="writing", difficulty="hard")).strip()
        return revised or merged
    except Exception as exc:  # noqa: BLE001
        log.info("synthesis self-eval skipped: %s", exc)
        return merged


def _make_section_runner(complete, board):
    async def _run(sec: Section) -> str:
        _mark(board, f"section:{sec.task}", sec.title)
        # Route this section to the model best suited to its task type.
        return await complete(
            [{"role": "user", "content": sec.prompt}],
            task_category=sec.task, difficulty="hard")
    return _run


def _mark(board, stage: str, detail: str = "") -> None:
    """Best-effort progress marker → a `tool` chip via the supervisor."""
    if board is None:
        return
    try:
        board.write(f"synthesis:{stage}", {"status": "done", "detail": detail},
                    agent="synthesis")
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "Section", "SynthesisResult", "orchestrate", "should_orchestrate",
    "enabled", "parse_plan", "plan_sections", "run_sections", "synthesize",
    "self_eval", "plan_and_run", "synthesize_stream",
]
