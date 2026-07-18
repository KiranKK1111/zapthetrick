"""Long-horizon planner — seeds the initial TODO checklist (P2-4).

Before a big build/edit run, ask the model to split the goal into a short,
dependency-ordered checklist (Claude proactively does this at the start of a
long task). The agent then maintains the list live via the `todo_write` tool.

`plan_todos` is best-effort and provider-agnostic: it routes through the app's
`LLMClient`, parses a JSON array of steps, and falls back to a single-item list
(the task itself) if the model is unavailable or the output is unparseable — so
the run never depends on the planner succeeding. Mockable in tests via the
injected `complete` callable.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable

from app.agent.todos import Todo, normalize_todos

log = logging.getLogger(__name__)

_PLAN_PROMPT = """You are planning a software task. Break it into a SHORT,
ordered checklist of concrete steps (3-7 items) a developer would do, in
dependency order. Keep steps high-level (not individual file edits).

TASK:
{task}
{context}
Reply with ONLY a JSON array, each item:
  {{"content": "<imperative step, e.g. 'Add the /users endpoint'>",
    "activeForm": "<present continuous, e.g. 'Adding the /users endpoint'>"}}
No prose, no code fences."""

# An async (messages, options) -> text callable. Defaults to the app client.
Completer = Callable[[list[dict], dict], Awaitable[str]]

# Step indicators that suggest a task is multi-step enough to warrant a plan.
_MULTISTEP_RE = re.compile(
    r"(\band then\b|\bthen\b|\band also\b|;|\n|\b\d+\.\s|\bfirst\b.*\bthen\b|"
    r"\brefactor\b|\bmigrate\b|\bimplement\b|\bbuild\b|\bcreate a\b|"
    r"\bset up\b|\bredesign\b|\bend[- ]to[- ]end\b)", re.I)


def looks_multistep(task: str, *, min_words: int = 12) -> bool:
    """Heuristic: is `task` big/multi-step enough to deserve an upfront plan?

    Keeps simple one-line edits ('fix the typo') cheap (no extra planning LLM
    call), while building-from-spec and broad refactors get a seeded checklist.
    """
    t = (task or "").strip()
    if not t:
        return False
    if len(t.split()) >= min_words:
        return True
    return bool(_MULTISTEP_RE.search(t))


def _parse_plan(text: str) -> list[Todo]:
    m = re.search(r"\[.*\]", text or "", re.S)
    raw = m.group(0) if m else (text or "")
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return []
    return normalize_todos(data)


async def plan_todos(
    task: str,
    *,
    context: str = "",
    completer: Completer | None = None,
    max_chars_context: int = 1500,
) -> list[Todo]:
    """Produce an initial ordered checklist for `task`. Best-effort.

    Returns a single-item fallback list when the planner can't run or its output
    is unusable, so callers always get a non-empty plan for a non-empty task."""
    task = (task or "").strip()
    if not task:
        return []
    fallback = [Todo(content=task[:200], active_form="Working on the task")]

    if completer is None:
        try:
            from app.core.llm_client import llm

            async def _default(messages: list[dict], options: dict) -> str:
                return await llm.complete(messages, None, options)
            completer = _default
        except Exception:  # noqa: BLE001
            return fallback

    ctx = ""
    if context.strip():
        ctx = "\nPROJECT CONTEXT (for grounding):\n" + context[:max_chars_context] + "\n"
    prompt = _PLAN_PROMPT.format(task=task, context=ctx)
    try:
        text = await completer(
            [{"role": "user", "content": prompt}],
            {"difficulty": "standard", "temperature": 0.1, "num_predict": 600})
    except Exception as exc:  # noqa: BLE001
        log.info("todo planner failed: %s", exc)
        return fallback
    todos = _parse_plan(text)
    return todos or fallback


__all__ = ["plan_todos", "looks_multistep", "Completer"]
