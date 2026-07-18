"""Iterative tool-use loop for chat (Architecture §13).

The single biggest accuracy lever: instead of guessing, let the model
compute/search *before it answers*, see the result, and continue. This runs a
small bounded reason-act loop **before** the persona streams its final answer:

  1. the model is shown the available tools + a strict JSON protocol and asked
     whether it needs one to answer accurately;
  2. it emits ONE JSON action — ``{"tool": "web_search", "args": {...}}`` — or
     ``{"tool": "final"}`` when it has enough;
  3. the backend runs the tool, frames the result as UNTRUSTED data (§11), and
     feeds it back; the model may call again, up to ``max_rounds``.

The accumulated tool results are returned as framed evidence blocks that the
persona appends to its context, so the *final answer stream stays one smooth
pass* with the evidence already in hand (no interleaving of tool calls into the
token stream). Reuses the Code-In JSON protocol and the orchestrator tool
registry; gated to hard/expert (and the intent profile's tool allow-list).

Everything is fail-open: any error returns whatever evidence was gathered so far
(possibly none), never blocking the answer. The LLM call and tool execution are
injectable so the loop is unit-testable without a provider or real tools.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

log = logging.getLogger(__name__)

# Difficulty ordering for the gate.
_DIFF_ORDER = {"trivial": 0, "standard": 1, "hard": 2, "expert": 3}

# Tools that must never be offered to the chat loop even if registered (the
# persona itself, recursion risks). The workspace-mutating file/bash tools live
# in a different registry (app/agent/tools.py) and aren't reachable here.
_NEVER = frozenset({"persona_answer"})

_PROTOCOL = (
    "You can use tools to gather facts BEFORE you answer. Reply with EXACTLY one "
    "JSON object and nothing else:\n"
    '  - to use a tool:  {"tool": "<name>", "args": { ... }}\n'
    '  - when you have enough to answer well:  {"tool": "final"}\n'
    "Call a tool only when it genuinely improves accuracy (a computation, a "
    "time-sensitive or external fact, a code/graph lookup). Prefer to finish "
    "quickly. Never wrap the JSON in prose or code fences."
)


@dataclass
class ToolLoopResult:
    """Framed UNTRUSTED evidence blocks + the tool-call records (for chips/trace)."""
    evidence: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.evidence)


def gate(difficulty: str | None, min_difficulty: str) -> bool:
    """True when this turn's difficulty is at/above the loop's threshold."""
    return (_DIFF_ORDER.get((difficulty or "standard"), 1)
            >= _DIFF_ORDER.get((min_difficulty or "hard"), 2))


def _config():
    from app.core.config_loader import cfg
    tl = cfg.tool_loop
    return (bool(getattr(tl, "enabled", False)),
            int(getattr(tl, "max_rounds", 3)),
            str(getattr(tl, "min_difficulty", "hard")),
            list(getattr(tl, "tools", []) or []))


def _resolve_tool_names(intent: str | None, default_tools: list[str]) -> list[str] | None:
    """Which tools the loop may offer. Returns None to mean "gated off — no
    tools" (an intent profile that explicitly allows none), else the ordered,
    registry-validated, deduped list.

    Precedence: an enabled intent profile's `tools` allow-list wins; otherwise
    the config default set.
    """
    names: Sequence[str] | None = default_tools
    try:
        from app.clarify import intent_profiles as ip
        if ip.enabled():
            prof = ip.resolve(intent)
            if prof.tools is not None:      # profile constrains tools
                if not prof.tools:          # explicitly no tools for this intent
                    return None
                names = prof.tools
    except Exception:  # noqa: BLE001 — fall back to the config default set
        names = default_tools

    from app.tools import registry
    out: list[str] = []
    for n in names or ():
        if n in _NEVER or n in out:
            continue
        if registry.get(n) is not None:
            out.append(n)
    return out or None


def _tool_docs(names: list[str]) -> str:
    from app.tools import registry
    lines = []
    for n in names:
        t = registry.get(n)
        if t is None:
            continue
        props = ((t.input_schema or {}).get("properties") or {})
        params = ", ".join(props.keys()) if props else "—"
        lines.append(f"- {t.name}: {t.description}  (args: {params})")
    return "\n".join(lines)


def _extract_action(text: str) -> dict | None:
    """Pull the first balanced ``{...}`` object containing a ``"tool"`` key.

    Tolerant of markdown fences and leading prose (models slip both in). Returns
    None when there's no parseable action.
    """
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        # drop a leading ```json / ``` fence
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                frag = s[start:i + 1]
                try:
                    obj = json.loads(frag)
                except json.JSONDecodeError:
                    start = -1
                    continue
                if isinstance(obj, dict) and "tool" in obj:
                    return obj
                start = -1
    return None


def _resolve_args(tool, args: dict, context: dict | None) -> dict:
    """Keep only params the tool declares; merge caller context (context wins for
    known params like conversation_id / resume_id / session)."""
    props = ((getattr(tool, "input_schema", None) or {}).get("properties") or {})
    allowed = set(props.keys())
    out = {k: v for k, v in (args or {}).items() if k in allowed}
    for k, v in (context or {}).items():
        if k in allowed and v is not None:
            out[k] = v
    return out


async def _default_complete(messages: list[dict], difficulty: str) -> str:
    from app.core.llm_client import llm
    return await llm.complete(
        messages, options={"difficulty": difficulty, "temperature": 0.1})


async def _default_run_tool(name: str, args: dict) -> object:
    from app.tools import registry
    tool = registry.get(name)
    if tool is None:
        raise KeyError(name)
    return await tool.handler(**args)


async def run_tool_loop(
    *,
    question: str,
    difficulty: str,
    intent: str | None = None,
    context: dict | None = None,
    history: list[dict] | None = None,
    board=None,
    force: bool = False,
    complete_fn: Callable[[list[dict], str], Awaitable[str]] | None = None,
    run_tool_fn: Callable[[str, dict], Awaitable[object]] | None = None,
) -> ToolLoopResult:
    """Run the bounded reason-act loop. Returns framed evidence + call records.

    Returns an empty result (falsy) when the loop is disabled, the difficulty is
    below threshold, or no tools are available for this intent. `force` bypasses
    the difficulty gate (G6: a time-sensitive turn needs a web lookup regardless
    of level). Never raises.
    """
    result = ToolLoopResult()
    try:
        enabled, max_rounds, min_diff, default_tools = _config()
        if not enabled or (not force and not gate(difficulty, min_diff)):
            return result
        names = _resolve_tool_names(intent, default_tools)
        if not names:
            return result

        complete = complete_fn or _default_complete
        run_tool = run_tool_fn or _default_run_tool
        from app.response_arch.trust import frame_untrusted

        sys = (
            "You are preparing to answer the user's question as accurately as "
            "possible.\n\nAvailable tools:\n" + _tool_docs(names) + "\n\n" + _PROTOCOL
        )
        convo: list[dict] = [{"role": "system", "content": sys}]
        for prior in (history or [])[-4:]:
            r, c = prior.get("role"), prior.get("content")
            if r and c:
                convo.append({"role": r, "content": c})
        convo.append({"role": "user", "content": question})

        for _round in range(max(1, max_rounds)):
            raw = await complete(convo, difficulty)
            action = _extract_action(raw)
            if not action:
                break
            tool = str(action.get("tool") or "").strip()
            if tool in ("final", "", "none"):
                break
            if tool not in names:
                # Model asked for a tool it isn't allowed / doesn't exist — stop
                # rather than loop on an impossible request.
                log.info("tool_loop: model requested unavailable tool %r", tool)
                break
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            from app.tools import registry
            call_args = _resolve_args(registry.get(tool), args, context)
            ok = True
            try:
                raw_result = await run_tool(tool, call_args)
            except Exception as exc:  # noqa: BLE001 — a failing tool shouldn't kill the turn
                ok = False
                raw_result = f"tool error: {exc}"
                log.info("tool_loop: tool %s failed: %s", tool, exc)
            body = (raw_result if isinstance(raw_result, str)
                    else json.dumps(raw_result, default=str))
            if len(body) > 4000:
                # Truncate LOUDLY: the model must know the result is partial
                # (a silent cut reads as "that's everything there was").
                dropped = len(body) - 4000
                body = (body[:4000]
                        + f"\n[... truncated {dropped} chars — result incomplete]")
                log.info("tool_loop: %s result truncated (%d chars dropped)",
                         tool, dropped)
            framed = frame_untrusted(body, label=f"{tool} result")
            result.evidence.append(framed)
            result.calls.append({"tool": tool, "args": call_args, "ok": ok})
            if board is not None:
                _write_board_marker(board, tool, ok)
            # feed the result back and let the model decide the next step
            convo.append({"role": "assistant", "content": raw.strip()[:2000]})
            convo.append({"role": "user", "content": framed})
        return result
    except Exception as exc:  # noqa: BLE001 — the loop must never block the answer
        log.info("tool_loop: aborted (%s) — returning %d evidence blocks",
                 exc, len(result.evidence))
        return result


def _write_board_marker(board, tool: str, ok: bool) -> None:
    """Best-effort tool-chip marker on the blackboard (surfaced by the
    supervisor as a `tool` SSE event). Never raises."""
    try:
        board.write(f"tool:{tool}", {"status": "done" if ok else "error"},
                    agent="tool_loop")
    except Exception:  # noqa: BLE001
        pass


__all__ = ["ToolLoopResult", "run_tool_loop", "gate"]
