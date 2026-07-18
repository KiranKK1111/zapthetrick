"""LLM-driven tool execution — makes the tool registry LIVE.

The registry was built so the orchestrator could "pick tools heuristically and
run them", but nothing actually dispatched it. This is that executor: given a
question + context, ask the fast model which registered tools (if any) would
help, then call their handlers. The model picks tool-specific arguments; the
caller's `context` (e.g. conversation_id) is merged in and wins, so tools that
need it work without the model knowing it.

Returns ``[{tool, result}]``; empty when no tool helps or anything fails.

Dispatch is **reliability-aware** (roadmap Phase 5 #11): `app.tools.reliability`
already measured which tools actually work, so we let that steer the choice —
the catalog is ordered most-reliable-first, currently-degraded tools are flagged
to the model, and when more calls come back than `max_tools` allows, healthy
tools win the slots. It is a *preference*, never a block: a degraded tool that is
the only path to a capability still runs (loudly). Fail-open — if the reliability
store is empty or raises, dispatch behaves exactly as it did before.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

_PROMPT = (
    "You decide which TOOLS (if any) would help answer the user's message. Most "
    "messages need NO tool — return an empty list then. Only pick a tool when it "
    "clearly helps and you can supply its arguments.\n\n"
    "Tools are listed most-reliable-first. A tool marked [unreliable] has been "
    "failing lately — prefer another tool that can do the job, and only pick it "
    "when nothing else fits.\n\n"
    "Available tools (name — description):\n{catalog}\n\n"
    "User message:\n{question}\n\n"
    "Reply with ONLY compact JSON: "
    '{"calls": [{"name": "<tool>", "arguments": {...}}]}'
)


def _routing_enabled() -> bool:
    """Reliability-aware ordering is ON by default. A config owner may add
    `tools.reliability_routing: false` to fall back to raw registry order."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "tools", None),
                            "reliability_routing", True))
    except Exception:  # noqa: BLE001 — no config → keep the new behaviour
        return True


def _degraded_names(names: list[str]) -> set[str]:
    """Names the reliability store currently considers degraded. A tool with no
    history is never degraded (`is_degraded` needs `min_attempts` of history)."""
    from app.tools import reliability
    try:
        return {n for n in names if reliability.is_degraded(n)}
    except Exception:  # noqa: BLE001 — telemetry must never break dispatch
        return set()


def _order_tools(tools: list) -> list:
    """Candidate tools ordered most-reliable-first (`reliability.rank`). Stable
    for equal scores, so a fresh registry keeps its declared order."""
    if not _routing_enabled():
        return tools
    from app.tools import reliability
    try:
        by_name = {t.name: t for t in tools}
        ordered = [by_name[n] for n in reliability.rank(list(by_name)) if n in by_name]
        return ordered if len(ordered) == len(tools) else tools
    except Exception:  # noqa: BLE001
        return tools


def _prioritise(calls: list[dict], degraded: set[str], max_tools: int) -> list[dict]:
    """Deprioritise degraded tools among the model's chosen calls: healthy calls
    keep their order and take the `max_tools` slots first; degraded calls fall to
    the back (best-of-the-bad first). When every call is degraded — i.e. there is
    no healthy alternative — they still run: we never hard-block the only path to
    a capability."""
    if not calls:
        return []
    if not _routing_enabled() or not degraded:
        return calls[:max_tools]
    from app.tools import reliability
    try:
        healthy = [c for c in calls if c["name"] not in degraded]
        sick = [c for c in calls if c["name"] in degraded]
        order = reliability.rank([c["name"] for c in sick])
        sick.sort(key=lambda c: order.index(c["name"]) if c["name"] in order else 0)
        return (healthy + sick)[:max_tools]
    except Exception:  # noqa: BLE001
        return calls[:max_tools]


def _resolve_args(input_schema: dict, call_args: dict, context: dict | None) -> dict:
    """Final kwargs for a handler: only keys the tool DECLARES (handlers take no
    **kwargs), with caller context winning over the model's guess. This stops
    conversation_id being injected into tools that don't accept it."""
    props = set((input_schema or {}).get("properties", {}) or {})
    merged = {**(call_args or {}), **(context or {})}
    return {k: v for k, v in merged.items() if k in props}


def _parse_calls(raw: str) -> list[dict]:
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        return []
    calls = obj.get("calls") if isinstance(obj, dict) else None
    if not isinstance(calls, list):
        return []
    out = []
    for c in calls:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            out.append({"name": c["name"],
                        "arguments": c.get("arguments") if isinstance(c.get("arguments"), dict) else {}})
    return out


async def run_relevant_tools(question: str, *, context: dict | None = None,
                             allow: set[str] | None = None,
                             exclude: set[str] | None = None,
                             max_tools: int = 2) -> list[dict]:
    """Ask the model which registered tools help, then dispatch them. `allow`
    restricts to a whitelist; `exclude` drops tools handled elsewhere (e.g. the
    code-graph tools, already covered by the retriever's code evidence)."""
    if not (question or "").strip():
        return []
    from app.core.config_loader import cfg
    from app.core.llm_client import LLMError, llm
    from app.tools import registry  # importing app.tools registers all tools
    from app.tools import reliability  # per-tool success/failure tracking (#11)

    tools = [t for t in registry.all_tools()
             if (allow is None or t.name in allow)
             and (exclude is None or t.name not in exclude)]
    if not tools:
        return []
    from app.core.prompt import fill
    # Reliability steers the model's choice: most-reliable-first, and degraded
    # tools are flagged — but only when a healthy alternative is on offer, so a
    # capability with a single (flaky) tool isn't talked out of being used.
    degraded = _degraded_names([t.name for t in tools])
    tools = _order_tools(tools)
    flag = degraded if len(degraded) < len(tools) else set()
    catalog = "\n".join(
        f"- {t.name} — {t.description}" + (" [unreliable]" if t.name in flag else "")
        for t in tools)
    prompt = fill(_PROMPT, catalog=catalog[:4000], question=question[:2000])
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            model=(cfg.llm.classifier_model or cfg.llm.model),
            options={"temperature": cfg.temperature.classifier,
                     "num_predict": cfg.output_tokens.short_json},
        )
    except (LLMError, Exception):  # noqa: BLE001
        return []

    results: list[dict] = []
    # The model may return more calls than we'll run: give the slots to the
    # tools that actually work (degraded ones sink to the back, and off the end
    # of the `max_tools` cap when a healthy alternative was also chosen).
    for call in _prioritise(_parse_calls(raw), degraded, max_tools):
        tool = registry.get(call["name"])
        if tool is None:
            continue
        if tool.name in degraded:
            # Nothing healthier was picked for this capability — run it anyway,
            # but say so: silent use of a known-flaky tool is how bad evidence
            # sneaks into an answer.
            log.warning("tool %s is degraded (reliability %.2f) — running it "
                        "anyway: no healthy alternative was selected",
                        tool.name, reliability.reliability(tool.name))
        # Only pass arguments the tool actually declares — handlers don't take
        # **kwargs, so injecting e.g. conversation_id into web_search (which
        # doesn't accept it) would TypeError.
        args = _resolve_args(tool.input_schema, call["arguments"], context)
        try:
            res = await tool.handler(**args)
            results.append({"tool": tool.name, "result": res})
            reliability.record(tool.name, True)
        except Exception as exc:  # noqa: BLE001 — one bad tool mustn't sink the rest
            log.info("tool %s failed: %s", tool.name, exc)
            reliability.record(tool.name, False)
    return results
