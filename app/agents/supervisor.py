"""The Conductor: parses intent, builds the plan, schedules the mesh.

Pseudocode mirrors Architecture.md §5:

    async def supervise(message, session):
        intent = await planner.classify_intent(message)
        if needs_clarification(intent) and mode != "live":
            return await clarifier.ask(message, intent)
        plan = await planner.plan(intent, session.context)
        spawn_p0_agents(plan)
        spawn_p1_agents(plan)
        spawn_p2_in_background()
        async for token in persona.stream():
            # halt + revise on grounder flag, adjust on critic hint
            yield token
        cancel(p1.still_running())
        schedule_reflection(session)

The streaming layer (SSE / WS) consumes [Supervisor.stream], which
yields a sequence of structured events the UI renders as tool chips +
streamed tokens.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterable

from ..blackboard.board import Blackboard
from ..blackboard.schema import (
    KEY_DRAFTS,
    KEY_EVIDENCE,
    KEY_GROUNDING,
    KEY_INTENT,
    KEY_PLAN,
    KEY_QUESTION,
    Drafts,
    Meta,
)
from ..blackboard.scheduler import P0, P1, P2, PriorityScheduler

log = logging.getLogger(__name__)
from .base import Agent, AgentRegistry


@dataclass
class SupervisorEvent:
    """One unit emitted by [Supervisor.stream] for the SSE layer."""
    kind: str                  # 'meta' | 'tool' | 'token' | 'done' | 'error'
    data: dict


# Slot the ClarifierAgent writes alongside `clarifying_questions`.
_KEY_CLARIFY_META = "clarify_meta"

_CLARIFY_META_DEFAULTS = {
    "confidence": 1.0,
    "blocking": False,
    "reason": "",
    "estimated_questions_saved": 0,
    "mode": "ask",
    "assumptions": [],
    "preview": False,
}


def _clarify_payload(questions: list, meta: dict | None) -> dict:
    """Build the `clarify` SSE payload: questions plus turn-level metadata."""
    m = {**_CLARIFY_META_DEFAULTS, **(meta or {})}
    return {
        "questions": questions or [],
        "confidence": m.get("confidence", 1.0),
        "blocking": bool(m.get("blocking", False)),
        "reason": m.get("reason", ""),
        "estimated_questions_saved": m.get("estimated_questions_saved", 0),
        "mode": m.get("mode", "ask"),
        "assumptions": m.get("assumptions", []),
        "preview": bool(m.get("preview", False)),
        # advanced-intent-reasoning additive fields (R5/R7/R2).
        "suggestions": m.get("suggestions", []),
        "calibrated_confidence": m.get("calibrated_confidence",
                                       m.get("confidence", 1.0)),
        "interpretations": m.get("interpretations", []),
    }


def _clarify_decision(meta: dict | None, questions: list, mode: str) -> str:
    """Decide what to do with a clarifier result: 'answer' | 'block' | 'refine'.

    A turn with no questions and no assumptions → answer. Otherwise the
    metadata's `blocking` flag decides: blocking → withhold the answer and ask;
    non-blocking → stream the answer and show the panel as a refinement. Live
    mode never blocks (R39)."""
    m = meta or {}
    has_content = bool(questions) or (
        m.get("mode") == "assume" and bool(m.get("assumptions"))
    )
    if not has_content:
        return "answer"
    blocking = bool(m.get("blocking", False))
    if mode == "live":
        blocking = False  # R39: Live allows non-blocking clarification only
    return "block" if blocking else "refine"


def _clarify_grace_s() -> float:
    """Grace window (seconds) for the clarifier to interrupt a fast answer."""
    try:
        from ..core.config_loader import cfg
        return max(0.0, int(cfg.advanced_rag.clarify_grace_ms) / 1000.0)
    except Exception:  # noqa: BLE001 — never let config break the turn
        return 1.5


def _engine_last_model_for(extras: dict | None) -> str | None:
    """The model the engine just routed to (for the 'model' event at stream
    start). Keyed by conversation if known, else the global last route."""
    try:
        from ..llm.engine import get_last_model
        return get_last_model((extras or {}).get("conversation_id"))
    except Exception:  # noqa: BLE001
        return None


def _clarify_block_s() -> float:
    """Max seconds to block on the clarifier for an ambiguous build request.
    Kept snappy — if the (strong) gate model is slower than this, the caller
    falls back to deterministic default questions, so the user is never left
    waiting long before being asked."""
    try:
        from ..core.config_loader import cfg
        return min(6.0, max(2.0, float(cfg.agents.deadlines_ms.total) / 1000.0))
    except Exception:  # noqa: BLE001
        return 6.0


async def _await_clarifier_blocking(clarify_task, board) -> list:
    """Fully await the clarifier (bounded) before answering — used for
    ambiguous build/project requests that should ask FIRST (Claude-style).
    Returns its question list, or [] on timeout/error (answer proceeds)."""
    try:
        await asyncio.wait_for(
            asyncio.shield(clarify_task), timeout=_clarify_block_s()
        )
    except asyncio.TimeoutError:
        clarify_task.cancel()
        return []
    except Exception:  # noqa: BLE001
        return []
    return board.get("clarifying_questions", []) or []


async def _await_clarifier_grace(clarify_task, board) -> list:
    """When the first answer token is ready but the clarifier is still
    deciding, wait up to the grace window for its verdict. Returns the
    question list (possibly empty). On timeout we cancel the clarifier and
    commit to the answer (returns [])."""
    grace = _clarify_grace_s()
    if grace <= 0:
        clarify_task.cancel()
        return []
    try:
        await asyncio.wait_for(asyncio.shield(clarify_task), timeout=grace)
    except asyncio.TimeoutError:
        clarify_task.cancel()
        return []
    except Exception:  # noqa: BLE001
        return []
    return board.get("clarifying_questions", []) or []


class Supervisor:
    """Owns one turn end-to-end: intent → plan → agents → stream → reflect."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        deadlines_ms: dict[str, int] | None = None,
        latency_budget_ms: int = 8_000,
    ) -> None:
        self.registry = registry
        self.deadlines_ms = deadlines_ms or {}
        self.latency_budget_ms = latency_budget_ms

    # ------------------------------------------------------------------
    async def stream(
        self,
        question: str,
        *,
        mode: str = "chat",
        extras: dict | None = None,
    ) -> AsyncIterator[SupervisorEvent]:
        """Run one turn and yield events for the SSE/WS layer.

        `extras` is a free-form bag the route layer uses to thread per-
        request context (resume_id, db_session, prior_messages, profile,
        session_id) onto the blackboard. Agents read it via
        `board.get("extras")`.
        """
        board = Blackboard()
        board.write(
            "meta",
            Meta(
                latency_budget_ms=self.latency_budget_ms,
                started_at_ms=int(time.time() * 1000),
            ),
            agent="supervisor",
        )
        board.write(KEY_QUESTION, question, agent="supervisor")
        if extras:
            board.write("extras", extras, agent="supervisor")

        # ---- intent + plan (P0, blocking) ----------------------------
        planner = self.registry.get("planner")
        if planner is not None:
            await planner.run(board)
        intent = board.get(KEY_INTENT)
        yield SupervisorEvent("meta", {"intent": _intent_to_dict(intent)})

        # Clarification — the Clarifier is the SOLE decision-maker. We run it
        # (concurrently, Option B) on EVERY substantive turn and let IT decide
        # whether a clarifying question genuinely helps — no upstream keyword /
        # planner gate that could suppress a useful question. It declines on
        # most turns (greetings/clear asks → empty list), so this is "smart in
        # every case" without ever blocking a normal answer.
        #   • trivial turns (greetings/acks) skip it entirely — instant.
        #   • an ambiguous build request (clarify_priority) BLOCKS until it
        #     answers, so it asks language/framework first (Claude-style).
        #   • every other non-trivial turn RACES it against the first token, so
        #     it interrupts only if a question is ready before we commit.
        difficulty = (extras or {}).get("difficulty", "standard")
        clarify_priority = bool((extras or {}).get("clarify_priority"))
        # A REQUIRED missing choice (code request with no language, etc.) — must
        # ask FIRST, never race, or the fast first token answers in a default.
        clarify_required = bool((extras or {}).get("clarify_required"))
        # A download/zip turn packages an existing deliverable — never clarify.
        suppress_clarify = bool((extras or {}).get("suppress_clarify"))
        planner_flagged = bool(
            intent is not None and getattr(intent, "needs_clarification", False)
        )
        # Turns we're CONFIDENT are ambiguous (an unspecified build, a required
        # missing slot, or the planner flagged it) ask FIRST — we block on the
        # clarifier. Every other non-trivial turn still RACES (zero added
        # latency, best-effort interrupt).
        clarify_block = (
            (clarify_priority or clarify_required or planner_flagged)
            and not suppress_clarify)
        clarify_task = None
        should_clarify = (
            mode != "live"
            and not suppress_clarify
            and (clarify_block or difficulty != "trivial")
        )
        if should_clarify:
            clarifier = self.registry.get("clarifier")
            if clarifier is not None:
                # Option B: run the gate CONCURRENTLY with the answer; we only
                # interrupt to ask if it decides BEFORE the first answer token,
                # so a normal answer never waits on it (fast AND smart).
                clarify_task = _safe_create_task(clarifier.run(board))

        # ---- schedule the rest ---------------------------------------
        sched = PriorityScheduler(board, deadlines_ms=self.deadlines_ms)
        for agent in self._p0_agents():
            sched.add(agent)
        for agent in self._p1_agents():
            sched.add(agent)

        # Stream the user-facing response while the scheduler works in
        # parallel on memory / critic / suggester.
        responder = self.registry.get("persona") or self.registry.get("coder")
        scheduled = _safe_create_task(
            sched.run_p0_p1(latency_budget_ms=self.latency_budget_ms)
        )

        emitted_any_token = False
        if responder is not None:
            agen = responder.stream(board)
            # Ambiguous turn (unspecified build OR planner-flagged) → ask FIRST
            # (Claude-style): fully await the clarifier before streaming. If it
            # asks, we interrupt; if it declines, we answer normally.
            if clarify_block:
                questions = []
                if clarify_task is not None:
                    questions = await _await_clarifier_blocking(
                        clarify_task, board
                    )
                meta = board.get(_KEY_CLARIFY_META, {}) or {}
                # GUARANTEE the ask for an ambiguous BUILD request — even if the
                # gate model was slow (blocking timed out) or declined. (Other
                # flagged turns have no generic fallback, so they just proceed.)
                if not questions and clarify_priority:
                    from .clarifier import default_build_questions
                    questions = default_build_questions()
                    meta = {
                        "confidence": 0.3, "blocking": True,
                        "reason": "These choices shape the whole project.",
                        "estimated_questions_saved": 5, "mode": "ask",
                        "assumptions": [], "preview": True,
                    }
                if questions:
                    scheduled.cancel()
                    with contextlib.suppress(BaseException):
                        await scheduled
                    with contextlib.suppress(BaseException):
                        await agen.aclose()
                    # The "ask first" path is always blocking by definition.
                    payload = _clarify_payload(questions, meta)
                    payload["blocking"] = True
                    yield SupervisorEvent("clarify", payload)
                    board.close()
                    return
                clarify_task = None  # decided (declined) — don't race again
            # Race the FIRST token against the clarifier. If the clarifier
            # produces questions first, interrupt and ask; otherwise we commit to
            # the answer and the clarifier result is dropped (too late to ask).
            first_chunk, questions, anext_task = await self._race_first_token(
                agen, clarify_task, board
            )
            clarify_meta = board.get(_KEY_CLARIFY_META, {}) or {}
            decision = _clarify_decision(clarify_meta, questions, mode)
            if decision == "block":
                scheduled.cancel()
                if anext_task is not None:
                    anext_task.cancel()
                    with contextlib.suppress(BaseException):
                        await anext_task
                with contextlib.suppress(BaseException):
                    await agen.aclose()
                payload = _clarify_payload(questions, clarify_meta)
                payload["blocking"] = True
                yield SupervisorEvent("clarify", payload)
                board.close()
                return
            if decision == "refine":
                # Non-blocking: show the panel as a refinement AND stream the
                # answer. Emit the clarify event now, then recover the first
                # token (still pending from the race) and continue normally.
                payload = _clarify_payload(questions, clarify_meta)
                payload["blocking"] = False
                yield SupervisorEvent("clarify", payload)
                if anext_task is not None:
                    try:
                        first_chunk = await anext_task
                    except (StopAsyncIteration, Exception):  # noqa: BLE001
                        first_chunk = None
            elif decision == "answer" and clarify_meta.get("suggestions"):
                # Answer-first, suggest-later (R7): no question, but offer
                # non-blocking follow-up suggestions alongside the answer.
                _sg = _clarify_payload([], clarify_meta)
                _sg["blocking"] = False
                yield SupervisorEvent("clarify", _sg)

            def _flush_tools() -> list[SupervisorEvent]:
                evs: list[SupervisorEvent] = []
                for be in board.drain_pending():
                    payload = _tool_event(be)
                    if payload is not None:
                        evs.append(SupervisorEvent("tool", payload))
                grounding = board.get(KEY_GROUNDING)
                if grounding is not None and getattr(grounding, "unverified", []):
                    evs.append(SupervisorEvent(
                        "tool", {"name": "grounder", "status": "flagged"}))
                return evs

            if first_chunk is not None:
                emitted_any_token = True
                for ev in _flush_tools():
                    yield ev
                # Announce the model that's answering, the moment streaming
                # starts (so the UI shows it BEFORE the answer completes).
                _m = _engine_last_model_for(extras)
                if _m:
                    yield SupervisorEvent("model", {"model": _m})
                yield SupervisorEvent("token", {"text": first_chunk})
            async for token in agen:
                emitted_any_token = True
                for ev in _flush_tools():
                    yield ev
                yield SupervisorEvent("token", {"text": token})

            board.write(
                KEY_DRAFTS,
                Drafts(current=board.get("drafts_current", "")),
                agent=responder.name,
            )
        elif clarify_task is not None:
            clarify_task.cancel()

        if not emitted_any_token:
            yield SupervisorEvent(
                "token",
                {"text": "(no responder agent is enabled — check config.yaml)"},
            )

        # Wait for the scheduler to drain so we can emit final tool events —
        # but BOUND it. A P0/P1 agent (retriever/grounder/memory/critic/
        # suggester) that makes a hanging LLM call must never block the turn's
        # final `done` event: without a cap the response streams fully but the
        # client stays stuck on "Waiting for reply…". On timeout we cancel the
        # stragglers and finish the turn.
        budget_s = (self.latency_budget_ms or 8000) / 1000.0
        try:
            await asyncio.wait_for(scheduled, timeout=budget_s)
        except asyncio.TimeoutError:
            log.warning(
                "supervisor: P0/P1 agents exceeded %.1fs budget — cancelling "
                "stragglers and finishing the turn", budget_s)
            scheduled.cancel()
            with contextlib.suppress(BaseException):
                await scheduled
        except Exception:
            pass

        # Flush any tool events that arrived after the last token.
        for be in board.drain_pending():
            payload = _tool_event(be)
            if payload is not None:
                yield SupervisorEvent("tool", payload)

        # ---- Grounder AUTO-CORRECT (product decision 2026-07-08) ----
        # The grounder runs concurrently with streaming, so a wrong claim can
        # already be on screen. Claude's own pattern: correct yourself IN the
        # same turn rather than retro-editing — so when unverified claims were
        # flagged, ONE bounded model pass writes a short visible correction
        # appended after the answer. Never in live mode; flag-gated;
        # fail-open; hard-capped so `done` is never held hostage.
        try:
            from app.core.config_loader import cfg as _qcfg
            _mode = str((board.get("extras", {}) or {}).get("mode") or "")
            _grounding = board.get(KEY_GROUNDING)
            _claims = list(getattr(_grounding, "unverified", []) or [])
            _min_claims = int(getattr(
                _qcfg.quality, "grounder_autocorrect_min_claims", 2))
            if (len(_claims) >= _min_claims and _mode != "live"
                    and emitted_any_token
                    and bool(getattr(_qcfg.quality, "grounder_autocorrect",
                                     True))):
                # Visible work, not dead air: the UI shows the fact-check is
                # running during the (bounded) tail pause.
                yield SupervisorEvent(
                    "tool", {"name": "grounder", "status": "verifying"})
                _note = await asyncio.wait_for(
                    self._grounder_correction(board, _claims),
                    timeout=float(getattr(
                        _qcfg.quality, "grounder_autocorrect_timeout_s", 3.5)))
                if _note:
                    yield SupervisorEvent(
                        "token", {"text": "\n\n> **Correction:** " + _note})
                    yield SupervisorEvent(
                        "tool", {"name": "grounder", "status": "corrected"})
        except Exception:  # noqa: BLE001 — the correction is best-effort
            pass

        # ---- P2 background (reflection, skill extraction) -----------
        sched.run_p2_background(self._p2_agents())

        yield SupervisorEvent("done", {"latency_ms": self._elapsed_ms(board)})
        board.close()

    async def _grounder_correction(self, board, claims: list) -> str:
        """One bounded model pass: given the flagged claims + the evidence,
        return a 1-3 sentence correction — or '' when the claims are merely
        unverifiable (not wrong): silence beats a nitpick note."""
        try:
            from app.core.llm_client import llm
            evidence = ""
            try:
                ev = board.get(KEY_EVIDENCE)
                chunks = list(getattr(ev, "chunks", []) or [])[:4]
                evidence = "\n".join(
                    str(getattr(c, "text", c))[:400] for c in chunks)
            except Exception:  # noqa: BLE001
                evidence = ""
            claim_lines = "\n".join(f"- {str(c)[:200]}" for c in claims[:5])
            reply = await llm.complete_routed(
                [{"role": "system", "content": (
                    "You verify claims a just-streamed answer made. Given the "
                    "flagged claims and the available evidence, respond with "
                    "a correction of 1-3 sentences ONLY if a claim is "
                    "factually wrong or clearly contradicted. If the claims "
                    "are merely unverifiable, respond with exactly: OK")},
                 {"role": "user", "content": (
                     f"Flagged claims:\n{claim_lines}\n\n"
                     f"Evidence:\n{evidence or '(none retrieved)'}")}],
                options={"purpose": "classifier"})
            text = (reply if isinstance(reply, str) else str(reply or "")).strip()
            if not text or text.upper().startswith("OK") or len(text) > 600:
                return ""
            return text
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------
    async def _race_first_token(self, agen, clarify_task, board):
        """Resolve the race between the answer's first token and the clarifier.

        Returns (first_chunk, questions, anext_task):
          • questions (non-empty) → the clarifier decided to ask BEFORE the
            answer began (or within the grace window). `anext_task` is the still
            -pending first-token future when the clarifier won outright (so a
            non-blocking refine can still stream the answer); it is None when the
            first token was already produced.
          • first_chunk → the first answer token (None for an empty stream),
            with questions None and anext_task None; the clarifier (if any) is
            cancelled — too late to interrupt once we're answering.
        A clarifier that DECLINES (empty list) simply stops racing, so we keep
        waiting for the first token with zero added latency to the answer.
        """
        anext_task = asyncio.ensure_future(agen.__anext__())
        pending = {anext_task}
        if clarify_task is not None:
            pending.add(clarify_task)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
            if clarify_task is not None and clarify_task in done:
                qs = board.get("clarifying_questions", []) or []
                if qs:
                    # Clarifier won; leave the first-token task pending so a
                    # non-blocking refine can still stream the answer.
                    return None, qs, anext_task
                clarify_task = None  # declined — keep waiting for the token
                continue
            if anext_task in done:
                # First token is ready. If the clarifier is still deciding, give
                # it a brief grace window to (possibly) interrupt with questions.
                if clarify_task is not None:
                    qs = await _await_clarifier_grace(clarify_task, board)
                    if qs:
                        try:
                            chunk = anext_task.result()
                        except StopAsyncIteration:
                            chunk = None
                        return chunk, qs, None
                try:
                    return anext_task.result(), None, None
                except StopAsyncIteration:
                    return None, None, None
        return None, None, None

    # ------------------------------------------------------------------
    def _p0_agents(self) -> Iterable[Agent]:
        names = ["retriever", "grounder"]
        return [a for a in (self.registry.get(n) for n in names) if a is not None]

    def _p1_agents(self) -> Iterable[Agent]:
        names = ["memory", "critic", "suggester"]
        return [a for a in (self.registry.get(n) for n in names) if a is not None]

    def _p2_agents(self) -> list[Agent]:
        names = ["reflector"]
        return [a for a in (self.registry.get(n) for n in names) if a is not None]

    def _elapsed_ms(self, board: Blackboard) -> int:
        meta = board.get("meta")
        if meta is None:
            return 0
        return int(time.time() * 1000) - meta.started_at_ms


_NOISY_TOOL_KEYS = frozenset({"meta", "question", "extras"})

# Slots whose value we forward to the UI so the context pane can render
# the actual content (suggestion text, memory hits, citations, critic
# issues), not just a "this agent fired" chip.
_UI_FORWARDED_KEYS = frozenset({
    "evidence", "memory_hits", "grounding", "critiques", "suggestions",
})


def _tool_event(be) -> dict | None:
    """Shape a [BlackboardEvent] for the SSE `tool` frame.

    Returns None for events the UI shouldn't show as tool-chips —
    supervisor housekeeping writes (meta / question / extras), which
    are plumbing rather than agent activity.

    For the [_UI_FORWARDED_KEYS] whitelist, attach a `data` field
    carrying the serialized slot value so the context pane can render
    the actual content (suggestion text, citation chunks, etc).
    """
    if be.agent == "supervisor" and be.key in _NOISY_TOOL_KEYS:
        return None
    payload: dict = {
        "name": be.agent,
        "key": be.key,
        "ts_ms": be.ts_ms,
        "status": "done",
    }
    if be.key in _UI_FORWARDED_KEYS:
        data = _serialize_slot(be.value)
        if data is not None:
            payload["data"] = data
    return payload


def _serialize_slot(value):
    """Convert a typed blackboard slot into a JSON-safe dict.

    Handles dataclasses (which is what we publish for typed slots) and
    falls back to the value unchanged for dicts / primitives. Returns
    None for anything we can't round-trip safely so [json.dumps] in
    the route layer never trips on it.
    """
    if value is None:
        return None
    from dataclasses import asdict, is_dataclass

    try:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, (dict, list, str, int, float, bool)):
            return value
    except Exception:
        return None
    return None


def _intent_to_dict(intent) -> dict:
    if intent is None:
        return {}
    return {
        "type": getattr(intent, "type", "general"),
        "topic": getattr(intent, "topic", ""),
        "urgency": getattr(intent, "urgency", "normal"),
    }


def _safe_create_task(coro):
    import asyncio
    return asyncio.create_task(coro)
