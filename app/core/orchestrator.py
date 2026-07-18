"""
The orchestrator: route a detected question through the right pipeline.

For Phase 1 we use *heuristic routing* — pick the tool set by question
type — rather than full LLM tool-calling. Tool-calling support varies
across local Ollama models and noisy classification yields broken
loops. Heuristic routing is more reliable today; the same Orchestrator
class is the seam where tool-calling will plug in later.

Flow per question:
  1. Classify (question_detection.classifier).
  2. Pick tools by type:
       coding             -> code_solver
       behavioral/concept -> resume_lookup -> persona_answer
       smalltalk          -> persona_answer
       else               -> persona_answer
  3. If follow-up, splice the prior Q+A into the answer prompt.
  4. Stream tokens out for SSE/WS consumers.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config_loader import cfg
from app.core.llm_client import LLMError
from app.question_detection.classifier import QuestionMeta, classify
from app.question_detection.context_tracker import ContextTracker, get_tracker
from app.rag import embedder
from app.tools import code_solver, persona_answer, resume_lookup  # registers tools


@dataclass
class AnswerEvent:
    """One event from the orchestrator's streaming output."""
    kind: str                      # "meta" | "token" | "tool" | "done" | "error"
    data: dict = field(default_factory=dict)


@dataclass
class AnswerContext:
    """Everything the orchestrator needs to handle one question."""
    question: str
    session_id: str
    profile: dict | None = None
    resume_id: str | None = None
    db_session: AsyncSession | None = None
    # Optional override for the question type (skip classification).
    forced_type: str | None = None
    # Optional override for difficulty (trivial|standard|hard|expert). The
    # live path supplies this from the prediction agent, which already judged
    # difficulty in the SAME call that cleaned the question — so the
    # orchestrator skips its own difficulty-classification LLM round trip and
    # the answer starts sooner.
    forced_difficulty: str | None = None
    # When True, only utterances classified as a question get an answer;
    # statements, acknowledgements, and small talk are transcribed + tagged
    # (the `meta` event still fires) but produce no answer. The Live Listen
    # path sets this so it doesn't answer the candidate's own filler or the
    # interviewer's non-question remarks. The Resume /ask path leaves it
    # False (every /ask is a question by construction).
    answer_only_questions: bool = False
    # Architecture.md §"Multi-modal question detection" — raw audio
    # behind this turn. When supplied, the classifier fuses prosody
    # features into the decision. Live WS path populates this; the
    # Resume /ask path leaves it None.
    audio: object | None = None
    # Live Listen: skip the bge-m3 follow-up embedding on the hot path. The
    # 2GB model's cold load stalls the event loop and adds 20s+ latency, and
    # follow-ups are already handled by the LLM classifier's is_followup plus
    # the full conversation thread fed to the answer model. The Resume /ask
    # path leaves it False.
    skip_embedding: bool = False
    # Live audio path: use the CONCISE real-time answer prompt, cap the answer
    # length, and guard the first token with a timeout so a stalled model can't
    # hang the turn. The Resume /ask path leaves it False (deep answers).
    live: bool = False
    # Live deliberation (live-conversational-intelligence Phase 3): an optional
    # answer DIRECTIVE (strategy scaffold + plan + knowledge-gap hedge) injected
    # into the SAME generation call (no second blocking LLM call), and a target
    # depth ('concise'|'standard'|'detailed') that maps to the answer length.
    answer_directive: str | None = None
    forced_depth: str | None = None
    # Verifier-driven ESCALATION retry: bypass the pinned fast live model and let
    # the auto-router pick the strongest model for the (forced-expert) difficulty,
    # so a weak/garbled answer is retried on a different, more capable model.
    escalate: bool = False


async def answer_question(
    ctx: AnswerContext,
) -> AsyncGenerator[AnswerEvent, None]:
    """Run the pipeline for one question and yield events as they happen."""
    tracker: ContextTracker = get_tracker(ctx.session_id)

    # 1. Classify.
    recent_qs = tracker.recent_questions()
    if ctx.forced_type:
        meta = QuestionMeta(
            is_question=True,
            type=ctx.forced_type,  # type: ignore[arg-type]
            is_followup=False,
            topic="",
            confidence=1.0,
            source="forced",
        )
    else:
        meta = await classify(ctx.question, recent_qs, audio_np=ctx.audio)

    # 2. Embedding for follow-up detection + tracker.
    # MUST run off the event loop: the embedder lazily loads bge-m3 (~2GB) on
    # first use, and a synchronous call here froze the whole async server
    # (handshakes/other answers stalled) — fatal once answers run concurrently.
    # to_thread keeps the loop responsive; a short timeout means a slow/cold
    # embedder never delays the answer (follow-up detection is best-effort —
    # the LLM classifier already provides is_followup).
    q_emb: list[float] = []
    if not ctx.skip_embedding:
        try:
            q_emb = await asyncio.wait_for(
                asyncio.to_thread(embedder.embed_one, ctx.question), timeout=3.0
            )
        except Exception:  # noqa: BLE001 -- timeout/unavailable; classifier still works
            q_emb = []
    if q_emb:
        meta.is_followup = meta.is_followup or tracker.is_followup(q_emb)

    # Reuse the embedding we just computed (don't make the tracker embed again).
    turn = await tracker.add_question(
        ctx.question, meta.type, meta.topic, embedding=q_emb
    )

    yield AnswerEvent("meta", {
        "type": meta.type,
        "is_question": meta.is_question,
        "is_followup": meta.is_followup,
        "topic": meta.topic,
        "confidence": meta.confidence,
        "source": meta.source,
    })

    # Gate: on the live path, don't answer utterances that aren't questions
    # (the interviewer thinking out loud, the candidate's own "yeah, right",
    # small talk). The transcript + meta already went out so the UI can show
    # the detection; we just skip generating an answer.
    if ctx.answer_only_questions and not meta.is_question:
        yield AnswerEvent("done", {"skipped": "not_a_question"})
        return

    # 3. Decide pipeline by type.
    try:
        if meta.type == "coding":
            # Prefer a resume language + make the code runnable when the live
            # code sandbox is on, and fold in any regeneration critique (the
            # sandbox error on a repair pass) so it reaches the solver.
            _problem = ctx.question
            _extra: list[str] = []
            _code_lang = None
            if ctx.live and getattr(cfg.live, "code_sandbox", False):
                try:
                    from app.live import code_run as _cr
                    _code_lang, _runnable = _cr.pick_language(
                        ctx.question, ctx.profile)
                    _extra.append(f"Write the solution in {_code_lang}.")
                    if _runnable:
                        _extra.append(_cr.runnable_directive(_code_lang))
                except Exception:  # noqa: BLE001
                    _code_lang = None
            if ctx.answer_directive:
                _extra.append(ctx.answer_directive)
            if _extra:
                _problem = ctx.question + "\n\n" + "\n".join(_extra)
            if _code_lang:
                yield AnswerEvent("meta", {"code_language": _code_lang})
            answer_text = ""
            async for chunk in code_solver.solve_text(_problem, language=_code_lang):
                chunk = _as_text(chunk)
                answer_text += chunk
                yield AnswerEvent("token", {"text": chunk})
            tracker.complete_answer(turn, answer_text)
            yield AnswerEvent("done", {})
            return

        # All non-coding paths go through persona_answer.
        context_snippets: list[str] = []
        _profile_q = False
        try:
            from app.live.profile import is_profile_question
            _profile_q = is_profile_question(ctx.question)
        except Exception:  # noqa: BLE001
            _profile_q = False
        # Resume retrieval: typed questions AND any question about the
        # candidate themselves — a promoted/implicit "tell me about your
        # projects" must ground in the resume even though its qtype isn't in
        # the classic list.
        # Latency batch 2026-07-11 (#5): LIVE profile questions skip RAG —
        # the spoken profile prompt embeds the full (compacted) profile, so
        # the chunks were a redundant second copy of the same resume.
        if ctx.resume_id and ctx.db_session \
                and not (ctx.live and _profile_q) \
                and (_profile_q or meta.type in (
            "behavioral",
            "technical_concept",
            "clarification",
        )):
            try:
                # Hard-bounded: retrieval sits on the pre-first-token critical
                # path in live mode; a slow embed/DB round trip must degrade
                # to an ungrounded answer, never delay the first token by
                # more than this (live-latency report 2026-07-08).
                import asyncio as _aio
                hits = await _aio.wait_for(
                    resume_lookup.lookup(
                        query=ctx.question,
                        resume_id=ctx.resume_id,
                        session=ctx.db_session,
                    ),
                    timeout=2.0 if ctx.live else 8.0,
                )
                context_snippets = [h["text"] for h in hits]
                yield AnswerEvent("tool", {
                    "name": "resume_lookup",
                    "hits": len(context_snippets),
                })
            except Exception as exc:  # noqa: BLE001
                yield AnswerEvent("tool", {"name": "resume_lookup", "error": str(exc)})

        # Conversation continuity: pass the recent answered Q+A turns (not
        # just the single previous one) so the model has the whole interview
        # thread and follow-ups like "and how does that scale?" resolve
        # correctly. The current question is the last turn (no answer yet),
        # so we skip it and keep the most recent answered exchanges.
        prior_qa = _recent_thread(tracker, max_pairs=6)

        context_block = "\n\n".join(f"- {s}" for s in context_snippets) or None

        # Capability-aware routing for live answers: classify the question so a
        # hard/expert one escalates to a stronger model. (Live stays single-shot
        # for latency — no multi-round loop here.) When the caller already knows
        # the difficulty (live prediction agent), use it and skip the extra LLM
        # round trip entirely.
        _difficulty = ctx.forced_difficulty or "standard"
        if not ctx.forced_difficulty:
            try:
                from app.core.config_loader import cfg as _cfgd
                if _cfgd.advanced_rag.difficulty_aware_routing:
                    from app.chat.difficulty import classify_difficulty
                    _difficulty = await classify_difficulty(ctx.question)
            except Exception:  # noqa: BLE001
                _difficulty = "standard"

        answer_text = ""
        try:
            # Live deliberation (Phase 3): map the target depth to a length cap
            # and inject the strategy/plan/hedge directive — same call, no extra
            # round trip. Depth absent → today's live cap.
            _depth = ctx.forced_depth
            _max = cfg.llm.live_max_tokens if ctx.live else None
            if _depth == "concise":
                _max = min(_max or 400, 400)
            elif _depth == "detailed":
                # Longer for hard/expert turns — but LIVE keeps its ceiling:
                # an uncapped live answer rides the 10k global default, which
                # is exactly the "never-ending response" the user reported.
                _max = cfg.llm.live_max_tokens if ctx.live else None
            # A live PROFILE answer is dictated aloud verbatim (60-90s spoken
            # ≈ 220 words) — cap it regardless of depth so it can't ramble.
            if ctx.live and _profile_q:
                _max = min(_max or 500, 500)
            # Live answers are DETAILED by default (same prompt as chat) unless
            # the deliberation layer explicitly asked for a concise answer or
            # the operator turned live_detailed off. The fast-model pin +
            # first-token deadline keep the FIRST token quick regardless.
            _concise = ctx.live and (
                _depth == "concise"
                or not bool(getattr(cfg.llm, "live_detailed", True))
            )
            # ONE resume representation per turn (#5): the profile JSON rides
            # the system prompt only where the prompt actually uses it —
            # profile/behavioral turns. General technical questions drop it.
            _needs_profile = _profile_q or meta.type in ("behavioral",
                                                         "candidate")
            stream = persona_answer.stream(
                question=ctx.question,
                profile=(ctx.profile or {}) if _needs_profile else {},
                context=context_block,
                prior_qa=prior_qa,
                qtype=meta.type,
                # Escalation retry: drop the pinned fast model so the auto-router
                # picks the strongest model for the forced-expert difficulty.
                model=(None if ctx.escalate else _live_model()),
                difficulty=_difficulty,
                concise=_concise,
                max_tokens=_max,
                directive=ctx.answer_directive,
                profile_q=ctx.live and _profile_q,
            )
            # First-token watchdog (live only): a stalled / rate-limited free
            # model must not hang the turn at "Thinking" for the full 120s
            # provider timeout. If no first token arrives in time, surface a
            # clear error instead of waiting it out.
            agen = stream.__aiter__()
            first = True
            while True:
                try:
                    if first and ctx.live:
                        chunk = await asyncio.wait_for(
                            agen.__anext__(),
                            timeout=cfg.llm.live_first_token_timeout,
                        )
                    else:
                        chunk = await agen.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    yield AnswerEvent("error", {"detail": (
                        "The model is taking too long to respond (it may be "
                        "rate-limited). Please try again."
                    )})
                    return
                first = False
                chunk = _as_text(chunk)
                answer_text += chunk
                yield AnswerEvent("token", {"text": chunk})
        except LLMError as exc:
            yield AnswerEvent("error", {"detail": str(exc)})
            return

        tracker.complete_answer(turn, answer_text)
        yield AnswerEvent("done", {})

    except Exception as exc:  # noqa: BLE001 -- surface anything else for the UI
        yield AnswerEvent("error", {"detail": f"Unexpected error: {exc}"})


def _live_model() -> str | None:
    """The pinned fast answer model for live interviews (cfg.llm.live_model),
    or None to use the normal auto-router chain."""
    m = getattr(cfg.llm, "live_model", None)
    return m.strip() if isinstance(m, str) and m.strip() else None


def _as_text(chunk) -> str:
    """Coerce a streamed LLM chunk to text. Some provider adapters yield a
    list of content parts or a dict ({"type":"text","text":...}) instead of a
    plain string; without this the live path raised "can only concatenate str
    (not list) to str" and the whole answer errored out."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, list):
        return "".join(_as_text(c) for c in chunk)
    if isinstance(chunk, dict):
        return str(chunk.get("text", "") or "")
    return str(chunk) if chunk is not None else ""


def _recent_thread(tracker: ContextTracker, max_pairs: int = 6) -> str | None:
    """Build a transcript of the most recent ANSWERED Q+A turns in this
    session, oldest first, for conversational continuity. Excludes the
    current (unanswered) turn. Returns None when there's no history yet."""
    turns = [t for t in list(tracker._turns)[:-1] if t.answer]  # noqa: SLF001
    turns = turns[-max_pairs:]
    if not turns:
        return None
    return "\n\n".join(f"Q: {t.question}\nA: {t.answer}" for t in turns)
