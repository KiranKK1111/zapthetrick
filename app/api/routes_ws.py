"""
Live-listen WebSocket endpoint.

Protocol:
  Client opens WS at cfg.server.ws_path (default `/ws/live`) with optional
  query params `?resume_id=...&session_id=...`. Then sends one of:

  - Binary frame: raw PCM audio (float32 or int16 mono, sample_rate from
    config). The server VAD-gates, transcribes, classifies, and streams
    an answer back as JSON text frames.
  - Text frame:   JSON control messages, e.g.
        {"type": "text", "content": "Tell me about yourself"}
        {"type": "flush"}                      (force-emit current utterance)
        {"type": "stop"}                       (close the session cleanly)
        {"type": "start_capture",              (server-side audio capture —
         "source": "system_loopback"}           captures the OTHER party's
                                                 voice for interview/meeting
                                                 help; no client audio needed)
        {"type": "stop_capture"}               (stop server-side capture)

  Server -> client text frames (all JSON):
    {"type": "transcript", "text": "..."}      (final utterance text)
    {"type": "meta", "intent": "...", ...}     (classifier output)
    {"type": "token", "text": "..."}           (answer token stream)
    {"type": "tool", "name": "...", ...}       (orchestrator tool event)
    {"type": "capture", "state": "...", ...}   (server capture status)
    {"type": "done"}
    {"type": "error", "detail": "..."}

  The session_id keys the context tracker, so follow-ups within the same
  session correctly carry prior Q+A.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.audio.stream import AudioStreamSegmenter
from app.core.config_loader import cfg
from app.core.orchestrator import AnswerContext, answer_question
from app.database import Resume
from storage.db import get_session_factory
import json as _json

router = APIRouter()




class _SpecHolder:
    """Frame buffer for one SPECULATIVE answer (started from a '?'-complete
    PARTIAL transcript while the speaker is still pausing). All frames the
    answer pipeline emits are buffered here until the utterance finalizes;
    on a transcript match they are flushed instantly (the LLM ran during the
    end-of-speech silence, so the first tokens are already waiting)."""

    __slots__ = ("frames", "live")

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.live = False


# Set inside a speculation task's context; `send` consults it so the entire
# answer pipeline (detection → generation → verifier) transparently buffers
# without any signature changes.
import contextvars as _contextvars

_SPEC_HOLDER: _contextvars.ContextVar = _contextvars.ContextVar(
    "live_spec_holder", default=None)


def _norm_question(t: str) -> str:
    """Normalize for speculated-vs-final transcript comparison."""
    import re as _re
    return _re.sub(r"[^a-z0-9 ]+", "", (t or "").lower()).strip()


# First words that mark a fragment as a CONTINUATION of the previous
# question rather than a new one ("in spring boot", "and for large
# payloads?", "versus RabbitMQ"). Prepositions/conjunctions only — an
# interrogative opener always reads as a new question.
_CONT_STARTERS = frozenset({
    "in", "on", "for", "with", "and", "or", "versus", "vs", "of", "to",
    "about", "into", "under", "between", "using", "across", "against",
    "via", "without", "within", "per", "from", "as", "at", "by", "over",
    "specifically", "especially", "particularly", "like", "regarding",
    "compared", "plus", "also",
})


def _looks_like_continuation(text: str) -> bool:
    """Does this utterance read as the TAIL of the previous question?

    True for fragments that start on a preposition/conjunction ("in spring
    boot") or short non-question phrases that don't close a thought
    ("various stereotype annotations"). A fragment that opens like a real
    question ("How does…", "What about…") is NOT a continuation."""
    import re as _re
    t = (text or "").strip()
    words = _re.findall(r"[a-z0-9']+", t.lower())
    if not words:
        return False
    if words[0] in _CONT_STARTERS:
        return True
    try:
        from app.question_detection.classifier import heuristic_classify
        if heuristic_classify(t).is_question:
            return False
    except Exception:  # noqa: BLE001
        return False
    try:
        from app.live.hypothesis import completeness
        return len(words) <= 6 and completeness(t) != "complete"
    except Exception:  # noqa: BLE001
        return False


def _merge_continuation(prev: str, cont: str) -> str:
    """Stitch a continuation fragment onto the previously committed question
    as ONE sentence. The head's terminal '?'/'.' was the ASR's guess on a
    premature endpoint — the continuation proves it wrong, so it is dropped;
    only the tail's own punctuation survives. The boundary word is de-duped
    ("…tell me" + "me about…")."""
    a = (prev or "").strip().rstrip(" ?.!,;:")
    b = (cont or "").strip()
    aw, bw = a.split(), b.split()
    if aw and bw and aw[-1].lower().strip("?.!,;:") == bw[0].lower().strip("?.!,;:"):
        b = " ".join(bw[1:])
    return (a + " " + b).strip()


def _speculation_worthy(text: str) -> bool:
    """Should this partial transcript start a speculative answer?

    A trailing '?' plus the question heuristic is the strong signal. (The
    old gate also demanded heuristic confidence ≥ 0.7, which requires a
    RECOGNIZED question type — that silently excluded most real questions,
    e.g. "How would you scale Kafka?", so speculation almost never fired.)
    Without the '?' (ASR drops it sometimes, and imperative questions —
    "Tell me about…", "Explain…" — never carry one) we still speculate when
    the heuristic reads it as a question AND the tail is not grammatically
    dangling ("How would you…" mid-sentence) AND it is long enough to be a
    whole thought. A wrong guess costs one cancelled LLM call; a right one
    hides the entire first-token wait inside the endpoint silence."""
    t = (text or "").rstrip()
    if not t:
        return False
    words = t.split()
    # A continuation-shaped fragment ("in spring boot?") belongs to the
    # previous question — speculating on it standalone is always wasted.
    if words and words[0].lower().strip("?.!,;:") in _CONT_STARTERS:
        return False
    try:
        from app.question_detection.classifier import heuristic_classify
        h = heuristic_classify(t)
    except Exception:  # noqa: BLE001
        return False
    if not h.is_question:
        return False
    try:
        from app.live.hypothesis import completeness
        # `completeness` sees through an ASR-guessed '?': a dangling stem
        # ("Can you tell me?") classifies incomplete and must not speculate.
        if completeness(t) == "incomplete":
            return False
    except Exception:  # noqa: BLE001
        return False
    if t.endswith("?"):
        return len(words) >= 3
    return len(words) >= 4


def _layer_failed() -> None:
    """A degraded optional layer is skipped, never fatal — but LOUDLY: the
    silent except/pass pattern made feature failures invisible (a layer could
    be broken for weeks with zero signal). Logs the failing call site + the
    exception; called only from `except` blocks."""
    import logging
    import sys as _sys
    frame = _sys._getframe(1)
    logging.getLogger("zapthetrick.live").info(
        "live layer degraded (routes_ws.py:%d)", frame.f_lineno, exc_info=True)


# Human-readable intent labels for the answer bubble header. The classifier's
# `qtype` is a machine token (technical_concept, behavioral, …); this turns it
# into the clear, accurate phrase the UI shows on top of each answer.
_INTENT_LABELS = {
    "coding": "Coding problem",
    "technical_concept": "Technical concept",
    "behavioral": "Behavioral (STAR)",
    "clarification": "Clarification",
    "system_design": "System design",
    "smalltalk": "Small talk",
    "implicit": "Open-ended prompt",
    "hypothetical": "Scenario question",
    "rhetorical": "Rhetorical",
}

# Topic tags that carry no information — appending them produces the ugly
# "Question — unknown" header the UI used to show.
_EMPTY_TOPICS = {"", "unknown", "none", "n/a", "general", "misc"}

# Difficulty ladder (weak → strong) for verification-retry escalation. Higher
# difficulty routes to a more capable model (app/chat/difficulty.py + router).
_DIFFICULTY_ORDER = ("trivial", "standard", "hard", "expert")


def _escalate_difficulty(current: str | None, stage: int, max_retries: int,
                         gibberish: bool = False) -> str:
    """The difficulty to route a verification retry at. A garbled answer, or the
    FINAL retry, forces 'expert' (strongest tier → different, stronger model);
    earlier retries bump one tier up from the current difficulty."""
    if gibberish or stage >= max(1, max_retries):
        return "expert"
    try:
        i = _DIFFICULTY_ORDER.index((current or "standard").strip().lower())
    except ValueError:
        i = 1
    return _DIFFICULTY_ORDER[min(i + 1, len(_DIFFICULTY_ORDER) - 1)]


def _intent_label(qtype: str | None, *, topic: str | None = None,
                  is_followup: bool = False) -> str:
    """A concise, human-readable intent for the answer header, e.g.
    'Follow-up · Technical concept — Kafka'."""
    base = _INTENT_LABELS.get((qtype or "").strip().lower(), "Question")
    if is_followup:
        base = f"Follow-up · {base}"
    t = (topic or "").strip()
    if (t and t.lower() not in _EMPTY_TOPICS
            and t.lower() not in base.lower()):
        base = f"{base} — {t}"
    return base


# Words-per-second treated as "fast" when normalizing the prosody speech rate
# into the [0,1] range app/live/emotion.py expects.
_SPEECH_RATE_FAST_WPS = 4.0


def _emotion_signal(audio_np, utterance: str) -> tuple[dict | None, str]:
    """ADVISORY Emotion_Signal (app/live/emotion.py) derived from the
    utterance's prosody — calm / stressed / rushed / hesitant.

    Returns `(meta_dict, delivery_note)`; `(None, "")` when the flag is off,
    there is no audio, or the prosody analyzer is unavailable (its Praat/librosa
    backends are OPTIONAL native deps — the numpy fallback is used when they're
    missing, and a hard failure simply yields no signal).

    The module is explicit that this is advisory and NEVER decisive: it must not
    gate whether we answer and must not override the decision engine. Callers
    surface it as additive meta and, at most, a soft delivery hint.

    This is CPU work (numpy/DSP), so it is never called inline on the answer
    path — `_run_answer` fires it into a thread as a background task and
    `_generate_answer` consumes the result non-blockingly.
    """
    if not getattr(cfg.live, "emotion_signal", False) or audio_np is None:
        return None, ""
    try:
        from app.live import emotion as _emotion
        from app.question_detection.prosody_analyzer import analyze as _prosody

        feats = _prosody(audio_np)
        rate = float(getattr(feats, "speech_rate_wps", 0.0) or 0.0)
        if rate <= 0.0:
            # The analyzer's backends don't all fill speech_rate — derive it
            # from the transcript's word count over the measured duration.
            dur_s = float(getattr(feats, "duration_ms", 0) or 0) / 1000.0
            words = len((utterance or "").split())
            rate = (words / dur_s) if (dur_s >= 0.2 and words) else 0.0
        sig = _emotion.analyze(
            energy=float(getattr(feats, "energy_peak_at_end", 0.0) or 0.0),
            pitch_var=float(getattr(feats, "pitch_rise_end", 0.0) or 0.0),
            speech_rate=(max(0.0, min(1.0, rate / _SPEECH_RATE_FAST_WPS))
                         if rate > 0.0 else None),
            # Disfluencies are the candidate-coaching layer's job (and the
            # transcript repair strips fillers before we get here), so we feed
            # prosody only — exactly what this module's docstring describes.
        )
        return sig.to_dict(), _emotion.delivery_note(sig)
    except Exception:  # noqa: BLE001
        _layer_failed()
        return None, ""


def _apply_emotion(directive, extra: dict, cached) -> str | None:
    """Fold an advisory emotion signal into the answer metadata, and at most a
    SOFT tone hint into the answer directive. Returns the (possibly extended)
    directive; `extra` is mutated additively.

    Never decisive: a signal that doesn't declare itself advisory is ignored
    outright, and nothing here can skip/abort a turn. Never raises."""
    try:
        if not cached:
            return directive
        sig, note = cached
        if not sig or sig.get("advisory") is not True:
            return directive        # not advisory → refuse to act on it at all
        extra["emotion"] = sig
        if note:
            return (directive + "\n" + note).strip() if directive else note
        return directive
    except Exception:  # noqa: BLE001
        _layer_failed()
        return directive


def _coaching_tips(text: str) -> list[str]:
    """Candidate DELIVERY coaching (app/live/coach.py) for the candidate's own
    utterance — fillers / length / missing concrete example.

    This is feedback FOR the candidate, not interview content: it is surfaced as
    its own meta frame and MUST NEVER enter an answer prompt. Never raises."""
    if not getattr(cfg.live, "delivery_coaching", False):
        return []
    try:
        from app.live import coach as _coach
        return list(_coach.coach(text or ""))
    except Exception:  # noqa: BLE001
        _layer_failed()
        return []


@router.websocket("/ws/live")
async def live_listen(
    websocket: WebSocket,
    resume_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    mode: str | None = Query(default=None),
):
    """Long-lived audio + Q&A WebSocket. One per Live Listen tab.

    `mode` selects the capture/answering behaviour:
      • "standard" (default) — real interview: the interviewer (system loopback)
        is answered, the candidate's own mic is absorbed/never answered, and the
        candidate-echo suppressor is active. Relies on audio-CHANNEL separation.
      • "solo" — testing: ONE audio source drives everything. Whatever comes in
        and reads as a question is answered, regardless of who/what source it
        came from. The candidate-absorb branch and the echo suppressor are
        DISABLED (in a solo test the same voice asks and reads answers aloud, so
        those would wrongly suppress). The is-question gate stays ON in both.
    """
    await websocket.accept()
    sid = session_id or str(uuid.uuid4())
    # Per-connection mode (query param wins; else the config default).
    solo_mode = ((mode or "").strip().lower() == "solo"
                 if mode else bool(getattr(cfg.live, "solo_mode", False)))
    # RESUME SCOPING (per-session isolation): the resume that grounds this
    # session's answers is the one PERSISTED on the Session row, not whatever
    # the client sent on the query string. The client-supplied `resume_id`
    # resolves to the globally "active" (last-uploaded) resume, so trusting
    # it leaked one session's resume into every other session. When this
    # session has its own persisted link we use THAT; the query param is only
    # a fallback for an ad-hoc session that has no row yet.
    if session_id:
        try:
            _has_row, _linked = await _load_session_resume_id(session_id)
            if _has_row:
                # The session row is authoritative — even a NULL link means
                # "no resume for THIS session", so a stale query-param resume
                # (the global active one) must NOT bleed in.
                resume_id = _linked
        except Exception:  # noqa: BLE001 — never block connect on the lookup
            import logging
            logging.getLogger("zapthetrick.live").exception(
                "Live: session resume lookup failed for session_id=%s",
                session_id)
    # Loading the resume profile must NEVER kill the connection. If the
    # lookup fails (bad id, DB hiccup, schema drift), log it and continue
    # Per-connection near-duplicate question guard (see _generate_answer).
    from app.live.dedup import QuestionDeduper
    _deduper = QuestionDeduper(
        window_s=float(getattr(cfg.live, "question_dedup_window_s", 20.0)),
        similarity=float(getattr(cfg.live, "question_dedup_similarity", 0.87)),
        semantic=bool(getattr(cfg.live, "question_dedup_semantic", True)),
        semantic_similarity=float(
            getattr(cfg.live, "question_dedup_semantic_sim", 0.90)))
    # with no profile — Live still works, just without resume grounding.
    # An unguarded failure here closed the socket before "ready" (close
    # code 1006), which looked like "Connect does nothing" on the client.
    profile = None
    try:
        profile = await _load_profile(resume_id)
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger("zapthetrick.live").exception(
            "Live: resume profile load failed for resume_id=%s", resume_id
        )

    # (Vocabulary boosting removed — cloud STT needs no hardcoded/biasing
    # term list; the question-prediction agent cleans up the transcript.)

    # Concurrent answers stream over one socket, so every send goes through a
    # lock — Starlette's WebSocket.send is not safe for overlapping writers.
    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        # A speculation task buffers its frames until its transcript is
        # confirmed; everything else sends straight to the socket.
        holder = _SPEC_HOLDER.get()
        if holder is not None and not holder.live:
            holder.frames.append(payload)
            return
        async with send_lock:
            await websocket.send_text(_json.dumps(payload))

    # Background answer tasks, one per detected question. Decoupling answer
    # generation from the segmenter is what lets TWO questions asked close
    # together be answered SIMULTANEOUSLY — the segmenter keeps listening and
    # transcribing while earlier answers are still streaming. Each answer's
    # events are tagged with a `qid` so the client can route interleaved
    # token streams to the right bubble.
    answer_tasks: set[asyncio.Task] = set()

    # Live event bus (one per session): every pipeline stage publishes
    # structured events here (UTTERANCE_FINALIZED → QUESTION_DETECTED →
    # ANSWER_STARTED/DONE/VERIFIED, TOPIC_CHANGED, …) and in-flight answers
    # register for targeted cancellation. Doubles as the replayable event
    # trail when the event log is enabled.
    from app.live.bus import (
        ANSWER_DONE, ANSWER_STARTED, ANSWER_VERIFIED, LiveEventBus,
        PARTIAL_TRANSCRIPT, QUESTION_DETECTED, QUESTION_SKIPPED,
        TOPIC_CHANGED, UTTERANCE_FINALIZED,
    )
    _elog = None
    if getattr(cfg.live, "event_log", False):
        try:
            from app.live.eventlog import get_log as _get_log
            _elog = _get_log(sid)
        except Exception:  # noqa: BLE001
            _elog = None
    bus = LiveEventBus(event_log=_elog)

    # Organization context captured at session creation (org name, job role,
    # job description, notes) — grounds "why us / fit / why hire you" answers.
    org_ctx: dict = {}
    if session_id:
        try:
            org_ctx = await _load_org_ctx(sid)
        except Exception:  # noqa: BLE001
            _layer_failed()
    # 2C-24 Cross-round memory: mark a new interview round for this company so a
    # later round can build on what earlier rounds covered. Durable, fail-open.
    if getattr(cfg.live, "cross_round_memory", True) and org_ctx.get("org_name"):
        try:
            from app.live import cross_round as _cr0
            _cr0.start_round(org_ctx.get("org_name", ""),
                             role=org_ctx.get("job_role", ""))
        except Exception:  # noqa: BLE001
            _layer_failed()

    await send({
        "type": "ready",
        "session_id": sid,
        "resume_loaded": profile is not None,
    })

    # Session-state resilience: if this is a RECONNECT into a fresh process
    # (backend restarted mid-interview), rebuild the conversational context
    # from the persisted snapshot so answers stay grounded. Gated + additive.
    if session_id and getattr(cfg.live, "session_resume", False):
        try:
            from app.live.state_persist import restore_state
            _restored = await restore_state(sid)
            if _restored:
                await send({"type": "meta", "restored_turns": _restored})
        except Exception:  # noqa: BLE001
            _layer_failed()

    # Consent gate + disclaimer (Phase 6): surfaced before capture when enabled
    # (additive; legacy clients ignore the frame). Off → no gate (today).
    if getattr(cfg.live, "consent", False):
        try:
            from app.live import consent as _consent
            frame = _consent.consent_frame()
            if frame is not None:
                await send(frame)
        except Exception:  # noqa: BLE001
            _layer_failed()

    # Advisory emotion signals, keyed by qid: written by the background prosody
    # task fired in `_run_answer`, read NON-BLOCKINGLY by `_generate_answer`.
    # Bounded — a skipped turn's entry is never consumed, so old ones are evicted.
    _emotion_by_qid: dict[str, tuple[dict, str]] = {}
    _EMOTION_CACHE_MAX = 32

    async def _run_answer(qid: str, utterance: str, audio_np,
                          stt_conf=None) -> None:
        """Predict the question from the transcript, then generate + stream one
        answer concurrently. Tagged with `qid`."""
        from app.question_detection import agent as _agent
        from app.question_detection.context_tracker import get_tracker

        is_audio = audio_np is not None

        # ADVISORY EMOTION (R43): prosody → calm/stressed/rushed/hesitant.
        # Fired here as a BACKGROUND task on a worker thread and never awaited,
        # so this DSP work can add zero latency to first token. `_generate_answer`
        # picks up whatever is ready by the time it builds its meta; anything that
        # lands later rides out on a post-answer meta frame. Strictly advisory —
        # it gates nothing and never touches the decision engine's verdict.
        if is_audio and getattr(cfg.live, "emotion_signal", False):
            try:
                async def _emotion_bg(_qid=qid, _audio=audio_np,
                                      _utt=utterance) -> None:
                    try:
                        sig, note = await asyncio.to_thread(
                            _emotion_signal, _audio, _utt)
                    except Exception:  # noqa: BLE001
                        return
                    if not sig:
                        return
                    while len(_emotion_by_qid) >= _EMOTION_CACHE_MAX:
                        _emotion_by_qid.pop(next(iter(_emotion_by_qid)), None)
                    _emotion_by_qid[_qid] = (sig, note)

                _emo_task = asyncio.create_task(_emotion_bg())
                answer_tasks.add(_emo_task)
                _emo_task.add_done_callback(answer_tasks.discard)
            except Exception:  # noqa: BLE001
                _layer_failed()

        # CANDIDATE SELF-ANSWER ECHO: in a dual-source session we also hear the
        # candidate. When they answer the interviewer — usually by reading or
        # paraphrasing the answer we just showed — that voice is NOT a new
        # question, so skip it instead of transcribing + re-answering. Audio
        # only (typed input is deliberate); fully fail-open.
        if (is_audio and not solo_mode
                and getattr(cfg.live, "candidate_echo_skip", False)):
            # SOLO mode disables this: the tester's single voice both asks and
            # reads answers aloud, so echo-matching would suppress real questions.
            try:
                from app.live import echo as _echo
                _ethr = float(
                    getattr(cfg.live, "candidate_echo_threshold", 0.72) or 0.72)
                _is_echo, _esim = _echo.is_candidate_echo(
                    sid or "", utterance, _ethr)
                if _is_echo:
                    await send({"type": "done", "qid": qid,
                                "skipped": "candidate_echo",
                                "echo_similarity": round(_esim, 3)})
                    await send({"type": "skipped", "qid": qid,
                                "text": utterance, "reason": "candidate_echo"})
                    try:
                        from app.live import ledger as _eled
                        _eled.record(sid or "", qid, utterance, _eled.SKIPPED,
                                     reason="candidate_echo")
                    except Exception:  # noqa: BLE001
                        pass
                    return
            except Exception:  # noqa: BLE001 — never drop a real question
                pass

        # Domain context for STT question repair: the resume skills, target role
        # + JD, and recent topics let the cleaner confidently fix mis-heard
        # technical terms ("Q proxy" -> "kube-proxy", "spring" -> "string").
        # Empty when there's no resume/role or the feature is off.
        _domain = ""
        if getattr(cfg.live, "context_repair", True):
            try:
                from app.live import domain as _dom
                _domain = _dom.build_domain(
                    profile, org_ctx,
                    recent=get_tracker(sid).recent_questions()).prompt_block()
            except Exception:  # noqa: BLE001
                _domain = ""

        # 0. FAST QUESTION PATH (latency): a transcript that is UNAMBIGUOUSLY
        #    a question ("… in Kafka?" — terminal '?' + interrogative shape +
        #    confident heuristic) doesn't need the detection LLM round-trip
        #    (seconds on free tiers) before answering. Build the structured
        #    event DETERMINISTICALLY (same split/boundary logic the LLM path
        #    uses) and go straight to generation — the answer's own first
        #    token becomes the only remaining LLM wait. Anything ambiguous
        #    falls through to the full LLM typing below.
        event = None
        # When domain repair is available, skip the fast (no-cleanup) path so the
        # question flows through the domain-aware cleaner and mis-transcriptions
        # get fixed. Falls back to the fast path when there's no domain context.
        if (getattr(cfg.live, "fast_question_path", True)
                and not _domain
                and utterance.rstrip().endswith("?")):
            try:
                from app.live import events as _live_events
                from app.question_detection.classifier import heuristic_classify
                h = heuristic_classify(utterance)
                if h.is_question and h.confidence >= 0.7:
                    ctx_sents, q_text = _live_events.split_boundary(
                        utterance, utterance)
                    qs = _live_events.split_questions(q_text or utterance)
                    if qs:
                        event = _live_events.UtteranceEvent(
                            kind=_live_events.QUESTION,
                            questions=qs,
                            context=ctx_sents,
                            topic=getattr(h, "topic", "") or "",
                            difficulty="standard",
                            confidence=round(float(h.confidence), 2),
                            source="fast-path",
                            qtype=(h.type if h.type != "unknown"
                                   else "technical_concept"),
                        )
            except Exception:  # noqa: BLE001 — fall through to the LLM path
                event = None

        # 1. Structured event typing (live-conversational-intelligence Phase 1).
        #    Reuses the SINGLE agent.predict call (no second blocking call):
        #    when `cfg.live.structured_events` is on we derive the event kind +
        #    multi-question split + boundary; otherwise we fall back to the
        #    classic agent.predict path below (byte-for-byte today's behavior).
        if event is None and getattr(cfg.live, "structured_events", False):
            try:
                from app.live import events as _live_events
                recent = get_tracker(sid).recent_questions()
                event = await _live_events.type_utterance(
                    utterance, recent, audio_np, domain=_domain)
            except Exception:  # noqa: BLE001 — fail open to the classic path
                event = None

        if event is not None:
            await _answer_from_event(qid, utterance, audio_np, event, is_audio,
                                     stt_conf=stt_conf)
            return

        # 1b. Classic path: agent predicts the clean question + type from the
        #     raw transcript (no heuristics / hardcoded term lists).
        try:
            recent = get_tracker(sid).recent_questions()
            pred = await _agent.predict(utterance, recent, domain=_domain)
        except Exception:  # noqa: BLE001 — never drop a turn on predictor error
            pred = _agent.Prediction(True, utterance.strip(), "technical_concept", "")

        # Tell the UI what was detected (incl. the cleaned question).
        # NOTE: the classification goes under "qtype" — "type" is reserved for
        # the event envelope kind, and a second "type" key would clobber it.
        _pred_intent = _intent_label(
            pred.type, topic=pred.topic,
            is_followup=getattr(pred, "is_followup", False))
        await send({
            "type": "meta", "qid": qid,
            "is_question": pred.is_question, "qtype": pred.type,
            "topic": pred.topic, "question": pred.question, "source": "agent",
            "intent": _pred_intent,
        })

        # On the audio path, only answer real questions (stay quiet on filler /
        # the candidate's own words). Typed input is always answered.
        if is_audio and not pred.is_question:
            await send({"type": "done", "qid": qid, "skipped": "not_a_question"})
            await send({"type": "skipped", "qid": qid, "text": utterance,
                        "reason": "not_a_question"})
            try:
                from app.live import ledger as _ledger
                _ledger.record(sid, qid, utterance, _ledger.SKIPPED,
                               reason="not_a_question")
            except Exception:  # noqa: BLE001
                pass
            return

        await _generate_answer(
            qid, pred.question or utterance, pred.type, pred.difficulty,
            topic=pred.topic, stt_conf=stt_conf, intent_label=_pred_intent,
        )

    async def _answer_from_event(qid, utterance, audio_np, event, is_audio,
                                 stt_conf=None) -> None:
        """Answer using a structured UtteranceEvent (Phase 1). Emits additive
        `meta.kind`/`questions`/`context` + an optional `state` frame, and
        answers EACH question of a multi-question utterance with its own
        `qid` (the first reuses the incoming qid)."""
        import uuid as _uuid

        # Topic graph + drift (Phase 2): widen the per-session tracker with a
        # topic tree; a drift away from the current topic is surfaced additively
        # and resolves "back to X" references. Fail-open: errors → no change.
        topic_drift = False
        topic_ref = None
        if getattr(cfg.live, "topic_graph", False):
            try:
                from app.question_detection.context_tracker import get_tracker
                from app.live import topic_graph as _tg
                graph = _tg.for_tracker(get_tracker(sid))
                topic_ref = graph.resolve_reference(utterance)
                if event.topic:
                    topic_drift = graph.observe(event.topic)
                if topic_drift:
                    bus.publish(TOPIC_CHANGED, qid=qid, to_topic=event.topic)
            except Exception:  # noqa: BLE001
                topic_drift, topic_ref = False, None

        # Unified DECISION ENGINE (post-detection): the ensemble FP-gate
        # (rules + agent + prosody) and the answerability rule are decided in
        # app/live/decision.py; this path only acts on the verdict.
        detection_conf = None
        ev_verdict = None
        try:
            from app.live import decision as _dec
            from app.question_detection.context_tracker import get_tracker
            ev_verdict = _dec.decide_event(
                event, is_audio=is_audio, utterance=utterance,
                audio_np=audio_np, tracker=get_tracker(sid))
        except Exception:  # noqa: BLE001
            ev_verdict = None
        if ev_verdict is not None:
            detection_conf = ev_verdict.signals.get("detection_confidence")
            if (ev_verdict.action == _dec.SKIP
                    and ev_verdict.reason == "ensemble_not_question"):
                await send({"type": "done", "qid": qid,
                            "skipped": "ensemble_not_question",
                            "detection_confidence": detection_conf})
                await send({"type": "skipped", "qid": qid, "text": utterance,
                            "reason": "ensemble_not_question"})
                bus.publish(QUESTION_SKIPPED, qid=qid,
                            reason="ensemble_not_question")
                try:
                    from app.live import ledger as _ledger
                    _ledger.record(sid, qid, utterance, _ledger.SKIPPED,
                                   reason="ensemble_not_question",
                                   signals=dict(ev_verdict.signals))
                except Exception:  # noqa: BLE001
                    pass
                return

        # Interviewer-style learning (Phase 4): observe the question for a
        # rolling per-session style estimate that tunes detection thresholds.
        if getattr(cfg.live, "style_learning", False):
            try:
                from app.question_detection.context_tracker import get_tracker
                from app.live import style as _style
                _style.for_tracker(get_tracker(sid)).observe(
                    question=(event.questions[0] if event.questions else utterance),
                    is_followup=(event.kind == "FOLLOWUP"),
                    topic_switch=topic_drift,
                )
            except Exception:  # noqa: BLE001
                _layer_failed()

        # Event log (Phase 4): append the typed event (bounded, in-process).
        if getattr(cfg.live, "event_log", False):
            try:
                from app.live.eventlog import get_log
                get_log(sid).append("event", {"kind": event.kind, "topic": event.topic,
                                              "questions": len(event.questions)})
            except Exception:  # noqa: BLE001
                _layer_failed()

        # Conversational depth (Phase 10): world-model + diarization + answer
        # revision + contradiction/temporal — all additive + fail-open.
        revision_qid = None
        if getattr(cfg.live, "world_model", False) or getattr(cfg.live, "answer_revision", False) \
                or getattr(cfg.live, "contradiction", False) or getattr(cfg.live, "diarization", False):
            try:
                from app.question_detection.context_tracker import get_tracker
                tr = get_tracker(sid)
                wm = None
                if getattr(cfg.live, "world_model", False):
                    from app.live import world_model as _wm
                    wm = _wm.for_tracker(tr)
                    _wm.extract_world(utterance, wm)
                    if event.questions:
                        wm.set_active(event.questions[0], qid=qid, topic=event.topic)
                # State validation (Phase 11): on a detected context gap, rebuild
                # the active topic from the rolling session summary.
                if getattr(cfg.live, "state_validation", False) and wm is not None:
                    try:
                        from app.live import validate as _val
                        from app.live import memory as _mem2
                        summ = _mem2.for_tracker(tr).l3()
                        gap, recovered = _val.validate_and_recover(
                            wm, summary=summ, recent_questions=tr.recent_questions())
                        if recovered:
                            await send({"type": "meta", "qid": qid, "recovered": True})
                    except Exception:  # noqa: BLE001
                        _layer_failed()
                # Diarization role (additive meta; default interviewer).
                if getattr(cfg.live, "diarization", False):
                    from app.live import diarize as _dia
                    role, _rc = _dia.for_tracker(tr).attribute(text=utterance)
                    meta_role = role
                else:
                    meta_role = None
                # Answer revision: a reinterpretation targets the prior qid.
                if getattr(cfg.live, "answer_revision", False) and wm is not None:
                    from app.live import revise as _rev
                    rq = _rev.detect_reinterpretation(utterance, wm)
                    if rq:
                        revision_qid = rq
                        await send({"type": "revision", "qid": rq,
                                    "question": _rev.revised_question(utterance, wm)})
                # Contradiction / temporal reference (additive meta).
                challenge = False
                temporal_ref = None
                if getattr(cfg.live, "contradiction", False):
                    from app.live import contradiction as _con
                    from app.live import topic_graph as _tg2
                    challenge = _con.is_challenge(utterance, wm)
                    temporal_ref = _con.resolve_temporal(utterance, _tg2.for_tracker(tr))
            except Exception:  # noqa: BLE001
                meta_role, challenge, temporal_ref = None, False, None
        else:
            meta_role, challenge, temporal_ref = None, False, None

        # Advance + surface the interview state (additive; never gates work).
        if getattr(cfg.live, "state_machine", False):
            try:
                from app.live.state_machine import get_state_machine
                sm = get_state_machine(sid)
                sm.advance(event)
                await send({"type": "state", "qid": qid, **sm.snapshot()})
            except Exception:  # noqa: BLE001
                _layer_failed()

        # A not-answerable event the decision engine PROMOTED (indirect /
        # hypothetical / tonal question) is answered like a real question.
        _promoted = (ev_verdict.signals.get("promoted")
                     if ev_verdict is not None else None)
        _promoted_qtype = (ev_verdict.signals.get("promoted_qtype")
                           if ev_verdict is not None else None)

        # Additive structured meta (legacy clients ignore unknown fields).
        _qtype = ("question" if event.questions
                  else (_promoted_qtype or event.kind.lower()))
        _ev_intent = _intent_label(
            event.qtype if getattr(event, "qtype", None)
            else (_promoted_qtype or _qtype),
            topic=event.topic, is_followup=(event.kind == "FOLLOWUP"))
        meta = {
            "type": "meta", "qid": qid,
            "is_question": bool(event.is_answerable or _promoted),
            "qtype": _qtype,
            "kind": event.kind,
            "questions": event.questions,
            "context": event.context,
            "topic": event.topic,
            "question": event.questions[0] if event.questions else "",
            "source": ("promotion:" + str(_promoted)) if _promoted
            else "live-events",
            # Clear human-readable intent for the answer bubble header.
            "intent": _ev_intent,
        }
        if topic_drift:
            meta["topic_drift"] = True
        if topic_ref:
            meta["topic_ref"] = topic_ref
        if detection_conf is not None:
            meta["detection_confidence"] = detection_conf
        if meta_role:
            meta["speaker"] = meta_role
        if challenge:
            meta["challenge"] = True
        if temporal_ref:
            meta["temporal_ref"] = temporal_ref
        await send(meta)

        # Non-answerable on the audio path → stay quiet (Property 2). The
        # verdict was computed by the decision engine above; a PROMOTED
        # utterance (indirect / hypothetical / tonal question) answers.
        if is_audio and not event.is_answerable and not _promoted:
            await send({"type": "done", "qid": qid, "skipped": event.kind})
            await send({"type": "skipped", "qid": qid, "text": utterance,
                        "reason": str(event.kind).lower()})
            bus.publish(QUESTION_SKIPPED, qid=qid, reason=str(event.kind))
            try:
                from app.live import ledger as _ledger
                _ledger.record(sid, qid, utterance, _ledger.SKIPPED,
                               reason=str(event.kind).lower())
            except Exception:  # noqa: BLE001
                pass
            return
        if _promoted:
            try:
                from app.live import ledger as _ledger
                _ledger.record(sid, qid, utterance, _ledger.PROMOTED,
                               reason=str(_promoted),
                               qtype=str(_promoted_qtype or ""),
                               signals=dict(ev_verdict.signals
                                            if ev_verdict else {}))
            except Exception:  # noqa: BLE001
                pass
        if not event.questions:
            # Typed non-question / promoted probe → answer the raw utterance.
            await _generate_answer(qid, utterance, "technical_concept", event.difficulty,
                                   topic=event.topic, stt_conf=stt_conf,
                                   intent_label=_ev_intent)
            return

        # ONE bubble for conjoined questions (enhancement #4, 2026-07-08):
        # "What is Kafka and how do you use it?" used to split into TWO
        # answers/bubbles. When the parts are short and arrived as one
        # utterance, answer them as ONE composite question with per-part
        # headings — reads like a person handling a compound question.
        if (len(event.questions) > 1
                and getattr(cfg.live, "combine_multi_questions", True)
                and all(len(p.split()) <= 18 for p in event.questions)):
            _parts = list(event.questions)
            _composite = " ".join(
                f"({_n}) {_p}" for _n, _p in enumerate(_parts, 1))
            await _generate_answer(
                qid, _composite, "technical_concept", event.difficulty,
                topic=event.topic, stt_conf=stt_conf, intent_label=_ev_intent,
                directive_extra=(
                    "This utterance contains multiple questions. Answer ALL "
                    "of them in ONE response, each under a short bold "
                    "heading, in the order asked."))
            return

        # Answer each question; first reuses `qid`, extras get fresh qids so
        # interleaved token streams route to distinct bubbles (concurrency).
        # `seq` orders siblings of one utterance: interleaved answers can land
        # in any order over the socket, so the client sorts by (utterance, seq)
        # instead of trusting frame arrival order.
        for i, q in enumerate(event.questions):
            this_qid = qid if i == 0 else _uuid.uuid4().hex
            _seq_intent = _ev_intent
            if i > 0:
                _seq_intent = _intent_label(
                    event.qtype if getattr(event, "qtype", None) else None,
                    topic=event.topic,
                    is_followup=(event.kind == "FOLLOWUP"))
                await send({"type": "transcript", "qid": this_qid, "text": q,
                            "seq": i, "parent_qid": qid})
                await send({
                    "type": "meta", "qid": this_qid, "is_question": True,
                    "qtype": "question", "kind": event.kind, "questions": [q],
                    "context": event.context, "topic": event.topic,
                    "question": q, "source": "live-events",
                    "seq": i, "parent_qid": qid,
                    "intent": _seq_intent,
                })
            await _generate_answer(this_qid, q, "technical_concept", event.difficulty,
                                   topic=event.topic, stt_conf=stt_conf,
                                   intent_label=_seq_intent)

    async def _generate_answer(qid, question, qtype, difficulty, topic="",
                               stt_conf=None, directive_extra=None,
                               is_retry=False, intent_label=None,
                               retry_stage=0) -> None:
        """Generate + stream one answer for a cleaned question, tagged `qid`,
        and persist it. Shared by the classic and event-driven paths.
        `directive_extra` folds an extra instruction into the generation call
        (used by the verifier's regeneration pass). `retry_stage` is the
        verification-retry depth (0 = original); it gates re-verification (stops
        the loop at `answer_max_retries`) and, on the final stage, ESCALATES to a
        different/stronger model. `is_retry` is derived from it."""
        # Any retry stage is a "retry" for the dedup-window bypass (a regen must
        # re-answer the same question on purpose).
        is_retry = is_retry or retry_stage > 0
        # NEAR-DUPLICATE GUARD (user report 2026-07-08: one spoken question
        # answered multiple times): endpoint splits / spec-final mismatches /
        # re-transcriptions of the SAME question inside a short window are
        # skipped instead of re-answered. Retries/regenerations bypass (they
        # re-answer on purpose); a genuine re-ask after the window answers
        # normally. Fail-open.
        # Speculative pass? (contextvar set only inside _maybe_speculate's
        # task). Speculation must neither consult nor seed the dedup window —
        # a CANCELLED speculation would otherwise suppress the real answer —
        # and its ledger rows are tagged so `answered` isn't inflated (#5).
        _is_spec = _SPEC_HOLDER.get(None) is not None
        if not is_retry and not _is_spec \
                and getattr(cfg.live, "question_dedup", True):
            try:
                if _deduper.is_duplicate(str(question)):
                    from app.live import ledger as _ledger
                    _ledger.record(sid, qid, question, _ledger.SKIPPED,
                                   reason="duplicate_question",
                                   qtype=str(qtype or ""))
                    # `skipped` frame first: the FE pre-created a question
                    # card + empty answer bubble on the transcript — without
                    # this, `done` alone leaves a forever-empty bubble.
                    await send({"type": "skipped", "qid": qid,
                                "text": str(question),
                                "reason": "duplicate_question"})
                    await send({"type": "done", "qid": qid,
                                "skipped": "duplicate_question"})
                    bus.publish(QUESTION_SKIPPED, qid=qid,
                                reason="duplicate_question")
                    return
            except Exception:  # noqa: BLE001
                pass
        # PREPARED-ANSWER fast path (latency batch 2026-07-11): a profile
        # question matching a pre-generated, resume-grounded answer streams
        # INSTANTLY — zero model latency. Only real (non-retry, non-spec)
        # passes; the long tail falls through to live generation.
        if not is_retry and not _is_spec and resume_id:
            try:
                from app.live import prepared as _prep
                from app.live import profile as _prof_gate2
                if (_prep.enabled()
                        and _prof_gate2.is_profile_question(str(question))):
                    _hit = _prep.match(str(resume_id), str(question))
                    if _hit is not None:
                        from app.live import ledger as _ledger
                        _ledger.record(sid, qid, question, _ledger.ANSWERED,
                                       reason="prepared",
                                       qtype=str(qtype or ""))
                        _deduper.note_answered(str(question))
                        await send({"type": "meta", "qid": qid,
                                    "prepared": True,
                                    "intent": "Question — about you"})
                        # Stream in a few chunks so the FE reveal animates.
                        _txt = str(_hit["answer"])
                        for _i in range(0, len(_txt), 400):
                            await send({"type": "token", "qid": qid,
                                        "text": _txt[_i:_i + 400]})
                        await send({"type": "done", "qid": qid,
                                    "prepared": True})
                        bus.publish(ANSWER_STARTED, qid=qid,
                                    question=str(question)[:200],
                                    retry=False)
                        try:
                            await _persist_live_qa(sid, str(question), _txt)
                        except Exception:  # noqa: BLE001
                            pass
                        return
            except Exception:  # noqa: BLE001 — fall through to generation
                pass
        # Accuracy ledger: every generated answer is one 'answered' decision
        # (speculative passes tagged so the raw counter stays honest).
        if not is_retry:
            try:
                from app.live import ledger as _ledger
                _ledger.record(sid, qid, question, _ledger.ANSWERED,
                               reason="speculative" if _is_spec else "",
                               qtype=str(qtype or ""))
                if not _is_spec:
                    _deduper.note_answered(str(question))
            except Exception:  # noqa: BLE001
                pass
        # Admission via the Decision Engine (session budget: concurrent-answer
        # cap). At the cap, shed rather than launch unbounded generations.
        from app.live import decision as _dec_admit
        _ok, budget = _dec_admit.admit_answer(sid)
        if not _ok:
            # Same contract as every other skip: a `skipped` frame converts
            # the FE's placeholder bubble into a muted, force-answerable row.
            await send({"type": "skipped", "qid": qid,
                        "text": str(question), "reason": "budget_cap"})
            await send({"type": "done", "qid": qid, "skipped": "budget_cap"})
            bus.publish(QUESTION_SKIPPED, qid=qid, reason="budget_cap")
            return
        bus.publish(ANSWER_STARTED, qid=qid, question=str(question)[:200],
                    retry=bool(is_retry))

        sm = None
        if getattr(cfg.live, "state_machine", False):
            try:
                from app.live.state_machine import get_state_machine
                sm = get_state_machine(sid)
                sm.on_answer_start()
            except Exception:  # noqa: BLE001
                sm = None

        # Qid registry (Phase 7): track this answer's lifecycle so a cancel
        # targets exactly its own qid and a duplicate/late final is a no-op.
        reg = None
        if getattr(cfg.live, "session_resume", False):
            try:
                from app.live.resume import get_registry
                reg = get_registry(sid)
                reg.open(qid)
            except Exception:  # noqa: BLE001
                reg = None

        # FACTUAL FAST PATH (enhancement #6, 2026-07-08): a short factual
        # question ("what is X?", "difference between A and B") leads with a
        # ONE-SENTENCE direct answer and stays concise — the interviewer
        # hears the answer immediately instead of waiting through a
        # long-form composition. Deterministic gate; folded into the same
        # single generation call.
        _fast_factual = False
        if getattr(cfg.live, "factual_fast_path", True) and not is_retry:
            try:
                import re as _re
                _ql = str(question).strip().lower()
                if (len(_ql.split()) <= 14 and _re.match(
                        r"^\(?\d*\)?\s*(what\s+is|what\s+are|what's|define|"
                        r"difference\s+between|what\s+does|when\s+would|"
                        r"is\s+|are\s+|does\s+|can\s+)", _ql)):
                    _fast_factual = True
            except Exception:  # noqa: BLE001
                _fast_factual = False
        if _fast_factual:
            _ff = ("FAST FACTUAL MODE: begin with ONE sentence that directly "
                   "answers the question, then expand with at most 3-5 short "
                   "bullets. No preamble, never restate the question.")
            directive_extra = (f"{directive_extra}\n{_ff}"
                               if directive_extra else _ff)

        # Deliberation (Phase 3): phase + strategy + plan + knowledge-gap guard
        # + adaptive depth, each flag-gated. Produces an answer directive folded
        # into the SAME generation call (no second blocking LLM call) + a target
        # depth + an answer confidence. Fail-open → today's generic answer.
        directive = None
        forced_depth = None
        # Calibration meta persisted with the answer (band/track/target) so a
        # reloaded transcript still shows the seniority pill. Set in the
        # calibration layer below; None until then (defined before the try so
        # it's always bound on the persist path).
        _calib_meta: dict | None = None
        # Whether the advisory emotion signal made it into the pre-answer meta.
        # When it didn't (the background prosody task hadn't finished — we never
        # WAIT for it), a post-answer meta frame carries it instead.
        _emotion_surfaced = False
        # Intent header persisted with the answer so a reloaded transcript shows
        # the SAME "what's really being asked" label the live view showed. Use
        # the string the caller already computed for the meta frame; fall back
        # to deriving it from qtype + topic. Stored on `sources` (JSONB) — not
        # the String(50) `intent` column — so a long topic suffix is kept whole,
        # never mid-word truncated.
        try:
            _answer_intent = (intent_label
                              or _intent_label(qtype, topic=topic)) or None
        except Exception:  # noqa: BLE001
            _answer_intent = None
        try:
            from app.live import deliberate as _delib
            from app.question_detection.context_tracker import get_tracker
            recent = get_tracker(sid).recent_questions()
            d = _delib.deliberate(question, qtype, difficulty, topic, recent)
            directive = d.directive or None
            forced_depth = d.depth
            _conf = d.confidence

            # Conversational depth (Phase 10): objective/depth + false-premise +
            # world-model honored context, all folded into the SAME call.
            _tr = get_tracker(sid)
            _add: list[str] = []
            extra = {}
            if getattr(cfg.live, "objective_depth", False):
                try:
                    from app.live import objective as _obj
                    if getattr(cfg.live, "multi_pass", False):
                        obj, dep = _obj.multi_pass(question, qtype, d.phase, difficulty, recent)
                    else:
                        obj, dep = _obj.estimate(question, qtype, d.phase, difficulty)
                    od = _obj.directive(obj, dep)
                    if od:
                        _add.append(od)
                    extra["objective"] = obj
                    extra["expected_depth"] = dep
                except Exception:  # noqa: BLE001
                    _layer_failed()
            if getattr(cfg.live, "false_premise", False):
                try:
                    from app.live import premise as _prem
                    pv = _prem.check_premise(question)
                    if pv.false_premise:
                        _add.append(_prem.directive(pv))
                        extra["false_premise"] = round(pv.confidence, 2)
                except Exception:  # noqa: BLE001
                    _layer_failed()
            if getattr(cfg.live, "world_model", False):
                try:
                    from app.live import world_model as _wm2
                    hc = _wm2.for_tracker(_tr).honored_context()
                    if hc:
                        _add.append("Honor the established context — " + hc)
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # Interview-knowledge (Phase 11): topic-triggered angles folded into
            # the SAME call (no second blocking call).
            if getattr(cfg.live, "interview_knowledge", False) and topic:
                try:
                    from app.live import knowledge as _know
                    snips = _know.interview_knowledge(topic, _know.configured_pack())
                    kd = _know.directive(snips)
                    if kd:
                        _add.append(kd)
                        extra["knowledge"] = len(snips)
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # NO-RESUME GUARD (user report 2026-07-08): a question about the
            # candidate THEMSELVES with no resume uploaded must NEVER be
            # answered with fabricated specifics. Warn the client (visible
            # banner) and pin the answer to a safe, generic, first-person
            # response that says what it can't know. Runs BEFORE the profile
            # block (which is skipped entirely when profile is None).
            try:
                from app.live import profile as _prof_gate
                if profile is None and _prof_gate.is_profile_question(question):
                    await send({
                        "type": "meta", "qid": qid, "resume_required": True,
                        "warning": ("Resume not uploaded — answers about "
                                    "your background will stay generic. "
                                    "Upload your resume for personalized, "
                                    "grounded answers."),
                    })
                    _add.append(
                        "NO RESUME IS UPLOADED. This question is about the "
                        "candidate personally. Answer in first person with a "
                        "SAFE GENERIC structure (role-appropriate strengths, "
                        "approach, enthusiasm) and NEVER invent employers, "
                        "project names, dates, technologies, or metrics. "
                        "Where a specific would normally go, keep it general "
                        "('in my recent projects', 'in my experience').")
                    extra["resume_required"] = True
            except Exception:  # noqa: BLE001
                _layer_failed()
            # Candidate intelligence (Phase 12): ground resume-based answers
            # in the structured profile and enforce resume-reality. Folded
            # into the SAME call.
            _cp = None
            if getattr(cfg.live, "candidate_profile", False) and profile:
                try:
                    from app.live import profile as _prof
                    _cp = _prof.build_profile(profile)
                    if _prof.is_profile_question(question):
                        # A question about the candidate THEMSELVES ("tell me
                        # about yourself", "your projects", "your experience")
                        # gets the FULL structured profile + a first-person,
                        # detail-demanding directive — topic slices are far
                        # too thin for these.
                        summary = _prof.profile_summary(_cp)
                        if summary:
                            _add.append("Candidate profile (ground truth):\n"
                                        + "\n".join(summary))
                        _add.append(_prof.first_person_directive())
                        extra["profile_question"] = True
                    else:
                        slices = _prof.scoped_retrieve(_cp, topic)
                        if slices:
                            _add.append("From the candidate's background: "
                                        + "; ".join(slices))
                    if getattr(cfg.live, "interview_assets", False):
                        from app.live import assets as _assets
                        rd = _assets.reality_directive(_cp)
                        if rd:
                            _add.append(rd)
                        akey = _assets.match_asset(question)
                        if akey:
                            extra["asset"] = akey
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # Organization intelligence: grounded in the session's intake form
            # (org name + role + pasted JD + notes) captured at create time.
            # Runs with OR WITHOUT a resume — fit analysis just degrades to
            # org/role guidance when no candidate profile exists.
            if getattr(cfg.live, "org_intelligence", False):
                try:
                    from app.live import org as _org
                    org_name = (org_ctx.get("org_name")
                                or (profile.get("org_name", "")
                                    if isinstance(profile, dict) else ""))
                    o = _org.build_org(
                        org_name,
                        org_ctx.get("job_description", ""),
                        org_ctx.get("job_role", ""),
                        notes=org_ctx.get("notes", ""),
                    )
                    od = _org.fit_directive(o, _cp)
                    if od:
                        _add.append(od)
                        extra["org_grounded"] = True
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # Seniority-band calibration: classify the candidate's real band
            # (fresher → principal) from the resume + target role and pitch the
            # answer at that level — truthful to real experience, framed toward
            # the target role's expectations where genuine. Folded into the SAME
            # call via the ANSWER GUIDANCE channel (no extra LLM call).
            if getattr(cfg.live, "answer_calibration", False):
                try:
                    from app.live import calibration as _cal
                    _cal_res = _cal.build_calibration(
                        profile if isinstance(profile, dict) else None,
                        org_ctx,
                        cp=_cp,
                        override=org_ctx.get("experience_level", ""),
                    )
                    cd = _cal.calibration_directive(_cal_res)
                    if cd:
                        _add.append(cd)
                        if _cal_res is not None:
                            extra["seniority_band"] = _cal_res.real_band.slug
                            _calib_meta = {"band": _cal_res.real_band.slug}
                            if _cal_res.target_band is not None:
                                extra["target_band"] = _cal_res.target_band.slug
                                _calib_meta["target"] = _cal_res.target_band.slug
                            if _cal_res.track is not None:
                                extra["career_track"] = _cal_res.track.slug
                                _calib_meta["track"] = _cal_res.track.slug
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # HR / negotiation / specialized modes (Phase 13): bias the answer
            # by operating MODE, add fact-based negotiation guidance on HR
            # questions, and fold an advisory emotion delivery note. Additive.
            if getattr(cfg.live, "interview_modes", False):
                try:
                    from app.live import modes as _modes
                    mt = _modes.for_tracker(_tr) if _tr is not None else None
                    if mt is not None:
                        mode = mt.update(question, qtype=qtype, topic=topic)
                    else:
                        mode = _modes.detect_mode(question, qtype=qtype, topic=topic)
                    md = _modes.directive(mode)
                    if md:
                        _add.append(md)
                        extra["mode"] = mode
                except Exception:  # noqa: BLE001
                    _layer_failed()
            if getattr(cfg.live, "negotiation", False):
                try:
                    from app.live import negotiate as _negot
                    from app.live import phase as _ph
                    if _ph.detect_phase(question, qtype=qtype, topic=topic) == _ph.HR:
                        strengths = []
                        if isinstance(profile, dict) and profile:
                            from app.live import profile as _prof2
                            strengths = sorted(_prof2.reality_terms(_prof2.build_profile(profile)))[:6]
                        # Dual-source: react to what the candidate actually SAID
                        # (stated salary / competing offer / notice) + interviewer
                        # pushback signals, from the world-model commitments.
                        cand_stated: dict = {}
                        interviewer_signal = None
                        if getattr(cfg.live, "commitment_tracking", False):
                            from app.live import world_model as _wmn
                            _wmm = _wmn.for_tracker(_tr)
                            _comms = _wmn.commitments_for(_wmm, topic or "salary")
                            for _k, _v in _comms.items():
                                if _v.get("role") == "candidate":
                                    cand_stated[_k] = _v.get("value")
                                elif _k == "salary_signal" and _v.get("role") == "interviewer":
                                    interviewer_signal = _v.get("value")
                        strat = _negot.negotiation_strategy(
                            question, strengths=strengths,
                            candidate_stated=cand_stated,
                            interviewer_signal=interviewer_signal)
                        nd = _negot.directive(strat)
                        if nd:
                            _add.append(nd)
                            extra["hr_intent"] = strat.intent
                            if cand_stated:
                                extra["candidate_stated"] = cand_stated
                            if strat.risk_flag:
                                extra["negotiation_risk"] = strat.risk_flag
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # Precision hardening (Phase 15): record recurring topics for skill-
            # gap detection + boosted retrieval, fold a knowledge-gap hedge when
            # evidence is thin, and adapt depth to cognitive load. All additive,
            # folded into the SAME answer call.
            if getattr(cfg.live, "skill_gap", False):
                try:
                    from app.live import world_model as _wmsg
                    from app.live import knowledge as _knowsg
                    wm = _wmsg.for_tracker(_tr)
                    _wmsg.record_topic(wm, topic)
                    gaps = _wmsg.skill_gaps(wm)
                    if gaps and topic:
                        boost = _knowsg.skill_gap_boost(topic, gaps, _knowsg.configured_pack())
                        kd = _knowsg.directive(boost)
                        if kd:
                            _add.append(kd)
                            extra["skill_gap"] = True
                except Exception:  # noqa: BLE001
                    _layer_failed()
            if getattr(cfg.live, "evidence", False):
                try:
                    from app.live import evidence as _ev
                    binding = _ev.EvidenceBinding()
                    for s in _add:
                        binding.add(s, source="directive")
                    hedge = _ev.hedge_directive(binding)
                    if hedge:
                        _add.append(hedge)
                        extra["evidence_hedge"] = True
                except Exception:  # noqa: BLE001
                    _layer_failed()
            if getattr(cfg.live, "cognitive_load", False):
                try:
                    from app.live import style as _styleload
                    style_lbl = _styleload.for_tracker(_tr).label()
                    load = _styleload.cognitive_load(interviewer_style=style_lbl)
                    note = _styleload.depth_for_load(load)
                    if note:
                        _add.append(note)
                        extra["cognitive_load"] = load
                except Exception:  # noqa: BLE001
                    _layer_failed()
            if _add:
                directive = ("\n".join([directive] + _add).strip() if directive
                             else "\n".join(_add).strip())
            # Uncertainty propagation (Phase 4): a known topic lifts, an unknown
            # one lowers, the surfaced answer confidence — and a poorly-heard
            # utterance (low STT confidence) drags it down too, so a garbled
            # transcript is never presented as a high-confidence answer.
            if getattr(cfg.live, "uncertainty_tracking", False) and _conf is not None:
                try:
                    from app.live import uncertainty as _unc
                    _conf = _unc.propagate(_conf, stt_conf=stt_conf,
                                           topic_conf=(0.9 if topic else 0.6))
                    if stt_conf is not None:
                        extra["stt_confidence"] = round(float(stt_conf), 3)
                except Exception:  # noqa: BLE001
                    _layer_failed()
            if d.phase:
                extra["phase"] = d.phase
                # Per-session phase PROGRESSION (Phase 2 #3/#14): smooth the
                # per-utterance phase into a session arc + surface progress.
                # Additive + fail-open; mirrors the state_machine wiring below.
                if getattr(cfg.live, "phase_detection", False):
                    try:
                        from app.live.phase_tracker import get_phase_tracker
                        _pt = get_phase_tracker(sid)
                        _pt.observe(d.phase)
                        extra["phase_progress"] = _pt.progress()
                        _pnext = _pt.predict_next()  # forward trajectory (Phase 2 #14)
                        if _pnext:
                            extra["predicted_next_phase"] = _pnext
                        if _pt.is_late_stage():
                            extra["late_stage"] = True
                    except Exception:  # noqa: BLE001
                        _layer_failed()
            # Conversation signals (Phase 2 #13 readiness · #23 contract ·
            # #31 rhythm/fatigue · #32 steering). Additive + fail-open.
            if getattr(cfg.live, "conversation_signals", False):
                try:
                    from app.live import contract as _contract
                    from app.live import readiness as _readiness
                    from app.live import rhythm as _rhythm
                    from app.live import steer as _steer
                    _rt = _rhythm.get_rhythm(sid)
                    _rt.observe_now()
                    _rsnap = _rt.snapshot()
                    extra["cadence"] = _rsnap["cadence"]
                    extra["fatigue"] = _rsnap["fatigue"]
                    extra["answer_readiness"] = _readiness.readiness_score(confidence=_conf)
                    _ct = _contract.ensure_contract(sid, phase=(d.phase or ""))
                    extra["max_answer_seconds"] = _ct.max_answer_seconds
                    _cbit = f"Keep the spoken answer under about {_ct.max_answer_seconds} seconds."
                    directive = (directive + "\n" + _cbit).strip() if directive else _cbit
                    _sd = _steer.steering_directive(question)
                    if _sd:
                        directive = (directive + "\n" + _sd).strip() if directive else _sd
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # ── Phase 2 completion signals (2A-5/6/7, 2B-9/10/15, 2C-18/19/20/
            #    21/24). All deterministic + fail-open; each surfaced additively
            #    and, at most, appends a short directive to the SAME answer call.
            # 2A-5 Silence taxonomy (thinking vs done vs hesitation).
            if getattr(cfg.live, "silence_taxonomy", True):
                try:
                    from app.live import silence as _sil
                    _ss = _sil.classify(question)
                    if _ss.label != _sil.UNKNOWN:
                        extra["silence_type"] = _ss.to_dict()
                        _sd2 = _sil.directive(_ss)
                        if _sd2:
                            directive = (directive + "\n" + _sd2).strip() if directive else _sd2
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # 2A-6 Acoustic adaptation: a degraded channel lowers answer conf.
            if getattr(cfg.live, "acoustic_adaptation", True) and _conf is not None:
                try:
                    from app.live import acoustic as _ac
                    _acp = _ac.assess(stt_conf=stt_conf)
                    if _acp.condition != _ac.UNKNOWN:
                        _conf = _ac.adjust_confidence(_conf, _acp)
                        extra["acoustic"] = _acp.to_dict()
                        if _ac.needs_reconfirmation(_acp):
                            extra["acoustic_reconfirm"] = True
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # 2A-7 Code-switch: reply in the dominant language, terms intact.
            if getattr(cfg.live, "multilingual", False):
                try:
                    from app.live import language as _lang2
                    if _lang2.is_code_switched(question):
                        _csd = _lang2.code_switch_directive(question)
                        if _csd:
                            directive = (directive + "\n" + _csd).strip() if directive else _csd
                        extra["code_switch"] = _lang2.languages_present(question)
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # 2B-9 Interviewer hidden-goal / probe intent.
            if getattr(cfg.live, "interviewer_intent", True):
                try:
                    from app.live import interviewer_intent as _ii
                    _pi = _ii.probe_intent(question, qtype=qtype)
                    if _pi.label != _ii.NEUTRAL:
                        extra["interviewer_intent"] = _pi.to_dict()
                        _iid = _ii.directive(_pi)
                        if _iid:
                            directive = (directive + "\n" + _iid).strip() if directive else _iid
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # 2B-15 Multi-hypothesis interpretation of ambiguous questions.
            if getattr(cfg.live, "multi_hypothesis", True):
                try:
                    from app.live import interpret as _interp
                    _hyps = _interp.interpretations(question, topic=topic)
                    if _hyps and _interp.is_ambiguous(question):
                        extra["interpretations"] = [h.to_dict() for h in _hyps]
                        _hd = _interp.directive(_hyps)
                        if _hd:
                            directive = (directive + "\n" + _hd).strip() if directive else _hd
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # 2C-18/19 Evidence-strength ranking → confidence calibration.
            if getattr(cfg.live, "evidence", False):
                try:
                    from app.live import evidence as _ev2
                    _binding = _ev2.EvidenceBinding()
                    if _cp is not None:
                        _binding.add("profile", source="profile")
                    for _s in _add:
                        _binding.add(_s, source="directive")
                    _strength = _ev2.strength_label(_binding)
                    extra["evidence_strength"] = _strength
                    _stdir = _ev2.strength_directive(_binding)
                    if _stdir:
                        directive = (directive + "\n" + _stdir).strip() if directive else _stdir
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # 2C-20/24 Company style bias + cross-round memory (need the org).
            if getattr(cfg.live, "org_intelligence", False):
                try:
                    from app.live import cross_round as _cr
                    from app.live import org as _org2
                    _org_name = (org_ctx.get("org_name")
                                 or (profile.get("org_name", "")
                                     if isinstance(profile, dict) else "")) or ""
                    _role_name = org_ctx.get("job_role", "") or ""
                    _cs = _org2.company_style_directive(_org_name)
                    if _cs:
                        directive = (directive + "\n" + _cs).strip() if directive else _cs
                        extra["company_style"] = True
                    if getattr(cfg.live, "cross_round_memory", True) and _org_name:
                        _crd = _cr.link_directive(_org_name, topic or "", role=_role_name)
                        if _crd:
                            directive = (directive + "\n" + _crd).strip() if directive else _crd
                        _prior = _cr.prior_topics(_org_name, role=_role_name)
                        if _prior:
                            extra["prior_rounds"] = _prior
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # 2C-21 Adaptive length / multi-level answer tier.
            if getattr(cfg.live, "conversation_signals", False):
                try:
                    from app.live import contract as _ctr2
                    _ctr = _ctr2.ensure_contract(sid, phase=(d.phase or ""))
                    _ld = _ctr2.length_directive(_ctr)
                    if _ld:
                        directive = (directive + "\n" + _ld).strip() if directive else _ld
                    extra["answer_tier"] = _ctr2.variant_for_seconds(_ctr.max_answer_seconds)
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # 2B-10 Consume a speculative pre-draft anticipated last turn.
            if getattr(cfg.live, "predictive_drafting", True):
                try:
                    from app.live import predict as _pred2
                    _pdd = _pred2.consume_directive(sid, question)
                    if _pdd:
                        directive = (directive + "\n" + _pdd).strip() if directive else _pdd
                        extra["predraft_hit"] = True
                except Exception:  # noqa: BLE001
                    _layer_failed()
            if _conf is not None:
                extra["answer_confidence"] = round(_conf, 3)
            # Multilingual (Phase 8): detect language, fold an answer directive
            # into the SAME call, and surface it additively.
            if getattr(cfg.live, "multilingual", False):
                try:
                    from app.live import language as _lang
                    lng = _lang.target_language(_lang.detect_language(question))
                    ld = _lang.answer_directive(lng)
                    if ld:
                        directive = (directive + "\n" + ld).strip() if directive else ld
                    extra["language"] = lng
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # ADVISORY emotion (R43): the prosody signal computed off the hot
            # path in `_run_answer`. Consumed NON-BLOCKINGLY — if the background
            # task hasn't finished we do NOT wait for it (a post-answer meta
            # carries it instead), so first-token latency is untouched. Surfaced
            # as additive `meta.emotion` plus, at most, a soft delivery hint in
            # the answer directive; it never gates the answer, never overrides
            # the decision engine, and never changes `_conf`.
            if getattr(cfg.live, "emotion_signal", False):
                try:
                    _emo_cached = _emotion_by_qid.pop(qid, None)
                    if _emo_cached:
                        directive = _apply_emotion(directive, extra, _emo_cached)
                        _emotion_surfaced = "emotion" in extra
                except Exception:  # noqa: BLE001
                    _layer_failed()
            # Live clarification (R59, carry-forward): when confidence is low,
            # surface a non-blocking hint the FE shows dismissibly (never pauses
            # the stream). Reuses the deliberation confidence — no extra call.
            if _conf is not None and _conf < float(
                    getattr(cfg.live, "knowledge_gap_threshold", 0.5) or 0.5):
                extra["clarify"] = (
                    "Not fully confident I caught this question — rephrase if it's off.")
            if extra:
                await send({"type": "meta", "qid": qid, **extra})
        except Exception:  # noqa: BLE001
            directive, forced_depth = None, None
        # Verifier critique (regeneration pass) — folded into the SAME call.
        if directive_extra:
            directive = ((directive + "\n" + directive_extra).strip()
                         if directive else str(directive_extra))

        factory = get_session_factory()
        if factory is None:
            await send({
                "type": "error", "qid": qid,
                "detail": "Database not ready — check Settings -> Database.",
            })
            if sm is not None:
                sm.on_answer_done()
            return

        # Trust/privacy (Phase 6): the LLM egress copy is sanitized (prompt-
        # injection guard) + PII-redacted; the persisted/displayed transcript
        # uses the ORIGINAL `question` (unchanged unless purged). Sanitization
        # fails open; PII redaction fails CLOSED (§11) — if the redaction
        # module itself can't run, the raw transcript must not reach a
        # third-party LLM.
        llm_question = question
        if getattr(cfg.live, "transcript_sanitization", False):
            try:
                from app.live import sanitize as _san
                llm_question = _san.sanitize(llm_question)
            except Exception:  # noqa: BLE001
                _layer_failed()
        if getattr(cfg.live, "pii_redaction", False):
            try:
                from app.live import privacy as _priv
                llm_question, _ = _priv.redact(llm_question)
            except Exception:  # noqa: BLE001 — withhold rather than leak
                import logging
                logging.getLogger("zapthetrick.live").error(
                    "PII redaction unavailable — withholding transcript "
                    "for qid %s", qid)
                llm_question = ("[transcript withheld: PII redaction "
                                "unavailable]")
        try:
            async with factory() as db_session:
                ctx = AnswerContext(
                    # Answer the CLEANED (+ sanitized/redacted egress) question.
                    question=llm_question,
                    session_id=sid,
                    profile=profile or {},
                    resume_id=resume_id,
                    db_session=db_session,
                    audio=None,
                    forced_type=qtype,       # skip the orchestrator's classifier
                    forced_difficulty=difficulty,  # skip its difficulty LLM call
                    answer_only_questions=False,
                    skip_embedding=True,     # keep the live path fast
                    live=True,               # concise prompt + first-token watchdog
                    answer_directive=directive,
                    forced_depth=("concise" if _fast_factual
                                  else forced_depth),
                    # A retry forced to 'expert' (final escalation, or a garbled
                    # answer) drops the pinned fast model so the auto-router picks
                    # a different, stronger one. Original expert questions keep
                    # the pinned model (retry_stage 0).
                    escalate=(retry_stage > 0
                              and str(difficulty).strip().lower() == "expert"),
                )
                answer_text = ""
                done_data: dict = {}
                # Bounded generation: a hung/slow LLM must not stall the
                # interview past this budget — the partial answer is kept,
                # the qid finalizes, and the session moves on.
                _budget_s = float(getattr(cfg.live, "answer_timeout_s", 60.0)
                                  or 60.0)
                _t0 = asyncio.get_event_loop().time()
                _gen = answer_question(ctx)
                _timed_out = False
                while True:
                    _left = _budget_s - (asyncio.get_event_loop().time() - _t0)
                    if _left <= 0:
                        _timed_out = True
                        break
                    try:
                        ev = await asyncio.wait_for(_gen.__anext__(),
                                                    timeout=_left)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        _timed_out = True
                        break
                    # We already sent a richer meta (with the cleaned question).
                    if ev.kind == "meta":
                        continue
                    # Defer the terminal `done` so additive post-answer meta
                    # (talking points) lands BEFORE the client finalizes the qid.
                    if ev.kind == "done":
                        done_data = dict(ev.data)
                        continue
                    if ev.kind == "token":
                        answer_text += str(ev.data.get("text", ""))
                    await send({"type": ev.kind, "qid": qid, **ev.data})
                if _timed_out:
                    import contextlib as _ctx
                    with _ctx.suppress(Exception):
                        await _gen.aclose()
                    done_data.setdefault("timeout", True)
                    await send({"type": "meta", "qid": qid, "timeout": True,
                                "detail": ("answer generation exceeded "
                                           f"{int(_budget_s)}s — showing what "
                                           "was generated")})
                # Persist the Q&A to this live session's history (org sidebar).
                if answer_text.strip():
                    # ADVISORY emotion that wasn't ready when the answer went out
                    # (we never block on it — see `_run_answer`). Surface it now
                    # as its own additive meta frame, before the terminal `done`.
                    if (not _emotion_surfaced
                            and getattr(cfg.live, "emotion_signal", False)):
                        try:
                            _emo_late = _emotion_by_qid.pop(qid, None)
                            if _emo_late and _emo_late[0]:
                                await send({"type": "meta", "qid": qid,
                                            "emotion": _emo_late[0]})
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # Glanceable surface (Phase 8): emit concise talking-point
                    # bullets as additive meta (the full answer is unchanged).
                    if getattr(cfg.live, "glanceable_surface", False):
                        try:
                            from app.live import surface as _surface
                            pts = _surface.talking_points(answer_text)
                            if pts:
                                await send({"type": "meta", "qid": qid,
                                            "talking_points": pts})
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # Predicted follow-ups (Phase 11): likely next questions for
                    # the current topic (the copilot "likely follow-ups" surface).
                    if getattr(cfg.live, "question_prediction", False):
                        try:
                            from app.live import predict as _pred
                            from app.live import topic_graph as _tg3
                            from app.live import world_model as _wm3
                            preds = _pred.predict_next(
                                topic_graph=_tg3.for_tracker(get_tracker(sid)),
                                world_model=_wm3.for_tracker(get_tracker(sid)))
                            if preds:
                                await send({"type": "meta", "qid": qid,
                                            "predicted_followups": preds})
                                # 2B-10 Speculative pre-drafting: stash answer
                                # scaffolds for the top predicted follow-ups so a
                                # matching next question starts already structured
                                # (consumed as a directive at answer entry). Runs
                                # POST-answer, off the hot path; deterministic.
                                if getattr(cfg.live, "predictive_drafting", True):
                                    _drafts = _pred.predraft(sid, preds)
                                    if _drafts:
                                        await send({"type": "meta", "qid": qid,
                                                    "predrafted": len(_drafts)})
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # 2C-24 Cross-round memory: durably record this topic against
                    # the target company so a later interview round can build on
                    # it. Persisted to a small local JSON file; fail-open.
                    if getattr(cfg.live, "cross_round_memory", True):
                        try:
                            from app.live import cross_round as _cr2
                            _cr_org = (org_ctx.get("org_name") or "") if isinstance(org_ctx, dict) else ""
                            if _cr_org and topic:
                                _cr2.record_topic(
                                    _cr_org, topic,
                                    role=(org_ctx.get("job_role", "") if isinstance(org_ctx, dict) else ""),
                                    qtype=qtype or "")
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # 2E-35 Voice output (minimal): surface speech-ready plain
                    # text so a client with its own TTS can voice the answer.
                    if getattr(cfg.live, "voice_output", True):
                        try:
                            from app.live import tts as _tts
                            _speech = _tts.speech_markup(answer_text)
                            if _speech:
                                await send({"type": "meta", "qid": qid,
                                            "speech_text": _speech[:4000]})
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # 2D-29 Developer overlay: role + model + latency + phase +
                    # band as a single additive meta.dev frame (prod ignores it).
                    if getattr(cfg.live, "dev_mode", True):
                        try:
                            from app.live import devmode as _dev
                            _ov = _dev.overlay(
                                model=(done_data.get("model") if isinstance(done_data, dict) else None),
                                latency_ms=((asyncio.get_event_loop().time() - _t0) * 1000.0),
                                phase=(d.phase if d is not None else None),
                                band=(extra.get("seniority_band") if isinstance(extra, dict) else None),
                                qtype=qtype)
                            if _ov:
                                await send({"type": "meta", "qid": qid, "dev": _ov})
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # Session health (Phase 11): additive non-blocking warning.
                    if getattr(cfg.live, "session_health", False):
                        try:
                            from app.live import health as _health
                            hw = _health.session_health(
                                latency_ms=_health.latency_ms_estimate())
                            if hw is not None:
                                # #3: how many duplicate questions this
                                # session's guard suppressed — visible proof
                                # the double-answer symptom is handled.
                                try:
                                    from app.live import ledger as _lg
                                    _dups = _lg.session_counts(sid).get(
                                        "skipped:duplicate_question", 0)
                                    if _dups:
                                        hw = {**hw,
                                              "duplicates_suppressed": _dups}
                                except Exception:  # noqa: BLE001
                                    pass
                                await send({**hw, "qid": qid})
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    await _persist_live_qa(sid, question, answer_text,
                                           calibration=_calib_meta,
                                           intent=_answer_intent)
                    # Remember this answer so a candidate reading/paraphrasing
                    # it back is recognized as an echo and skipped (item: skip
                    # candidate self-answers). Fail-open.
                    if getattr(cfg.live, "candidate_echo_skip", False):
                        try:
                            from app.live import echo as _echo2
                            _echo2.remember_answer(sid or "", answer_text)
                        except Exception:  # noqa: BLE001
                            pass
                    # Snapshot the in-process conversational state (tracker
                    # turns + conversation log + world model) so a backend
                    # restart mid-interview can restore it on reconnect.
                    if getattr(cfg.live, "session_resume", False):
                        try:
                            from app.live.state_persist import save_state
                            _sp = asyncio.create_task(save_state(sid))
                            answer_tasks.add(_sp)
                            _sp.add_done_callback(answer_tasks.discard)
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # Role-aware graph (dual-source): record the assistant's
                    # answer so later turns stay consistent with what we already
                    # suggested. Additive; fail-open.
                    if getattr(cfg.live, "role_memory", False):
                        try:
                            from app.live import conversation as _conv2
                            from app.question_detection.context_tracker import get_tracker
                            _conv2.for_tracker(get_tracker(sid)).add(
                                "assistant", answer_text[:500], topic)
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # Outcome analytics (Phase 14): record a lightweight answer
                    # event so the advisory Outcome_Estimate can derive an
                    # answered/total ratio at session end. Additive; fail-open.
                    if getattr(cfg.live, "outcome_estimate", False):
                        try:
                            from app.live.eventlog import get_log
                            get_log(sid).append("answer", {
                                "topic": topic,
                                "confidence": (extra.get("answer_confidence")
                                               if isinstance(extra, dict) else None),
                            })
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                    # Refresh the rolling session summary in the background
                    # (deterministic, no LLM call) so it never blocks the answer.
                    if getattr(cfg.live, "multi_level_memory", False):
                        try:
                            from app.question_detection.context_tracker import get_tracker
                            from app.live import memory as _mem
                            mem = _mem.for_tracker(get_tracker(sid))
                            asyncio.create_task(asyncio.to_thread(_mem.refresh_summary, mem))
                        except Exception:  # noqa: BLE001
                            _layer_failed()
                # Now emit the deferred terminal done.
                await send({"type": "done", "qid": qid, **done_data})
                bus.publish(ANSWER_DONE, qid=qid, chars=len(answer_text),
                            retry=bool(is_retry))
                # VERIFIER stage: score the finished answer (non-blocking —
                # tokens already streamed) and, on a weak/garbled verdict,
                # regenerate with the critique folded in — escalating to a
                # stronger model on the final retry. Keep verifying retries up to
                # `answer_max_retries` so the escalation chain can run; the final
                # (escalated) stage is not re-verified, which stops the loop.
                _max_retries = int(getattr(cfg.live, "answer_max_retries", 2) or 0)
                _code_max = int(getattr(cfg.live, "code_max_fix", 1) or 0)
                _is_code = (qtype == "coding"
                            and getattr(cfg.live, "code_sandbox", False))
                if _is_code:
                    # CODING answers verify by RUNNING the code in the sandbox
                    # (compile/run) — the authoritative check — plus the same
                    # leak/gibberish gate. Replaces the prose verifier for code.
                    if answer_text.strip() and retry_stage < _code_max:
                        _cvt = asyncio.create_task(_verify_code_and_maybe_regen(
                            qid, question, answer_text, difficulty, retry_stage))
                        answer_tasks.add(_cvt)
                        _cvt.add_done_callback(answer_tasks.discard)
                elif (answer_text.strip() and retry_stage < _max_retries
                        and getattr(cfg.live, "answer_verify", False)):
                    _vt = asyncio.create_task(_verify_and_maybe_regen(
                        qid, question, answer_text, topic, difficulty, qtype,
                        retry_stage))
                    answer_tasks.add(_vt)
                    _vt.add_done_callback(answer_tasks.discard)
        except asyncio.CancelledError:
            # Backend-initiated cancel (interruption / continuation merge /
            # Stop) — the client has a live bubble for this qid that would
            # otherwise spin in "streaming" forever. Best-effort terminal
            # frame; on a disconnect-cancel the socket is gone and this is a
            # harmless no-op.
            try:
                await asyncio.shield(
                    send({"type": "done", "qid": qid, "cancelled": True}))
            except Exception:  # noqa: BLE001
                pass
            raise
        except Exception as exc:  # noqa: BLE001
            await send({"type": "error", "qid": qid, "detail": str(exc)})
        finally:
            if sm is not None:
                sm.on_answer_done()
            if reg is not None:
                reg.close(qid)
            if budget is not None:
                budget.release()

    async def _verify_and_maybe_regen(qid, question, answer_text, topic,
                                      difficulty, qtype, retry_stage=0) -> None:
        """Post-answer VERIFIER: badge the answer with a verification verdict
        and (when `cfg.live.answer_regenerate` is on) regenerate on a weak /
        garbled score. Escalates each retry — bumping the difficulty tier, and
        on the FINAL retry forcing 'expert' so the orchestrator drops the pinned
        fast model and the auto-router picks a different, stronger model.
        Capped at `answer_max_retries`. Fail-open — no verdict, no badge, no
        regen."""
        try:
            from app.live import verify as _verify
            verdict = await _verify.verify_answer(
                question, answer_text, topic, session_key=sid)
            if verdict is None:
                return
            await send({"type": "meta", "qid": qid,
                        "verify": verdict.to_meta()})
            bus.publish(ANSWER_VERIFIED, qid=qid, **verdict.to_meta())
            if verdict.ok or not getattr(cfg.live, "answer_regenerate", False):
                return
            _max_retries = int(getattr(cfg.live, "answer_max_retries", 2) or 0)
            next_stage = retry_stage + 1
            if next_stage > _max_retries:
                return  # exhausted retries — keep the best answer, badged weak
            esc_difficulty = _escalate_difficulty(
                difficulty, next_stage, _max_retries, verdict.gibberish)
            new_qid = uuid.uuid4().hex
            # The revised answer streams under a fresh qid linked to the
            # original, so the client renders it as a follow-up bubble.
            await send({"type": "transcript", "qid": new_qid,
                        "text": question, "revised_of": qid})
            await send({"type": "meta", "qid": new_qid, "is_question": True,
                        "qtype": qtype, "question": question,
                        "revised_of": qid, "source": "verifier",
                        "retry_stage": next_stage,
                        "escalated": next_stage >= _max_retries})
            await _generate_answer(
                new_qid, question, qtype, esc_difficulty, topic=topic,
                directive_extra=_verify.critique_directive(verdict, question),
                retry_stage=next_stage)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _layer_failed()

    async def _verify_code_and_maybe_regen(qid, question, answer_text,
                                           difficulty, retry_stage=0) -> None:
        """Post-answer CODE verifier: extract the coding answer's code, RUN it in
        the sandbox (compile + execute), badge the result, and on a failure —
        or a leaked/garbled answer — regenerate once with the sandbox error (or
        critique) folded in. The regen prefers a resume language and asks for
        runnable code. Fail-open — no badge, no regen on any error."""
        async def _regen(critique: str) -> None:
            _cm = int(getattr(cfg.live, "code_max_fix", 1) or 0)
            nstage = retry_stage + 1
            if nstage > _cm:
                return
            nqid = uuid.uuid4().hex
            await send({"type": "transcript", "qid": nqid,
                        "text": question, "revised_of": qid})
            await send({"type": "meta", "qid": nqid, "is_question": True,
                        "qtype": "coding", "question": question,
                        "revised_of": qid, "source": "code-sandbox",
                        "retry_stage": nstage})
            await _generate_answer(nqid, question, "coding", difficulty,
                                   directive_extra=critique, retry_stage=nstage)

        try:
            from app.live import code_run as _cr
            from app.live import verify as _verify
            # 1) Same leak/gibberish gate as prose answers (code can leak too).
            _gib = _verify.looks_incoherent(answer_text)
            _leak = _verify.looks_like_leaked_reasoning(answer_text)
            if _gib or _leak:
                await send({"type": "meta", "qid": qid,
                            "code_verify": {"status": "leaked" if _leak
                                            else "gibberish", "ran": False}})
                await _regen(_verify.critique_directive(
                    _verify.Verdict(0.0, 1.0, "garbled/leaked output",
                                    gibberish=_gib, leaked=_leak), question))
                return
            # 2) Extract + run the code.
            _prof = profile if isinstance(profile, dict) else None
            lang, _ = _cr.pick_language(question, _prof)
            code, clang = _cr.extract_code(answer_text, prefer_lang=lang)
            run_lang = clang or lang
            if not code or run_lang not in _cr.RUNNABLE:
                await send({"type": "meta", "qid": qid,
                            "code_verify": {"status": "not_run",
                                            "language": run_lang}})
                return
            from app.orchestration.sandbox import verify_snippet
            res = await verify_snippet(code, run_lang)
            await send({"type": "meta", "qid": qid, "code_verify": {
                "status": res.status, "language": run_lang,
                "ran": bool(res.ran)}})
            if res.verified or res.status in ("unavailable", "disabled"):
                return  # verified, or sandbox couldn't run it → keep the answer
            # 3) Compile/run failure → regenerate with the sandbox error.
            err = (res.output or res.repair_feedback or "").strip()[:600]
            await _regen(
                "Your previous code FAILED to compile/run in a sandbox with "
                f"this error:\n{err}\nReturn corrected {run_lang} code that "
                "compiles and runs cleanly on the example input(s); keep the "
                "same overall approach and the runnable driver.")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _layer_failed()

    # Worker the segmenter calls when an utterance finalises. It sends the
    # transcript and SPAWNS the answer, then returns immediately so the
    # segmenter is free to detect the next question while this one answers.
    #
    # Turn-taking (live Phase 1, `cfg.live.turn_taking`): hold a finalized
    # utterance for a short settle window so continued speech merges into the
    # SAME question instead of being answered twice. Off (default) → answer per
    # finalized utterance, byte-for-byte today's behavior.
    from time import monotonic as _monotonic
    from app.live.hypothesis import HypothesisBuffer

    _turn: dict = {"buf": None, "task": None, "audio": None}
    # Dual-source: which speaker the incoming audio frames are attributed to.
    # Flipped by the `source_role` control frame (candidate|interviewer). Default
    # interviewer (today's single loopback source).
    channel_role: dict = {"role": "interviewer"}
    # The most recently COMMITTED question — the continuation-merge safety
    # net compares late-arriving fragments against it (no settle window can
    # beat every thinking pause; this heals whatever slips through).
    _last_commit: dict = {"qid": None, "text": "", "at": 0.0}
    # Filled in after the segmenter is constructed below; the settle timer
    # peeks at it to HOLD a commit while the speaker has resumed talking.
    _seg_holder: dict = {"seg": None}

    def _note_commit(qid: str, text: str) -> None:
        _last_commit.update(qid=qid, text=text, at=_monotonic())

    def _spawn_answer(utterance: str, audio_np, stt_conf=None) -> asyncio.Task:
        qid = uuid.uuid4().hex
        bus.publish(QUESTION_DETECTED, qid=qid, text=utterance[:200])
        _note_commit(qid, utterance)

        async def _go() -> None:
            await send({"type": "transcript", "qid": qid, "text": utterance})
            await _run_answer(qid, utterance, audio_np, stt_conf=stt_conf)

        task = asyncio.create_task(_go())
        answer_tasks.add(task)
        task.add_done_callback(answer_tasks.discard)
        # Register on the bus so an interruption cancels THIS qid's work.
        bus.register_answer_task(qid, task)
        return task

    async def _settle_then_answer(generation: int) -> None:
        # DYNAMIC endpointing: wait longer when the question is grammatically
        # incomplete (speaker paused mid-thought), shorter when it already
        # reads as complete — instead of a fixed silence gap that splits a
        # question the moment the interviewer pauses to think.
        buf0 = _turn["buf"]
        if buf0 is not None:
            settle_ms = buf0.required_settle_ms()
        else:
            settle_ms = int(getattr(cfg.live, "turn_settle_ms", 600) or 0)
        try:
            await asyncio.sleep(max(0, settle_ms) / 1000.0)
            # HOLD: if the speaker has already resumed (voiced audio is
            # buffered in the segmenter but not finalized yet), the coming
            # fragment belongs to THIS turn — committing now would answer a
            # half question. When that fragment finalizes, buf.add bumps the
            # generation and this task retires in the check below. This
            # closes the race a fixed settle window can never win: the
            # continuation's own endpoint + STT take longer than any
            # reasonable settle wait.
            seg = _seg_holder.get("seg")
            if seg is not None:
                deadline = (_monotonic()
                            + float(cfg.audio.max_utterance_ms) / 1000.0 + 5.0)
                while seg.utterance_pending() and _monotonic() < deadline:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return
        buf = _turn["buf"]
        if buf is None or buf.generation != generation:
            return  # superseded — the speaker continued, a newer settle is queued
        text, _had_audio = buf.take()
        audio = _turn["audio"]
        _turn["audio"] = None
        conf = _turn.pop("stt_conf", None)
        if text.strip():
            _spawn_answer(text, audio, stt_conf=conf)

    async def handle_utterance(utterance: str, audio_np=None, role=None,
                               *, stt_conf=None) -> None:
        # Dual-source (hear both voices, act on one): resolve the speaker role.
        # Default is interviewer (today's single loopback source). The candidate
        # channel is ABSORBED into the shared graph + commitments, never answered.
        _role = (role or channel_role.get("role") or "interviewer").strip().lower()
        # 2D-30 Live confidence recovery: track rolling STT confidence and, when
        # it stays degraded, trigger a ONE-TIME switch to the fallback ASR engine
        # (via app/stt/switch). Background + single-shot + fail-open — never
        # blocks the turn, never thrashes engines.
        if getattr(cfg.live, "confidence_recovery", True):
            try:
                from app.live import stt_recovery as _rec
                _rec.observe(sid, stt_conf)
                if _rec.should_recover(sid):
                    _tgt = _rec.recover(sid)   # decision only (latched)
                    if _tgt:
                        async def _do_recover(_target=_tgt) -> None:
                            try:
                                from app.stt import switch as _sw
                                await _sw.start_switch(_target)
                            except Exception:  # noqa: BLE001
                                return
                            await send({"type": "meta", "role": "system",
                                        "stt_recovery": _target})
                        _rt2 = asyncio.create_task(_do_recover())
                        answer_tasks.add(_rt2)
                        _rt2.add_done_callback(answer_tasks.discard)
            except Exception:  # noqa: BLE001
                _layer_failed()
        if getattr(cfg.live, "role_memory", False) or getattr(cfg.live, "commitment_tracking", False):
            try:
                from app.question_detection.context_tracker import get_tracker
                _trk = get_tracker(sid)
                _topic_now = ""
                try:
                    from app.live import world_model as _wm0
                    _topic_now = _wm0.for_tracker(_trk).topic
                except Exception:  # noqa: BLE001
                    _topic_now = ""
                if getattr(cfg.live, "role_memory", False):
                    from app.live import conversation as _conv
                    _conv.for_tracker(_trk).add(_role, utterance, _topic_now)
                if getattr(cfg.live, "commitment_tracking", False):
                    from app.live import world_model as _wm0b
                    _wm0 = _wm0b.for_tracker(_trk)
                    for _slot, _val in _wm0b.extract_commitments(utterance, role=_role).items():
                        _wm0b.record_commitment(_wm0, _slot, _role, _val,
                                                topic=(_topic_now or "salary"))
                    if _role == "candidate":
                        _wm0.mark_candidate_answered()
            except Exception:  # noqa: BLE001
                _layer_failed()
        if (not solo_mode and _role == "candidate"
                and getattr(cfg.live, "candidate_channel", False)):
            # Candidate's own speech: absorb (awareness) and STOP — never answer
            # it. Skipped in SOLO mode: there, every question is answered
            # regardless of role.
            try:
                from app.question_detection.context_tracker import get_tracker
                if getattr(cfg.live, "candidate_awareness", False):
                    from app.live import surface as _surf
                    _surf.for_tracker(get_tracker(sid)).observe_candidate(utterance)
            except Exception:  # noqa: BLE001
                _layer_failed()
            await send({"type": "meta", "role": "candidate", "absorbed": True,
                        "text": utterance})
            # DELIVERY COACHING (R38): feedback about the CANDIDATE's own
            # delivery (fillers / length / missing concrete example). It rides
            # out on its OWN meta frame and is never folded into an answer
            # directive — and this branch answers nothing anyway, so it cannot
            # reach the interview answer. Additive + fail-open (helper returns
            # [] on any error). Off the answer path entirely → zero latency cost.
            _tips = _coaching_tips(utterance)
            if _tips:
                await send({"type": "meta", "role": "candidate",
                            "coaching": _tips})
            return
        # Phase 4 robustness pre-processing (flag-gated, fail-open).
        # Conservative domain-term transcript repair BEFORE detection.
        if getattr(cfg.live, "transcript_repair", False):
            try:
                from app.live import repair as _repair
                from app.live import topic_graph as _tg
                from app.question_detection.context_tracker import get_tracker
                vocab = (cfg.stt.prompt or "").split()
                utterance = _repair.repair(
                    utterance, vocab=vocab, topic_graph=_tg.for_tracker(get_tracker(sid)))
                # Hesitation fillers ("uh", "um", "ahh") the ASR faithfully
                # transcribes make the question noisy and confuse the
                # completeness/continuation logic — strip them here so every
                # downstream consumer sees the clean question.
                utterance = _repair.strip_fillers(utterance)
            except Exception:  # noqa: BLE001
                _layer_failed()
        # Unified DECISION ENGINE (app/live/decision.py): interruption,
        # satisfaction/feedback, rhetorical suppression and implicit-question
        # promotion — previously four inline gates — decided in ONE place.
        bus.publish(UTTERANCE_FINALIZED, text=utterance[:200], role=_role,
                    audio=audio_np is not None)
        verdict = None
        try:
            from app.live import decision as _dec
            verdict = _dec.decide_utterance(
                utterance, is_audio=audio_np is not None)
        except Exception:  # noqa: BLE001
            _layer_failed()
        if verdict is not None:
            from app.live import decision as _dec
            for frame in verdict.frames:
                await send(frame)
            if verdict.action == _dec.CANCEL_THEN_ANSWER:
                # Self-correction / topic switch: cancel everything in flight,
                # then fall through and answer the NEW utterance.
                _spec_cancel()
                bus.cancel_all_answers(reason="interruption")
                for t in list(answer_tasks):
                    t.cancel()
                if getattr(cfg.live, "state_machine", False):
                    try:
                        from app.live.state_machine import get_state_machine
                        get_state_machine(sid).mark_interrupted()
                    except Exception:  # noqa: BLE001
                        _layer_failed()
                if getattr(cfg.live, "event_log", False):
                    try:
                        from app.live.eventlog import get_log
                        get_log(sid).append("interrupted", {"text": utterance[:80]})
                    except Exception:  # noqa: BLE001
                        _layer_failed()
            elif verdict.action == _dec.SKIP:
                _spec_cancel()  # a suppressed utterance must not flush a spec
                bus.publish(QUESTION_SKIPPED, reason=verdict.reason,
                            text=utterance[:120])
                # Accuracy ledger + UI transparency: a skipped utterance is
                # VISIBLE (muted row with the reason) and correctable — the
                # client can force-answer it, which is logged as feedback.
                _skip_qid = uuid.uuid4().hex
                await send({"type": "skipped", "qid": _skip_qid,
                            "text": utterance, "reason": verdict.reason})
                try:
                    from app.live import ledger as _ledger
                    _ledger.record(sid, _skip_qid, utterance,
                                   _ledger.SKIPPED, reason=verdict.reason)
                except Exception:  # noqa: BLE001
                    pass
                if (getattr(cfg.live, "event_log", False)
                        and verdict.reason == "feedback" and verdict.frames):
                    try:
                        from app.live.eventlog import get_log
                        get_log(sid).append(
                            "feedback", {"state": verdict.frames[0].get("state")})
                    except Exception:  # noqa: BLE001
                        _layer_failed()
                return

        # CONTINUATION MERGE (retroactive safety net): a fragment arriving
        # shortly after a committed question that reads as its TAIL ("in
        # spring boot", "various stereotype annotations") means the commit
        # fired too early — no settle window can beat every thinking pause.
        # Cancel the in-flight answer and re-answer the MERGED question.
        # Audio-only: typed text is deliberate, never a stray fragment.
        if (audio_np is not None
                and getattr(cfg.live, "continuation_merge", True)
                and _last_commit.get("text")):
            _cont_window = float(
                getattr(cfg.live, "continuation_window_s", 8.0) or 8.0)
            if ((_monotonic() - _last_commit.get("at", 0.0)) <= _cont_window
                    and _looks_like_continuation(utterance)):
                _spec_cancel()
                prev_qid = _last_commit.get("qid")
                if prev_qid:
                    bus.cancel_answer(prev_qid, reason="continuation")
                merged_q = _merge_continuation(
                    _last_commit["text"], utterance)
                await send({"type": "meta", "continuation_of": prev_qid,
                            "merged_question": merged_q})
                utterance = merged_q
                # Consumed — the merged question's own commit re-records it,
                # so a THIRD fragment can chain onto the merged text.
                _last_commit.update(qid=None, text="", at=0.0)

        # SPECULATION FLUSH: if a speculative answer was started from this
        # utterance's partial and the FINAL transcript matches it, hand the
        # utterance to the speculation — flush its buffered frames (transcript,
        # meta, any tokens already generated) and let the rest stream live.
        # The end-of-speech wait and the LLM first-token wait have then been
        # fully overlapped. A mismatch (speaker kept going / STT differed)
        # cancels the speculation and answers normally.
        if audio_np is not None and _spec_state.get("task") is not None:
            spec_text = _spec_state.get("text", "")
            a, b = _norm_question(utterance), _norm_question(spec_text)
            matched = a == b
            if not matched and a and b:
                import difflib as _difflib
                matched = _difflib.SequenceMatcher(None, a, b).ratio() >= 0.92
            if matched:
                holder = _spec_state.get("holder")
                task = _spec_state.get("task")
                spec_qid = _spec_state.get("qid")
                _spec_state.update(task=None, text="", holder=None, qid=None)
                if spec_qid:
                    _note_commit(spec_qid, utterance)
                if holder is not None:
                    # Show the FINAL (repaired) transcript, not the partial.
                    for f in holder.frames:
                        if f.get("type") == "transcript" and f.get("speculative"):
                            f["text"] = utterance
                            f.pop("speculative", None)
                            break
                    sent_n = 0
                    while sent_n < len(holder.frames):
                        f = holder.frames[sent_n]
                        sent_n += 1
                        await send(f)
                    # No await between the length check above and this flip, so
                    # nothing can slip into the buffer unsent.
                    holder.live = True
                    if task is not None:
                        answer_tasks.add(task)
                return  # the speculation owns this utterance
            _spec_cancel()

        if getattr(cfg.live, "turn_taking", False):
            now = _monotonic()
            buf = _turn["buf"]
            if buf is None:
                buf = HypothesisBuffer(
                    settle_ms=int(getattr(cfg.live, "turn_settle_ms", 600) or 0))
                _turn["buf"] = buf
            gen = buf.add(utterance, now, has_audio=audio_np is not None)
            if audio_np is not None:
                _turn["audio"] = audio_np
            if stt_conf is not None:
                _turn["stt_conf"] = stt_conf
            # Reschedule the end-of-turn check; continued speech extends the turn.
            prev = _turn["task"]
            if prev is not None and not prev.done():
                prev.cancel()
            settle = asyncio.create_task(_settle_then_answer(gen))
            _turn["task"] = settle
            answer_tasks.add(settle)
            settle.add_done_callback(answer_tasks.discard)
            return
        # Turn-taking off → today's behavior: answer each finalized utterance.
        qid = uuid.uuid4().hex
        bus.publish(QUESTION_DETECTED, qid=qid, text=utterance[:200])
        _note_commit(qid, utterance)
        await send({"type": "transcript", "qid": qid, "text": utterance})
        task = asyncio.create_task(
            _run_answer(qid, utterance, audio_np, stt_conf=stt_conf))
        answer_tasks.add(task)
        task.add_done_callback(answer_tasks.discard)
        bus.register_answer_task(qid, task)

    # SPECULATIVE ANSWERING — the last big latency lever. A partial that
    # already reads as a complete question ("… in Kafka?") means the LLM can
    # start DURING the end-of-speech silence instead of after it. The whole
    # answer pipeline runs with its frames buffered; when the utterance
    # finalizes and the transcript matches, the buffer flushes instantly —
    # first tokens are typically already waiting, so speech-end → visible
    # answer collapses to just the endpoint wait (~300 ms).
    _spec_state: dict = {"task": None, "text": "", "holder": None, "qid": None}

    def _spec_cancel() -> None:
        t = _spec_state.get("task")
        if t is not None and not t.done():
            t.cancel()
        _spec_state.update(task=None, text="", holder=None, qid=None)

    def _maybe_speculate(text: str) -> None:
        if not getattr(cfg.live, "speculative_answers", True):
            return
        t = (text or "").rstrip()
        if not _speculation_worthy(t):
            return
        prev = _spec_state.get("task")
        if prev is not None and not prev.done():
            if _norm_question(_spec_state.get("text", "")) == _norm_question(t):
                return  # already speculating on this exact question
            _spec_cancel()  # superseded by a longer/different question
        holder = _SpecHolder()
        qid = uuid.uuid4().hex
        _spec_state.update(text=t, holder=holder, qid=qid)

        async def _go() -> None:
            token = _SPEC_HOLDER.set(holder)
            try:
                await send({"type": "transcript", "qid": qid, "text": t,
                            "speculative": True})
                await _run_answer(qid, t, None)
            finally:
                _SPEC_HOLDER.reset(token)

        task = asyncio.create_task(_go())
        _spec_state["task"] = task
        answer_tasks.add(task)
        task.add_done_callback(answer_tasks.discard)
        bus.register_answer_task(qid, task)

    async def resume_answer(qid: str, question: str, partial: str,
                            mode: str) -> None:
        """Continue or retry an interrupted answer, streaming into the SAME
        `qid` bubble. `mode == 'continue'` resumes from `partial` (aware of
        what was already said and what's left); otherwise it regenerates the
        answer fresh. Reuses the normal answer engine via an answer directive,
        so a NEW model (after a rate-limit failover) picks up with full
        context."""
        q = (question or "").strip()
        if not q:
            await send({"type": "done", "qid": qid})
            return
        factory = get_session_factory()
        if factory is None:
            await send({"type": "error", "qid": qid,
                        "detail": "Database not ready — check Settings."})
            await send({"type": "done", "qid": qid})
            return
        if mode == "continue" and (partial or "").strip():
            directive = (
                "You already began answering this interview question but were "
                "cut off mid-response. This is exactly what you have said so "
                "far:\n\n\"\"\"\n" + partial.strip() + "\n\"\"\"\n\n"
                "Continue the answer seamlessly from precisely where it "
                "stopped. Do NOT repeat, summarize, or re-introduce anything "
                "already said — output only the remaining continuation so the "
                "whole reads as one coherent answer.")
        else:
            directive = ("Answer this interview question again, clearly and "
                         "completely from the start.")
        try:
            async with factory() as db_session:
                ctx = AnswerContext(
                    question=q, session_id=sid, profile=profile or {},
                    resume_id=resume_id, db_session=db_session, audio=None,
                    answer_only_questions=False, skip_embedding=True,
                    live=True, answer_directive=directive,
                )
                saw_done = False
                new_text: list[str] = []
                async for ev in answer_question(ctx):
                    # Skip the classifier meta (we already know it's a
                    # question); pass tokens + the terminal done straight
                    # through, tagged with this qid so the client appends.
                    if ev.kind == "meta":
                        continue
                    if ev.kind == "token":
                        new_text.append(str(ev.data.get("text", "")))
                    if ev.kind == "done":
                        saw_done = True
                    await send({"type": ev.kind, "qid": qid, **ev.data})
                if not saw_done:
                    await send({"type": "done", "qid": qid})
                # Persist the completed answer so session history doesn't keep
                # only the pre-cut partial (continue) or nothing (retry).
                tail = "".join(new_text).strip()
                if tail:
                    full = ((partial.rstrip() + "\n" + tail)
                            if mode == "continue" and (partial or "").strip()
                            else tail)
                    try:
                        await _persist_resumed_answer(sid, q, partial, full)
                    except Exception:  # noqa: BLE001
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await send({"type": "error", "qid": qid, "detail": str(exc)})
            await send({"type": "done", "qid": qid})

    def spawn_resume_answer(qid: str, question: str, partial: str,
                            mode: str) -> None:
        """Run [resume_answer] as a tracked, cancellable background task so the
        Stop button (cancel_answer) and a disconnect can tear it down."""
        task = asyncio.create_task(resume_answer(qid, question, partial, mode))
        answer_tasks.add(task)
        task.add_done_callback(answer_tasks.discard)
        try:
            bus.register_answer_task(qid, task)
        except Exception:  # noqa: BLE001
            pass

    # STREAMING STT partials: while the speaker is still talking, interim
    # text from the fast provider streams to the client as `partial` frames
    # (the final utterance still goes through the accurate primary chain).
    async def _on_partial(text: str) -> None:
        bus.publish(PARTIAL_TRANSCRIPT, text=text[:200])
        await send({"type": "partial", "text": text})
        _maybe_speculate(text)

    # Surface a swallowed STT failure/empty so a hot mic that produces no
    # transcript is EXPLAINABLE (was: silent nothing → "mic doesn't work").
    async def _on_stt_status(kind: str, detail: str) -> None:
        await send({"type": "stt_status", "state": kind, "detail": detail})

    # Speech-start hook: pre-open the live LLM provider connection while the
    # speaker is still talking, so the answer's first request never pays TLS
    # setup (the pool's keepalive expires between questions). Rate-limited —
    # one warm per ~20s covers a whole question/answer exchange.
    _warm_state = {"at": 0.0}

    async def _on_speech_start() -> None:
        now = _monotonic()
        if now - _warm_state["at"] < 20.0:
            return
        _warm_state["at"] = now
        from app.perceived.prefetch import warm_live_provider
        await warm_live_provider()

    segmenter = AudioStreamSegmenter(
        on_utterance=handle_utterance,
        prompt_provider=lambda: _stt_bias_prompt(sid),
        on_partial=(_on_partial
                    if (getattr(cfg.stt, "partial_provider", "") or "")
                    else None),
        on_stt_status=_on_stt_status,
        on_speech_start=_on_speech_start,
    )
    # The settle timer peeks at the segmenter to hold a commit while the
    # speaker has resumed talking (see _settle_then_answer).
    _seg_holder["seg"] = segmenter

    # SEPARATE candidate segmenter (dual-source fix). On desktop "capture both",
    # the server grabs system loopback (the INTERVIEWER) into `segmenter`, while
    # the client streams its own mic (the CANDIDATE) as PCM bytes. Feeding both
    # into ONE segmenter interleaved the two PCM streams into garbled
    # utterances AND answered the candidate's own speech as interviewer
    # questions. So client PCM gets its own segmenter, tagged `candidate`, whenever
    # server capture is active; the candidate channel is absorbed, never answered.
    async def _candidate_utterance(text, audio_np=None, role=None,
                                   *, stt_conf=None) -> None:
        await handle_utterance(text, audio_np, "candidate", stt_conf=stt_conf)

    candidate_segmenter = AudioStreamSegmenter(
        on_utterance=_candidate_utterance,
        prompt_provider=lambda: _stt_bias_prompt(sid),
        on_stt_status=_on_stt_status,
    )

    # Holds the server-side capture task (system-loopback / mic) when the
    # client asks the backend to capture audio itself. Mutable so the
    # control handler and the disconnect path can both reach it.
    capture_state: dict = {"task": None}
    # Surface an audio-pipeline failure to the client at most once per session
    # (instead of crashing the socket on every frame → reconnect loop).
    audio_error_sent = False

    try:
        while True:
            msg = await websocket.receive()
            # Raw `receive()` RETURNS the disconnect message — it does not
            # raise WebSocketDisconnect (that's the receive_text/bytes
            # wrappers). Without this check the loop spun once more and the
            # second receive() raised RuntimeError, skipping cleanup entirely
            # — leaking the loopback capture + in-flight answers per drop.
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                # Audio frame. Treat as 16-bit PCM little-endian and convert.
                raw = msg["bytes"]
                # FE-tagged framing (remote backend): a pod can't capture the
                # interview's system-loopback audio itself, so the Flutter
                # client captures BOTH the interviewer (system loopback) and
                # the candidate (mic) locally and streams them over this one
                # socket, prefixing each frame with a 1-byte role tag
                # (0x00 = interviewer/answered, 0x01 = candidate/absorbed). We
                # split the tag here and route to the matching segmenter — the
                # server-side analogue of the desktop dual-capture split below.
                _seg = None
                if capture_state.get("fe_tagged"):
                    role, raw = _split_fe_frame(raw)
                    # SOLO answers every source; STANDARD answers only the
                    # interviewer role and absorbs the candidate.
                    _seg = segmenter if solo_mode else (
                        candidate_segmenter if role == 1 else segmenter)
                chunk = _decode_pcm(raw)
                if chunk is not None:
                    # Untagged legacy path: client PCM is the CANDIDATE mic only
                    # when the server is also capturing loopback (desktop
                    # dual-source). Otherwise (mobile) the client mic IS the
                    # interviewer — main segmenter. SOLO: one source drives
                    # everything → always the main (answerable) segmenter.
                    if _seg is None:
                        _seg = (segmenter if solo_mode else (
                                candidate_segmenter
                                if capture_state.get("task") is not None
                                else segmenter))
                    try:
                        await _seg.push(chunk)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        # A failure deep in the audio pipeline (VAD/STT) must
                        # NOT tear down the WebSocket — that caused the client's
                        # "live audio keeps failing" reconnect loop. Log it,
                        # tell the client once, and keep the socket alive.
                        import logging
                        logging.getLogger("zapthetrick.live").exception(
                            "live audio frame processing failed"
                        )
                        if not audio_error_sent:
                            audio_error_sent = True
                            await send({
                                "type": "error",
                                "detail": f"Audio processing failed on the server: {exc}",
                            })
                            # Mobile runtime (Phase 7): also surface a clear,
                            # classified audio status (no silent failure).
                            if getattr(cfg.live, "mobile_runtime", False):
                                try:
                                    from app.live import mobile as _mobile
                                    await send(_mobile.classify_audio_error(str(exc)))
                                except Exception:  # noqa: BLE001
                                    _layer_failed()
            elif msg.get("text") is not None:
                await _handle_control(
                    websocket, json.loads(msg["text"]), segmenter,
                    handle_utterance, capture_state, send, channel_role,
                    sid=sid, bus=bus, resume_answer=spawn_resume_answer,
                    profile=profile, org_ctx=org_ctx,
                )
    except WebSocketDisconnect:
        pass  # normal disconnect — cleanup runs in the finally below
    finally:
        await _stop_capture(capture_state)
        # Cancel any in-flight answer tasks so they don't keep streaming into
        # a dead socket.
        for task in list(answer_tasks):
            task.cancel()
        # Drop the per-session live state machine (in-process; no DB) — UNLESS
        # session resume is on, in which case we keep it so a reconnect with the
        # same session_id continues the interview without re-answering (R23).
        if not getattr(cfg.live, "session_resume", False):
            try:
                from app.live.state_machine import forget_session as _forget_sm
                _forget_sm(sid)
                from app.live.eventlog import forget_session as _forget_log
                _forget_log(sid)
                from app.live.resume import forget_session as _forget_reg
                _forget_reg(sid)
                from app.live.budget import forget_session as _forget_budget
                _forget_budget(sid)
                from app.live.echo import forget_session as _forget_echo
                _forget_echo(sid)
                from app.live.phase_tracker import forget_session as _forget_phase
                _forget_phase(sid)
                from app.live.contract import forget_session as _forget_contract
                _forget_contract(sid)
                from app.live.rhythm import forget_session as _forget_rhythm
                _forget_rhythm(sid)
                from app.live.stt_recovery import forget_session as _forget_rec
                _forget_rec(sid)
                from app.live.predict import forget_session as _forget_pred
                _forget_pred(sid)
            except Exception:  # noqa: BLE001
                _layer_failed()
        # Socket is gone — drop the audio buffer and cancel pending
        # transcriptions rather than flushing (which would transcribe only to
        # fail the send back).
        await segmenter.cancel()
        try:
            await candidate_segmenter.cancel()
        except Exception:  # noqa: BLE001
            pass


async def _handle_control(
    websocket: WebSocket,
    payload: dict,
    segmenter: AudioStreamSegmenter,
    handle_utterance,
    capture_state: dict,
    send,
    channel_role: dict | None = None,
    sid: str | None = None,
    bus=None,
    resume_answer=None,
    profile: dict | None = None,
    org_ctx: dict | None = None,
) -> None:
    """Process a non-audio control message from the client. `send` is the
    lock-guarded sender shared with the concurrent answer tasks."""
    kind = payload.get("type")
    if kind == "text":
        content = payload.get("content", "")
        if content.strip():
            # No audio when the user typed the message — prosody fusion
            # skips automatically when audio_np is None. An optional `role`
            # attributes typed input to a speaker (dual-source).
            await handle_utterance(content.strip(), None, payload.get("role"))
    elif kind == "candidate_text":
        # Explicit candidate speech (typed): absorbed, never answered.
        content = payload.get("content", "")
        if content.strip():
            await handle_utterance(content.strip(), None, "candidate")
    elif kind == "source_role":
        # Dual-source: attribute subsequent audio frames to this speaker
        # (candidate|interviewer). The tagged mic stream sets this before
        # streaming the candidate's audio.
        if channel_role is not None:
            role = (payload.get("role") or "interviewer").strip().lower()
            channel_role["role"] = role if role in ("candidate", "interviewer") else "interviewer"
            await send({"type": "meta", "channel_role": channel_role["role"]})
    elif kind in ("screen_text", "multimodal"):
        # 2E-34 Screen / vision understanding: shared-screen OCR text (or any
        # non-audio modality) is normalized into the SAME utterance shape and
        # fed through the live pipeline. Gated by cfg.live.multimodal; fail-open.
        if getattr(cfg.live, "multimodal", False):
            try:
                from app.live import multimodal as _mm
                modality = payload.get("modality") or _mm.SCREEN_TEXT
                mi = _mm.to_utterance(modality, payload.get("content", ""),
                                      meta=payload.get("meta"))
                if mi is not None and mi.text:
                    await send({"type": "meta", "modality": mi.modality,
                                "ingested": True})
                    await handle_utterance(mi.text, None,
                                           payload.get("role") or "interviewer")
            except Exception:  # noqa: BLE001
                pass
    elif kind == "mock_start":
        # 2D-28 Self-practice / mock: the app acts as the interviewer, generating
        # practice questions from the profile + org and pushing them as
        # `mock_question` frames (each explicitly labeled practice). The
        # candidate answers through the SAME live pipeline. Gated; fail-open.
        if getattr(cfg.live, "mock_mode", False):
            try:
                from app.live import mock as _mock
                from app.live import org as _morg
                from app.live import profile as _mprof
                _cp = _mprof.build_profile(profile) if isinstance(profile, dict) and profile else None
                _oc = org_ctx or {}
                _org = _morg.build_org(_oc.get("org_name", ""),
                                       _oc.get("job_description", ""),
                                       _oc.get("job_role", ""),
                                       notes=_oc.get("notes", ""))
                limit = int(payload.get("limit", 8) or 8)
                qs = _mock.generate_questions(_cp, _org, limit=limit)
                await send({"type": "meta", "mock_started": True,
                            "count": len(qs)})
                for i, q in enumerate(qs):
                    await send({"type": "mock_question", "seq": i,
                                "question": q.get("question", ""),
                                "category": q.get("category", ""),
                                "label": q.get("label", "practice")})
            except Exception:  # noqa: BLE001
                pass
    elif kind == "research_brief":
        # 2E-36 Pre-interview research (minimal): a deterministic prep brief from
        # the intake (company + role + JD skills) — no browsing. Gated; fail-open.
        if getattr(cfg.live, "pre_interview_research", True):
            try:
                from app.live import org as _rorg
                from app.live import research as _research
                _oc = org_ctx or {}
                _org = _rorg.build_org(_oc.get("org_name", ""),
                                       _oc.get("job_description", ""),
                                       _oc.get("job_role", ""))
                brief = _research.build_brief(_org.company, _org.role, _org.jd_skills)
                await send({"type": "research_brief", **brief.to_dict()})
            except Exception:  # noqa: BLE001
                pass
    elif kind == "flush":
        await segmenter.flush()
    elif kind == "start_capture":
        # Server-side capture: the backend grabs system-loopback audio (the
        # other party in a Zoom/Teams/Meet call) and feeds it through the
        # same VAD -> STT -> classify -> answer pipeline. No client mic needed.
        source = payload.get("source") or cfg.audio.source
        await _start_capture(segmenter, source, capture_state, send)
    elif kind == "client_capture":
        # Remote-backend (pod) path: the client can't rely on the server to
        # capture the interview's system audio, so it captures BOTH sources
        # locally and streams them role-tagged (see the binary-frame handler).
        # This just flips the per-connection framing flag; the audio itself
        # arrives as tagged binary frames. Fail-open — defaults to tagged.
        capture_state["fe_tagged"] = bool(payload.get("tagged", True))
        await send({"type": "capture",
                    "state": "client" if capture_state["fe_tagged"] else "server"})
    elif kind == "stop_capture":
        await _stop_capture(capture_state)
        await segmenter.flush()
        await send({"type": "capture", "state": "stopped"})
    elif kind == "stop":
        await _stop_capture(capture_state)
        await websocket.close()
    elif kind == "cancel_answer":
        # User tapped Stop while an answer was streaming — cancel the in-flight
        # generation(s) so we stop burning tokens. A specific qid cancels just
        # that answer; otherwise cancel everything in flight. Keeps the socket
        # open (the session continues).
        if bus is not None:
            qid = payload.get("qid")
            try:
                if qid:
                    bus.cancel_answer(str(qid), reason="user_stop")
                else:
                    bus.cancel_all_answers(reason="user_stop")
            except Exception:  # noqa: BLE001
                pass
        await send({"type": "meta", "cancelled": True,
                    "qid": payload.get("qid")})
    elif kind in ("continue_answer", "retry_answer"):
        # An answer was interrupted (rate-limit / error / user Stop). Resume it
        # in the SAME bubble: 'continue_answer' picks up from the partial the
        # client still shows; 'retry_answer' regenerates fresh. Either way the
        # (possibly new, post-failover) model gets the question + partial so it
        # knows what's done and what's left.
        qid = (payload.get("qid") or "").strip()
        question = payload.get("question") or ""
        partial = payload.get("partial") or ""
        mode = "continue" if kind == "continue_answer" else "retry"
        if qid and resume_answer is not None:
            resume_answer(qid, question, partial, mode)
    elif kind == "detection_feedback":
        # Accuracy ledger: the user flagged a decision as wrong ("this
        # shouldn't have been answered" / "this should have been").
        verdict = (payload.get("verdict") or "").strip()
        if verdict in ("should_have_answered", "should_not_have_answered"):
            try:
                from app.live import ledger as _ledger
                _ledger.feedback(sid or "", payload.get("qid"), verdict,
                                 utterance=payload.get("text", "") or "")
            except Exception:  # noqa: BLE001
                pass
            await send({"type": "meta", "feedback_recorded": True,
                        "qid": payload.get("qid")})
    elif kind == "answer_feedback":
        # Answer-quality feedback: the user gave a delivered answer a thumbs
        # up/down (or cleared it). Feeds the same accuracy ledger.
        rating = (payload.get("rating") or "").strip()
        if rating in ("thumb_up", "thumb_down", ""):
            try:
                from app.live import ledger as _ledger
                _ledger.answer_feedback(
                    sid or "", payload.get("qid"), rating,
                    utterance=payload.get("text", "") or "",
                    answer=payload.get("answer", "") or "")
            except Exception:  # noqa: BLE001
                pass
            await send({"type": "meta", "answer_feedback_recorded": True,
                        "qid": payload.get("qid")})
    elif kind == "force_answer":
        # The user tapped "Answer" on a skipped utterance: answer it now via
        # the typed path (audio-only gates don't apply), and record the skip
        # as a detection miss so the ledger learns from it.
        text = (payload.get("text") or "").strip()
        if text:
            try:
                from app.live import ledger as _ledger
                _ledger.feedback(sid or "", payload.get("qid"),
                                 "should_have_answered", utterance=text)
            except Exception:  # noqa: BLE001
                pass
            await handle_utterance(text, None, payload.get("role"))
    elif kind == "ping":
        await send({"type": "pong"})


async def _start_capture(
    segmenter: AudioStreamSegmenter,
    source: str,
    capture_state: dict,
    send,
) -> None:
    """Spawn a background task that pumps server-captured audio chunks into
    the segmenter. Idempotent — a running capture is stopped first."""
    await _stop_capture(capture_state)

    async def _pump() -> None:
        from app.audio import capture as _capture
        try:
            async for chunk in _capture.read_chunks(source=source):
                await segmenter.push(chunk)
        except asyncio.CancelledError:
            raise
        except _capture.CaptureError as exc:
            await send({"type": "capture", "state": "error", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            await send({
                "type": "capture", "state": "error",
                "detail": f"Audio capture failed: {exc}",
            })

    task = asyncio.create_task(_pump())
    capture_state["task"] = task
    await send({"type": "capture", "state": "started", "source": source})


async def _persist_live_qa(session_id, question: str, answer: str, *,
                           calibration: dict | None = None,
                           intent: str | None = None) -> None:
    """Persist one Q&A turn to a live session's history (user transcript +
    assistant answer), so it shows in the org sidebar like a chat. No-op if
    the session_id isn't a real persisted live session.

    `calibration` ({band, track?, target?}) and `intent` (the answer's header
    label) are both stored on the assistant message's `sources` JSONB so a
    reloaded transcript shows the same pill + intent header — `intent` goes in
    `sources`, not the String(50) column, so a long label is never truncated."""
    factory = get_session_factory()
    if factory is None:
        return
    try:
        sid_uuid = uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        return  # ad-hoc/typed session, not a persisted live session
    try:
        from storage.repos import MessageRepo, SessionRepo

        async with factory() as db:
            repo = SessionRepo(db)
            sess = await repo.get(sid_uuid)
            if sess is None or sess.type != "live":
                return
            mr = MessageRepo(db)
            await mr.append(session_id=sid_uuid, role="user", content=question)
            await repo.record_message(sid_uuid)
            _src: dict = {}
            if calibration:
                _src["calibration"] = calibration
            if intent:
                _src["intent"] = intent
            await mr.append(
                session_id=sid_uuid, role="assistant", content=answer,
                sources=(_src or None),
            )
            await repo.record_message(sid_uuid)
            await db.commit()
    except Exception:  # noqa: BLE001 — persistence must never break the live answer
        import logging
        logging.getLogger("zapthetrick.live").exception("live Q&A persist failed")


async def _persist_resumed_answer(session_id, question: str, partial: str,
                                  full_answer: str) -> None:
    """Persist a Continue/Retry result. If the truncated partial was already
    saved (the 60s-timeout path persists it), REPLACE that assistant message
    with the completed answer instead of appending a duplicate Q&A pair;
    otherwise append a fresh pair. Best-effort."""
    factory = get_session_factory()
    if factory is None:
        return
    try:
        sid_uuid = uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        return
    try:
        from sqlalchemy import select as _select

        from app.database import Message
        from storage.repos import SessionRepo

        async with factory() as db:
            sess = await SessionRepo(db).get(sid_uuid)
            if sess is None or sess.type != "live":
                return
            last = (
                await db.execute(
                    _select(Message)
                    .where(Message.session_id == sid_uuid,
                           Message.role == "assistant")
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )
            ).scalars().first()
            if (last is not None and (partial or "").strip()
                    and last.content.strip() == partial.strip()):
                last.content = full_answer
                last.incomplete = False
                await db.commit()
                return
    except Exception:  # noqa: BLE001 — fall through to a plain append
        pass
    await _persist_live_qa(session_id, question, full_answer)


async def _stop_capture(capture_state: dict) -> None:
    """Cancel the running server-side capture task, if any."""
    task = capture_state.get("task")
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    capture_state["task"] = None


def _split_fe_frame(raw: bytes) -> tuple[int, bytes]:
    """Split an FE-tagged audio frame into (role, pcm_bytes).

    The Flutter client prefixes each PCM frame with a single role byte when it
    captures audio locally and streams both sources over one socket:
    0x00 = interviewer (system loopback, answered), 0x01 = candidate (mic,
    absorbed). An empty frame yields (0, b"") so it degrades to the interviewer
    role rather than raising.
    """
    if not raw:
        return 0, b""
    return raw[0], raw[1:]


def _decode_pcm(raw: bytes) -> np.ndarray | None:
    """Convert int16 PCM bytes to float32 [-1, 1].

    The client is expected to send 16-bit little-endian PCM at cfg.audio.sample_rate.
    Web `MediaRecorder` output (opus/webm) is NOT supported on this path — the
    Flutter client must decode/resample to int16 before sending.
    """
    if not raw:
        return None
    try:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return None
        return arr
    except Exception:  # noqa: BLE001
        return None


def _stt_bias_prompt(session_id: str) -> str | None:
    """Build the Whisper biasing prompt for this session: the configured
    technical-vocabulary seed plus the most recent interviewer questions, so
    the recogniser adapts to the actual interview's terminology.

    Whisper caps the prompt at ~224 tokens; we keep it well under by using the
    seed plus the last few questions, newest last (most influential)."""
    from app.question_detection.context_tracker import get_tracker

    seed = (cfg.stt.prompt or "").strip()
    try:
        recent = get_tracker(session_id).recent_questions()[-3:]
    except Exception:  # noqa: BLE001
        recent = []
    if recent:
        seed = (seed + " Recent questions: " + " ".join(recent)).strip()
    return seed or None


async def _load_org_ctx(session_id: str) -> dict:
    """Load the organization intake captured at session creation (org name,
    job role, pasted job description, notes) from `Session.session_metadata`.
    Empty dict on any failure — org grounding simply degrades."""
    import uuid as _uuid
    from storage.models import Session as _SessionRow
    factory = get_session_factory()
    if factory is None:
        return {}
    async with factory() as db:
        row = await db.get(_SessionRow, _uuid.UUID(str(session_id)))
        meta = getattr(row, "session_metadata", None) if row else None
    if not isinstance(meta, dict):
        return {}
    return {
        "org_name": str(meta.get("org_name") or ""),
        "job_role": str(meta.get("job_role") or ""),
        "job_description": str(meta.get("job_description") or ""),
        "notes": str(meta.get("notes") or ""),
        # Manual seniority override from the interview setup dialog ("auto" or a
        # band slug); empty/"auto" → the calibration layer infers the band.
        "experience_level": str(meta.get("experience_level") or ""),
    }


async def _load_session_resume_id(session_id: str) -> tuple[bool, str | None]:
    """Resolve the resume linked to THIS live session from its own row.

    Returns (has_row, resume_id): `has_row` is False only when no Session
    row exists (a brand-new ad-hoc session), in which case the caller keeps
    the client-supplied query param. When the row exists its `resume_id` is
    authoritative — including NULL, which means "no resume for this session"
    and must override a stale global query-param resume."""
    import uuid as _uuid
    from storage.models import Session as _SessionRow
    factory = get_session_factory()
    if factory is None:
        return (False, None)
    try:
        key = _uuid.UUID(str(session_id))
    except (ValueError, AttributeError):
        return (False, None)
    async with factory() as db:
        row = await db.get(_SessionRow, key)
        if row is None:
            return (False, None)
        rid = getattr(row, "resume_id", None)
        return (True, str(rid) if rid is not None else None)


async def _load_profile(resume_id: str | None) -> dict | None:
    """Resolve the profile JSON for the resume scoped to this WS session."""
    if not resume_id:
        return None
    factory = get_session_factory()
    if factory is None:
        return None
    async with factory() as session:
        resume = await session.get(Resume, resume_id)
        if resume is None:
            return None
        # `profile` is a JSONB column — SQLAlchemy already gives us a dict.
        # Fall back to a raw-text summary if it's empty or not yet parsed.
        profile = resume.profile
        if isinstance(profile, dict) and profile:
            return profile
        if isinstance(profile, str) and profile.strip():
            try:
                return json.loads(profile)
            except json.JSONDecodeError:
                pass
        if resume.raw_text:
            return {"summary": resume.raw_text[:2000]}
        return None


@router.get("/api/live/replay/{sid}")
async def live_replay(sid: str, summary: bool = False) -> dict:
    """Read-only Session_Replay (live-conversational-intelligence R45) built from
    the in-process Event_Log for a session. Dev/debug only — no runtime effect on
    the live path. Gated by `live.session_replay` (fail-open → empty)."""
    from app.core.config_loader import cfg
    from app.live import replay as _replay

    try:
        if not getattr(cfg.live, "session_replay", False):
            return {"session_id": sid, "enabled": False, "count": 0, "steps": []}
        return _replay.summary(sid) if summary else _replay.build_replay(sid)
    except Exception as exc:  # noqa: BLE001
        return {"session_id": sid, "error": str(exc), "count": 0, "steps": []}


@router.get("/api/live/outcome/{sid}")
async def live_outcome(sid: str) -> dict:
    """Advisory Outcome_Estimate (live-conversational-intelligence R44) derived
    from the in-process Event_Log. Explicitly NOT a hiring decision. Dev/
    read-only; gated by `live.outcome_estimate` (fail-open → unknown)."""
    from app.core.config_loader import cfg
    from app.live import eventlog as _eventlog
    from app.live import outcome as _outcome

    try:
        if not getattr(cfg.live, "outcome_estimate", False):
            return {"session_id": sid, "enabled": False, **_outcome.OutcomeEstimate().to_dict()}
        events = _eventlog.get_log(sid).events()
        questions = sum(1 for e in events if e.get("type") == "event"
                        and (e.get("data") or {}).get("questions", 0))
        answers = [e for e in events if e.get("type") == "answer"]
        confs = [(e.get("data") or {}).get("confidence") for e in answers]
        confs = [c for c in confs if isinstance(c, (int, float))]
        avg_conf = (sum(confs) / len(confs)) if confs else None
        fb = [e for e in events if e.get("type") == "feedback"]
        pos = sum(1 for e in fb if (e.get("data") or {}).get("state") in ("satisfied", "positive"))
        satisfaction = (pos / len(fb)) if fb else None
        est = _outcome.estimate(
            answered=len(answers), total=max(questions, len(answers)),
            avg_confidence=avg_conf, satisfaction=satisfaction,
        )
        return {"session_id": sid, **est.to_dict()}
    except Exception as exc:  # noqa: BLE001
        return {"session_id": sid, "error": str(exc), **_outcome.OutcomeEstimate().to_dict()}


@router.get("/api/live/career/{sid}")
async def live_career(sid: str) -> dict:
    """Advisory Career_Intelligence (live-conversational-intelligence R61),
    standalone/read-only and DISABLED by default. Builds career-prep coaching
    from the candidate profile + fit + session replay. Explicitly NOT
    professional/legal/financial advice. Gated by `live.career_intelligence`."""
    from app.core.config_loader import cfg
    from app.live import career as _career

    try:
        if not getattr(cfg.live, "career_intelligence", False):
            return {"session_id": sid, "enabled": False, **_career.CareerIntelligence().to_dict()}
        from app.live import replay as _replay

        # Standalone/read-only: build from the session replay summary. Profile/
        # org enrichment is opt-in via the mock/profile path; absent → advisory
        # coaching from session activity only (fail-open).
        rs = _replay.summary(sid)
        ci = _career.analyze(None, None, fit=None, replay_summary=rs)
        return {"session_id": sid, **ci.to_dict()}
    except Exception as exc:  # noqa: BLE001
        return {"session_id": sid, "error": str(exc), **_career.CareerIntelligence().to_dict()}
