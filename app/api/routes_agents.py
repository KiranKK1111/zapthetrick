"""Multi-agent chat endpoint — drives [Supervisor] through SSE.

POST /api/agents/stream
  Runs the full agent mesh (Planner → Retriever → Persona → Grounder,
  with Memory/Critic/Suggester in parallel and Reflector after). Mirrors
  the SSE contract from /api/chat/stream so the Flutter client can use
  the same parser:

    event: meta     data: {"conversation_id": ..., "intent": {...}}
    event: tool     data: {"name": ..., "status": ...}      (repeated)
    event: token    data: {"text": "..."}                   (repeated)
    event: clarify  data: {"questions": [...], "confidence": .., "blocking": .., "reason": ".."}
    event: done     data: {"message_id": ..., "episode_id": ..., "latency_ms": ...}
    event: error    data: {"detail": "..."}

POST /api/agents/episodes/{episode_id}/feedback
  Attaches 👍/👎/edit signals to a past episode. Drives the
  Reflector / Critic learning loops downstream.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import (
    AgentRegistry,
    ClarifierAgent,
    CoderAgent,
    CriticAgent,
    GrounderAgent,
    MemoryAgent,
    PersonaAgent,
    PlannerAgent,
    ReflectorAgent,
    RetrieverAgent,
    SuggesterAgent,
    Supervisor,
    VisionAgent,
    WebAgent,
)
from app.core.config_loader import cfg
from app.database import Conversation, Message, get_session
from storage.db import get_session_factory
from app.memory import Episode, attach_feedback_db, record_episode
from app.schemas import AgentsStreamRequest, FeedbackRequest


import logging

router = APIRouter(prefix="/api/agents")
log = logging.getLogger(__name__)

# Injected into the answering model's prompt when the turn asks to zip /
# archive / download the project. The app packages the answer into a real,
# downloadable archive (a button appears below the message), so the model must
# explain + emit the files rather than refuse with "I can't send a zip".
_DOWNLOAD_DIRECTIVE = (
    "FILE DOWNLOAD CAPABILITY: This application automatically packages the "
    "project into a downloadable ZIP and shows a Download button directly "
    "below your message — you never paste or attach the archive yourself. So "
    "NEVER tell the user you are unable to create, send, or attach a ZIP/file.\n"
    "How to respond to a zip / archive / download request:\n"
    "- If the project's source files were ALREADY provided earlier in this "
    "conversation: do NOT repeat the code. Reply with just 1-3 sentences that "
    "confirm the project is packaged and tell the user to use the Download "
    "button below. Be brief.\n"
    "- ONLY if the project has NOT yet been produced in this conversation: give "
    "a short explanation, then include the complete source as fenced code "
    "blocks — one file per block, each starting with a filename comment (e.g. "
    "`// src/App.tsx`, `# app/main.py`) so it becomes a real file at that path "
    "in the archive.\n"
    "Keep the prose concise."
)

# Injected when the turn asks for a single downloadable DOCUMENT (pdf / word /
# excel / csv / markdown / text). The app renders the answer into that exact
# file format with a Download button + live preview below the message, so the
# model must produce the document CONTENT — never tell the user to convert it.
_DOC_FILE_DIRECTIVE = (
    "DOCUMENT DOWNLOAD CAPABILITY: This application automatically converts your "
    "answer into the requested downloadable file (PDF, Word, Excel, CSV, "
    "Markdown, or text) and shows a Download button with a live preview "
    "directly below your message. You do NOT attach or send the file yourself.\n"
    "- Write the document's CONTENT directly, as clean well-structured Markdown "
    "(headings, lists, GFM tables, fenced code blocks). For a spreadsheet "
    "(Excel/CSV) request, put the data in Markdown tables.\n"
    "- NEVER tell the user to copy the text, convert it themselves, or use an "
    "external tool (e.g. Pandoc, Typora, Obsidian, Word, an online converter). "
    "Do NOT append any 'to obtain a PDF / to create the file…' instructions — "
    "the app handles generation automatically.\n"
    "- Do not refuse or say you cannot produce a file. Just output the content."
)


def _blueprint_directive(bp) -> str:
    """Phase 2 planner → generation: when a NEW document is being AUTHORED for a
    recognized goal (design doc, proposal, research, …), tell the model to
    structure it along the planned section blueprint. Empty for a GENERAL goal
    (no imposed template) or when packaging existing content."""
    sections = getattr(bp, "sections", None)
    if not sections:
        return ""
    names = ", ".join(s.title for s in sections)
    required = ", ".join(s.title for s in sections if s.required)
    directive = (
        f"\nDOCUMENT STRUCTURE ({bp.goal.value} · {bp.depth.value}): organize the "
        f"content with clear ## Markdown headings in THIS order — {names}. Every "
        f"REQUIRED section ({required}) must be present; add the rest only when "
        f"the topic warrants it. Target roughly {bp.est_pages} page(s) of depth."
    )
    # Phase 8 (subset) — executive-summary-last: a summary written before the
    # body just guesses; drafting it AFTER makes it reflect the real content.
    if any("summary" in s.title.lower() or "abstract" in s.title.lower()
           for s in sections):
        directive += (" Draft the Executive Summary / Abstract LAST (after the "
                      "body) so it accurately reflects the finished content.")
    return directive


def _empty_target_directive(kind: str) -> str:
    """Directive that makes the model ASK a natural clarifying question for a
    deliverable request with no subject/target yet — the way Claude does it.

    It must NOT invent the deliverable, and must NOT emit a fixed canned
    sentence: a warm, brief, genuinely helpful question that offers a couple
    of concrete examples, phrased freshly each time (so regenerating gives a
    different, natural variant). `kind` tailors the examples."""
    if kind == "archive":
        what = ("an archive / ZIP of a project, but there's no project or "
                "code in this conversation yet to package")
        examples = ('"zip up a FastAPI backend with JWT auth" or "package my '
                    'React todo app"')
        ask = ("what you'd like the project to be (its purpose, language/"
               "framework, and any key pieces) — or paste/describe existing "
               "code you want archived")
    elif kind == "code":
        what = ("a source-code file, but you haven't said what it should do "
                "or in which language")
        examples = ('"a Python script that dedupes a CSV" or "a TypeScript '
                    'debounce hook"')
        ask = ("what the code should do and the language/framework")
    else:  # docs / generic document
        what = ("a document, but you haven't said what kind or what it should "
                "cover")
        examples = ('"a one-page resume for a backend engineer" or "a report '
                    'on my project\'s architecture"')
        ask = ("what type of document you want (report, letter, resume, "
               "spec, notes, …) and, in a sentence, what it should be about")
    return (
        "CLARIFY-FIRST (deliverable with no subject yet): The user asked for "
        f"{what}. Do NOT invent, fabricate, or produce any file, project, "
        "file tree, or document, and do NOT give generic how-to instructions "
        "for making one yourself. Instead, reply like a helpful colleague "
        "with ONE short, warm clarifying question: briefly acknowledge the "
        f"request, then ask {ask}. Offer a couple of concrete examples such "
        f"as {examples}. Keep it to 1-3 sentences, phrase it naturally in "
        "your own words (not a template), and once they tell you, you'll "
        "create it."
    )

# Injected when the turn is a project BUILD request. Models tend to show a full
# directory tree but then only write a few files ("…continue similarly…") and
# sometimes add files not in the tree — so the downloadable ZIP doesn't match
# the layout. This forces a complete, one-to-one, path-labelled output.
_BUILD_COMPLETE_DIRECTIVE = (
    "PROJECT GENERATION — COMPLETENESS IS MANDATORY: If you present a directory/"
    "file layout (a tree), you MUST then output the COMPLETE contents of EVERY "
    "file shown in that layout, each as its OWN fenced code block whose FIRST "
    "line is a filename comment with the exact path from the layout (e.g. "
    "`# app/routers/welcome.py`, `// src/App.tsx`). Rules:\n"
    "- Do NOT abbreviate, summarise, or use placeholders like "
    "\"...\", \"continue similarly\", \"rest of the files\", or \"same as "
    "above\". Write each file in full.\n"
    "- Do NOT include any file that isn't in the layout, and do NOT omit any "
    "file that is — the tree and the code blocks must match ONE-TO-ONE.\n"
    "- If the project is large, prefer FEWER files over incomplete ones, but "
    "whatever appears in the tree must be fully written.\n"
    "This is what makes the project downloadable as a correct ZIP."
)

# Strong refs to detached "save the partial on disconnect" tasks so the event
# loop doesn't garbage-collect them before they finish committing.
_BG_SAVES: set = set()

# An ARCHIVE deliverable is never re-emitted by the artifact re-delivery path
# below: the ZIP fast-path already rebuilds it from the conversation's code.
_ARCHIVE_FMTS = {"zip", "7z", "7zip", "tar", "tar.gz", "tgz", "rar"}


def _review_quality(text: str, blueprint=None) -> dict | None:
    """Deterministic quality review of a generated document (Phase 3), checked
    against the planner's Blueprint when there is one (Phase 2).

    The blueprint is what makes `review._check_completeness` live: it verifies
    the document actually contains the sections the planner intended, so a
    missing REQUIRED section is surfaced instead of shipping silently. None
    blueprint → the structural checks only. Fail-open → None (no quality block).
    """
    try:
        from app.documents.review import analyze_document
        return analyze_document(text, blueprint=blueprint).as_dict()
    except Exception:  # noqa: BLE001 — a review must never break a turn
        return None


def _reuses_existing_artifact(planner) -> bool:
    """True when the planner (app/documents/intent.py) classified this turn as a
    RE-DELIVERY of an artifact the conversation already produced — "where's the
    pdf?", "send me that file again". Its PlannerDecision says exactly that:
    DOWNLOAD_EXISTING with `reuse_response` and `requires_llm=False`.

    Such a turn must NOT run the model again — regenerating would author a NEW
    document the user already has. Archives are excluded (the ZIP fast-path owns
    them). Flag: `cfg.documents.reuse_existing_artifact` (default ON). Fail-open:
    any error → False → today's behavior.
    """
    try:
        from app.documents.intent import ArtifactIntent as _AI

        if not bool(getattr(cfg.documents, "reuse_existing_artifact", True)):
            return False
        return (
            planner is not None
            and planner.intent == _AI.DOWNLOAD_EXISTING
            and bool(planner.reuse_response)
            and not bool(planner.requires_llm)
            and (planner.artifact_type or "").lower() not in _ARCHIVE_FMTS
        )
    except Exception:  # noqa: BLE001
        return False


async def _existing_artifact(conversation_id, *, want_format: str | None = None,
                             fallback: dict | None = None) -> dict | None:
    """The conversation's most recent generated document — ``{content, format,
    title}`` — or None when there is nothing on file.

    Primary source is the versioned store (`app.documents.store.latest_for_session`).
    That store is best-effort (`record_generation` is fail-open), so when it has
    no row we fall back to the last artifact-flagged assistant message of the
    thread. NEVER invents an artifact: no content anywhere → None, and the caller
    then behaves exactly as it does today (generate). Fail-open on any error.
    """
    row = None
    try:
        from app.documents.store import latest_for_session

        _f = get_session_factory()
        if _f is not None:
            async with _f() as _s:
                row = await latest_for_session(_s, conversation_id)
    except Exception:  # noqa: BLE001 — the store is an enhancement, not a gate
        row = None

    content, fmt, title = "", "", ""
    if row is not None and (getattr(row, "content_md", "") or "").strip():
        content = row.content_md
        fmt = str(getattr(row, "doc_format", "") or "")
        title = str(getattr(row, "title", "") or "")
    elif isinstance(fallback, dict) and (fallback.get("content") or "").strip():
        content = str(fallback.get("content") or "")
        fmt = str(fallback.get("format") or "")
    if not (content or "").strip():
        return None
    # A format NAMED in this turn wins ("where's the word version?"); else the
    # format it was delivered in; else the app default.
    fmt = (want_format or fmt or "pdf").lower()
    if fmt in _ARCHIVE_FMTS:
        return None
    return {"content": content, "format": fmt, "title": title}


async def _redeliver_artifact(conversation_id, artifact: dict, planner, *,
                              doc_pending_sent: bool = False):
    """SSE frames that RE-DELIVER an existing artifact with no LLM call at all.

    Streams the stored document's source back (so the message is self-contained
    on reload — the export/preview routes render from the message content) and
    ends with the authoritative `done.document` the client draws the download
    card from. The persist is best-effort: a DB failure still delivers the card.
    """
    fmt = str(artifact.get("format") or "pdf")
    text = str(artifact.get("content") or "")
    if not doc_pending_sent:
        yield _sse("meta", {"doc_pending": fmt})
    doc = {
        "document": True, "format": fmt, "formats": [fmt],
        "artifact_intent": getattr(planner.intent, "value", str(planner.intent)),
        "reuse_response": True,
        # This turn re-emitted an EXISTING artifact — nothing was generated.
        "redelivered": True,
    }
    for i in range(0, len(text), 240):
        yield _sse("token", {"text": text[i:i + 240]})
    mid = None
    try:
        f = get_session_factory()
        if f is not None:
            async with f() as ws:
                msg = Message(
                    conversation_id=conversation_id, role="assistant",
                    content=text, intent="general", sources=doc,
                )
                ws.add(msg)
                crow = await ws.get(Conversation, conversation_id)
                if crow is not None:
                    crow.title = crow.title  # bump updated_at
                await ws.commit()
                await ws.refresh(msg)
                mid = msg.id
    except Exception as exc:  # noqa: BLE001 — the card must still be delivered
        log.warning("download-existing save failed (non-fatal): %s", exc)
    log.info("download-existing: re-emitted %s artifact (no LLM call)", fmt)
    yield _sse("done", {
        "message_id": mid, "episode_id": None, "latency_ms": 0,
        "document": doc,
    })


def _json_default(o):
    """JSON serializer for non-stdlib types that show up in SSE payloads.

    The supervisor's tool events carry blackboard slot values; those
    include `uuid.UUID` (Postgres ids), `datetime` (timestamps), and
    `set` (rare but possible). Without this hook every UUID in a frame
    would raise `TypeError: Object of type UUID is not JSON
    serializable` inside the generator — the server would silently
    close the SSE stream and the Flutter client would surface
    "Connection closed while receiving data" with no actionable hint.
    """
    if isinstance(o, uuid.UUID):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=_json_default)}\n\n"


async def _record_clarify_asked(conversation_id: str | None,
                                intent: str, confidence: float) -> None:
    """Best-effort: persist that we ASKED a clarification this turn so the next
    turn can resolve the outcome (advanced-intent-reasoning R1). Never raises."""
    try:
        from app.clarify import OutcomeStore
        from storage.device import ensure_device_user
        from storage.models import User

        factory = get_session_factory()
        if factory is None or not conversation_id:
            return
        uid = await ensure_device_user()
        if uid is None:
            return
        async with factory() as s:
            user = await s.get(User, uid)
            if user is None:
                return
            store = OutcomeStore(dict(user.preferences or {}))
            store.record_decision(conversation_id, intent or "unknown",
                                  confidence, asked=True)
            user.preferences = dict(store.root)
            await s.commit()
    except Exception:  # noqa: BLE001 — telemetry must never break a turn
        pass


async def _record_assumptions(conversation_id: str | None,
                              assumptions: list | None) -> None:
    """Best-effort: persist the assumptions an assume-mode answer just stated
    into the goal ledger (Phase-1 assumption persistence). The assumed slot is
    suppressed next turn like a confirmed one; the user's NEXT message settles
    it — objection clears, anything else promotes (silence = acceptance, which
    matches how the assumption was presented). Never raises.

    Self-contained load→record→save because the turn's single save_store runs
    BEFORE streaming — this event arrives mid-stream.
    """
    try:
        from app.core.config_loader import cfg as _cfg
        if not getattr(_cfg.decision_core, "persist_assumptions", True):
            return
        if not conversation_id or not assumptions:
            return
        from app.clarify import GoalLedger
        from storage.device import ensure_device_user
        from storage.models import User

        factory = get_session_factory()
        if factory is None:
            return
        uid = await ensure_device_user()
        if uid is None:
            return
        async with factory() as s:
            user = await s.get(User, uid)
            if user is None:
                return
            root = dict(user.preferences or {})
            GoalLedger(root, conversation_id).record_assumptions(
                [str(a) for a in assumptions if a])
            user.preferences = root
            await s.commit()
    except Exception:  # noqa: BLE001 — persistence must never break a turn
        pass


def _engine_last_model(session_key: str | None) -> str | None:
    """The model that actually produced this turn's answer (for the header)."""
    try:
        from app.llm.engine import get_last_model
        return get_last_model(session_key)
    except Exception:  # noqa: BLE001
        return None


def _quality_done_meta(deg_on: bool, deg_start: int, surface: bool, *,
                       critic_on: bool = False, answer: str = "",
                       asked_items: list | None = None,
                       decisions: dict | None = None) -> dict:
    """Optional additive `done`-frame fields for the reliability layer
    (evaluation-and-reliability R6.3/R7.3/R9.3). Surfaces only this turn's
    degradation events and a non-blocking critic report, and only when surfacing
    is enabled. Empty dict → legacy `done` shape unchanged."""
    if not surface:
        return {}
    out: dict = {}
    if deg_on:
        try:
            from app.quality.degrade import since
            events = since(deg_start)
            if events:
                out["degraded"] = [e.get("subsystem") for e in events]
        except Exception:  # noqa: BLE001
            pass
    if critic_on:
        try:
            from app.quality.critic import review
            rep = review(answer, asked_items=asked_items, decisions=decisions)
            # Only surface MATERIAL findings (R7.3); a clean/skipped report adds
            # nothing to the payload.
            if rep.has_findings:
                out["quality"] = rep.to_dict()
        except Exception:  # noqa: BLE001
            pass
    return out


async def _register_artifact(conversation_id: str | None, artifact_id: str,
                             title: str, kind: str) -> None:
    """Best-effort: mark the just-created artifact as the conversation's
    Current_Artifact in the ConversationState so a follow-up ("add X to it")
    targets it (workspace-and-artifacts R7). Shares the preferences root via
    load_store/save_store. Never raises."""
    try:
        from app.core.config_loader import cfg
        if not getattr(cfg.followup, "enabled", False) or not conversation_id:
            return
        from app.clarify import load_store, save_store
        from app.followup import ConversationState
        from app.artifacts.bridge import register_current_artifact
        from storage.device import ensure_device_user

        uid = await ensure_device_user()
        factory = get_session_factory()
        if uid is None or factory is None:
            return
        async with factory() as s:
            store, user = await load_store(s, uid, conversation_id=conversation_id)
            if store is None:
                return
            cstate = ConversationState(store.root, conversation_id)

            class _A:
                pass
            a = _A()
            a.id, a.title, a.kind = artifact_id, title, kind
            register_current_artifact(cstate, a)
            await save_store(s, user, store)
    except Exception:  # noqa: BLE001
        pass


async def _memory_commit(conversation_id: str | None) -> None:
    """Best-effort (memory-graph R5/R7): promote the conversation's durable
    decisions / preferences / confirmed tech into typed Memory_Objects, embed
    them, add to the store, run lifecycle maintenance, and persist. Background-
    only; never blocks or raises (R8.3)."""
    try:
        from app.core.config_loader import cfg
        if not getattr(cfg.memory, "graph_enabled", False) or not conversation_id:
            return
        from app.clarify import load_store, save_store
        from app.followup import ConversationState
        from app.memory.mstore import memory_store
        from app.memory.objects import (
            MemoryObject, SCOPE_GLOBAL, workspace_scope)
        from app.memory import lifecycle as _life
        from storage.device import ensure_device_user

        uid = await ensure_device_user()
        factory = get_session_factory()
        if uid is None or factory is None:
            return

        def _embed(text: str):
            try:
                from app.rag.embedder import embed_one
                return embed_one(text)
            except Exception:  # noqa: BLE001
                return None

        async with factory() as s:
            store, user = await load_store(s, uid, conversation_id=conversation_id)
            if store is None:
                return
            # §17/G5: scope new memory objects to the PROJECT (workspace) when the
            # conversation is in one, so the structured memory graph accretes
            # across the project's chats; else global (today's behavior).
            _obj_scope = SCOPE_GLOBAL
            with contextlib.suppress(Exception):
                import uuid as _uuidm
                from storage.models import Session as _SessionRow
                _srow = await s.get(_SessionRow,
                                    _uuidm.UUID(str(conversation_id)))
                if _srow is not None and getattr(_srow, "project_id", None):
                    _obj_scope = workspace_scope(str(_srow.project_id))
            cstate = ConversationState(store.root, conversation_id)
            mem = memory_store()
            mem.load_from(store.root)            # hydrate from prior persistence

            # Promote durable per-conversation items into typed memory objects.
            items: list[tuple[str, str]] = []
            for k, v in (cstate.decisions() or {}).items():
                items.append(("decision", f"{k}: {v}"))
            for k, v in (cstate.preferences() or {}).items():
                items.append(("preference", f"{k}: {v}"))
            for k, v in (cstate.confirmed_slots() or {}).items():
                if v:
                    items.append(("entity", f"{k}: {v}"))

            existing = {o.content for o in mem.all()}
            for kind, content in items:
                if content in existing:
                    continue
                obj = MemoryObject(content=content, kind=kind,
                                   scope=_obj_scope, importance=0.7,
                                   durable=True)
                obj.embedding = _embed(content)
                mem.add(obj)

            _life.maintain(mem)
            mem.export_to(store.root)
            await save_store(s, user, store)
    except Exception:  # noqa: BLE001 — memory must never break a turn
        pass


async def _followup_commit(conversation_id: str | None, user_text: str,
                           answer: str) -> None:
    """Best-effort: after the answer streamed, register entities + enumerated
    options from it into the ConversationState so the NEXT turn's selection
    references / pronouns resolve (followup-context-engine R7/R10). Shares the
    same preferences root via load_store/save_store. Never raises."""
    try:
        from app.core.config_loader import cfg
        if not getattr(cfg.followup, "enabled", False) or not conversation_id:
            return
        from app.clarify import load_store, save_store
        from app.followup import ConversationState
        from app.followup import update as _fu_update
        from storage.device import ensure_device_user

        uid = await ensure_device_user()
        factory = get_session_factory()
        if uid is None or factory is None:
            return
        async with factory() as s:
            store, user = await load_store(s, uid, conversation_id=conversation_id)
            if store is None:
                return
            cstate = ConversationState(store.root, conversation_id)
            _fu_update.commit(user_text, answer, cstate)
            await save_store(s, user, store)
    except Exception:  # noqa: BLE001 — telemetry/state must never break a turn
        pass



def _build_registry() -> AgentRegistry:
    """Build the agent set from `cfg.agents.enabled`.

    Disabled agents are skipped — the scheduler / supervisor just
    ignores their slot subscribers. This is how the user toggles
    individual agents off in settings without a restart.
    """
    enabled = cfg.agents.enabled
    registry = AgentRegistry()
    if enabled.planner:
        registry.register(PlannerAgent())
    if enabled.clarifier:
        registry.register(ClarifierAgent())
    if enabled.retriever:
        registry.register(RetrieverAgent())
    if enabled.memory:
        registry.register(MemoryAgent())
    if enabled.persona:
        registry.register(PersonaAgent())
    if enabled.coder:
        registry.register(CoderAgent())
    if enabled.vision:
        registry.register(VisionAgent())
    if enabled.web:
        registry.register(WebAgent())
    if enabled.grounder:
        registry.register(GrounderAgent())
    if enabled.critic:
        registry.register(CriticAgent())
    if enabled.suggester:
        registry.register(SuggesterAgent())
    if enabled.reflector:
        registry.register(ReflectorAgent())
    return registry


@router.post("/stream")
async def agents_stream(
    body: AgentsStreamRequest,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Drive the full agent mesh for one user message."""
    # Perceived-speed (R1.3/R1.4): consume any prefetch warmed while the user
    # was typing. Reuse on a matching submit, discard otherwise. Best-effort —
    # the real connection reuse happens transparently via the shared pool.
    try:
        from app.perceived.prefetch import manager as _prefetch
        if body.prefetch_token:
            _prefetch.reuse(body.prefetch_token, body.message)
    except Exception:  # noqa: BLE001 — never let prefetch bookkeeping break a turn
        pass
    # Hard-stop if the database isn't ready — without it we can't
    # create the session/message rows, and writing to a stalled
    # connection is what the user was seeing as "Connection closed
    # while receiving data" (asyncpg blocking inside the SSE
    # generator). Returning a clean 503 + JSON message is what the
    # Flutter ApiException renderer expects.
    from storage import bootstrap as _bs

    if not _bs.POSTGRES_READY:
        state = _bs.MIGRATION_STATE
        err = _bs.MIGRATION_ERROR
        msg = {
            "idle": "Database not configured. Open Settings -> Database.",
            "migrating": "Database is migrating. Try again in a few seconds.",
            "error": f"Database setup failed: {err or 'see backend logs'}.",
        }.get(state, "Database not ready.")
        raise HTTPException(status_code=503, detail=msg)

    # Resolve / create the conversation row up front so the meta event
    # can include its id, same as /api/chat/stream.
    if body.conversation_id:
        convo = await session.get(Conversation, body.conversation_id)
        if convo is None:
            raise HTTPException(404, detail="Conversation not found")
    else:
        title = " ".join(body.message[:500].split()[:6])[:200] or "New conversation"
        convo = Conversation(title=title)
        session.add(convo)
        await session.flush()

    user_msg = Message(
        conversation_id=convo.id, role="user", content=body.message
    )
    session.add(user_msg)

    # The full body is stored above (nothing lost); everything the model/mesh
    # sees uses a condensed copy so a huge typed/pasted body can't blow the
    # context window. Threaded — the salience pass is CPU work that can stall
    # on a giant paste. (Large pastes normally arrive as file attachments via
    # the upload route, but a big body can still land here.)
    import asyncio as _asyncio

    from app.chat.condense import condense_oversized

    user_text = (
        await _asyncio.to_thread(condense_oversized, body.message)
    )[0]

    # Pull prior messages, then WINDOW them to a token budget so a very long
    # thread doesn't resend its whole history (and blow the context window)
    # every turn. The dropped older turns are covered by the session's rolling
    # summary (built in the background — see app/chat/history.py).
    from app.chat.history import window_messages

    history_result = await session.execute(
        select(Message)
        .where(Message.conversation_id == convo.id)
        .order_by(Message.created_at)
    )
    all_prior = []
    # Phase 0 (artifact taxonomy): did an earlier assistant turn already produce
    # a downloadable artifact? Enables UPDATE_EXISTING / DOWNLOAD_EXISTING intents
    # (the `sources` dict is dropped from `prior` below, so capture it here).
    # `_prior_artifact` keeps the LATEST one's content+format as the fallback for
    # a DOWNLOAD_EXISTING re-delivery when the versioned store has no row (its
    # persistence is best-effort).
    _has_prior_artifact = False
    _prior_artifact: dict | None = None
    for m in history_result.scalars().all():
        # Exclude the user message we just added — Persona adds it back.
        if m.id == user_msg.id:
            continue
        c = m.content
        analysis = None
        if isinstance(getattr(m, "sources", None), dict):
            analysis = m.sources.get("image_analysis")
            if m.role == "assistant" and m.sources.get("document"):
                _has_prior_artifact = True
                _prior_artifact = {
                    "content": m.content or "",
                    "format": str(m.sources.get("format") or ""),
                }
        # Prior image turns persist their vision analysis in `sources` (not in
        # `content`), so a text-only follow-up still carries the screenshot's
        # problem — e.g. after we asked which language to solve it in.
        if m.role == "user" and analysis:
            c = f"{c}\n\n[Attached image content]:\n{analysis}"
        all_prior.append({"role": m.role, "content": c})
    # Threaded: window_messages condenses each message (CPU work that could
    # stall the loop if an old turn stored a huge body).
    prior, dropped = await _asyncio.to_thread(window_messages, all_prior)
    # Capture the summary BEFORE commit (commit expires ORM attributes, and a
    # lazy reload inside the async generator would error).
    history_summary = (convo.summary or "").strip() if dropped > 0 else ""
    await session.commit()

    # `convo.id` is a UUID under the new schema. Stringify it so every
    # SSE frame stays JSON-serializable and the Flutter client gets a
    # plain string in `conversation_id`.
    conversation_id = str(convo.id)
    session_id = body.session_id or f"sess-{uuid.uuid4().hex[:12]}"

    # Explicit Stop signal (the FE's HTTP client can't abort a request in-flight,
    # so it POSTs /conversations/{id}/cancel). Clear any STALE flag from a prior
    # turn so this one isn't killed at birth; the stream loop polls it.
    from app.api.replay import (
        is_cancelled as _is_cancelled, clear_cancel as _clear_cancel)
    _clear_cancel(conversation_id)

    def _stream_cancelled() -> bool:
        return _is_cancelled(conversation_id)

    # Flag the assistant turn as a document ONLY when the user explicitly asked
    # to generate one — in THIS message OR the immediately preceding
    # clarification exchange (the "Format: PDF" / "Title: …" answers that
    # precede generation). Persisted in `sources.document`; drives the UI's
    # document card / preview panel. Plain answers/summaries stay unflagged.
    # LLM-driven (no keyword rules), run CONCURRENTLY with the stream so it adds
    # no latency before the first token — awaited only at save time, by which
    # point it's done. Considers the recent window so a "Format: PDF" follow-up
    # to a generate request is still recognised.
    # Single-call turn triage: difficulty + document-intent in ONE LLM round-
    # trip (replaces two separate classifier calls). Considers the recent window
    # so a "Format: PDF" follow-up — or a difficulty-inheriting follow-up — is
    # recognised. Started CONCURRENTLY; difficulty is awaited before streaming,
    # the doc flag only at save time.
    from app.chat.triage import triage as _triage

    # A file/document/archive request is judged strictly PER-TURN. The ONLY
    # time a prior turn's request carries forward is when THIS turn is a
    # clarification answer (the user answered a clarify card) — otherwise
    # "give me a .py file" (an earlier turn) would keep forcing files onto
    # every later program request. `skip_clarify` is the client's explicit
    # "this is a clarification answer" flag.
    _skip_clarify = bool(getattr(body, "skip_clarify", False))
    _allow_recent_doc = _skip_clarify

    # Is there anything in THIS chat to archive/deliver as a file? An archive
    # or download is only meaningful when code/content already exists (prior
    # turns, or pasted in this message). On a first prompt "get me the archive"
    # there's nothing to package — so we must NOT flag a ZIP document (no
    # download card) and NOT inject the "emit the files" directive; the model
    # just answers ("which project?").
    _has_prior_code = any("```" in (m.get("content") or "") for m in prior)
    _has_prior_content = any(
        m.get("role") == "assistant" and (m.get("content") or "").strip()
        for m in prior)
    _archivable = _has_prior_code or ("```" in (user_text or ""))

    # Phase 0 — Artifact Intent Taxonomy (deterministic planner decision). This
    # CLASSIFIES the turn's desired outcome (CHAT / ANSWER_AND_ARTIFACT /
    # ARTIFACT_ONLY / UPDATE_EXISTING / DOWNLOAD_EXISTING); it is threaded onto
    # the doc metadata for observability + as the foundation for reuse-without-
    # regeneration (Phase 5). It does NOT decide WHETHER a document generates —
    # triage remains the single source of truth for that (overriding it caused
    # false docs). Fail-open to CHAT.
    try:
        from app.documents.intent import classify_artifact_intent as _classify_ai
        _planner = _classify_ai(
            user_text, has_prior_artifact=_has_prior_artifact,
            has_prior_content=_has_prior_content)
    except Exception:  # noqa: BLE001
        from app.documents.intent import ArtifactIntent, PlannerDecision
        _planner = PlannerDecision(ArtifactIntent.CHAT)

    # Phase 2 — plan the document's structure (goal → section blueprint) when a
    # NEW document is being AUTHORED (ANSWER_AND_ARTIFACT). The directive built
    # from it steers the model to well-structured sections. Empty for a general
    # goal or when packaging existing content. Fail-open.
    # The Blueprint itself is kept (not just its directive): it is the PLAN the
    # document is checked against — `review.analyze_document(..., blueprint=…)`
    # verifies the generated document actually contains the sections we planned
    # (`_check_completeness`). Without threading it through, that check was dead.
    _bp_directive = ""
    _blueprint = None
    try:
        from app.documents.intent import ArtifactIntent as _AI
        if _planner.intent == _AI.ANSWER_AND_ARTIFACT:
            from app.documents.planner import plan_document as _plan_doc
            _blueprint = _plan_doc(user_text)
            _bp_directive = _blueprint_directive(_blueprint)
    except Exception:  # noqa: BLE001
        _bp_directive = ""
        _blueprint = None

    # Phase 7 — audience persona: "write this for my manager / the CTO / the
    # client" shapes tone + depth. Appended to the doc directive; empty for a
    # general audience. Fail-open.
    _persona_directive = ""
    try:
        from app.documents.profiles import persona_directive as _persona
        _persona_directive = _persona(user_text)
    except Exception:  # noqa: BLE001
        _persona_directive = ""

    _recent_user = [m["content"] for m in prior if m.get("role") == "user"][-2:]
    _recent_user.append(user_text)
    # A clarification-answer turn where the user picked a delivery format from a
    # chip ("Format: Word (.docx)", "PDF", "a zip"). explicit_doc_request misses
    # a bare format answer, so parse it directly. A chosen DOCUMENT format (docx/
    # pdf/xlsx/…) is authoritative — it must win over the archive/zip packaging
    # path even when there's prior code in the chat (the Word→ZIP bug).
    from app.documents.detect import format_answer as _fmt_answer
    _clar_fmt = _fmt_answer(user_text) if _allow_recent_doc else None
    _clar_arch = {"zip", "7z"}
    _triage_task = _asyncio.create_task(
        _triage(user_text, recent=" ".join(_recent_user[:-1]),
                allow_recent_doc=_allow_recent_doc)
    )

    async def _doc_sources(text: str):
        from app.documents.detect import explicit_doc_formats as _doc_fmts
        from app.documents.detect import infer_code_ext as _infer_code

        # SINGLE SOURCE OF TRUTH: the turn triage decides whether a document is
        # produced (explicit-only + artifact-verified — see app/chat/triage.py).
        # We no longer re-decide here or apply a length gate: an explicit request
        # generates regardless of answer length (a code file can be short), and
        # a non-explicit turn never generates. This removes the old split-brain
        # where a regex here could disagree with the classifier (docs appearing
        # unrequested, or explicit requests silently dropped).
        tri = await _triage_task
        # Triage is the SINGLE SOURCE OF TRUTH for whether a document is
        # produced. On a clarification answer it already inherits the original
        # request's intent via the recent window (allow_recent_doc → triage runs
        # explicit_doc_request over the prior turn), so a genuine "put this in a
        # document" → "Format: Word" flow returns wants_document=True here without
        # any override. We do NOT force generation off the chosen format alone —
        # that would produce a file whenever a clarification answer merely
        # mentions a format word (e.g. a topic answered "just the PDF part"). If
        # triage says no document, there was no document request.
        if not tri.wants_document:
            return None
        # Prefer the explicit multi-format list ("a txt AND a pdf" → both) from
        # THIS turn; the recent window is consulted only on a clarification
        # answer (per-turn intent otherwise); else the single triage format.
        # A clarification answer that named a DOCUMENT format is authoritative:
        # the user was asked "which format?" and answered — honor it verbatim
        # rather than re-deriving (which drops "Format: Word (.docx)" to pdf/zip).
        if _clar_fmt and _clar_fmt not in _clar_arch:
            _fmts = [_clar_fmt]
        else:
            _fmts = (_doc_fmts(user_text)
                     or (_doc_fmts(" ".join(_recent_user)) if _allow_recent_doc
                         else None)
                     or [tri.doc_format])
        # A generic "code file" (language unspecified) → resolve to the actual
        # language extension from the answer's first fenced block.
        _fmts = [(_infer_code(text) if x == "code" else x) for x in _fmts]
        # An export/archive/document of "the project" with NOTHING in this
        # chat to package (no prior code/content, none pasted) must not
        # produce ANY file — whatever format triage guessed, the document
        # would be pure invention (the "Project Export" hallucination).
        if _suppress_empty_target:
            return None
        # An ARCHIVE deliverable (zip/7z) needs something to archive. On a first
        # prompt with nothing built yet, suppress the ZIP card entirely — the
        # answer is conversational, not a package.
        _arch_fmts = {"zip", "7z", "7zip", "tar", "tar.gz", "tgz", "rar"}
        if _fmts and _fmts[0].lower() in _arch_fmts and not _archivable:
            return None
        # Phase 0: tag the artifact with its classified intent + whether it
        # reuses the prior answer (no new reasoning). Persisted in `sources` and
        # echoed in the `done.document` event — observability now, the hook for
        # reuse-without-regeneration + existing-artifact ops in later phases.
        _out = {
            "document": True, "format": _fmts[0], "formats": _fmts,
            "artifact_intent": _planner.intent.value,
            "reuse_response": _planner.reuse_response,
        }
        # Phase 3: run the deterministic quality review at GENERATION time (not
        # only on download) so the answer's structural health rides the
        # `done.document` event. The planner's Blueprint (Phase 2 — present only
        # when a NEW document was AUTHORED) is passed in, so completeness is
        # checked against the sections we actually PLANNED: a missing required
        # section is surfaced as a `missing_section` issue instead of shipping
        # silently. None elsewhere → the structural checks only, as today.
        # NON-blocking — the inline path never refuses a doc. Fail-open.
        _q = _review_quality(text, _blueprint)
        if _q:
            _out["quality"] = _q
        # Phase 6: goal-completion detector — surface concrete missing
        # deliverables (tests / Dockerfile / CI / Swagger …) for a project-shaped
        # answer. Self-suppresses on a no-code chat (empty suggestions), so it
        # rides the event only when it has something useful to add. Fail-open.
        try:
            from app.documents.completion import completion_report
            _cmpl = completion_report(text)
            if _cmpl.get("suggestions"):
                _out["completion"] = _cmpl
        except Exception:  # noqa: BLE001
            pass
        return _out

    # Download/ZIP intent: a fast, zero-latency regex gate (separate from the
    # LLM doc-intent classifier used for FLAGGING). When the user asks to zip /
    # archive / download the project or its files, we inject a directive so the
    # answering model EXPLAINS the project and emits its files instead of
    # refusing with "I can't send a zip". The app packages the answer into a
    # real downloadable archive with a button below the message.
    # Per-turn: download/archive intent is read from THIS message only, unless
    # this turn is a clarification answer (then the prior request carries).
    _dl_text = (" ".join(_recent_user) if _allow_recent_doc else user_text).lower()
    _wants_download = (
        bool(re.search(r"\b(zip|\.zip|archive|compressed?)\b", _dl_text))
        or (
            "download" in _dl_text
            and re.search(
                r"\b(project|projects|code|codebase|app|application|source|"
                r"files?|everything|repo|repository)\b",
                _dl_text,
            )
            is not None
        )
    ) and _archivable  # only when there's actually something to package
    # Which SUPPORTED archive format did the user name? Only zip / 7z can be
    # created; a request naming tar/rar/etc. (or none) leaves this None so the
    # Clarifier asks which of the two to use.
    if re.search(r"\b(7z|7-?zip|sevenz)\b", _dl_text):
        _archive_fmt: str | None = "7z"
    elif re.search(r"\b(zip|\.zip)\b", _dl_text):
        _archive_fmt = "zip"
    else:
        _archive_fmt = None
    _archive_fmt_named = _archive_fmt is not None
    # (_skip_clarify is computed earlier, before the triage task.)

    # A single-document request (pdf/word/excel/csv/md/txt — NOT a zip): detect
    # it deterministically and synchronously (regex, no latency) so we can tell
    # the answering model to emit the document content directly, with no
    # "convert it yourself" boilerplate. Mirrors the doc flag the UI uses.
    from app.documents.detect import explicit_doc_request as _explicit_doc

    _det_doc, _det_fmt = _explicit_doc(user_text)
    if not _det_doc:
        _det_doc, _det_fmt = _explicit_doc(_dl_text)
    _wants_doc_file = _det_doc and _det_fmt not in (None, "zip")
    # Did the user actually NAME a format ("as a PDF", "word doc"), or did we
    # default to PDF for a generic "give me this in a document"? Used so the
    # pre-stream progress label stays honest ("Generating document…") instead of
    # presuming a format the user never asked for. The delivered file still
    # defaults to PDF (with the download-as menu for other formats).
    try:
        from app.documents.detect import mentions_format as _mentions_fmt
        # A concrete format token must actually appear ("word", "pdf", "xlsx").
        # explicit_doc_formats can't be used here: it defaults to pdf for a
        # generic "put this in a document", which would wrongly read as "named".
        _fmt_named = bool(_mentions_fmt(user_text)
                          or (_allow_recent_doc and _mentions_fmt(_dl_text)))
    except Exception:  # noqa: BLE001
        _fmt_named = True  # be safe: don't hide a named format on error

    # Apply the clarification-answer format (chip the user tapped when asked
    # "which format?"). This is the fix for the Word→ZIP bug: a bare answer like
    # "Format: Word (.docx)" is invisible to explicit_doc_request, so without
    # this the archive path (prior code ⇒ _archivable) would silently win and
    # deliver a zip. A chosen DOCUMENT format is authoritative and overrides the
    # packaging path; a chosen ARCHIVE format (zip/7z) resolves the archive.
    if _clar_fmt:
        _fmt_named = True
        if _clar_fmt in _clar_arch:
            _archive_fmt = _clar_fmt
            _archive_fmt_named = True
            _wants_download = _wants_download or _archivable
            _wants_doc_file = False
        else:
            _det_doc = True
            _det_fmt = _clar_fmt
            _wants_doc_file = True
            _wants_download = False  # a document is not a zip — never package it

    # Ambiguous build/project request → ask FIRST (Claude-style). When the user
    # asks to BUILD something but hasn't named a language/framework anywhere in
    # the recent window, we tell the Supervisor to BLOCK on the Clarifier before
    # answering, so it asks "which language / framework?" instead of silently
    # picking one. The Clarifier still makes the final call (it declines if the
    # request is already specific or the choice was made earlier).
    from app.chat.difficulty import is_ambiguous_build_request

    _clarify_priority = is_ambiguous_build_request(
        user_text, " ".join(_recent_user)
    )

    # A REQUIRED missing choice (e.g. "write a program …" with NO language) must
    # be ASKED FIRST, never raced — otherwise the fast first token wins the race
    # and we silently pick a default (Python). Deterministic + cheap (no LLM):
    # the same pre-gate the Clarifier uses. If a language IS named, this is
    # False and the turn answers normally. Skipped when the user is answering a
    # prior clarify (skip_clarify) so we don't loop.
    # "Operate on existing content" intents (archive / "document this" / "give
    # me the file") only make sense when there IS prior content in THIS chat.
    # On a first prompt ("get me the archive of this project" with nothing
    # built yet), asking "which archive format?" is nonsensical — let the model
    # answer naturally. (`_has_prior_code`/`_has_prior_content` computed above.)
    _clarify_required = False
    # Suppress ALL clarification when the turn wants to operate on existing
    # content (archive / "document this") but this chat has none yet — the
    # model should just answer ("there's nothing to archive yet"), not ask.
    _suppress_empty_target = False
    # "archive" | "docs" | "code" — tailors the clarifying-question directive.
    _empty_target_kind = ""
    # Captured so the unified TurnState (Phase 3 #1/#2) can be BUILT from — and
    # CONSUMED by — this route, not just written to the blackboard and dropped.
    _turn_assessment = None
    if not getattr(body, "skip_clarify", False):
        try:
            from app.clarify.intent_pipeline import (
                CLARIFY, INTENT_ARCHIVE, INTENT_CODE_GEN, INTENT_DOCS,
                assess as _assess, detect_intent as _detect_intent)
            # `assess` expects `recent` as a STRING (it does `recent.strip()`).
            # Passing the `_recent_user` LIST threw AttributeError, which this
            # try/except swallowed — so empty-target detection silently never
            # fired on a first prompt (a non-empty 1-element list), and a
            # clarifying-question turn still generated a stray PDF. Join to a
            # string; the prior turns are the context (exclude the current).
            _a = _assess(user_text, " ".join(_recent_user[:-1]), {})
            _turn_assessment = _a
            _intent = _detect_intent(user_text)
            # `assess` is the oracle for "is this specified enough to act on?":
            # a deliverable ask it marks CLARIFY has no real subject yet
            # ("can you get me a document", "give me a source-code file").
            _underspecified = _a.decision == CLARIFY
            # `_archivable` (not just prior code): code pasted in THIS message
            # is a real target — "zip this: ```…```" must still package.
            if _intent == INTENT_ARCHIVE and not _archivable:
                _suppress_empty_target = True
                _empty_target_kind = "archive"
            elif (_intent == INTENT_DOCS and not _has_prior_content
                    and _underspecified):
                # A vague document ask with nothing in the chat → clarify like
                # Claude ("what kind, and about what?"), not a fixed string.
                _suppress_empty_target = True
                _empty_target_kind = "docs"
            elif (_intent == INTENT_CODE_GEN and not _archivable
                    and not _has_prior_content and _underspecified
                    # Only when the SUBJECT itself is missing ("give me a
                    # program", "can you write some code") — a concrete ask
                    # that's merely missing its LANGUAGE ("program for finding
                    # nth max from an array") must keep the structured
                    # language card; this suppression used to swallow it and
                    # the answer silently defaulted to Python.
                    and any(k in ("task_details", "subject", "artifact")
                            for k in (_a.missing_required or []))):
                # "give me a source-code file" / "can you give me a program"
                # with no subject → ask what it should do.
                _suppress_empty_target = True
                _empty_target_kind = "code"
            _req = list(_a.missing_required or [])
            if not _has_prior_code and "archive_format" in _req:
                _req.remove("archive_format")
            _clarify_required = _a.decision == CLARIFY and bool(_req)
        except Exception:  # noqa: BLE001 — never block a turn on the pre-gate
            _clarify_required = False
    # The empty-target clarifying question is driven by the answer model's own
    # directive (below), so the structured Clarifier card + the required-slot
    # pre-gate must stand down — otherwise both fire.
    if _suppress_empty_target:
        _clarify_required = False
        _clarify_priority = False
    # An empty-session export/document turn must not emit a file directive:
    # the model would fabricate document content for a project that doesn't
    # exist (the "Project Export" hallucination).
    if _suppress_empty_target:
        _wants_doc_file = False

    # A project build → demand COMPLETE, layout-consistent file output so the
    # downloadable ZIP matches the directory tree the model shows.
    from app.chat.difficulty import is_build_request as _is_build_req_fn

    _is_build_request = _is_build_req_fn(user_text)

    # Explicit performance/complexity constraints in a code request ("within
    # 500ms", "worst case", "O(1) space") become a hard requirement directive.
    from app.chat.perf_constraints import extract_performance_constraints

    _perf_directive = extract_performance_constraints(user_text)


    # Capability-aware routing: difficulty comes from the triage call above
    # (context-aware), so the Persona routes hard/expert work to the strongest
    # model + adds a rigor directive.
    from app.chat.difficulty import HARD, STANDARD, TRIVIAL, is_level
    _difficulty = STANDARD
    try:
        from app.core.config_loader import cfg as _cfgd
        if _cfgd.advanced_rag.difficulty_aware_routing:
            _difficulty = (await _triage_task).difficulty
    except Exception:  # noqa: BLE001
        _difficulty = STANDARD

    # A project BUILD is a substantial code-generation task — route it to the
    # LARGE models (the strong tier the router rotates across), not the fast
    # small model standard routing would pick. Heavy "whole project" asks are
    # already 'expert'; bump the rest up to 'hard'.
    if _is_build_request and _difficulty in (TRIVIAL, STANDARD):
        _difficulty = HARD

    # Unified semantic Understanding pass (the 'brain'): ONE embedding → intent,
    # difficulty, task category, implicit topic-shift, capabilities, output
    # complexity. Gated (`understanding.enabled`); its embedding is remembered
    # per-conversation so the NEXT turn can detect a topic-shift by distance.
    _understanding = None
    try:
        from app import understanding as _u
        if _u.enabled():
            # Bounded: the embedder runs off-loop with a hard 2s SLA so a slow
            # (cold-loading) model can never stall the P0 slot.
            _understanding = await asyncio.wait_for(
                asyncio.to_thread(
                    _u.understand,
                    user_text,
                    prev_embedding=_u.last_embedding(conversation_id),
                    has_image=False,
                ),
                timeout=2.0,
            )
            _u.remember_embedding(conversation_id, _understanding.embedding)
            # G4: on a low-confidence intent, ask the model to disambiguate
            # instead of trusting the keyword net. Gray-zone turns only.
            try:
                from app.understanding.disambiguate import (
                    disambiguate_intent as _disamb, enabled as _disamb_on)
                _thr = float(getattr(cfg.semantic_intent,
                                     "primary_threshold", 0.5))
                if (_disamb_on()
                        and _understanding.intent_confidence < _thr):
                    _better = await _disamb(user_text)
                    if _better:
                        _understanding.intent = _better
                        _understanding.source = "llm_disambiguated"
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 — the brain must never block a turn
        _understanding = None

    # Manual UI override (HIGHEST precedence): an explicit, valid difficulty from
    # the client ("think harder" / "keep it light") wins over both the classifier
    # and the build-bump. Invalid/absent → keep the derived value (fail-open).
    _ui_difficulty = getattr(body, "difficulty", None)
    if _ui_difficulty and is_level(_ui_difficulty):
        _difficulty = _ui_difficulty.strip().lower()
    elif (_understanding is not None and not _is_build_request
          and _understanding.difficulty_confidence >= 0.5):
        # Brain's difficulty when it's confident and no UI/build override applies.
        _difficulty = _understanding.difficulty

    extras_base: dict = {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "prior_messages": prior,
        # Rolling summary of older (windowed-out) turns; Persona folds it into
        # the system prompt. Empty unless the thread is long enough to trim.
        "history_summary": history_summary,
        # trivial|standard|hard|expert — drives capability-aware model routing.
        "difficulty": _difficulty,
        # Non-empty when the turn asks to zip/archive/download the project —
        # tells the answering model to explain + emit files, not refuse. For a
        # single-document request (pdf/word/excel/…) the doc directive tells it
        # to emit the content directly (no "convert it yourself" note). An
        # export/archive/doc/code turn in an EMPTY session instead gets the
        # CLARIFY-FIRST directive: ask a natural, varied clarifying question
        # (Claude-style) — never invent a deliverable, never a fixed string.
        "download_directive": _empty_target_directive(_empty_target_kind)
        if _suppress_empty_target
        else _DOWNLOAD_DIRECTIVE
        if _wants_download
        else (_DOC_FILE_DIRECTIVE + _bp_directive + _persona_directive
              if _wants_doc_file else ""),
        # Non-empty on a project build — forces COMPLETE, layout-consistent file
        # output so the directory tree and the downloadable ZIP match. Never
        # on an empty-target turn (that must ASK, not emit files).
        "build_directive": (_BUILD_COMPLETE_DIRECTIVE
                            if _is_build_request and not _suppress_empty_target
                            else ""),
        # Explicit performance/complexity requirements ("within 500ms",
        # "worst-case O(n)", "constant space") the solution MUST satisfy.
        "perf_directive": _perf_directive,
        # A clarifying question is generated hotter so each regenerate offers a
        # genuinely different, natural phrasing (Claude-style variety). None
        # everywhere else → the model's default temperature.
        "answer_temperature": 0.7 if _suppress_empty_target else None,
        # True for an ambiguous build request missing a language/framework —
        # the Supervisor blocks on the Clarifier so it asks before answering.
        "clarify_priority": _clarify_priority,
        # True when the request has a REQUIRED missing choice (e.g. a code
        # request with no language) — the Supervisor BLOCKS on the Clarifier
        # (ask first) instead of racing, so a fast first token can't answer in
        # a defaulted language before the "which language?" card appears.
        "clarify_required": _clarify_required,
        # Whether THIS chat already has content to archive / document / turn
        # into a file. The Clarifier uses these to NOT ask "which archive
        # format?" / "which doc format?" on a first prompt when there is
        # nothing to act on yet.
        "has_prior_code": _has_prior_code,
        "has_prior_content": _has_prior_content,
        # A zip/archive/download request packages the EXISTING project — never
        # clarify on it (the user wants the deliverable, not more questions).
        # Also suppressed when the client says this turn IS a clarification
        # answer (so the Clarifier can't ask the same thing again in a loop).
        # A download/compress turn that NAMES a format is suppressed (just make
        # it); one with NO format is NOT suppressed, so the Clarifier can ask
        # which archive format to use first.
        "suppress_clarify": (_wants_download and _archive_fmt_named)
        or _skip_clarify or _suppress_empty_target,
        # db_session is injected *inside* event_generator from a fresh
        # session — FastAPI closes the request-scoped one as soon as
        # the route returns the StreamingResponse, which is before the
        # generator runs.
    }
    if body.resume_id:
        extras_base["resume_id"] = body.resume_id

    # §Understanding: hand the brain's read to the mesh + the model router. The
    # persona forwards task_category/capabilities into the router's `options`;
    # the compact meta rides the envelope trace.
    if _understanding is not None:
        extras_base["understanding"] = _understanding.as_meta()
        extras_base["task_category"] = _understanding.task_category
        extras_base["needs_tool"] = "tools" in _understanding.capabilities
        extras_base["needs_json"] = "json" in _understanding.capabilities
        extras_base["topic_shift"] = _understanding.topic_shift
        # The query embedding keys Phase-2 semantic routing (in-process only).
        extras_base["understanding_embedding"] = _understanding.embedding

    # The instruction actually sent to the answer mesh. Defaults to the user's
    # turn; the follow-up engine may replace it with a confident, self-contained
    # rewrite (followup-context-engine R5). The STORED user message + episode
    # always keep the original text.
    model_text = user_text
    _fu_meta: dict = {}
    _quality_meta: dict = {}
    _route_meta: dict = {}
    _policy_meta: dict = {}
    _usermodel_meta: dict = {}

    # Clarification preference memory (Phase 4): inject the device user's known
    # choices (this conversation + durable cross-session) so the gate skips
    # already-decided choices, and record this turn's answers when it IS a
    # clarification answer (skip_clarify). Best-effort — never break a turn.
    try:
        from app.clarify import load_store, save_store
        from app.clarify import OutcomeStore, parse_answer_lines
        from storage.device import ensure_device_user

        _clar_uid = await ensure_device_user()
        # §17/§18: the device user id, for project-scoped memory + account-level
        # export/delete of the turn's episode.
        if _clar_uid:
            extras_base["user_id"] = str(_clar_uid)
        _clar_store, _clar_user = await load_store(
            session, _clar_uid, conversation_id=conversation_id)
        # §17: the user's own standing instructions ride into the persona prompt
        # (below the safety boundary). Empty → nothing changes. Fail-open.
        with contextlib.suppress(Exception):
            from app.personalization.instructions import load_custom_instructions
            _ci = load_custom_instructions(
                getattr(_clar_user, "preferences", None))
            if _ci:
                extras_base["custom_instructions"] = _ci
        # §17 Projects: a conversation in a project gains its project-level
        # instructions + a project-scoped KG. Ungrouped → nothing changes.
        with contextlib.suppress(Exception):
            from app.personalization.projects import load_project_context
            _proj_ctx = await load_project_context(session, convo)
            if _proj_ctx.get("project_id"):
                extras_base["project_id"] = _proj_ctx["project_id"]
            if _proj_ctx.get("instructions"):
                extras_base["project_instructions"] = _proj_ctx["instructions"]
        if _clar_store is not None:
            extras_base["clarify_prefs"] = _clar_store.known_choices()
            # Outcome telemetry shares the SAME preferences root (one save
            # persists both). Expose calibration buckets to the gate (R2) and
            # resolve any clarification the user is now responding to (R1).
            _outcomes = OutcomeStore(_clar_store.root)
            extras_base["clarify_calibration"] = _outcomes.calibration_buckets()
            _skip = bool(getattr(body, "skip_clarify", False))
            if _skip and parse_answer_lines(user_text):
                _outcomes.record_response(conversation_id, "answered")
            elif _outcomes.has_pending(conversation_id):
                # A new, unrelated turn after an asked clarification → the user
                # bypassed it (answered directly / moved on).
                _outcomes.record_response(conversation_id, "overridden")
            else:
                _outcomes.decay_recent()
            # Fatigue + trust counters drive the clarifier's adaptive answer
            # band (R3/R4): recent volume + skip/answer history → ask less.
            _ctr = _outcomes.counters()
            extras_base["clarify_fatigue"] = {
                "recent": int(_ctr.get("recent", 0)),
                "skips": int(_ctr.get("skipped", 0)) + int(_ctr.get("overridden", 0)),
                "answers": int(_ctr.get("answered", 0)),
            }
            # Goal ledger (R6): accumulate confirmed slots so they are never
            # re-asked, and classify the conversation state to set the base
            # answer band (more permissive in discovery, stricter in execution).
            from app.clarify import GoalLedger, classify_state, threshold_for
            _recent_join = " ".join(_recent_user[:-1])
            _ledger = GoalLedger(_clar_store.root, conversation_id)
            _ledger.observe(user_text, _recent_join)
            _merged_prefs = dict(extras_base.get("clarify_prefs") or {})
            _merged_prefs.update(_ledger.confirmed_slots())
            extras_base["clarify_prefs"] = _merged_prefs
            _state = classify_state(_recent_join, user_text)
            extras_base["clarify_state"] = _state
            extras_base["clarify_answer_band"] = threshold_for(_state)
            # Follow-up / conversation-state engine (followup-context-engine
            # R1/R2/R3/R5/R6). Flag-gated; extends the SAME ledger record
            # (shared root) so this turn's single save_store persists it.
            # Deterministic + fail-open: any error → today's continuity prompt.
            if getattr(cfg.followup, "enabled", False):
                try:
                    from app.followup import ConversationState
                    from app.followup import acts as _fu_acts
                    from app.followup import reference as _fu_ref
                    from app.followup import rewrite as _fu_rewrite
                    from app.followup import update as _fu_update

                    _cstate = ConversationState(_clar_store.root, conversation_id)
                    # Classify the act + resolve references BEFORE observe() so
                    # selection refs resolve against the PRIOR enumerations.
                    _act, _fu_conf = _fu_acts.classify(user_text, _cstate)
                    _shift = _fu_acts.is_topic_shift(user_text)
                    _res = None
                    # EXPLICIT topic shift: abandon the prior thread's context so
                    # the new subject answers cleanly. Reset the topic-scoped
                    # state, seed it from THIS turn, and skip reference-rewrite /
                    # continuation directives (which would answer for the old
                    # topic). Standing slot preferences are kept (see reset_topic).
                    if _shift:
                        _act = _fu_acts.NEW_TOPIC
                        _cstate.reset_topic()
                        _cstate.observe(user_text, _recent_join)
                    else:
                        _res = _fu_ref.resolve(user_text, _cstate)
                        # State updates (corrections / reversals / negative
                        # constraints) for this turn.
                        _fu_update.apply_turn(user_text, _act, _res, _cstate)
                        # Record the turn's slots/entities/goal.
                        _cstate.observe(user_text, _recent_join)

                    # Build the model-facing instruction. A low-confidence
                    # reference is NOT rewritten — it falls through to the
                    # existing clarifier gate (R3.3, no second LLM call). Skipped
                    # entirely on a topic shift (there is no prior ref to resolve).
                    if (not _shift
                            and not getattr(_res, "needs_clarification", False)):
                        _rw, _rw_conf = _fu_rewrite.rewrite(
                            user_text, _act, _res, _cstate)
                        if _act == _fu_acts.CONTINUATION:
                            _dir = _fu_update.continuation_directive(_cstate)
                            model_text = f"{_dir}\n\n{_rw or user_text}"
                        elif _act == _fu_acts.APPROVAL:
                            model_text = (
                                "(The user approves the previous proposal.)\n"
                                + _fu_update.continuation_directive(_cstate))
                        elif _rw and _rw != user_text:
                            model_text = _rw

                    _fu_summary = _cstate.summary()
                    if _fu_summary:
                        extras_base["followup_state_summary"] = _fu_summary
                    # Additive interpretation meta for FE surfacing (Phase 4).
                    _fu_meta = {
                        "act": _act,
                        "followup_confidence": round(float(_fu_conf), 3),
                    }
                    if model_text != user_text:
                        _fu_meta["resolved_prompt"] = model_text
                except Exception as _fexc:  # noqa: BLE001
                    log.info("followup engine skipped: %s", _fexc)
                    model_text = user_text
            # Resolve the active clarification mode (Phase 5): a per-turn
            # override wins, else the stored mode; a "decide automatically"
            # collaboration contract behaves like Autopilot (R19).
            _eff_mode = (getattr(body, "clarify_mode", None)
                         or _clar_store.mode())
            _autonomy = (_clar_store.contract().get("autonomy") or "").lower()
            if not _eff_mode and _autonomy in ("auto", "autopilot", "automatic"):
                _eff_mode = "autopilot"
            if _eff_mode:
                extras_base["clarify_mode"] = _eff_mode
            if getattr(body, "clarify_mode", None):
                _clar_store.set_mode(body.clarify_mode)
            if bool(getattr(body, "skip_clarify", False)):
                _answers = parse_answer_lines(user_text)
                if _answers:
                    _clar_store.record_answers(_answers)
                    extras_base["clarify_prefs"] = _clar_store.known_choices()
            # One save persists prefs + outcomes (shared root).
            await save_store(session, _clar_user, _clar_store)
    except Exception as _exc:  # noqa: BLE001
        log.info("clarify preference memory skipped: %s", _exc)

    # ── Unified world state (Phase 3 #1/#2) — BUILD + CONSUME ────────────────
    # TurnState is projected from the clarifier's assessment (goal/intent/
    # decision/constraints/horizon/capabilities). The audit's gap was that it was
    # PRODUCED (written to the blackboard) but read by NO runtime consumer. Here
    # the chat route actually consumes it: (a) its answer_directive shapes the
    # model prompt so the reply honors the turn's extracted output constraints +
    # planning horizon, (b) it rides `extras` so every mesh agent reads ONE
    # picture, and (c) it is surfaced + constraint-checked against the final
    # answer. Additive + flag-gated + fail-open.
    _turn_state = None
    _turn_state_dict: dict = {}
    _interaction_meta: dict = {}
    if getattr(cfg.decision_core, "turn_state_enabled", True):
        try:
            from app.core.world_state import TurnState
            _turn_state = TurnState.from_assessment(
                _turn_assessment if _turn_assessment is not None else object(),
                goal=user_text)
            _turn_state_dict = _turn_state.as_dict()
            # (b) one shared picture for the mesh + trace.
            extras_base["turn_state"] = _turn_state_dict
            # (a) genuine answer shaping: honor the turn's constraints/horizon.
            if getattr(cfg.decision_core, "turn_state_directive", True):
                _ts_dir = _turn_state.answer_directive()
                if _ts_dir:
                    model_text = f"{_ts_dir}\n\n{model_text}"
        except Exception as _tsexc:  # noqa: BLE001
            log.info("turn state skipped: %s", _tsexc)

    # Human-interaction model + density (Phase 3 #12): SELECT the interaction
    # move (ask/proceed/summarize) + presentation shape (prose/table/steps/
    # diagram) and, when a denser shape fits, nudge the answer's FORMAT to it.
    # Deterministic + fail-open; surfaced as additive `interaction` meta.
    if getattr(cfg.understanding, "interaction_engine", True):
        try:
            from app.understanding import interaction as _ix
            _missing = list(getattr(_turn_assessment, "missing_required", []) or []) \
                if _turn_assessment is not None else []
            _horizon = (_turn_state_dict.get("horizon")
                        if _turn_state_dict else None)
            _plan = _ix.select(user_text, missing_required=_missing,
                               horizon=_horizon)
            _interaction_meta = _plan.as_dict()
            _shape_dir = _plan.shape_directive()
            # Never override an explicit deliverable/build turn's own directives.
            if (_shape_dir and not _wants_download and not _wants_doc_file
                    and not _is_build_request and not _suppress_empty_target):
                model_text = f"{model_text}\n\n{_shape_dir}"
        except Exception as _ixexc:  # noqa: BLE001
            log.info("interaction engine skipped: %s", _ixexc)

    # Aggregate confidence (evaluation-and-reliability R4). Flag-gated +
    # fail-open + additive: blends the available subsystem signals (intent
    # pre-gate + routing difficulty + any follow-up reference resolution) into
    # one band. A low band defers to the EXISTING clarifier — no new ask path,
    # no second LLM call. Surfaced only as optional meta.
    if getattr(cfg.quality, "aggregate_confidence", False):
        try:
            from app.quality import confidence as _qc
            _signals = [_qc.from_routing(_difficulty)]
            # Fold the intent pre-gate confidence (real signal now that the
            # TurnState/assessment is captured on the chat path).
            if _turn_assessment is not None:
                _signals.append(_qc.from_assessment(_turn_assessment))
            _r = locals().get("_res")
            if _r is not None:
                _signals.append(_qc.from_resolution(_r))
            _agg = _qc.aggregate(_signals)
            _quality_meta = {
                "band": _agg.band,
                "score": _agg.score,
                "decision": _qc.gate(_agg),
            }
        except Exception as _qexc:  # noqa: BLE001
            log.info("quality confidence skipped: %s", _qexc)

    # Request governor (evaluation-and-reliability R5). Flag-gated + fail-open:
    # selects a fast vs deep pipeline from the existing difficulty. Surfaced as
    # additive meta + an extras hint; the mesh's own capability routing +
    # deadlines remain authoritative (full stage-skip is consumed downstream).
    if getattr(cfg.quality, "governor", False):
        try:
            from app.quality.governor import select_pipeline, Budgets
            _pipe = select_pipeline(_difficulty, Budgets())
            extras_base["pipeline"] = _pipe.kind
            _quality_meta = {**_quality_meta, "pipeline": _pipe.kind}
        except Exception as _gexc:  # noqa: BLE001
            log.info("governor skipped: %s", _gexc)

    # Meta-router (intelligent-model-routing R7): unify task classification +
    # strategy into one decision for explainability. Flag-gated + fail-open; the
    # actual model+key pick still delegates to route_request inside the engine.
    if getattr(cfg.routing, "meta_router", False):
        try:
            from app.llm.meta_router import decide as _mdecide
            _rd = _mdecide(
                {"text": user_text, "difficulty": _difficulty},
                enabled=True,
                escalation_enabled=bool(getattr(cfg.routing, "escalation", False)),
                multi_model_enabled=bool(getattr(cfg.routing, "multi_model", False)),
            )
            _route_meta = {
                "category": _rd.task_category,
                "strategy": _rd.strategy,
                "difficulty": _rd.difficulty,
            }
        except Exception as _mexc:  # noqa: BLE001
            log.info("meta-router skipped: %s", _mexc)

    # Personalization + governance (personalization-and-governance R2/R4).
    # Flag-gated + fail-open + additive; safety guards already ran upstream and
    # keep precedence — the policy gate only ADDS caution. No second LLM call.
    if getattr(cfg.personalization, "topic_policy", False):
        try:
            from app.personalization.policy import classify as _risk_classify
            from app.personalization.policy import strategy_for as _risk_strategy
            _risk = _risk_classify(user_text)
            if _risk != "general":
                _strat = _risk_strategy(_risk)
                if _strat.directive:
                    # Prepend the additive caution directive to the model-facing
                    # instruction (never weakens the existing guards).
                    model_text = f"{_strat.directive}\n\n{model_text}"
                    _policy_meta = {"risk": _risk}
        except Exception as _pexc:  # noqa: BLE001
            log.info("topic policy skipped: %s", _pexc)
    if getattr(cfg.personalization, "user_model_enabled", False):
        try:
            from app.personalization.user_model import infer as _um_infer
            from app.personalization.adapt import adapt_signals as _um_adapt
            _um_signals = {
                "recent_user_texts": list(_recent_user),
                "depth_pref": getattr(body, "depth", None),
            }
            _um = _um_infer(_um_signals)
            _um_sig = _um_adapt(_um)
            if _um_sig:
                extras_base["user_model"] = _um_sig
                _usermodel_meta = {k: _um_sig[k] for k in
                                   ("expertise", "preferred_depth", "comm_style")
                                   if k in _um_sig}
        except Exception as _uexc:  # noqa: BLE001
            log.info("user model skipped: %s", _uexc)

    # Memory-graph (memory-graph R3): inject relevance-ranked, scope-isolated
    # memories into the prompt context. Flag-gated + fail-open; ranked set is
    # appended to the rolling summary (which Persona already folds into the
    # system prompt) — no second LLM call. Surfaced as additive `memory` meta.
    _memory_meta: dict = {}
    _mems: list = []   # recalled memory items — reused for memory_graph suggestions
    if (getattr(cfg.memory, "graph_enabled", False)
            and getattr(cfg.memory, "inject_into_context", False)):
        try:
            from app.memory.mstore import memory_store
            from app.memory.retriever import relevant
            # G5: retrieve from the project's workspace scope (+ global) when the
            # conversation belongs to a project; else global only, as today.
            _mems = relevant(user_text, memory_store(),
                             extras_base.get("project_id"))
            if _mems:
                _mem_lines = "\n".join(f"- {m.content}" for m in _mems[:6])
                _base = (extras_base.get("history_summary") or "").strip()
                # §11 trust boundary: recalled memory is UNTRUSTED (a poisoned
                # memory could carry an injection) — frame it as data.
                from app.response_arch.trust import frame_untrusted
                extras_base["history_summary"] = (
                    (_base + "\n\n" if _base else "")
                    + frame_untrusted(_mem_lines, label="recalled memory"))
                _memory_meta = {"recalled": len(_mems)}
        except Exception as _memexc:  # noqa: BLE001
            log.info("memory retrieval skipped: %s", _memexc)

    # Cross-session "open threads" from prior EPISODES (Architecture §3.2/§6):
    # related prior questions the user could resume → memory_graph suggestions.
    # Recalled here (clean request session, before the mesh's work_session).
    _episode_threads: list = []
    if getattr(cfg.memory, "episodic_enabled", False):
        try:
            from app.memory.episodic import search_episodes_similar
            _eps = await search_episodes_similar(session, user_text, top_k=4)
            _episode_threads = [
                {"question": getattr(e, "question", ""),
                 "intent": getattr(e, "intent", None)}
                for e in (_eps or [])
                if (getattr(e, "question", "") or "").strip()
            ]
        except Exception as _epexc:  # noqa: BLE001
            log.info("episode recall skipped: %s", _epexc)

    registry = _build_registry()
    supervisor = Supervisor(
        registry,
        latency_budget_ms=cfg.agents.deadlines_ms.total,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # First frame so the client can lock in the conversation id.
        log.info("agents-stream (TEXT path, no image upload): msg=%r",
                 (user_text or "")[:60])
        yield _sse(
            "meta",
            {"conversation_id": conversation_id, "session_id": session_id},
        )
        # Follow-up interpretation (R11): additive + optional. Only when the
        # engine produced meta AND surfacing is enabled. Legacy clients ignore
        # the extra `interpretation` event entirely (Property 11).
        if _fu_meta and getattr(cfg.followup, "surface_interpretation", False):
            yield _sse("interpretation", _fu_meta)
        # Aggregate confidence (R4.5): additive + optional; legacy clients ignore.
        if _quality_meta and getattr(cfg.quality, "surface_meta", False):
            yield _sse("aggregate_confidence", _quality_meta)
        # Routing explainability (intelligent-model-routing R10): additive `route`
        # meta when route_trace is on. Legacy clients ignore the extra event.
        if _route_meta and getattr(cfg.routing, "route_trace", False):
            yield _sse("route", _route_meta)
        # Unified world state (Phase 3 #1) + interaction plan (Phase 3 #12):
        # additive + optional; legacy clients ignore. Off by default.
        if _turn_state_dict and getattr(cfg.decision_core,
                                        "surface_turn_state", False):
            yield _sse("turn_state", _turn_state_dict)
        if _interaction_meta and getattr(cfg.understanding,
                                         "surface_interaction", False):
            yield _sse("interaction", _interaction_meta)
        # Phase 6 — response orchestrator: a formal plan (upcoming sections +
        # stream mode) emitted BEFORE the first token (first-meaningful-paint),
        # then per-token logical-block + progressive-artifact frames, then
        # end-of-stream analytics. Entirely additive + fail-open: any failure
        # sets _orch = None and the stream behaves exactly as before.
        _orch = None
        try:
            if getattr(cfg.response_arch, "enabled", True):
                from app.response_arch.orchestrator import ResponseOrchestrator
                _orch = ResponseOrchestrator(user_text)
                _ev, _data = _orch.plan_frame()
                yield _sse(_ev, _data)
        except Exception:  # noqa: BLE001
            _orch = None
        collected: list[str] = []
        # Optional token coalescing (R16): off by default (threshold 0). When
        # the client signals rendering load, scale the chunk size up (R46
        # adaptive chunk sizing) so a struggling client gets fewer, larger frames.
        from app.chat.stream_coalesce import TokenCoalescer, effective_threshold
        _base_chunk = int(getattr(cfg.advanced_rag, "stream_chunk_chars", 0) or 0)
        _coalescer = TokenCoalescer(
            effective_threshold(_base_chunk, getattr(body, "client_load", None)))
        intent_payload: dict = {}
        tools_called: list[str] = []
        latency_ms = 0

        # ── Perceived-speed wiring (all flag-gated; OFF = unchanged) ─────────
        import time as _time
        _t0 = _time.monotonic()
        # Degradation snapshot (evaluation-and-reliability R6): capture the
        # event sequence so the done frame can surface only THIS turn's events.
        _deg_on = bool(getattr(cfg.quality, "degradation", False))
        _deg0 = 0
        _artifact_meta: dict = {}
        if _deg_on:
            with contextlib.suppress(Exception):
                from app.quality.degrade import snapshot as _deg_snap
                _deg0 = _deg_snap()
        # Latency observatory (R16): read-only per-stage TTFT telemetry.
        _obs_on = bool(getattr(cfg.perceived, "observatory_enabled", False))
        _obs_rid = uuid.uuid4().hex
        if _obs_on:
            with contextlib.suppress(Exception):
                from app.perceived.observatory import observatory as _obs
                _obs.record(_obs_rid, "submit", 0.0)
        # TTFT acknowledgment (R7.3): emit an immediate "ack" frame when the
        # first token is slower than the budget; 0 disables.
        _ack_budget = float(getattr(cfg.perceived, "ttft_ack_threshold_s", 0.0) or 0.0)
        _ack_sent = False
        _first_token_seen = False
        # Answer reuse cache (R14/R21): only a plain Q&A turn is cacheable —
        # never a build/doc/zip/clarify turn (side-effecting / context-specific).
        _answer_cache_on = bool(getattr(cfg.perceived, "answer_cache", False))
        _cacheable_turn = (
            _answer_cache_on
            and not _wants_download and not _wants_doc_file
            and not _is_build_request and not _clarify_priority
            and not _skip_clarify
            # A clarifying question should read fresh on each regenerate —
            # never serve the same cached phrasing back (Claude-style variety).
            and not _suppress_empty_target
        )
        _cache_scope = ""
        if _cacheable_turn:
            with contextlib.suppress(Exception):
                from storage.device import ensure_device_user as _edu
                _uid = await _edu()
                _cache_scope = str(_uid) if _uid else ""

        # One fresh session lives for the entire stream. It's the
        # db_session every P0/P1 agent reads from `extras` and the
        # session we use for the final persist + episode write.
        #
        # `get_session_factory()` is late-bound: migrations run as a
        # background task, so a module-level `from … import SessionFactory`
        # would have captured the pre-bootstrap `None` and held onto
        # it forever (the bug behind "Database not bootstrapped").
        factory = get_session_factory()
        if factory is None:
            yield _sse(
                "error",
                {
                    "detail": (
                        "Database not ready — migrations are still running "
                        "or Postgres is unreachable. Check Settings -> Database."
                    )
                },
            )
            return

        # Best-effort save of (possibly partial) assistant text in a FRESH
        # session — safe to call even while this request is being torn down
        # (client disconnect / Stop), unlike `work_session` which is mid-exit.
        async def _save_assistant(text: str, *, incomplete: bool) -> None:
            f = get_session_factory()
            if f is None or not text.strip():
                return
            async with f() as ws:
                msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=text,
                    intent=intent_payload.get("type") or "general",
                    incomplete=incomplete,
                    sources=await _doc_sources(text),
                )
                ws.add(msg)
                convo_row = await ws.get(Conversation, conversation_id)
                if convo_row is not None:
                    convo_row.title = convo_row.title  # bump updated_at
                await ws.commit()

        assistant_saved = False
        stream_completed = False
        # §15: set when a mid-stream error/exhaustion cuts the turn short but we
        # already showed a partial answer — persist it as `incomplete` + offer
        # Continue/Retry instead of replacing it with a raw provider error.
        _incomplete_reason: str | None = None
        assistant_msg = None
        episode_id = None
        # The authoritative document decision for this turn (from triage via
        # _doc_sources). Computed once at save time, reused in the `done` event
        # so the client uses the SAME decision it will see on reload — no
        # client-side re-guessing with its own regex.
        _doc_meta = None
        # The unified response.v1 envelope, built + persisted at save time and
        # reused in the `done` event (live == reload).
        _envelope = None
        # Follow-up suggestions captured from the mesh's clarify event, so they
        # become first-class `suggestions[]` in the envelope (Architecture §5/§6).
        _captured_suggestions: list = []

        # NOTE: an empty-target deliverable request (archive/doc/code with no
        # subject or content yet) is NOT short-circuited here. It streams
        # through the normal answer path with the CLARIFY-FIRST directive
        # (extras.download_directive) so the model asks a natural, VARIED
        # clarifying question — exactly like Claude, and different on each
        # regenerate — instead of a robotic fixed string. File generation is
        # already suppressed for it (`_wants_doc_file`/download off), and the
        # turn is marked non-cacheable below so regenerations don't repeat.

        # ── ZIP fast-path ────────────────────────────────────────────────
        # A zip/download request for an ALREADY-built project: skip the LLM
        # answer entirely (no slow self-refine, no flashing the whole codebase
        # again). Show progressive steps + a real project brief (name, what it
        # is, how to run it), flag the turn as a ZIP document, and let the
        # client build the .zip from the conversation's code + a Download card.
        _has_prior_code = any("```" in (m.get("content") or "") for m in prior)
        if _wants_download and _has_prior_code and (
                _archive_fmt_named or _skip_clarify):
            _dl_fmt = _archive_fmt or "zip"   # default to zip if unspecified
            # Visible, ordered progress steps while we package.
            for _step in ("Collecting project files",
                          "Building directory structure",
                          f"Creating {_dl_fmt.upper()} archive"):
                yield _sse("stage", {"name": _step})
                await asyncio.sleep(0.5)
            # Compose an accurate brief from the project already in the thread.
            _proj = "\n\n".join(
                (m.get("content") or "") for m in prior
                if m.get("role") == "assistant" and "```" in (m.get("content") or "")
            )
            from app.chat.project_brief import build_brief
            _name, confirmation = build_brief(_proj)
            yield _sse("token", {"text": confirmation})
            _zip_mid = None
            try:
                async with factory() as ws:
                    _m = Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=confirmation,
                        intent="general",
                        sources={"document": True, "format": _dl_fmt,
                                 "project_name": _name},
                    )
                    ws.add(_m)
                    _crow = await ws.get(Conversation, conversation_id)
                    if _crow is not None:
                        _crow.title = _crow.title  # bump updated_at
                    await ws.commit()
                    await ws.refresh(_m)
                    _zip_mid = _m.id
            except Exception as exc:  # noqa: BLE001
                log.warning("zip fast-path save failed: %s", exc)
            yield _sse("done", {
                "message_id": _zip_mid, "episode_id": None, "latency_ms": 0,
                "project_name": _name,
                # The FE shows the download card ONLY when `done.document` is
                # present — without this the live turn rendered plain text and
                # the archive only appeared after a reload.
                "document": {"document": True, "format": _dl_fmt,
                             "formats": [_dl_fmt]},
            })
            return

        # ── Progressive deliverable signal ───────────────────────────────
        # The user explicitly asked for a downloadable file/archive: tell the
        # client NOW (before any tokens) so the bubble shows "Generating
        # <format>…" instead of raw streaming markdown, backend-driven rather
        # than the FE's regex guess. The terminal `done.document` decision
        # still gates actual delivery.
        _doc_pending_sent = False
        if _wants_doc_file or (_wants_download and _archivable):
            _pending_fmt = (_det_fmt if _wants_doc_file
                            else (_archive_fmt or "zip"))
            if _pending_fmt:
                _doc_pending_sent = True
                # Honest label: when the user asked for "a document" but named no
                # format (we defaulted to PDF), show the generic "Generating
                # document…" instead of presuming "PDF". A named format (or an
                # archive) shows its real type.
                _label = ("document"
                          if (_wants_doc_file and not _fmt_named)
                          else str(_pending_fmt))
                yield _sse("meta", {"doc_pending": _label})

        # ── DOWNLOAD_EXISTING fast-path ──────────────────────────────────
        # "where's the pdf?" / "send me that file again". The planner (Phase 0)
        # classified this turn as a RE-DELIVERY of an artifact this conversation
        # already produced (reuse_response, requires_llm=False) — so skip the LLM
        # entirely and re-emit the STORED document. Regenerating here would
        # author a NEW document the user already has.
        # Guards: never on an empty-target turn (that must stay a guidance
        # answer — no hallucinated artifact), never for archives (the ZIP
        # fast-path above owns those), and never when there is nothing on file:
        # `_existing_artifact` returns None and we fall straight through to the
        # normal answer path, exactly as today.
        if _reuses_existing_artifact(_planner) and not _suppress_empty_target:
            _prev_art = None
            with contextlib.suppress(Exception):
                _prev_art = await _existing_artifact(
                    conversation_id, want_format=_planner.artifact_type,
                    fallback=_prior_artifact)
            if _prev_art:
                async for _frame in _redeliver_artifact(
                        conversation_id, _prev_art, _planner,
                        doc_pending_sent=_doc_pending_sent):
                    yield _frame
                return

        # ── Answer reuse cache (R14/R21) — serve before generating ──────────
        # A previously-generated, revalidated answer for the SAME prompt (per-
        # user scope) is streamed instantly with no model call. Flag-gated and
        # restricted to plain Q&A turns; an error-marked cache entry is rejected
        # by `validate` so a fresh answer is produced (R14.3/R14.4).
        if _cacheable_turn:
            _cached = None
            with contextlib.suppress(Exception):
                from app.perceived.cache import answer_cache as _acache
                # Semantic tier (latency batch 2026-07-11 #3): when the
                # shared embedder is warm, a PARAPHRASED repeat of a cached
                # prompt also serves instantly — scope-isolated, cosine ≥
                # cfg.perceived.cache_similarity_threshold.
                _embed_fn = None
                with contextlib.suppress(Exception):
                    from app.rag import embedder as _semb
                    if _semb.is_ready():
                        _embed_fn = lambda t: _semb.embed([t])[0]  # noqa: E731
                _cached = _acache().serve(
                    _cache_scope, user_text,
                    validate=lambda a: bool(a) and "[LLM error:" not in a
                    and "[Persona could not" not in a,
                    embed_fn=_embed_fn,
                )
            if _cached:
                if _obs_on:
                    with contextlib.suppress(Exception):
                        from app.perceived.observatory import observatory as _obs
                        _obs.record_ttft(_obs_rid, 0.0)
                # Chunk for smooth client rendering, then persist as a normal
                # assistant turn so history stays consistent.
                _cstep = 240
                for _ci in range(0, len(_cached), _cstep):
                    yield _sse("token", {"text": _cached[_ci:_ci + _cstep]})
                with contextlib.suppress(Exception):
                    await _save_assistant(_cached, incomplete=False)
                yield _sse("done", {
                    "message_id": None, "episode_id": None,
                    "latency_ms": 0, "cached": True,
                })
                return

        try:
            async with factory() as work_session:
                extras = {**extras_base, "db_session": work_session}

                try:
                    # Manual pull loop (instead of `async for`) so we can emit
                    # SSE keepalive comments during long silences — an expert
                    # turn's multi-round self-refine on a big model can be silent
                    # for a minute+, and an idle HTTP response gets dropped
                    # ("connection closed while receiving data"). A `:`-comment
                    # line keeps it alive and is ignored by the client parser.
                    # We create the next-pull task at the TOP of each iteration
                    # and never cancel a RUNNING generator (no aclose-while-busy).
                    _sup = supervisor.stream(model_text, extras=extras)
                    # Total wall-clock budget for the turn (2026-07-09): the
                    # keepalives used to hold a stuck stream open FOREVER —
                    # content-level runaway is stopped by the stream guard,
                    # this bounds silent/stalled supervisors.
                    _budget_s = float(
                        getattr(cfg.llm, "chat_stream_budget_s", 300.0) or 0)
                    _deadline = ((_time.monotonic() + _budget_s)
                                 if _budget_s > 0 else None)
                    while True:
                        _nxt = _asyncio.ensure_future(_sup.__anext__())
                        _cancel_acc = 0.0  # elapsed since last ack/keepalive
                        while True:
                            if (_deadline is not None
                                    and _time.monotonic() > _deadline):
                                _nxt.cancel()
                                with contextlib.suppress(BaseException):
                                    await _sup.aclose()
                                if collected:
                                    yield _sse("token", {
                                        "text": "\n\n*…answer stopped — "
                                                "turn time budget "
                                                "exceeded.*"})
                                yield _sse("error", {
                                    "detail": (f"Turn exceeded its "
                                               f"{int(_budget_s)}s stream "
                                               "budget.")})
                                return
                            # Before the first token, poll on the ack budget so
                            # we can emit an immediate acknowledgment when the
                            # model is slow to start (R7.3); else a 10s keepalive.
                            _wait_t = 10.0
                            if (_ack_budget > 0 and not _ack_sent
                                    and not _first_token_seen):
                                _wait_t = min(10.0, _ack_budget)
                            # Poll every ~0.6s so an explicit Stop is noticed
                            # near-instantly even while the model is silent, but
                            # only emit the ack/keepalive at the original cadence
                            # (accumulate elapsed) so the wire stays quiet.
                            _done, _ = await _asyncio.wait({_nxt}, timeout=0.6)
                            if _done:
                                break
                            # Explicit Stop (FE POSTed /conversations/{id}/cancel):
                            # tear down now instead of waiting for a disconnect.
                            if _stream_cancelled():
                                _nxt.cancel()
                                with contextlib.suppress(BaseException):
                                    await _sup.aclose()
                                raise asyncio.CancelledError()
                            _cancel_acc += 0.6
                            if _cancel_acc < _wait_t:
                                continue
                            _cancel_acc = 0.0
                            if (_ack_budget > 0 and not _ack_sent
                                    and not _first_token_seen):
                                _ack_sent = True
                                yield _sse("ack", {"text": "Working on it…"})
                            else:
                                yield ": keepalive\n\n"  # idle keepalive
                        try:
                            evt = _nxt.result()
                        except StopAsyncIteration:
                            break
                        # Stop pressed while tokens are actively streaming → end
                        # between chunks (the finally/except saves the partial).
                        if _stream_cancelled():
                            with contextlib.suppress(BaseException):
                                await _sup.aclose()
                            raise asyncio.CancelledError()
                        if evt.kind == "meta":
                            intent_payload = evt.data.get("intent", {}) or {}
                            yield _sse("meta", {"intent": intent_payload})
                        elif evt.kind == "tool":
                            name = evt.data.get("name") or "tool"
                            if name not in tools_called:
                                tools_called.append(name)
                            yield _sse("tool", evt.data)
                        elif evt.kind == "model":
                            yield _sse("model", evt.data)
                        elif evt.kind == "clarify":
                            # A blocking clarification withholds the answer — emit
                            # and end the stream. A non-blocking one is a
                            # refinement shown alongside a streaming answer, so we
                            # forward it and keep going.
                            # Capture follow-up suggestions for the envelope (§6).
                            if evt.data.get("suggestions"):
                                _captured_suggestions = list(
                                    evt.data.get("suggestions") or [])
                            # Phase-1: an assume-mode answer states assumptions —
                            # persist them so the assumed slot is never re-asked
                            # and the user's next turn settles it.
                            if evt.data.get("assumptions"):
                                with contextlib.suppress(BaseException):
                                    await _record_assumptions(
                                        conversation_id,
                                        evt.data.get("assumptions"))
                            if evt.data.get("blocking", True):
                                with contextlib.suppress(BaseException):
                                    await _sup.aclose()
                                # Telemetry: record the asked clarification so
                                # the next turn resolves its outcome (R1).
                                with contextlib.suppress(BaseException):
                                    await _record_clarify_asked(
                                        conversation_id,
                                        str(intent_payload.get("intent")
                                            or "unknown"),
                                        float(evt.data.get("confidence", 0.0)
                                              or 0.0))
                                yield _sse("clarify", evt.data)
                                return
                            yield _sse("clarify", evt.data)
                        elif evt.kind == "token":
                            text = evt.data.get("text", "")
                            if text:
                                # Lazy doc_pending: the deterministic gate
                                # missed but the (concurrent) triage LLM
                                # verified a document ask — surface the
                                # "Generating…" signal as soon as it's known.
                                if (not _doc_pending_sent
                                        and _triage_task.done()):
                                    _doc_pending_sent = True
                                    with contextlib.suppress(BaseException):
                                        _tri = _triage_task.result()
                                        if (_tri.wants_document
                                                and not _suppress_empty_target
                                                and not (
                                                    (_tri.doc_format or "")
                                                    in ("zip", "7z")
                                                    and not _archivable)):
                                            yield _sse("meta", {
                                                # Don't GUESS a specific format
                                                # (was "pdf") when the format is
                                                # still unknown — that showed
                                                # "Generating PDF…" even when the
                                                # user picked Word. A generic
                                                # marker renders "Generating
                                                # document…"; the final
                                                # `done.document` carries the
                                                # real format.
                                                "doc_pending":
                                                    str(_tri.doc_format
                                                        or "document")})
                                if not _first_token_seen:
                                    _first_token_seen = True
                                    if _obs_on:
                                        with contextlib.suppress(Exception):
                                            from app.perceived.observatory import (
                                                observatory as _obs,
                                            )
                                            _obs.record_ttft(
                                                _obs_rid,
                                                (_time.monotonic() - _t0) * 1000.0,
                                            )
                                collected.append(text)
                                chunk = _coalescer.push(text)
                                if chunk:
                                    yield _sse("token", {"text": chunk})
                                    if _orch is not None:
                                        try:
                                            for _ev, _data in _orch.on_token(
                                                    chunk):
                                                yield _sse(_ev, _data)
                                        except Exception:  # noqa: BLE001
                                            _orch = None
                        elif evt.kind == "done":
                            rem = _coalescer.flush()
                            if rem:
                                yield _sse("token", {"text": rem})
                            latency_ms = evt.data.get("latency_ms", 0)
                        elif evt.kind == "error":
                            with contextlib.suppress(BaseException):
                                await _sup.aclose()
                            # §15: if a partial answer was already shown, don't
                            # replace it with a raw error — keep it and mark the
                            # turn incomplete so the UI offers Continue/Retry.
                            if "".join(collected).strip():
                                _incomplete_reason = (
                                    (evt.data or {}).get("detail")
                                    or "stream_error")
                                break
                            yield _sse("error", evt.data)
                            return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.exception("agents stream failed")
                    # §15: preserve a partial answer as incomplete rather than
                    # surfacing a raw error over good content.
                    if "".join(collected).strip():
                        _incomplete_reason = f"stream_error: {exc}"
                    else:
                        yield _sse("error", {"detail": f"Stream error: {exc}"})
                        return

                # Coalescer safety flush: if the supervisor ended via
                # StopAsyncIteration (no terminal `done` event), the last
                # buffered chunk would otherwise never reach the client — the
                # visible stream would stop a few characters short of the saved
                # text and look like a stall at the very end. No-op when
                # coalescing is disabled or the `done` branch already flushed.
                _rem = _coalescer.flush()
                if _rem:
                    yield _sse("token", {"text": _rem})

                stream_completed = True
                full_text = "".join(collected).strip()
                if not full_text:
                    # The orchestrated answer came back BLANK (a flaky free model
                    # produced no tokens). Never leave the user with an empty
                    # turn: fall back to a DIRECT answer on a STRONGER tier (a
                    # different model pool), using the windowed history for
                    # context. Retry across tiers before giving up.
                    _fb_msgs = [{"role": "system",
                                 "content": "You are a helpful, precise "
                                 "technical assistant. Answer the user's latest "
                                 "message clearly and completely."}]
                    _fb_msgs += [{"role": m.get("role", "user"),
                                  "content": m.get("content", "")}
                                 for m in prior if m.get("content")]
                    _fb_msgs.append({"role": "user", "content": model_text})
                    from app.core.llm_client import llm as _fllm
                    for _diff in ("standard", "hard", "expert"):
                        if _stream_cancelled():
                            raise asyncio.CancelledError()
                        try:
                            _txt, _ = await _fllm.complete_routed(
                                _fb_msgs, None, {"difficulty": _diff})
                        except Exception as _fexc:  # noqa: BLE001 — try next tier
                            log.info("empty-answer fallback tier=%s failed (%s)",
                                     _diff, _fexc)
                            continue
                        _txt = (_txt or "").strip()
                        if _txt:
                            for _ci in range(0, len(_txt), 240):
                                yield _sse("token", {"text": _txt[_ci:_ci + 240]})
                            collected.append(_txt)
                            full_text = _txt
                            log.info("empty-answer fallback recovered on tier=%s",
                                     _diff)
                            break
                    if not full_text:
                        yield _sse("error", {
                            "detail": "Every model returned an empty response — "
                                      "please try again."})
                        return
                # Phase 6 — flush any pending logical block + end-of-stream
                # analytics (TTFMU / first-code / artifact-ready / total).
                # Additive, fail-open.
                if _orch is not None:
                    try:
                        for _ev, _data in _orch.flush():
                            yield _sse(_ev, _data)
                        yield _sse(*_orch.analytics_frame())
                    except Exception:  # noqa: BLE001
                        _orch = None

                # §4 Intent Profile Registry: resolve one behavior profile for
                # this turn's intent. Each decision point below reads it ONLY
                # when the registry is enabled — off → today's behavior.
                from app.clarify import intent_profiles as _ip
                _profile = _ip.resolve((intent_payload or {}).get("type"))
                _profiles_on = _ip.enabled()

                # Architecture.md §"Response architecture" — run the
                # shaping layer so the persisted reply gets uniform
                # markdown polish + (when applicable) artifact splitting.
                # Failures fall through with the raw text — never block
                # a save on a shaper bug.
                try:
                    from app.core.config_loader import cfg as _cfg
                    from app.response_arch import finalize as _finalize

                    if _cfg.response_arch.enabled:
                        shaped = _finalize(
                            full_text,
                            question=user_text,
                            # Profile shape hint (§4) → prose/table/code/…; None
                            # keeps the auto pick_shape heuristic.
                            shape=(_profile.response_shape
                                   if (_profiles_on and _profile.response_shape)
                                   else None),
                            depth=((_profile.depth
                                    if (_profiles_on and _profile.depth) else None)
                                   or body.depth
                                   or extras_base.get("user_model", {}).get(
                                       "preferred_depth")
                                   or _cfg.response_arch.default_depth),
                        )
                        full_text = shaped.text.strip() or full_text
                        if shaped.artifacts:
                            yield _sse(
                                "artifacts",
                                {
                                    "items": [
                                        {
                                            "filename": a.filename,
                                            "language": a.language,
                                            "content": a.content,
                                        }
                                        for a in shaped.artifacts
                                    ]
                                },
                            )
                except Exception:  # noqa: BLE001
                    pass

                # A mid-stream provider drop is caught by the Persona agent,
                # which leaves an inline error marker — treat that as an
                # interrupted (resumable) turn too.
                incomplete = (
                    _incomplete_reason is not None
                    or ("[LLM error:" in full_text)
                    or ("[Persona could not" in full_text)
                )
                # §15: when the "always finishes" contract is on, a stream that
                # still ends on a length cut-off (continuations exhausted) is an
                # interrupted turn too — offer Continue/Retry.
                if not incomplete:
                    with contextlib.suppress(Exception):
                        if getattr(cfg.resilience,
                                   "mid_stream_continuation", False):
                            from app.llm import usage as _usage
                            from app.llm.continuation import is_cutoff
                            if is_cutoff(_usage.finish_reason()):
                                incomplete = True
                                _incomplete_reason = "length"

                # Persist the assistant message + the episode on the
                # same work_session — keeps the whole turn in one
                # transaction.
                intent_label = intent_payload.get("type") or "general"
                _doc_meta = await _doc_sources(full_text)
                # §4: an intent whose profile isn't doc-eligible never emits a
                # downloadable document (registry-gated; e.g. knowledge/chitchat)
                # — UNLESS the user EXPLICITLY asked for a file this turn (the
                # deterministic detector or the download gate fired): an explicit
                # ask always wins over the profile heuristic, otherwise a
                # knowledge-classified "explain X and give it as a pdf" silently
                # loses its document.
                if (_profiles_on and _doc_meta and not _profile.doc_eligible
                        and not (_det_doc or _wants_download)):
                    log.info("intent-profile[%s]: doc suppressed (not eligible)",
                             _profile.intent)
                    _doc_meta = None
                # Phase 5 — persist the generated document as a versioned
                # artifact (evolution timeline + incremental edits). FAIL-OPEN +
                # best-effort: opens its own session; any error is swallowed and
                # never affects the turn or the answer. On an UPDATE_EXISTING turn
                # the prior document is loaded and this turn's edit MERGED into it
                # (replace/append the edited section), then stored as the next
                # VERSION of the same doc_key — so the timeline stays coherent.
                if _doc_meta:
                    try:
                        from app.documents.intent import ArtifactIntent as _AI
                        from app.documents.store import (
                            latest_for_session, record_generation)
                        _content_to_store = full_text
                        _chain = False
                        if (_planner.intent == _AI.UPDATE_EXISTING
                                and _has_prior_artifact):
                            _chain = True
                            try:
                                from storage.db import get_session_factory as _gsf
                                _f = _gsf()
                                if _f is not None:
                                    async with _f() as _rs:
                                        _prev = await latest_for_session(
                                            _rs, conversation_id)
                                    if _prev is not None and _prev.content_md:
                                        from app.documents.lifecycle import (
                                            merge_update)
                                        _content_to_store = merge_update(
                                            _prev.content_md, full_text)
                            except Exception:  # noqa: BLE001
                                _content_to_store = full_text
                        _doc_title = ""
                        for _ln in (_content_to_store or "").splitlines():
                            _s = _ln.strip().lstrip("#").strip()
                            if _s:
                                _doc_title = _s[:120]
                                break
                        await record_generation(
                            conversation_id, _content_to_store, title=_doc_title,
                            fmt=str(_doc_meta.get("format") or "pdf"),
                            meta={"artifact_intent":
                                  _doc_meta.get("artifact_intent")},
                            chain_latest=_chain)
                    except Exception:  # noqa: BLE001 — persistence is optional
                        pass
                # Build the unified response.v1 envelope ONCE from the turn's
                # durable fields and persist it with the message, so a reload
                # reconstructs the SAME object it streamed live (Architecture §5).
                try:
                    from app.response_arch.envelope import (
                        build_envelope, structure_suggestions)
                    # Tag each suggestion with the intent it would trigger, using
                    # the FAST regex classifier (no embed cost at save time).
                    def _hint(_t: str):
                        from app.clarify.intent_pipeline import (
                            detect_intent, INTENT_UNKNOWN, INTENT_CHITCHAT)
                        _i = detect_intent(_t)
                        return None if _i in (INTENT_UNKNOWN, INTENT_CHITCHAT) else _i
                    _sugg = structure_suggestions(
                        _captured_suggestions, source="profile", intent_of=_hint)
                    # Blend graph-aware sources (§6): memory-graph cross-session
                    # threads (real today) + knowledge-graph neighbors (empty
                    # until the content KG is populated, roadmap #5). Fail-open.
                    from app.response_arch.suggestion_sources import (
                        blend as _blend, from_episodes as _from_eps,
                        from_kg as _from_kg, from_memory as _from_mem)
                    # Open-thread episodes first (cleaner signal), then raw
                    # recalled memory. blend() de-dupes + caps downstream.
                    _mem_sugg = _from_eps(
                        _episode_threads, current_question=user_text,
                        limit=1, intent_of=_hint)
                    _mem_sugg += _from_mem(
                        [getattr(m, "content", "") for m in (_mems or [])],
                        intent_of=_hint)
                    # knowledge_graph source (§3.1/§6). Prefer the CHEAP persistent
                    # doc-KG built at ingest (local query, no per-turn LLM); fall
                    # back to opt-in per-turn extraction. Fail-open → none.
                    _kg_names: list = []
                    _kg_relations: list = []
                    # §4: only consult the knowledge graph when this intent's
                    # profile lists it (registry-gated; else always try, as today).
                    _kg_allowed = (not _profiles_on) or _profile.consults(
                        _ip.GRAPH_KNOWLEDGE)
                    try:
                        from app.rag.kg_extract import (
                            graph_from_json as _kg_from_json,
                            neighbors_in_text as _kg_neighbors,
                            relations_in_text as _kg_relations_fn)
                        _kg_data = None
                        try:
                            import uuid as _uuidk
                            from storage.models import (
                                Project as _ProjectRow, Session as _SessionRow)
                            _srow = await work_session.get(
                                _SessionRow, _uuidk.UUID(str(conversation_id)))
                            # §17: read the PROJECT KG when this conversation is
                            # in a project (shared across its chats); else the
                            # per-conversation KG.
                            _pid = getattr(_srow, "project_id", None)
                            if _pid is not None:
                                _prow = await work_session.get(_ProjectRow, _pid)
                                if _prow is not None and isinstance(
                                        _prow.project_metadata, dict):
                                    _kg_data = _prow.project_metadata.get("kg")
                            if _kg_data is None and _srow is not None and \
                                    isinstance(_srow.session_metadata, dict):
                                _kg_data = _srow.session_metadata.get("kg")
                        except Exception:  # noqa: BLE001
                            _kg_data = None
                        if _kg_data and _kg_allowed:
                            _kg_graph = _kg_from_json(_kg_data)
                            _kg_names = _kg_neighbors(
                                _kg_graph, full_text, hops=1, limit=2)
                            _kg_relations = _kg_relations_fn(
                                _kg_graph, full_text, limit=4)
                        elif _kg_allowed:
                            from app.core.config_loader import cfg as _cfgkg
                            if getattr(_cfgkg.advanced_rag, "kg_suggestions", False):
                                from collections import Counter as _Counter
                                from app.rag.kg_extract import (
                                    build_graph as _build_kg,
                                    extract_graph as _extract_kg,
                                    related_concepts as _related)
                                _nodes, _edges = await _extract_kg(full_text)
                                if _nodes:
                                    _kg = _build_kg(_nodes, _edges)
                                    _deg: _Counter = _Counter()
                                    for _e in _edges:
                                        _deg[_e.src] += 1
                                        _deg[_e.dst] += 1
                                    _topic = (_deg.most_common(1)[0][0]
                                              if _deg else _nodes[0].id)
                                    _kg_names = _related(_kg, [_topic],
                                                         hops=1, limit=2)
                    except Exception:  # noqa: BLE001 — additive, never fatal
                        _kg_names, _kg_relations = [], []
                    _kg_sugg = _from_kg(_kg_names, intent_of=_hint)
                    # Envelope `knowledge` block (§5): related entities +
                    # the relations that grounded them. Absent when the KG
                    # is empty (fail-open).
                    _kg_knowledge = None
                    if _kg_names or _kg_relations:
                        _kg_knowledge = {
                            "related": [{"entity": n} for n in _kg_names],
                            "relations": list(_kg_relations),
                        }
                    _sugg = _blend(profile=_sugg, memory=_mem_sugg,
                                   kg=_kg_sugg, limit=3)
                    # Per-turn trace (Architecture §14): what ran, which graphs,
                    # which suggestion sources — surfaced in the envelope + logged.
                    import uuid as _uuidt
                    from app.obs.trace import build_trace as _build_trace
                    _trace = _build_trace(
                        trace_id=_uuidt.uuid4().hex[:12],
                        model=_engine_last_model(conversation_id),
                        difficulty=_difficulty, latency_ms=latency_ms,
                        tools=tools_called,
                        memory_recalled=int((_memory_meta or {}).get("recalled", 0)),
                        kg_neighbors=len(_kg_names or []),
                        episodes=len(_episode_threads or []),
                        suggestions=_sugg)
                    # §4: record which intent profile drove this turn (obs only).
                    if _profiles_on:
                        with contextlib.suppress(Exception):
                            _trace["profile"] = _profile.intent
                            _trace["suggestion_style"] = _profile.suggestions
                    # The brain's unified read (obs) — intent/difficulty/task/
                    # topic-shift/capabilities that drove routing this turn.
                    if _understanding is not None:
                        with contextlib.suppress(Exception):
                            _trace["understanding"] = _understanding.as_meta()
                    with contextlib.suppress(Exception):
                        import json as _jsont
                        log.info("turn trace %s", _jsont.dumps(_trace, default=str))
                    # Online eval sampling (§14): a fraction of live turns are
                    # recorded for offline grading. Off unless configured.
                    with contextlib.suppress(Exception):
                        from app.eval.online_sample import maybe_record as _samp
                        _samp(question=user_text, answer=full_text,
                              intent=intent_label, trace_id=_trace.get("id"),
                              rate=getattr(cfg.learning,
                                           "online_eval_sample_rate", 0.0))
                    # ── Evidence graph + per-source trust (Phase 3 #7) +
                    # cross-source contradiction (Phase 3 #11) + constraint
                    # verification (Phase 3 truth-maintenance). All additive +
                    # fail-open → folded into the envelope `grounding` block +
                    # the `confidence_band` annotation (Phase 3 #8) + the trace.
                    _grounding = None
                    _conf_band = (_quality_meta or {}).get("band")
                    if getattr(cfg.quality, "evidence_provenance", True):
                        with contextlib.suppress(Exception):
                            from app.rag import conflict as _cflt
                            from app.rag import provenance as _prov
                            _sources = _prov.assemble(
                                memory=_mems, episodes=_episode_threads,
                                kg_names=_kg_names, kg_relations=_kg_relations)
                            # Committed memory vs the produced answer + KG facts.
                            _mem_stmts = [getattr(m, "content", "")
                                          for m in (_mems or [])]
                            _other = (_cflt.split_sentences(full_text)
                                      + [str(r) for r in (_kg_relations or [])])
                            _conflicts = [c.as_dict()
                                          for c in _cflt.detect(_mem_stmts, _other)]
                            _grounding = _prov.grounding_block(
                                _sources, conflicts=_conflicts)
                            if _grounding:
                                _trace["evidence_sources"] = _grounding.get("count", 0)
                                _trace["evidence_trust"] = _grounding.get("trust", 0.0)
                                if _conflicts:
                                    _trace["conflicts"] = len(_conflicts)
                    # Verify the produced answer against the turn's extracted
                    # output constraints (chat-side constraint gate).
                    if _turn_state is not None:
                        with contextlib.suppress(Exception):
                            _crep = _turn_state.check_output(full_text)
                            if _crep and _crep.get("violations"):
                                _trace["constraint_violations"] = _crep["violations"]
                    with contextlib.suppress(Exception):
                        if _turn_state_dict:
                            _trace["horizon"] = _turn_state_dict.get("horizon")
                        if _interaction_meta:
                            _trace["interaction"] = {
                                "action": _interaction_meta.get("action"),
                                "shape": _interaction_meta.get("shape")}
                    _envelope = build_envelope(
                        conversation_id=conversation_id,
                        intent={"type": intent_label} if intent_label else None,
                        difficulty=_difficulty,
                        incomplete=bool(incomplete),
                        document=_doc_meta,
                        suggestions=_sugg,
                        knowledge=_kg_knowledge,
                        grounding=_grounding,
                        confidence_band=_conf_band,
                        model=_engine_last_model(conversation_id),
                        latency_ms=latency_ms,
                        trace=_trace,
                        # §12: /api/agents/stream is the text entry point; the
                        # multimodal entry point is /api/chat/upload-stream.
                        input_modality="text",
                    )
                except Exception:  # noqa: BLE001 — envelope is additive, never fatal
                    # A message persisted without its envelope loses suggestions/
                    # trace/document metadata on reload — loud, not silent.
                    log.warning("envelope build failed for conversation %s "
                                "(message persists without envelope)",
                                conversation_id, exc_info=True)
                    _envelope = None
                assistant_msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_text,
                    intent=intent_label,
                    incomplete=incomplete,
                    sources=_doc_meta,
                    envelope=_envelope,
                )
                work_session.add(assistant_msg)
                convo_row = await work_session.get(Conversation, conversation_id)
                if convo_row is not None:
                    # No-op assignment to trigger SQLAlchemy onupdate.
                    convo_row.title = convo_row.title
                await work_session.commit()
                await work_session.refresh(assistant_msg)
                assistant_saved = True

                # Answer reuse cache (R14): store only a high-quality, complete
                # answer for this user scope so a later identical prompt is served
                # instantly. Flag-gated + best-effort.
                if _cacheable_turn and not incomplete:
                    with contextlib.suppress(Exception):
                        from app.perceived.cache import answer_cache as _acache
                        _vec = None
                        with contextlib.suppress(Exception):
                            from app.rag import embedder as _semb2
                            if _semb2.is_ready():
                                _vec = list(map(float,
                                                _semb2.embed([user_text])[0]))
                        _acache().store(
                            _cache_scope, user_text, full_text,
                            quality_ok=True, embedding=_vec,
                        )
                # Observatory (R16): record the streaming-stage duration.
                if _obs_on:
                    with contextlib.suppress(Exception):
                        from app.perceived.observatory import observatory as _obs
                        _obs.record(_obs_rid, "streaming",
                                    (_time.monotonic() - _t0) * 1000.0)

                episode = Episode(
                    session_id=session_id,
                    user_id=extras_base.get("user_id"),
                    project_id=extras_base.get("project_id"),
                    question=user_text,
                    draft=full_text,
                    final=full_text,
                    intent=intent_label,
                    tools_called=tools_called,
                    latency_ms=latency_ms,
                )
                episode_id = await record_episode(work_session, episode)

                # Fold newly-aged turns into the rolling summary (background) so
                # long threads stay within the context window next turn.
                from app.chat.history import maybe_update_summary

                _st = asyncio.create_task(maybe_update_summary(conversation_id))
                _BG_SAVES.add(_st)
                _st.add_done_callback(_BG_SAVES.discard)

                # Follow-up engine (R7/R10): register answer-derived entities +
                # enumerations into ConversationState for the next turn. Gated +
                # best-effort; runs in the background.
                _fc = asyncio.create_task(
                    _followup_commit(conversation_id, user_text, full_text))
                _BG_SAVES.add(_fc)
                _fc.add_done_callback(_BG_SAVES.discard)

                # Memory-graph (R5/R7): promote durable items into typed memory,
                # run lifecycle maintenance, persist — background, gated.
                _mc = asyncio.create_task(_memory_commit(conversation_id))
                _BG_SAVES.add(_mc)
                _mc.add_done_callback(_BG_SAVES.discard)

                # Learning router (intelligent-model-routing R8): record this
                # turn's outcome (task category, model, success, latency) so a
                # later turn of the same kind can be biased toward what worked.
                # Flag-gated + best-effort; keyed by model_id to match scoring.
                if getattr(cfg.routing, "learning_router", False):
                    with contextlib.suppress(Exception):
                        from app.llm.engine import get_last_model_id
                        from app.llm.learning import record as _lrec
                        from app.llm.task_class import classify_task as _ctask
                        # MUST key by model_id (the router scores by model_id) —
                        # get_last_model() returns the display name, which never
                        # matched, so the learning/adaptive signals were inert.
                        _used = get_last_model_id(conversation_id)
                        if _used:
                            _cat = _ctask(user_text, intent_label, _difficulty)
                            _lrec(_cat, _used, success=not incomplete,
                                  latency_ms=latency_ms or None)
                            with contextlib.suppress(Exception):
                                from app.llm.adaptive import record_outcome
                                record_outcome(_used, not incomplete,
                                               latency_ms or None)
                # Phase 2 semantic routing: fold this turn's outcome into the
                # query's EMBEDDING CLUSTER, keyed by the model_id that answered
                # (matches the router's scoring key). Best-effort + gated.
                if getattr(cfg.routing, "semantic_learning", False):
                    with contextlib.suppress(Exception):
                        from app.llm.engine import get_last_model_id
                        from app.llm.semantic_routing import (
                            record as _srec, remember_turn as _sremember)
                        _emb = extras_base.get("understanding_embedding")
                        _mid = get_last_model_id(conversation_id)
                        if _emb and _mid:
                            _srec(_emb, _mid, not incomplete)
                            # G9: cache the turn so a later 👍/👎 can fold answer
                            # quality into the same cluster.
                            _sremember(str(episode_id), _mid, _emb)
                # G2: cache this turn's difficulty + task so a later 👍 reinforces
                # those classifiers (not just intent). Best-effort.
                if _understanding is not None:
                    with contextlib.suppress(Exception):
                        from app.understanding import remember_turn_meta as _rtm
                        _rtm(str(episode_id), _understanding.difficulty,
                             _understanding.task_category,
                             _understanding.intent_confidence)

                # Artifact runtime (workspace-and-artifacts R4/R5/R7): when the
                # answer is a substantial structured output, save it as a
                # versioned Artifact (reuses the blob store) + register it as the
                # conversation's Current_Artifact for follow-up edits. Flag-gated
                # + best-effort; a normal answer creates nothing (R4.3).
                _artifact_meta: dict = {}
                if getattr(cfg.workspace, "artifacts_enabled", False):
                    try:
                        from app.artifacts import (
                            should_create_artifact, artifact_store,
                        )
                        _src = assistant_msg.sources if isinstance(
                            assistant_msg.sources, dict) else None
                        _afmt = (_src or {}).get("format")
                        _akind = should_create_artifact(
                            full_text, _src, _afmt,
                            int(getattr(cfg.workspace, "artifact_min_chars", 400)))
                        if _akind and not incomplete:
                            _atitle = (" ".join(user_text.split())[:60]
                                       or "Artifact")
                            _art = await artifact_store().create(
                                "default", _akind, _atitle, full_text,
                                _afmt or "md")
                            _artifact_meta = {
                                "id": _art.id, "kind": _akind,
                                "title": _atitle, "version": _art.current_version,
                            }
                            _ar = asyncio.create_task(_register_artifact(
                                conversation_id, _art.id, _atitle, _akind))
                            _BG_SAVES.add(_ar)
                            _ar.add_done_callback(_BG_SAVES.discard)
                    except Exception as _aexc:  # noqa: BLE001
                        log.info("artifact creation skipped: %s", _aexc)
        except asyncio.CancelledError:
            # The client disconnected / pressed Stop mid-stream. Save whatever
            # was generated so it isn't lost — in a DETACHED task that outlives
            # this (cancelled) request. Marked incomplete when the stream
            # hadn't finished, so the UI offers Continue / Retry.
            if not assistant_saved and collected:
                partial = "".join(collected).strip()
                if partial:
                    t = asyncio.create_task(
                        _save_assistant(partial, incomplete=not stream_completed)
                    )
                    _BG_SAVES.add(t)
                    t.add_done_callback(_BG_SAVES.discard)
            raise
        except Exception as exc:  # noqa: BLE001
            # The turn shares ONE transaction with the agent mesh (db_session=
            # work_session), so a DB query by an agent that fails poisons it and
            # the message commit then dies with "current transaction is aborted".
            # Don't lose the answer: the poisoned session is rolled back as the
            # `async with` unwinds, so re-save the reply in a FRESH session.
            recovered = "".join(collected).strip()
            if not assistant_saved and recovered:
                try:
                    await _save_assistant(
                        recovered, incomplete=not stream_completed)
                    yield _sse("done", {
                        "message_id": None,
                        "episode_id": None,
                        "latency_ms": latency_ms,
                    })
                    log.warning(
                        "agents turn: shared txn failed (%s) — saved reply in a "
                        "fresh session", exc)
                    return
                except Exception:  # noqa: BLE001
                    pass
            # Couldn't recover — surface a clean SSE error frame.
            yield _sse("error", {"detail": f"Persist failed: {exc}"})
            return

        # Unified response.v1 envelope (Architecture.md §5) — the SAME object we
        # persisted at save time (so live == reload), with the now-known
        # message_id stamped in. Additive: old clients ignore it. Fail-open.
        _done_envelope = None
        try:
            if _envelope:
                _done_envelope = {**_envelope, "message_id": str(assistant_msg.id)}
        except Exception:  # noqa: BLE001 — envelope is additive, never fatal
            _done_envelope = None

        yield _sse(
            "done",
            {
                "message_id": assistant_msg.id,
                "episode_id": episode_id,
                "latency_ms": latency_ms,
                "model": _engine_last_model(conversation_id),
                # §15: interrupted turn → the client shows a Continue/Retry bar
                # instead of treating the partial as a finished answer.
                **({"incomplete": True} if incomplete else {}),
                **({"envelope": _done_envelope} if _done_envelope else {}),
                **_quality_done_meta(
                    _deg_on, _deg0,
                    getattr(cfg.quality, "surface_meta", False),
                    critic_on=getattr(cfg.quality, "critic", False),
                    answer=full_text,
                    asked_items=[user_text],
                    decisions=extras_base.get("clarify_prefs"),
                ),
                **({"artifact": _artifact_meta}
                   if _artifact_meta and getattr(cfg.workspace,
                                                 "surface_artifact", False)
                   else {}),
                **({"memory": _memory_meta}
                   if _memory_meta and getattr(cfg.memory,
                                               "surface_memory", False)
                   else {}),
                **({"policy": _policy_meta}
                   if _policy_meta and getattr(cfg.personalization,
                                               "surface_meta", False)
                   else {}),
                **({"user_model": _usermodel_meta}
                   if _usermodel_meta and getattr(cfg.personalization,
                                                  "surface_meta", False)
                   else {}),
                # Authoritative doc decision (or absent → NOT a document turn).
                # The client renders the download/preview strictly from this,
                # instead of re-guessing with its own regex.
                **({"document": _doc_meta} if _doc_meta else {}),
            },
        )

    # Keepalive: emit an SSE comment (": keepalive") whenever `interval`
    # seconds pass with no frame, so proxies/clients don't drop the socket
    # during a long P0 planning or slow-provider stall. Comment lines carry
    # no data and are ignored by SSE parsers (and by the replay buffer).
    async def _with_keepalive(gen, interval: float = 15.0):
        ait = gen.__aiter__()
        while True:
            nxt = asyncio.ensure_future(ait.__anext__())
            while True:
                done, _ = await asyncio.wait({nxt}, timeout=interval)
                if nxt in done:
                    break
                yield ": keepalive\n\n"
            try:
                frame = nxt.result()
            except StopAsyncIteration:
                return
            yield frame

    # §15 reconnect/replay: assign a stream id, tag each SSE frame with a
    # monotonic event id, and tee it into a bounded buffer so a dropped socket
    # can resume via /stream/{id}/events?after=<last id>. Additive + flag-gated;
    # off → the raw event_generator with no ids/buffering.
    _replay_on = True
    try:
        _replay_on = bool(getattr(cfg.resilience, "replay_enabled", True))
    except Exception:  # noqa: BLE001
        _replay_on = True

    if not _replay_on:
        return StreamingResponse(
            _with_keepalive(event_generator()),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    from app.api.replay import buffer as _replay_buffer, new_stream_id
    _stream_id = new_stream_id()

    async def _replayable():
        seq = 0
        async for frame in event_generator():
            # Keepalive comments (": …") carry no data — pass through unbuffered.
            if not frame.startswith("event:"):
                yield frame
                continue
            seq += 1
            framed = f"id: {seq}\n{frame}"
            with contextlib.suppress(Exception):
                _replay_buffer.append(_stream_id, seq, framed)
            yield framed

    return StreamingResponse(
        _with_keepalive(_replayable()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Stream-Id": _stream_id,
        },
    )


@router.get("/stream/{stream_id}/events")
async def agents_stream_replay(stream_id: str, after: int = 0) -> StreamingResponse:
    """Replay buffered SSE frames after `after` for a dropped stream (§15).

    Lets a client that lost the socket re-fetch everything it missed (including
    the terminal `done`/`error`) instead of losing output. 404 when the stream
    is unknown/expired — the client then falls back to reloading the
    conversation.
    """
    from app.api.replay import buffer as _replay_buffer
    frames = _replay_buffer.since(stream_id, after)
    if frames is None:
        raise HTTPException(status_code=404,
                            detail="stream not found or expired")

    async def _gen():
        for _eid, framed in frames:
            yield framed

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Stream-Id": stream_id,
        },
    )


@router.post("/episodes/{episode_id}/feedback")
async def post_feedback(
    episode_id: str,
    body: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Attach user feedback (👍/👎/edit) to a past episode.

    Drives the self-learning loops in Architecture.md §4: the Reflector
    reads these signals to extract patterns into semantic memory.
    """
    valid_kinds = {"up", "down", "edit", "redo"}
    if body.kind not in valid_kinds:
        raise HTTPException(
            400, detail=f"feedback kind must be one of {sorted(valid_kinds)}"
        )
    await attach_feedback_db(session, episode_id, body.kind, body.payload)
    # #12 self-improving intent: fold the signal into learned exemplars. The FE
    # passes the turn's question + classified intent in `payload`; 👍 reinforces
    # that phrasing→intent, 👎 demotes it (and reinforces a `corrected_intent`).
    _learned = None
    with contextlib.suppress(Exception):
        from app.clarify.learned_exemplars import learn_from_feedback
        _p = body.payload or {}
        # G2: reinforce difficulty + task too (from the turn's cached meta, or
        # whatever the FE echoed in the payload).
        _tm = None
        with contextlib.suppress(Exception):
            from app.understanding import turn_meta as _turn_meta
            _tm = _turn_meta(episode_id)
        _diff = (_tm[0] if _tm else None) or _p.get("difficulty")
        _task = (_tm[1] if _tm else None) or _p.get("task")
        _res = learn_from_feedback(
            body.kind, _p.get("question"), _p.get("intent"),
            _p.get("corrected_intent"), difficulty=_diff, task=_task)
        # G1: calibrate the intent threshold — a 👍 at score S says "trusting a
        # verdict at S was right", 👎 says "wrong". Never raises.
        if _tm and _tm[2] > 0 and body.kind in ("up", "down"):
            with contextlib.suppress(Exception):
                from app.core.calibration import observe as _cal
                _cal("intent_threshold", _tm[2], body.kind == "up")
        if _res.get("added"):
            _learned = _res
    # G9: fold explicit 👍/👎 into the semantic router (answer quality, not just
    # completeness). Best-effort; only fires when the turn was cached + learning on.
    if body.kind in ("up", "down"):
        with contextlib.suppress(Exception):
            from app.llm.semantic_routing import record_feedback as _sfb
            _sfb(episode_id, body.kind == "up")
    out = {"ok": True, "episode_id": episode_id, "feedback": body.kind}
    if _learned:
        out["learned"] = _learned
    return out


@router.get("/personalization/exemplars")
async def get_learned_exemplars() -> dict:
    """Stats on the learned intent exemplars (self-improving intent, #12)."""
    from app.clarify.learned_exemplars import enabled, stats
    return {"enabled": enabled(), **stats()}


@router.post("/personalization/exemplars/clear")
async def clear_learned_exemplars() -> dict:
    """Forget all learned intent exemplars (privacy control)."""
    from app.clarify.learned_exemplars import clear
    clear()
    return {"cleared": True}


# ---- Data lifecycle & privacy (Architecture §18) ------------------------

@router.get("/data/export")
async def data_export(session: AsyncSession = Depends(get_session)) -> dict:
    """Export EVERYTHING the device user owns as one JSON bundle (§18)."""
    from app.memory.data_lifecycle import export_all
    from storage.device import ensure_device_user
    uid = await ensure_device_user()
    return await export_all(session, user_id=uid)


@router.post("/data/delete-all")
async def data_delete_all(session: AsyncSession = Depends(get_session)) -> dict:
    """Erase EVERYTHING for the device user — conversations, memory, KG, vectors,
    blobs, learned exemplars. Deterministic and irreversible (§18)."""
    from app.memory.data_lifecycle import delete_all
    from storage.device import ensure_device_user
    uid = await ensure_device_user()
    return await delete_all(session, user_id=uid)


@router.post("/data/purge")
async def data_purge(session: AsyncSession = Depends(get_session)) -> dict:
    """Run the retention sweep — purge episodes/skills past `retention_days`
    (no-op when retention is disabled) (§18)."""
    from app.memory.data_lifecycle import purge_expired
    from storage.device import ensure_device_user
    uid = await ensure_device_user()
    return await purge_expired(session, user_id=uid)


@router.post("/data/forget/episode/{episode_id}")
async def data_forget_episode(
    episode_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Forget one remembered turn (episode) + its vector (§18)."""
    from app.memory.data_lifecycle import forget_episode
    ok = await forget_episode(session, episode_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"forgotten": True, "episode_id": episode_id}


@router.post("/data/forget/kg")
async def data_forget_kg_node(
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Forget one knowledge-graph node AND its incident edges from a
    conversation's or project's graph (§18). Body: ``{node_id, conversation_id? |
    project_id?}``."""
    from app.memory.data_lifecycle import forget_kg_node
    node_id = (body.get("node_id") or "").strip()
    if not node_id:
        raise HTTPException(status_code=400, detail="node_id required")
    import uuid as _uuidf
    from storage.models import Project as _P, Session as _S
    pid = body.get("project_id")
    cid = body.get("conversation_id")
    try:
        if pid:
            row = await session.get(_P, _uuidf.UUID(str(pid)))
            meta = dict(row.project_metadata or {}) if row else None
        elif cid:
            row = await session.get(_S, _uuidf.UUID(str(cid)))
            meta = dict(row.session_metadata or {}) if row else None
        else:
            raise HTTPException(
                status_code=400,
                detail="conversation_id or project_id required")
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid id")
    if row is None or meta is None:
        raise HTTPException(status_code=404, detail="Graph owner not found")
    before = meta.get("kg") or {"nodes": [], "edges": []}
    meta["kg"] = forget_kg_node(before, node_id)
    if pid:
        row.project_metadata = meta
    else:
        row.session_metadata = meta
    await session.commit()
    return {"forgotten": True, "node_id": node_id,
            "nodes": len(meta["kg"].get("nodes", [])),
            "edges": len(meta["kg"].get("edges", []))}


@router.get("/clarify/preferences")
async def get_clarify_preferences(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """View the device user's retained clarification preferences (R36)."""
    from app.clarify import load_store
    from storage.device import ensure_device_user

    uid = await ensure_device_user()
    store, _ = await load_store(session, uid)
    if store is None:
        return {"durable": {}, "mode": None, "contract": {}, "analytics": {}}
    return {
        "durable": store.durable_prefs(),
        "mode": store.mode(),
        "contract": store.contract(),
        "analytics": store.analytics(),
    }


@router.post("/clarify/preferences/clear")
async def clear_clarify_preferences(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Forget all retained clarification preferences / analytics (R36)."""
    from app.clarify import load_store, save_store
    from storage.device import ensure_device_user

    uid = await ensure_device_user()
    store, user = await load_store(session, uid)
    if store is None:
        return {"cleared": False}
    store.clear()
    await save_store(session, user, store)
    return {"cleared": True}


@router.post("/clarify/preferences")
async def set_clarify_preferences(
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Set the active clarification mode and/or collaboration contract (R18/R19).

    Body: {"mode": "explorer|builder|expert|autopilot|teacher"|null,
           "contract": {setting: value}}.
    """
    from app.clarify import load_store, save_store
    from storage.device import ensure_device_user

    uid = await ensure_device_user()
    store, user = await load_store(session, uid)
    if store is None:
        return {"ok": False}
    if "mode" in body:
        mode = body.get("mode")
        store.set_mode((str(mode).strip().lower() or None) if mode else None)
    if isinstance(body.get("contract"), dict):
        store.set_contract(body["contract"])
    await save_store(session, user, store)
    return {"ok": True, "mode": store.mode(), "contract": store.contract()}


@router.get("/personalization/instructions")
async def get_custom_instructions(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The device user's standing custom instructions (Architecture §17)."""
    from app.personalization.instructions import (
        MAX_CHARS, load_custom_instructions)
    from storage.device import ensure_device_user
    from storage.models import User

    uid = await ensure_device_user()
    user = await session.get(User, uid) if uid else None
    text = load_custom_instructions(getattr(user, "preferences", None))
    return {"custom_instructions": text, "max_chars": MAX_CHARS}


@router.post("/personalization/instructions")
async def set_custom_instructions_route(
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Set (or clear, with blank text) the device user's custom instructions.

    Body: ``{"custom_instructions": "<text>"}``. Trimmed + capped server-side.
    """
    from app.personalization.instructions import (
        load_custom_instructions, set_custom_instructions)
    from storage.device import ensure_device_user
    from storage.models import User

    uid = await ensure_device_user()
    if not uid:
        raise HTTPException(status_code=503, detail="No device user available")
    user = await session.get(User, uid)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.preferences = set_custom_instructions(
        user.preferences, body.get("custom_instructions"))
    await session.commit()
    return {"ok": True,
            "custom_instructions": load_custom_instructions(user.preferences)}
