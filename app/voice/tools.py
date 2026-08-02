"""Agent-stack tool bridge (design §4).

This is the mechanism that keeps the existing intelligence relevant under a
cloud speech model, and the reason "it can answer anything by voice" is true
without pretending a fast conversational model is also the best reasoner.

Every tool here is a **thin adapter over a capability that already exists**. No
new intelligence is written in this module — if a tool looks like it is doing
reasoning, it is in the wrong place.

Design points worth keeping in mind while reading:

* ``ask_reasoner`` is the escape hatch for depth. The realtime model answers
  conversationally and handles ordinary technical explanation itself; when a
  question deserves the full routed stack, it delegates. That division is what
  makes voice mode a peer of chat rather than a subset of it.
* Tool latency is made **audible, not hidden**. Each dispatch emits start/ok/fail
  so the client can show "looking that up" instead of dead air, and the session
  instructions tell the model to verbalise briefly before a slow tool.
* Tools are **allow-listed per session** and run under the authenticated user's
  scope, so a voice session can never reach data the user could not reach in
  chat (Requirement 8.3).
* A failing tool is **reported to the model, not to the user** (Requirement 8.4):
  the model is told the tool failed and answers without it.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

log = logging.getLogger("zapthetrick.voice.tools")

# Per-tool wall-clock deadline. Past this the model is told the tool failed and
# continues — a voice conversation must never stall on a slow backend.
DEFAULT_DEADLINE_S = 12.0
# `ask_reasoner` is expected to be slow (it is the full routed stack), so it
# gets its own, longer budget.
REASONER_DEADLINE_S = 30.0


@dataclass(frozen=True)
class VoiceTool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Awaitable[Any]]
    deadline_s: float = DEFAULT_DEADLINE_S

    def to_openai(self) -> dict:
        """Realtime session tool schema (flat form, not nested under
        `function` — that is the shape the realtime API expects)."""
        return {"type": "function", "name": self.name,
                "description": self.description, "parameters": self.parameters}


# ── Handlers ────────────────────────────────────────────────────────────────
# Each takes the session context as `ctx` plus its declared arguments, and
# returns something JSON-serialisable. Raising is fine — `dispatch` converts an
# exception into the model-visible failure.

async def _search_workspace(ctx, *, query: str, **_) -> dict:
    """The user's own documents, via the existing hybrid RAG retriever."""
    from storage.db import get_session_factory
    resume_id = getattr(ctx, "session_id", None) or ""
    if not resume_id:
        return {"hits": [], "note": "no workspace document is open"}
    factory = get_session_factory()
    if factory is None:
        return {"hits": [], "note": "storage unavailable"}
    from app.tools import resume_lookup
    async with factory() as session:
        hits = await resume_lookup.lookup(query=query, resume_id=resume_id,
                                          session=session)
    return {"hits": hits[:5]}


async def _recall_conversation(ctx, *, query: str, **_) -> dict:
    """"What did we decide earlier" — searches this conversation's history."""
    conv = getattr(ctx, "conversation_id", None) or ""
    if not conv:
        return {"hits": [], "citations": "No conversation to recall."}
    from storage.db import get_session_factory
    from app.tools.conversation_search import conversation_search
    factory = get_session_factory()
    if factory is None:
        return {"hits": [], "citations": "Storage unavailable."}
    async with factory() as session:
        return await conversation_search(query=query, conversation_id=conv,
                                         session=session)


async def _ask_reasoner(ctx, *, question: str, **_) -> dict:
    """Delegate a hard question to the full routed model stack.

    This is the depth escape hatch. It deliberately goes through `LLMClient`
    (not the low-level router) so routing, fallback, verification and the
    never-empty ladder all apply exactly as they do in chat.
    """
    from app.core.llm_client import LLMClient
    client = LLMClient()
    messages = [
        {"role": "system", "content":
         "Answer precisely and completely. Your answer will be READ ALOUD, so "
         "use plain prose: no markdown, no code fences, no bullet characters. "
         "Prefer a few well-formed sentences over a list."},
        {"role": "user", "content": question},
    ]
    text = await client.complete(messages)
    return {"answer": (text or "").strip()}


async def _run_code(ctx, *, code: str, language: str = "python", **_) -> dict:
    """Execute in the existing sandbox. Compute and verification, not display."""
    from app.sandbox.executor import verify_script
    res = await asyncio.to_thread(verify_script, code, language)
    return {
        "ok": bool(getattr(res, "ok", False)),
        "stdout": (getattr(res, "stdout", "") or "")[:2000],
        "stderr": (getattr(res, "stderr", "") or "")[:1000],
        "status": getattr(res, "status", ""),
        "reason": (getattr(res, "reason", "") or "")[:400],
    }


async def _web_search(ctx, *, query: str, **_) -> dict:
    """Freshness — the model's training cutoff is not the user's today."""
    from app.tools import web_search
    hits = await web_search.search(query=query)
    return {"results": hits[:5]}


async def _make_artifact(ctx, *, title: str, content: str,
                         format: str = "md", **_) -> dict:
    """"Write that up for me" — produces a downloadable artifact that lands in
    the associated chat conversation (Requirement 8.5)."""
    from app.documents.generators import render_document
    data, mime, ext = await asyncio.to_thread(
        render_document, content, format, title)
    return {"ok": True, "title": title, "ext": ext, "mime": mime,
            "bytes": len(data or b"")}


async def _remember(ctx, *, fact: str, **_) -> dict:
    """Durable user facts, through the existing typed-memory layer.

    Mirrors the persistence dance `routes_agents` already performs: hydrate the
    store from the user's preferences blob, add a durable `MemoryObject`, export
    it back, save. Memory is additive — a failure here never breaks a turn.
    """
    text = (fact or "").strip()
    if not text:
        return {"ok": False, "detail": "nothing to remember"}
    uid = getattr(ctx, "user_id", None)
    if not uid:
        return {"ok": False, "detail": "no signed-in user"}
    from storage.db import get_session_factory
    factory = get_session_factory()
    if factory is None:
        return {"ok": False, "detail": "storage unavailable"}

    from app.clarify.preferences import load_store, save_store
    from app.memory.mstore import memory_store
    from app.memory.objects import MemoryObject, SCOPE_GLOBAL

    def _embed(t: str):
        try:
            from app.rag.embedder import embed_one
            return embed_one(t)
        except Exception:  # noqa: BLE001
            return None

    async with factory() as session:
        store, user = await load_store(
            session, uid,
            conversation_id=getattr(ctx, "conversation_id", None))
        if store is None:
            return {"ok": False, "detail": "no preference store"}
        mem = memory_store()
        mem.load_from(store.root)
        if any(o.content == text for o in mem.all()):
            return {"ok": True, "note": "already remembered"}
        obj = MemoryObject(content=text, kind="fact", scope=SCOPE_GLOBAL,
                           importance=0.7, durable=True)
        obj.embedding = _embed(text)
        mem.add(obj)
        mem.export_to(store.root)
        await save_store(session, user, store)
    return {"ok": True}


# ── Catalogue ───────────────────────────────────────────────────────────────

_STR = {"type": "string"}


ALL_TOOLS: tuple[VoiceTool, ...] = (
    VoiceTool(
        "search_workspace",
        "Search the user's own uploaded documents and workspace for passages "
        "relevant to a query. Use when the user refers to their files, resume, "
        "or anything 'in my documents'.",
        {"type": "object", "properties": {"query": _STR}, "required": ["query"]},
        _search_workspace,
    ),
    VoiceTool(
        "recall_conversation",
        "Search earlier turns of this conversation. Use for 'what did we decide',"
        " 'what did I say about', or any reference to something discussed before.",
        {"type": "object", "properties": {"query": _STR}, "required": ["query"]},
        _recall_conversation,
    ),
    VoiceTool(
        "ask_reasoner",
        "Delegate a hard technical question to a stronger reasoning model. Use "
        "for deep architecture, algorithms, debugging, mathematics, or anything "
        "where being right matters more than being fast. Say something brief "
        "aloud first — this takes a few seconds.",
        {"type": "object", "properties": {"question": _STR},
         "required": ["question"]},
        _ask_reasoner,
        deadline_s=REASONER_DEADLINE_S,
    ),
    VoiceTool(
        "run_code",
        "Execute code in a sandbox and return its output. Use to compute an "
        "answer or to verify that code you are about to describe actually runs.",
        {"type": "object",
         "properties": {"code": _STR,
                        "language": {"type": "string", "default": "python"}},
         "required": ["code"]},
        _run_code,
        deadline_s=REASONER_DEADLINE_S,
    ),
    VoiceTool(
        "web_search",
        "Search the web for current information. Use for recent events, "
        "versions, prices, or anything that changes over time.",
        {"type": "object", "properties": {"query": _STR}, "required": ["query"]},
        _web_search,
    ),
    VoiceTool(
        "make_artifact",
        "Produce a downloadable document from content you have written. It "
        "appears in the user's chat conversation.",
        {"type": "object",
         "properties": {"title": _STR, "content": _STR,
                        "format": {"type": "string",
                                   "enum": ["md", "txt", "docx", "pdf", "xlsx"]}},
         "required": ["title", "content"]},
        _make_artifact,
        deadline_s=REASONER_DEADLINE_S,
    ),
    VoiceTool(
        "remember",
        "Store a durable fact about the user for future conversations.",
        {"type": "object", "properties": {"fact": _STR}, "required": ["fact"]},
        _remember,
    ),
)

_BY_NAME = {t.name: t for t in ALL_TOOLS}

# The default allow-list. Every tool is safe under the user's own scope, but the
# set is explicit so a deployment can narrow it without editing code.
DEFAULT_ALLOWED = frozenset(_BY_NAME)


def schema_for(allowed: frozenset[str] | None = None) -> list[dict]:
    """Tool schemas for a session, in the realtime API's shape."""
    names = allowed if allowed else DEFAULT_ALLOWED
    return [t.to_openai() for t in ALL_TOOLS if t.name in names]


async def dispatch(name: str, args: dict | str, ctx,
                   allowed: frozenset[str] | None = None) -> dict:
    """Run one tool call and return a JSON-serialisable result.

    NEVER raises. A rejected, unknown, failed or timed-out tool returns an
    ``{"error": ...}`` payload, which the caller hands back to the model — the
    model then answers without that tool rather than the turn dying
    (Requirement 8.4).
    """
    tool = _BY_NAME.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}"}
    names = allowed if allowed else DEFAULT_ALLOWED
    if name not in names:
        return {"error": f"tool {name!r} is not available in this session"}

    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except Exception:  # noqa: BLE001
            return {"error": "arguments were not valid JSON"}
    if not isinstance(args, dict):
        args = {}

    try:
        return await asyncio.wait_for(tool.handler(ctx, **args),
                                      timeout=tool.deadline_s)
    except asyncio.TimeoutError:
        log.info("voice tool %s timed out after %.0fs", name, tool.deadline_s)
        return {"error": f"{name} timed out after {tool.deadline_s:.0f}s"}
    except asyncio.CancelledError:
        raise
    except TypeError as exc:            # wrong/missing arguments from the model
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        log.info("voice tool %s failed", name, exc_info=True)
        return {"error": f"{name} failed: {str(exc)[:200]}"}


__all__ = [
    "VoiceTool", "ALL_TOOLS", "DEFAULT_ALLOWED", "schema_for", "dispatch",
    "DEFAULT_DEADLINE_S", "REASONER_DEADLINE_S",
]
