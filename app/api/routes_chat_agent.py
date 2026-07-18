"""Chat Agent-Run API (Spec §8.4-2, Phase 3) — bring the workspace tool LOOP
into the chat flow so a chat user can hand the agent a codebase (uploaded
archive, materialized by Phase 2) or a spec and have it plan → (clarify) →
edit/build → run/test → fix → return the modified project.

POST /api/chat/agent-run  (SSE)
  body: {conversation_id, task, kind?("build"|"edit"), mode?, history?,
         condition?, max_steps?, images?, files?}
  → resolve the conversation's workspace (fresh for a build, existing for an
    edit) → stream `run_goal(...)` events (`thought`/`tool_call`/`tool_result`/
    `plan`/`question`/`approval`/`goal_*`/`final`) → on `final` emit a `diff`
    event (git diff vs the materialized baseline) and flag the assistant turn
    as a downloadable project ZIP. Every event is also persisted to
    `agent_steps` AND the user/final turns to `messages`, both under the
    conversation (a Session row), so the run reloads in chat history.

GET /api/chat/agent-run/{conversation_id}/download
  → the modified workspace packaged as a .zip (skips .git/node_modules/…).

Approvals / questions reuse the existing /api/agent/approve + /api/agent/answer
endpoints (the loop's `ask_user`/`present_plan` await the same registries).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.agent_workspace import redact_event, redact_secrets

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat-agent"])

# Detached persist-on-disconnect tasks (GC guard).
_BG_SAVES: set = set()

# Concurrency cap (Phase 4.5): a single VPS can only run so many untrusted
# workspace agent loops at once (each spawns build/test subprocesses). A
# process-wide semaphore queues extra runs instead of overloading the box.
# Lazily built so a config reload picks up the limit.
_run_sem: "asyncio.Semaphore | None" = None
_run_sem_limit: int = -1
_active_runs: int = 0


def _semaphore() -> "asyncio.Semaphore":
    global _run_sem, _run_sem_limit
    try:
        from app.core.config_loader import cfg
        limit = max(1, int(getattr(cfg.advanced_rag,
                                   "max_concurrent_agent_runs", 3)))
    except Exception:  # noqa: BLE001
        limit = 3
    if _run_sem is None or limit != _run_sem_limit:
        _run_sem = asyncio.Semaphore(limit)
        _run_sem_limit = limit
    return _run_sem


def _redact_on() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.advanced_rag, "redact_agent_secrets", True))
    except Exception:  # noqa: BLE001
        return True


class ChatAgentRunBody(BaseModel):
    conversation_id: str
    task: str
    # "build" (fresh workspace) or "edit" (reuse the conversation's workspace).
    kind: str = "edit"
    mode: str = "acceptEdits"
    history: list[dict] = []
    # Completion condition for the goal loop; defaults to the task itself.
    condition: str = ""
    max_rounds: int = 4
    max_steps: int = 24
    images: list[str] = []
    files: list[str] = []


def _frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _resolve_workspace(conversation_id: str, kind: str) -> tuple[str | None, str]:
    """(workspace_path, error). For a build → a fresh empty dir; for an edit →
    the existing materialized workspace (must already exist from an upload)."""
    from app.agent_workspace import (
        fresh_workspace,
        workspace_exists,
        workspace_path,
    )

    if kind == "build":
        # Reuse a non-empty workspace if the user already uploaded a seed
        # (e.g. a starter zip); otherwise start clean.
        if workspace_exists(conversation_id):
            return workspace_path(conversation_id), ""
        return fresh_workspace(conversation_id), ""
    # edit / default — needs a materialized codebase.
    if not workspace_exists(conversation_id):
        return None, (
            "No project workspace for this conversation yet. Upload a code "
            "archive (zip/rar/7z/tar) first, then ask me to edit it."
        )
    return workspace_path(conversation_id), ""


def _resolve_kind(body: "ChatAgentRunBody") -> str:
    """Resolve an explicit kind, or auto-detect via the intent router when the
    caller passes ``kind="auto"`` (wires the chat intent router into the run)."""
    if body.kind in ("build", "edit"):
        return body.kind
    from app.agent_workspace import workspace_exists
    from app.documents.detect import detect_agentic_intent

    ws = workspace_exists(body.conversation_id)
    intent = detect_agentic_intent(
        body.task, has_archive=ws, workspace_exists=ws)
    # Default to "edit" when a workspace exists, else "build" from scratch.
    return intent.get("kind") or ("edit" if ws else "build")


async def _diff(workspace: str) -> str:
    try:
        from app.agent_workspace import diff_summary
        return await diff_summary(workspace)
    except Exception:  # noqa: BLE001
        return ""


async def _semantic(workspace: str) -> list[str]:
    """AST-level change summary (#106), best-effort. Call after _diff."""
    try:
        from app.agent_workspace import semantic_change_summary
        return await asyncio.wait_for(
            semantic_change_summary(workspace), timeout=30)
    except Exception:  # noqa: BLE001
        return []


async def _tests(workspace: str):
    """P2-5 test surface of the change (tests added + untested symbols),
    best-effort. Call after _diff (the tree is staged). Returns a TestSurface
    or None."""
    try:
        from app.agent.testgen import test_surface
        return await asyncio.wait_for(test_surface(workspace), timeout=30)
    except Exception:  # noqa: BLE001
        return None


async def _gitflow(workspace: str, task: str, summary: str):
    """P2-7 git workflow: branch + commit (+ opt-in push/PR). Best-effort →
    None. Run AFTER the diff/semantic/test passes (committing moves HEAD)."""
    try:
        import os as _os

        from app.agent_workspace import run_git_workflow
        from app.core.config_loader import cfg
        gw = getattr(cfg, "git_workflow", None)
        if gw is None or not getattr(gw, "enabled", True):
            return None
        token = (getattr(gw, "token", "") or "").strip() \
            or _os.environ.get("ZAPTHETRICK_GIT_TOKEN", "")
        return await asyncio.wait_for(run_git_workflow(
            workspace, task=task, summary=summary,
            branch_prefix=getattr(gw, "branch_prefix", "zapthetrick"),
            auto_push=bool(getattr(gw, "auto_push", False)),
            open_pr=bool(getattr(gw, "open_pr", False)),
            token=token), timeout=320)
    except Exception:  # noqa: BLE001
        return None


async def _trust_pass(task: str, final_text: str, diff: str,
                      ctx_files: list[str], sig: dict):
    """Phase 8: adversarial review → confidence band + provenance.
    Returns (review_dicts, ConfidenceResult|None, provenance_list)."""
    try:
        from app.core.config_loader import cfg
    except Exception:  # noqa: BLE001
        return [], None, []
    do_review = bool(getattr(cfg.advanced_rag, "red_team_review", True))
    do_conf = bool(getattr(cfg.advanced_rag, "surface_confidence", True))

    review_dicts: list[dict] = []
    high_risks = 0
    if do_review:
        try:
            from app.chat.redteam import (
                count_high,
                red_team_review,
                risks_to_dicts,
            )
            work = (final_text or "")
            if diff:
                work += "\n\n[diff]\n" + diff
            risks = await asyncio.wait_for(
                red_team_review(task, work), timeout=45)
            review_dicts = risks_to_dicts(risks)
            high_risks = count_high(risks)
        except Exception:  # noqa: BLE001
            review_dicts, high_risks = [], 0

    # Phase 12 — multi-provider council: a DIFFERENT free model (or N) judges
    # the result. Disagreement lowers confidence + is surfaced. Time-budgeted.
    verdict = None
    try:
        from app.chat.council import council_enabled, cross_model_verify
        _cv_on, _cv_size = council_enabled()
        if _cv_on:
            _work = (final_text or "")
            if diff:
                _work += "\n\n[diff]\n" + diff
            verdict = await asyncio.wait_for(
                cross_model_verify(task, _work, n=_cv_size),
                timeout=20 + 20 * _cv_size)
    except Exception:  # noqa: BLE001
        verdict = None

    conf = None
    prov: list[str] = []
    if do_conf:
        try:
            from app.chat.trust import (
                ConfidenceSignals,
                build_provenance,
                confidence_band,
            )
            conf = confidence_band(ConfidenceSignals(
                goal_passed=sig.get("goal_passed"),
                verify_attempted=bool(sig.get("verify_attempted")),
                verify_ok=sig.get("verify_ok"),
                rounds=int(sig.get("rounds") or 1),
                had_error=bool(sig.get("had_error")),
                high_risks=high_risks,
                tests_added=int(sig.get("tests_added") or 0),
                untested_changes=int(sig.get("untested_changes") or 0),
                cross_verify_disagree=bool(verdict is not None
                                           and verdict.votes and not verdict.agree),
            ))
            changed = diff.count("\n") if diff and "No file changes" not in diff \
                else 0
            prov = build_provenance(
                context_files=ctx_files,
                changed_files=(changed or None),
            )
        except Exception:  # noqa: BLE001
            conf, prov = None, []
    return review_dicts, conf, prov, verdict


async def _brain_finalize(workspace: str, task: str, kind: str,
                          metrics, final_text: str) -> None:
    """Phase 11: after a run, remember the model outcome (learning-lite) and
    append an auto entry to the decision ledger for continuity. Best-effort."""
    try:
        from app.core.config_loader import cfg
        if not getattr(cfg.advanced_rag, "project_brain", True):
            return
        from app.agent_workspace import record_decision, remember_run
        await asyncio.to_thread(
            remember_run, workspace, model=getattr(metrics, "model", None),
            kind=kind, success=bool(getattr(metrics, "success", False)))
        summary = (final_text or "").strip().splitlines()
        head = summary[0][:200] if summary else "(completed)"
        verdict = "verified" if getattr(metrics, "verify_ok", None) else (
            "completed" if getattr(metrics, "success", False) else "attempted")
        await asyncio.to_thread(
            record_decision, workspace, (task or "request")[:70], head,
            f"{verdict}; {getattr(metrics, 'rounds', 1)} round(s)"
            + (f", model {metrics.model}" if getattr(metrics, "model", None)
               else ""))
        # P2-11: a one-line reflection (lesson) into the brain ledger.
        if getattr(cfg.advanced_rag, "reflection_notes", True):
            from app.chat.self_improve import reflect
            note = reflect(
                task, success=bool(getattr(metrics, "success", False)),
                verify_ok=getattr(metrics, "verify_ok", None),
                rounds=int(getattr(metrics, "rounds", 1) or 1))
            await asyncio.to_thread(
                record_decision, workspace, "Reflection", note)
    except Exception:  # noqa: BLE001
        pass


@router.post("/agent-run")
async def chat_agent_run(body: ChatAgentRunBody) -> StreamingResponse:
    from app.agent.loop import run_agent, run_goal

    kind = _resolve_kind(body)
    workspace, err = _resolve_workspace(body.conversation_id, kind)

    async def gen() -> AsyncGenerator[str, None]:
        from storage.db import get_session_factory
        from storage.repos import AgentStepRepo, MessageRepo, SessionRepo

        # Workspace resolution failure → a clean error frame, no crash.
        if err or workspace is None:
            yield _frame("error", {"detail": err or "workspace unavailable"})
            yield "event: end\ndata: {}\n\n"
            return

        yield _frame("session", {"id": body.conversation_id,
                                 "workspace": workspace, "kind": kind})

        # agent-orchestration (R1/R2/R3): when enabled, surface the orchestration
        # PLAN (decomposed sub-tasks + selected role workflow + tool plan) as an
        # additive timeline event before the agent loop runs. Flag-gated +
        # fail-open; the existing run_agent loop still executes the work. Legacy
        # clients ignore the extra `orchestration` event.
        try:
            from app.core.config_loader import cfg as _cfgo
            if (getattr(_cfgo.orchestration, "enabled", False)
                    and getattr(_cfgo.orchestration, "surface_orchestration", False)):
                from app.orchestration import decompose, select_workflow, plan_tools
                _subs = decompose(body.task)
                _wf = select_workflow({
                    "difficulty": "expert" if _subs else "standard",
                    "task_category": "coding" if kind in ("edit", "build") else "general",
                    "multi_goal": bool(_subs),
                })
                _tools = []
                try:
                    from app.mcp.registry import registry as _mcp_reg
                    _tools = _mcp_reg.list_tools()
                except Exception:  # noqa: BLE001
                    _tools = []
                _tp = plan_tools(body.task, _subs, _tools)
                # Phase-4 #2/#3: wire RoleRunner — assign each role a
                # capability-matched model (real RoleRunner run), surfaced so the
                # UI shows who's doing what. Fail-open.
                _roles_used: dict = {}
                if _wf.kind != "single" and _wf.roles:
                    try:
                        from app.orchestration.goal_engine import plan_role_models
                        _roles_used = await plan_role_models(
                            _wf, "coding" if kind in ("edit", "build")
                            else "general")
                    except Exception:  # noqa: BLE001
                        _roles_used = {}
                if _subs or _wf.kind != "single" or _tp.names():
                    yield _frame("orchestration", {
                        "workflow": _wf.kind,
                        "roles": list(_wf.roles),
                        "roles_used": _roles_used,
                        "sub_tasks": [s.text for s in _subs],
                        "tools": _tp.names(),
                    })
        except Exception as _oexc:  # noqa: BLE001
            log.info("orchestration plan skipped: %s", _oexc)

        # P2-12 — misuse/content-safety guard: refuse offensive-tooling requests
        # (malware/keylogger/ransomware/phishing/…) up front, before running the
        # agent. Defensive/educational security work is allowed. Flag-gated.
        try:
            from app.core.config_loader import cfg as _cfgs
            if getattr(_cfgs.advanced_rag, "content_safety", True):
                from app.agent.safety import classify_misuse
                _mv = classify_misuse(body.task)
                if _mv.blocked:
                    yield _frame("security", {"blocked": True,
                                              "category": _mv.category})
                    yield _frame("final", {"message": _mv.reason})
                    yield "event: end\ndata: {}\n\n"
                    return
        except Exception as exc:  # noqa: BLE001 — never let the guard crash a run
            log.info("content-safety guard skipped: %s", exc)

        # Concurrency cap (Phase 4.5): bound simultaneous untrusted runs. If the
        # box is busy, tell the user we've queued and wait our turn.
        sem = _semaphore()
        if sem.locked():
            yield _frame("queued",
                         {"detail": "Another project build is running — "
                                    "queued, starting shortly…"})
        await sem.acquire()
        global _active_runs
        _active_runs += 1
        _red = _redact_on()

        def _emit(event: str, data: dict) -> str:
            return _frame(event, redact_event(data) if _red else data)

        condition = body.condition.strip() or (
            "The user's request is fully and correctly implemented, the code "
            f"builds, and tests pass: {body.task}"
        )

        # Phase 5: rank the workspace's files for THIS task and inject a
        # budget-bounded "relevant project context" preamble, so a large repo
        # fits the free model's window and the agent starts on the right files.
        # Edit runs only (a fresh build has nothing to rank). Best-effort.
        ctx = ""
        ctx_files: list[str] = []
        _sec_hits: list[str] = []   # P2-12 injection hits found in context
        # Phase 8 trust signals, captured from the goal-loop events.
        _sig = {"goal_passed": None, "rounds": 1, "verify_attempted": False,
                "verify_ok": None, "had_error": False}
        # Phase 9 observability — per-run metrics accumulated as events stream.
        from app.obs.metrics import RunMetrics
        _metrics = RunMetrics(kind=kind)
        try:
            from app.core.config_loader import cfg as _cfgctx
            if (kind == "edit"
                    and getattr(_cfgctx.advanced_rag, "use_context_builder", True)):
                from app.chat.context_builder import ContextBudget, build_context
                _budget = ContextBudget(max_tokens=int(getattr(
                    _cfgctx.advanced_rag, "context_budget_tokens", 6000)))
                # P2-3: prefer context_v2 (hierarchical digest + recency-aware
                # ranking + read compression); fall back to the Phase-5 builder.
                if getattr(_cfgctx.advanced_rag, "context_v2", True):
                    from app.chat.context_v2 import build_context_v2
                    _res = await asyncio.to_thread(
                        build_context_v2, workspace, body.task, budget=_budget,
                        include_digest=getattr(
                            _cfgctx.advanced_rag, "repo_digest", True))
                else:
                    _res = await asyncio.to_thread(
                        build_context, workspace, body.task, budget=_budget)
                ctx = _res.text
                ctx_files = _res.files
                if ctx:
                    yield _frame("context", {"files": _res.files,
                                             "tokens": _res.tokens,
                                             "truncated": _res.truncated})
        except Exception as exc:  # noqa: BLE001 — context is an optimization
            log.info("context builder skipped: %s", exc)
            ctx = ""

        # code-intelligence R5/R6: augment the selected slice with each file's
        # direct dependencies (symbol/import graph), and keep the symbol index
        # fresh. Flag-gated by use_code_knowledge_graph + fail-open; additive
        # `context_deps` frame (legacy clients ignore it).
        try:
            from app.core.config_loader import cfg as _cfgci
            if (kind == "edit"
                    and getattr(_cfgci.advanced_rag, "use_code_knowledge_graph", False)
                    and ctx_files):
                from app.codeintel.index import build_index
                from app.codeintel.graph import dependency_graph
                _ci = await asyncio.to_thread(
                    build_index, body.conversation_id, workspace)
                _dep = dependency_graph(_ci).get("internal", {})
                _have = set(ctx_files)
                _add: list[str] = []
                for _f in list(ctx_files):
                    for _d in _dep.get(_f, []):
                        if _d not in _have:
                            _have.add(_d)
                            _add.append(_d)
                if _add:
                    yield _frame("context_deps", {"files": _add[:10]})
        except Exception as exc:  # noqa: BLE001
            log.info("codeintel context augmentation skipped: %s", exc)

        # Phase 11 — project brain: continuity across turns. Read the workspace
        # memory (facts + decision ledger + preferred model) and prepend it to
        # the context preamble; ensure the file exists so the agent's
        # `record_decision` tool + the post-run auto-summary have somewhere to
        # write. Best-effort, flag-gated.
        _brain_on = False
        try:
            from app.core.config_loader import cfg as _cfgb
            if getattr(_cfgb.advanced_rag, "project_brain", True):
                _brain_on = True
                from app.agent_workspace import brain_context, ensure_brain
                await asyncio.to_thread(ensure_brain, workspace)
                _bctx = await asyncio.to_thread(brain_context, workspace)
                if _bctx:
                    ctx = (_bctx + "\n\n" + ctx) if ctx else _bctx
                    yield _frame("brain", {"present": True})
        except Exception as exc:  # noqa: BLE001
            log.info("project brain skipped: %s", exc)

        # P2-12 — prompt-injection guard: the context is built from the user's
        # UNTRUSTED files. Scan it for instructions aimed at the agent, surface
        # any hits, and frame the whole preamble as data-not-instructions.
        try:
            from app.core.config_loader import cfg as _cfgi
            if ctx and getattr(_cfgi.advanced_rag, "injection_guard", True):
                from app.agent.safety import scan_injection, wrap_untrusted
                _hits = scan_injection(ctx)
                if _hits:
                    _sec_hits = _hits[:8]
                    yield _frame("security", {"injection": _sec_hits})
                ctx = wrap_untrusted(ctx, source="project context")
        except Exception as exc:  # noqa: BLE001
            log.info("injection guard skipped: %s", exc)

        # P2-5 — test-first rigor: steer build/edit runs to add tests for new/
        # changed symbols (and characterization tests before a refactor), and
        # optionally gate 'done' on the change being tested. Flag-gated.
        _require_tests = False
        try:
            from app.core.config_loader import cfg as _cfgr
            if (kind in ("build", "edit")
                    and getattr(_cfgr.advanced_rag, "test_first_rigor", True)):
                from app.agent.testgen import TEST_FIRST_DIRECTIVE
                ctx = (ctx + "\n\n" + TEST_FIRST_DIRECTIVE) if ctx \
                    else TEST_FIRST_DIRECTIVE
                _require_tests = bool(
                    getattr(_cfgr.advanced_rag, "strict_test_gate", False))
        except Exception as exc:  # noqa: BLE001
            log.info("test-first rigor skipped: %s", exc)

        # P2-4 — long-horizon planner seed: for build/edit, split the goal into
        # an ordered TODO checklist up front and stream it as the first `todo`
        # event, so the user sees a plan from step one. The agent then maintains
        # it live via the `todo_write` tool. Best-effort, flag-gated.
        try:
            from app.core.config_loader import cfg as _cfgt
            if (kind in ("build", "edit")
                    and getattr(_cfgt.advanced_rag, "todo_list", True)
                    and getattr(_cfgt.advanced_rag, "todo_planner", True)):
                from app.agent.planner import looks_multistep, plan_todos
                # Only seed a plan for genuinely multi-step work — a fresh build
                # always; a simple one-line edit stays cheap (no extra LLM call).
                if kind == "build" or looks_multistep(body.task):
                    from app.agent.todos import (
                        clear_todos,
                        progress,
                        save_todos,
                        todos_to_dicts,
                    )
                    await asyncio.to_thread(clear_todos, workspace)
                    _todos = await asyncio.wait_for(
                        plan_todos(body.task, context=ctx), timeout=40)
                    if _todos:
                        await asyncio.to_thread(save_todos, workspace, _todos)
                        _d, _t = progress(_todos)
                        yield _frame("todo", {"todos": todos_to_dicts(_todos),
                                              "done": _d, "total": _t})
        except Exception as exc:  # noqa: BLE001 — planning is an optimization
            log.info("todo planner skipped: %s", exc)

        def _stream():
            return run_goal(
                body.task, condition, workspace=workspace,
                mode=body.mode, max_rounds=body.max_rounds,
                max_steps=body.max_steps, context=ctx,
                require_tests=_require_tests,
            ) if kind in ("build", "edit") else run_agent(
                body.task, workspace=workspace, mode=body.mode,
                max_steps=body.max_steps, history=body.history,
                images=body.images[:10], context=ctx,
            )

        factory = get_session_factory()
        sid = body.conversation_id
        seq = turn = 0
        last = time.monotonic()
        final_text = ""

        # No DB → still run the agent (best-effort), just without persistence.
        if factory is None:
            try:
                async for evt in _stream():
                    now = time.monotonic()
                    if isinstance(evt, dict):
                        evt["_elapsed_ms"] = int((now - last) * 1000)
                    last = now
                    _metrics.on_event(evt)
                    et0 = evt.get("type")
                    if et0 == "goal_done":
                        _sig["goal_passed"] = bool(evt.get("passed"))
                        _sig["rounds"] = int(evt.get("rounds") or _sig["rounds"])
                    elif et0 == "goal_eval" and evt.get("verify"):
                        _sig["verify_attempted"] = True
                        _sig["verify_ok"] = bool(evt.get("passed"))
                    elif et0 == "error":
                        _sig["had_error"] = True
                    if et0 == "final":
                        final_text = str(evt.get("message", ""))
                    yield _emit(evt.get("type", "event"), evt)
                d = await _diff(workspace)
                if _red:
                    d = redact_secrets(d)
                if d:
                    yield _emit("diff", {"summary": d})
                _sem = await _semantic(workspace)
                if _sem:
                    yield _frame("semantic_diff", {"items": _sem})
                _surface = await _tests(workspace)
                if _surface is not None:
                    _sig["tests_added"] = _surface.tests_added
                    _sig["untested_changes"] = _surface.untested_count
                    if _surface.added or _surface.changed:
                        yield _frame("tests", _surface.to_dict())
                review_dicts, conf, prov, verdict = await _trust_pass(
                    body.task, final_text, d, ctx_files, _sig)
                if review_dicts:
                    yield _frame("review", {"risks": review_dicts})
                if verdict is not None and verdict.votes:
                    yield _frame("cross_verify", verdict.to_dict())
                if conf is not None:
                    yield _frame("confidence",
                                 {"band": conf.band, "score": conf.score,
                                  "reasons": conf.reasons})
                if prov:
                    yield _frame("provenance", {"items": prov})
                _metrics.finalize(confidence=(conf.band if conf else None))
                yield _frame("metrics", _metrics.to_dict())
                await _brain_finalize(workspace, body.task, kind, _metrics, final_text)
                if kind in ("build", "edit"):
                    _gres = await _gitflow(workspace, body.task, final_text)
                    if _gres is not None and _gres.committed:
                        yield _frame("git", _gres.to_dict())
            except Exception as exc:  # noqa: BLE001
                log.exception("chat agent-run failed (no-persist)")
                yield _frame("error", {"detail": str(exc)})
            finally:
                _active_runs -= 1
                sem.release()
            yield "event: end\ndata: {}\n\n"
            return

        try:
            async with factory() as ws:
                steps = AgentStepRepo(ws)
                sess = SessionRepo(ws)
                msgs = MessageRepo(ws)
                persist = True

                async def _commit() -> None:
                    nonlocal persist
                    if not persist:
                        return
                    try:
                        await ws.commit()
                    except Exception:  # noqa: BLE001
                        log.warning("chat agent-step persist failed; disabling",
                                    exc_info=True)
                        persist = False
                        try:
                            await ws.rollback()
                        except Exception:  # noqa: BLE001
                            pass

                # The conversation IS a Session row; resume seq/turn after any
                # prior agent steps on it (Flow C follow-ups).
                if await sess.get(sid) is None:
                    yield _frame("error",
                                 {"detail": "Conversation not found"})
                    yield "event: end\ndata: {}\n\n"
                    return
                seq, turn = await steps.next_seq(sid)

                um = await msgs.append(session_id=sid, role="user",
                                       content=body.task)
                await sess.record_message(sid)
                await _commit()
                if persist:
                    await steps.append(
                        session_id=sid, seq=seq, turn=turn, event="user",
                        message_id=str(um.id),
                        payload={"type": "user", "text": body.task,
                                 "images": body.images[:10],
                                 "files": body.files, "kind": kind})
                    await _commit()
                    seq += 1

                async for evt in _stream():
                    et = evt.get("type", "event")
                    now = time.monotonic()
                    elapsed = int((now - last) * 1000)
                    last = now
                    if isinstance(evt, dict):
                        evt["_elapsed_ms"] = elapsed
                    if _red and isinstance(evt, dict):
                        evt = redact_event(evt)
                    _metrics.on_event(evt)
                    # Capture trust signals (Phase 8) from the goal-loop events.
                    if et == "goal_done":
                        _sig["goal_passed"] = bool(evt.get("passed"))
                        _sig["rounds"] = int(evt.get("rounds") or _sig["rounds"])
                    elif et == "goal_eval":
                        _v = evt.get("verify")
                        if _v:
                            _sig["verify_attempted"] = True
                            _sig["verify_ok"] = bool(evt.get("passed"))
                    elif et == "error":
                        _sig["had_error"] = True
                    yield _frame(et, evt)
                    if et == "final":
                        final_text = str(evt.get("message", ""))
                    if not persist:
                        continue
                    if et != "final":
                        await steps.append(
                            session_id=sid, seq=seq, turn=turn, event=et,
                            step=evt.get("step"), tool=evt.get("tool"),
                            kind=evt.get("kind"), elapsed_ms=elapsed,
                            payload=evt)
                        await _commit()
                        seq += 1

                # Result delivery (§8.9): diff vs baseline + downloadable zip.
                diff = await _diff(workspace)
                if _red:
                    diff = redact_secrets(diff)
                if diff:
                    yield _frame("diff", {"summary": diff})
                _sem = await _semantic(workspace)
                if _sem:
                    yield _frame("semantic_diff", {"items": _sem})
                _surface = await _tests(workspace)
                if _surface is not None:
                    _sig["tests_added"] = _surface.tests_added
                    _sig["untested_changes"] = _surface.untested_count
                    if _surface.added or _surface.changed:
                        yield _frame("tests", _surface.to_dict())

                # Quality & trust (Phase 8): adversarial review → confidence band
                # + provenance. Phase 12: cross-model council verdict.
                review_dicts, conf, prov, verdict = await _trust_pass(
                    body.task, final_text, diff, ctx_files, _sig)
                if review_dicts:
                    yield _frame("review", {"risks": review_dicts})
                if verdict is not None and verdict.votes:
                    yield _frame("cross_verify", verdict.to_dict())
                if conf is not None:
                    yield _frame("confidence",
                                 {"band": conf.band, "score": conf.score,
                                  "reasons": conf.reasons})
                if prov:
                    yield _frame("provenance", {"items": prov})

                _metrics.finalize(confidence=(conf.band if conf else None))
                yield _frame("metrics", _metrics.to_dict())
                await _brain_finalize(workspace, body.task, kind, _metrics,
                                      final_text)

                # P2-7 — git workflow: branch + commit (+ opt-in push/PR). Runs
                # AFTER diff/semantic/tests (committing moves HEAD).
                _gres = None
                if kind in ("build", "edit"):
                    _gres = await _gitflow(workspace, body.task, final_text)
                    if _gres is not None and _gres.committed:
                        yield _frame("git", _gres.to_dict())

                if persist:
                    am = await msgs.append(
                        session_id=sid, role="assistant",
                        content=final_text or "(no summary)",
                        intent="agent_build" if kind == "build"
                        else "agent_edit",
                        confidence=(conf.score if conf is not None else None),
                        sources={"document": True, "format": "zip",
                                 "workspace": True,
                                 "download": f"/api/chat/agent-run/{sid}/download",
                                 "diff": diff or None,
                                 "confidence": (conf.band if conf else None),
                                 "provenance": prov or None,
                                 "review": review_dicts or None,
                                 "metrics": _metrics.to_dict(),
                                 "cross_verify": (verdict.to_dict()
                                                  if verdict is not None
                                                  and verdict.votes else None),
                                 "semantic_diff": _sem or None,
                                 "tests": (_surface.to_dict() if _surface
                                           is not None and
                                           (_surface.added or _surface.changed)
                                           else None),
                                 "git": (_gres.to_dict() if _gres is not None
                                         and _gres.committed else None),
                                 "security": ({"injection": _sec_hits}
                                              if _sec_hits else None)})
                    await sess.record_message(sid)
                    await steps.append(
                        session_id=sid, seq=seq, turn=turn, event="final",
                        message_id=str(am.id), elapsed_ms=0,
                        payload={"type": "final", "message": final_text,
                                 "diff": diff})
                    # Observability ledger row (reuses `agent_runs`): one row
                    # per chat agent-run with latency/tokens/tool-calls/verify.
                    try:
                        from storage.repos import AgentRunRepo
                        _md = _metrics.to_dict()
                        _arr = AgentRunRepo(ws)
                        _run = await _arr.start(
                            agent="chat_agent", session_id=sid,
                            message_id=str(am.id),
                            input_summary={"task": body.task[:500],
                                           "kind": kind})
                        await _arr.finish(
                            _run.id,
                            status=("ok" if _metrics.success else "error"),
                            output_summary=_md,
                            tokens=_metrics.out_tokens)
                    except Exception:  # noqa: BLE001
                        pass
                    await _commit()
                    yield _frame("done", {"message_id": str(am.id),
                                          "conversation_id": sid})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("chat agent-run failed")
            yield _frame("error", {"detail": str(exc)})
        finally:
            _active_runs -= 1
            sem.release()

        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agent-run/{conversation_id}/download")
async def chat_agent_download(conversation_id: str) -> Response:
    """The conversation's (modified) workspace, packaged as a .zip."""
    from app.agent_workspace import package_workspace, workspace_exists

    if not workspace_exists(conversation_id):
        raise HTTPException(404, "No workspace for this conversation")
    data = await asyncio.to_thread(package_workspace, conversation_id)
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition":
                f'attachment; filename="project-{conversation_id[:8]}.zip"',
        },
    )


@router.get("/agent-run/{conversation_id}/metrics")
async def chat_agent_metrics(conversation_id: str) -> dict:
    """Observability (Phase 9): per-run + aggregate metrics for a conversation's
    chat agent-runs, from the `agent_runs` ledger."""
    from app.obs.metrics import aggregate_runs
    from storage.db import get_session_factory
    from storage.repos import AgentRunRepo

    factory = get_session_factory()
    if factory is None:
        return {"runs": 0, "items": []}
    async with factory() as ws:
        rows = await AgentRunRepo(ws).list_for_session(conversation_id)
    items = [
        {
            "agent": r.agent,
            "status": r.status,
            "tokens": int(r.tokens or 0),
            "started_at": (r.started_at.isoformat() if r.started_at else None),
            "output_summary": r.output_summary or {},
        }
        for r in rows if r.agent == "chat_agent"
    ]
    return {"summary": aggregate_runs(items), "items": items}
