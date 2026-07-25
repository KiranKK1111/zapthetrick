"""Verify-while-streaming for plain chat code answers (vNext §3.1).

The Solve/image path already sandbox-verifies its solution (routes_attachments);
this brings the SAME honest verdict to an ordinary "write me X in language Y"
chat answer. Chat reveals the draft immediately (the user is already reading),
then this lane — concurrently — parses (tree-sitter pre-gate), compiles + runs
the code against any visible examples, and either attaches a passing verdict or
hot-swaps a repaired block with an honest note. Only Live gates before reveal;
Chat never blocks the draft on verification.

This module is a thin, reusable orchestration layer over the existing
`app.codeintel.solution_verify.verify_and_maybe_repair` — it owns the fence
extraction, the sticky-language resolution ladder, the tree-sitter pre-gate, and
the deadline + cancellation + progress pattern (previously inlined only in the
image path). It is an **async generator** so the caller maps its events straight
onto its SSE stream; it **never raises** (a failure yields an un-ran result and
the turn is byte-identical to today).

Flag-gated by `code_solver.verify_chat_code` (default OFF).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

_FENCE = re.compile(r"```([\w+#.\-]*)[ \t]*\n(.*?)```", re.DOTALL)


@dataclass
class ChatVerifyResult:
    """Outcome of the lane. `ran` is False when verification was skipped (no
    code, un-gateable, cancelled, timed out, or errored) — the caller then leaves
    the answer untouched. When `ran`, `delta` is the text to append + stream and
    `updated_text` is the new full answer."""
    updated_text: str
    ran: bool = False
    delta: str = ""
    suffix: str = ""
    fixed_code: str | None = None


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.code_solver, "verify_chat_code", False))
    except Exception:  # noqa: BLE001
        return False


def _limits() -> tuple[int, float]:
    try:
        from app.core.config_loader import cfg
        return (int(getattr(cfg.code_solver, "verify_chat_max_repairs", 2)),
                float(getattr(cfg.code_solver, "verify_chat_deadline_s", 180.0)))
    except Exception:  # noqa: BLE001
        return (2, 180.0)


def extract_primary_code(text: str) -> tuple[str, str] | None:
    """The first substantial fenced code block → ``(code, fence_label)``, or None.
    "Substantial" = ≥ 2 non-blank lines, so a one-line inline snippet or an empty
    fence is skipped."""
    for m in _FENCE.finditer(text or ""):
        label = (m.group(1) or "").strip()
        body = (m.group(2) or "").rstrip("\n")
        if len([ln for ln in body.splitlines() if ln.strip()]) >= 2:
            return body, label
    return None


def resolve_language(*, fence_label: str | None = None,
                     explicit: str | None = None,
                     sticky: str | None = None) -> str | None:
    """The sticky-language resolution ladder (§3.1): an explicit user mention
    wins, then the answer's own fence label, then the conversation's last
    confirmed language. None → nothing to verify against (caller skips)."""
    for cand in (explicit, fence_label, sticky):
        c = (cand or "").strip()
        if c:
            return c
    return None


def should_verify(answer_text: str, *, language_label: str | None) -> bool:
    """Gate: there is a substantial code block AND it's real code in a language
    we can actually run. Pure + cheap (tree-sitter pre-gate); never raises."""
    if not language_label:
        return False
    got = extract_primary_code(answer_text)
    if not got:
        return False
    code, _ = got
    try:
        from app.codeintel import pregate
        return pregate.looks_like_code(code, language_label)
    except Exception:  # noqa: BLE001
        return True


def plan(answer_text: str, *, question: str | None = None,
         sticky: str | None = None) -> str | None:
    """Decide whether this answer is worth verifying and in which language.
    Returns the resolved language label, or None to skip. Pure + fail-open."""
    got = extract_primary_code(answer_text)
    if not got:
        return None
    _code, fence_label = got
    explicit = None
    try:
        from app.codeintel.code_language import requested_language
        explicit = requested_language(question or "")
    except Exception:  # noqa: BLE001
        explicit = None
    lang = resolve_language(fence_label=fence_label, explicit=explicit,
                            sticky=sticky)
    if not lang:
        return None
    if not should_verify(answer_text, language_label=lang):
        return None
    # Stage-4 §3.5 toolchain prefetch: warm this language's runtime OFF the
    # request path so the verifier that follows starts hot. Best-effort.
    try:
        from app.sandbox import pool as _pool
        _pool.prefetch_toolchain(lang)
    except Exception:  # noqa: BLE001
        pass
    return lang


async def verify_stream(
    problem: str,
    answer_text: str,
    *,
    language_label: str,
    is_cancelled: Callable[[], bool] | None = None,
    cancel_sandbox: Callable[[], None] | None = None,
    max_repairs: int | None = None,
    deadline_s: float | None = None,
):
    """Run the verify lane, yielding ``("stage", name)`` progress events as it
    goes; the FINAL yield is ``("result", ChatVerifyResult)``. Encapsulates the
    queue-drain + deadline + Stop-cancellation pattern. Never raises.

    The caller should tag the sandbox run-group (for Stop) BEFORE iterating, so
    the internal `ensure_future` captures it; `cancel_sandbox` is invoked to kill
    an in-flight exec on Stop/timeout.
    """
    result = ChatVerifyResult(updated_text=answer_text)
    _max = max_repairs if max_repairs is not None else _limits()[0]
    _deadline_s = deadline_s if deadline_s is not None else _limits()[1]

    got = extract_primary_code(answer_text)
    if not got:
        yield ("result", result)
        return
    code, _label = got

    # Tree-sitter pre-gate: don't spend a compile on prose-in-a-fence, and hand
    # a precise syntax error straight to repair feedback when one is definite.
    try:
        from app.codeintel import pregate
        if not pregate.looks_like_code(code, language_label):
            yield ("result", result)
            return
    except Exception:  # noqa: BLE001
        pass

    try:
        from app.codeintel.solution_verify import (
            _fence_tag as _ftag,
            verify_and_maybe_repair,
        )
    except Exception:  # noqa: BLE001 — verifier unavailable → skip, unchanged
        yield ("result", result)
        return

    _q: asyncio.Queue = asyncio.Queue()

    async def _on_stage(name: str) -> None:
        await _q.put(name)

    task: asyncio.Future = asyncio.ensure_future(
        verify_and_maybe_repair(problem, answer_text, language_label,
                                on_stage=_on_stage, max_repairs=_max))
    deadline = time.monotonic() + _deadline_s
    timed_out = False
    try:
        while not task.done():
            if is_cancelled and is_cancelled():
                task.cancel()
                if cancel_sandbox:
                    try:
                        cancel_sandbox()
                    except Exception:  # noqa: BLE001
                        pass
                yield ("result", result)      # cancelled → unchanged
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                task.cancel()
                if cancel_sandbox:
                    try:
                        cancel_sandbox()
                    except Exception:  # noqa: BLE001
                        pass
                timed_out = True
                break
            try:
                name = await asyncio.wait_for(_q.get(),
                                              timeout=min(0.6, remaining))
                yield ("stage", name)
            except asyncio.TimeoutError:
                pass
        while not _q.empty():
            yield ("stage", _q.get_nowait())
    except Exception:  # noqa: BLE001 — verification never breaks a turn
        pass

    suffix, fixed = "", None
    if task.done() and not task.cancelled():
        try:
            suffix, fixed = task.result()
        except Exception:  # noqa: BLE001
            suffix, fixed = "", None

    if not suffix:
        # Timed out / errored before a verdict — leave the answer untouched but
        # never silently claim success (the caller may add an honest ℹ️ note).
        if timed_out:
            result.suffix = ("\n\n---\nℹ️ Sandbox verification timed out — the "
                             "code above is unchanged.")
        yield ("result", result)
        return

    # Streamed branch: the draft is already on screen, so append the repaired
    # block (if any) + the verdict — the honest, un-rewindable equivalent of a
    # hot-swap. A passing verdict just appends the ✅ line.
    try:
        fence = _ftag(language_label)
    except Exception:  # noqa: BLE001
        fence = (language_label or "").lower()
    delta = ((f"\n\n```{fence}\n{fixed}\n```" if fixed else "") + suffix)
    result.ran = True
    result.suffix = suffix
    result.fixed_code = fixed
    result.delta = delta
    result.updated_text = (answer_text + delta).strip()
    yield ("result", result)


__all__ = ["ChatVerifyResult", "enabled", "extract_primary_code",
           "resolve_language", "should_verify", "plan", "verify_stream"]
