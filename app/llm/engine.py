"""Routing engine — the retry loop over `route_request` + adapters.

Ported from the orchestration in freellmapi's `routes/proxy.ts`. This is
what `LLMClient`'s `auto` path calls. For each request it:

  1. resolves a sticky preferred model (so a conversation doesn't hop
     models mid-thread), then
  2. loops up to MAX_RETRIES: route → call adapter → on a retryable error
     (429/503/timeout) penalize the model, cool down the key, skip it, and
     try the next model in the chain; on success record usage + sticky.

Streaming retries happen *before the first token is emitted* — once bytes
are on the wire we can't un-send them, so a mid-stream failure ends the
stream (the caller surfaces an error event).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from app.llm import ratelimit, router
from app.llm.providers import ProviderError, get_adapter
from app.obs import failure_kb as _failure_kb
from app.obs import failure_taxonomy as _failure_taxonomy
from app.obs import recovery as _recovery
from app.perceived.health import health as _ph

log = logging.getLogger(__name__)

# Backstop cap; the effective retry count is cfg.llm.routing.max_retries
# (read per-call via `_max_retries`). Kept as a hard ceiling so a bad config
# can't loop forever.
MAX_RETRIES = 20
_STICKY_TTL_S = 30 * 60


def _max_retries() -> int:
    """Effective fallback-attempt cap from config (bounded by MAX_RETRIES)."""
    try:
        from app.core.config_loader import cfg
        n = int(getattr(cfg.llm.routing, "max_retries", 6) or 6)
        return max(1, min(n, MAX_RETRIES))
    except Exception:  # noqa: BLE001
        return 6


def _first_token_deadline() -> float:
    """Seconds to wait for a model's FIRST token before abandoning it and
    falling to the next. 0 → disabled (unbounded). This is the core guard
    against a provider that connects then hangs."""
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.llm.routing, "first_token_deadline_s", 7.0) or 0.0)
    except Exception:  # noqa: BLE001
        return 7.0


def _no_route_retry() -> tuple[int, float]:
    """(#retries, base backoff seconds) for a TRANSIENT no-route — all models
    momentarily rate-limited by a concurrent burst. A short backoff lets the 60s
    RPM/TPM windows recover and de-syncs the burst so the retry finds a free
    model instead of erroring straight to the UI."""
    try:
        from app.core.config_loader import cfg
        r = cfg.resilience
        n = int(getattr(r, "route_no_route_retries", 3) or 0)
        ms = int(getattr(r, "route_backoff_ms", 400) or 0)
        return max(0, n), max(0.0, ms / 1000.0)
    except Exception:  # noqa: BLE001
        return 3, 0.4


def _recovery_on() -> bool:
    """Whether the RECOVERY PLAN drives the retry gate (Phase 4 #8) and its
    outcome feeds the failure KB (Phase 7 #3). Default ON; set
    `resilience.recovery_planner: false` to fall back to the pure
    `exc.retryable` gate."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.resilience, "recovery_planner", True))
    except Exception:  # noqa: BLE001 — fail-open to the new behaviour
        return True


def _recovery_backoff_cap_ms() -> int:
    """Ceiling on the backoff the plan asks for. The planner's COOLDOWN_WAIT
    curve grows 1s→2s→4s…, which is right when you'd retry the SAME endpoint —
    but the engine skips the rate-limited (model,key) and falls to a *different*
    model on the next attempt, so a long wait would be pure added latency (this
    is the live-answer hot path). The genuine "everything is rate-limited" wait
    lives in `_no_route_retry`. Default cap 1s; 0 disables plan backoff."""
    try:
        from app.core.config_loader import cfg
        return max(0, int(getattr(cfg.resilience, "recovery_backoff_max_ms",
                                  1000) or 0))
    except Exception:  # noqa: BLE001
        return 1000


# Failure classes whose TERMINAL recovery must never veto provider fallback in
# the engine. `internal_error` is the taxonomy's catch-all — every provider
# message it doesn't recognize (a bare "HTTP 502", "bad gateway", …) lands there
# with recovery=escalate; `network_error` matches loose signatures like "offline",
# which providers also use for a single dead model. Honoring those as terminal
# would abort the whole fallback chain on a failure the NEXT model could serve,
# so for these two we defer to `exc.retryable` (the transport-level truth).
_NO_VETO_CLASSES = {"internal_error", "network_error"}


def _plan_recovery(failure, *, attempt: int, max_attempts: int):
    """Classified, budgeted recovery decision for this failure (or None when the
    planner is off/unavailable → the loop keeps its legacy `exc.retryable` gate)."""
    if not _recovery_on():
        return None
    try:
        plan = _recovery.plan_recovery(failure, attempt=attempt,
                                       max_attempts=max_attempts)
        log.debug("recovery[%s]: action=%s (taking %s) retry=%s backoff=%dms — %s",
                  plan.failure_id, plan.action, plan.effective_action,
                  plan.should_retry, plan.backoff_ms, plan.rationale)
        return plan
    except Exception:  # noqa: BLE001 — planning must never break a call
        return None


def _plan_vetoes_retry(plan) -> bool:
    """True when the recovery plan says this failure is terminal — no model in
    the chain can recover it, so stop instead of burning the retry budget.

    Budget exhaustion is deliberately NOT a veto: the loop's own cap already
    ends the turn with the usual `NoRouteAvailable("Exhausted N …")`, and callers
    (e.g. `stream_with_continuation`) branch on that type. Fail-open → no veto."""
    try:
        if plan is None or not _recovery_on():
            return False
        if plan.failure_id in _NO_VETO_CLASSES:
            return False
        return bool(plan.terminal)
    except Exception:  # noqa: BLE001 — the gate must never break a call
        return False


def _plan_backoff_s(plan, action: str) -> float:
    """Seconds to wait before the plan's retry (0 when disabled/terminal)."""
    try:
        if plan is None or not _recovery_on() or not plan.should_retry:
            return 0.0
        ms = _recovery.backoff_for(action, plan.attempt) if action != plan.action \
            else plan.backoff_ms
        cap = _recovery_backoff_cap_ms()
        return max(0, min(int(ms or 0), cap)) / 1000.0
    except Exception:  # noqa: BLE001
        return 0.0


class _RecoveryTracker:
    """Closes the failure-KB learn loop (Phase 7 #3).

    The KB can only recommend a recovery if it's told whether one WORKED. The
    engine is the only place that knows: it holds the action taken for the most
    recent classified failure and resolves it when the request finishes —
    succeeded on a later attempt → that action recovered the failure; another
    failure / exhausted retries / a raise → it didn't.

    Fail-open by construction: every method swallows its own errors, so KB
    bookkeeping can never break the LLM call it's observing.
    """

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending: tuple[str, str] | None = None

    def note(self, failure_id: str, action: str) -> None:
        """We're about to attempt `action` to recover `failure_id`."""
        try:
            # A fresh classified failure means the PREVIOUS recovery didn't work.
            self.resolve(False)
            self._pending = (failure_id, action)
        except Exception:  # noqa: BLE001
            self._pending = None

    def resolve(self, success: bool) -> None:
        """The attempted recovery (if any) did/didn't get us to a good answer."""
        try:
            pending, self._pending = self._pending, None
            if pending and _recovery_on():
                _failure_kb.record_outcome(pending[0], pending[1], bool(success))
        except Exception:  # noqa: BLE001 — never break the call over bookkeeping
            self._pending = None


# session_key -> (model_db_id, expiry_epoch)
_sticky: dict[str, tuple[int, float]] = {}

# Most-recent model that actually STREAMED, per session (+ a "global" slot) so
# the UI can show the real per-turn model instead of the static health-check
# one. Set on the first streamed chunk.
_last_model: dict[str, str] = {}
# Parallel to _last_model but the model_id (stable routing key, not display name)
# — Phase-2 semantic routing records outcomes by this so scoring keys match.
_last_model_id: dict[str, str] = {}

# P2-10: the model db id that most recently SUCCEEDED at a given difficulty, so
# routing can prefer what's been working (extends learning-lite into routing).
_diff_success: dict[str, int] = {}


def _prefer_recent_on() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.advanced_rag, "prefer_recent_model", True))
    except Exception:  # noqa: BLE001
        return True


def _record_diff_success(difficulty: str | None, model_db_id: int) -> None:
    if difficulty and _prefer_recent_on():
        _diff_success[difficulty] = model_db_id


def _diff_pref(difficulty: str | None) -> int | None:
    if not difficulty or not _prefer_recent_on():
        return None
    return _diff_success.get(difficulty)


def _apply_reasoning_mode(difficulty: str) -> str:
    """P5 #26: apply the user's Fast/Balanced/Thorough reasoning mode to the
    routing difficulty band. BALANCED (the default) leaves it unchanged, so this
    is byte-identical to today unless a mode was explicitly set for the request.
    Fail-open."""
    try:
        from app.llm.reasoning_mode import effective_difficulty
        return effective_difficulty(difficulty)
    except Exception:  # noqa: BLE001
        return difficulty


def _apply_operational_mode(options: dict) -> None:
    """P5 #27: stamp reproducible sampling params (temperature 0 + fixed seed)
    onto the provider options when reproducible mode is on. No-op by default.
    Fail-open."""
    try:
        from app.llm.operational import apply_to_options
        apply_to_options(options)
    except Exception:  # noqa: BLE001
        pass


def _record_last_model(session_key: str | None, display_name: str,
                       model_id: str | None = None) -> None:
    _last_model["global"] = display_name
    if session_key:
        _last_model[session_key] = display_name
    if model_id:
        _last_model_id["global"] = model_id
        if session_key:
            _last_model_id[session_key] = model_id


def get_last_model(session_key: str | None = None) -> str | None:
    """The model that most recently produced a streamed answer (for this
    session if known, else the global last)."""
    if session_key and session_key in _last_model:
        return _last_model[session_key]
    return _last_model.get("global")


def record_winner_model(session_key: str | None, display_name: str | None,
                        model_id: str | None) -> None:
    """Public: re-assert the model that actually answered (speculation winner),
    overwriting any last-model an in-flight losing draft may have stamped."""
    if display_name:
        _record_last_model(session_key, display_name, model_id)


def get_last_model_id(session_key: str | None = None) -> str | None:
    """The model_id (routing key) that most recently answered — for keying
    learned-outcome records consistently with scoring."""
    if session_key and session_key in _last_model_id:
        return _last_model_id[session_key]
    return _last_model_id.get("global")


# Validate the first slice of a model's output for degeneration before we
# commit to it (so a broken model can be skipped without showing garbage).
# Kept small so the user sees the first words fast — 96 chars is enough to
# catch <unk>/replacement-token gibberish and a whitespace-free mash, while
# cutting perceived time-to-first-token ~2.5x vs the old 240.
_DEGEN_HEAD = 96


def _looks_degenerate(text: str) -> bool:
    """True when model output looks broken: unknown/replacement tokens, or a
    very long whitespace-free 'word' (gibberish mash), or near-zero spacing."""
    if not text:
        return False
    if "<unk>" in text or text.count("\ufffd") >= 3:
        return True
    longest = max((len(w) for w in text.split()), default=0)
    if longest >= 50:
        return True
    if len(text) >= 400:
        ws = sum(1 for c in text if c.isspace())
        if ws / len(text) < 0.03:
            return True
    return False


def estimate_tokens(messages: list[dict]) -> int:
    """Rough input-token estimate (chars/4), matching freellmapi."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(seg.get("text", "")) for seg in content if isinstance(seg, dict))
    return max(1, total // 4)


def _real_total_tokens(est: int, produced_chars: int) -> int:
    """Real prompt+completion tokens from the provider's `usage` when available,
    else the chars//4 estimate (G6.1). Task-local via `app.llm.usage`."""
    try:
        from app.llm import usage as _usage
        pt, ct, tt = _usage.tokens()
        if tt:
            return max(1, tt)
        prompt = pt if pt is not None else est
        completion = ct if ct is not None else max(0, produced_chars // 4)
        return max(1, prompt + completion)
    except Exception:  # noqa: BLE001 — accounting must never break a call
        return max(1, est + produced_chars // 4)


def _get_sticky(session_key: str | None) -> int | None:
    if not session_key:
        return None
    entry = _sticky.get(session_key)
    if not entry:
        return None
    model_db_id, expiry = entry
    if time.time() > expiry:
        _sticky.pop(session_key, None)
        return None
    return model_db_id


def _set_sticky(session_key: str | None, model_db_id: int) -> None:
    if session_key:
        _sticky[session_key] = (model_db_id, time.time() + _STICKY_TTL_S)


_DAY_MS = 24 * 60 * 60 * 1000
# A 404 that isn't a recognised dead-model marker (a provider's "function not
# found for account" / a misconfigured endpoint) rarely recovers within a turn —
# but a one-off gateway/CDN 404 does, so cool it down for a few minutes rather
# than disabling a possibly-healthy model.
_NOT_FOUND_COOLDOWN_MS = 5 * 60 * 1000


def _penalize(route: "router.RouteResult", status: int | None, dead: bool) -> None:
    """Record a retryable failure against the chosen model+key for this request.

    A `dead` model (gone/invalid id) is also disabled in the DB by the caller;
    we still cool it down for a day as a belt-and-suspenders. A 429 / 413 gets
    the escalating cooldown so it recovers when the provider window resets; a
    stray 404 gets a short cooldown so it isn't re-picked next request.
    """
    router.record_rate_limit_hit(route.model_db_id)
    if dead or status in (402, 403):
        # Gone model, out of credits, or behind a credit-card/verification wall —
        # none recover soon, so cool it down for a day instead of retrying it on
        # every request.
        ratelimit.set_cooldown(route.platform, route.model_id, route.key_id, _DAY_MS)
    elif status in (429, 413):
        # 429 (rate limit) AND 413 ("request too large" — a free-tier per-minute
        # TPM cap this turn's tokens exceed) both recover when the provider's
        # window resets. Cooling the model down stops us re-picking it and
        # burning the whole retry budget on a limit it can't satisfy right now.
        ratelimit.set_cooldown(route.platform, route.model_id, route.key_id)
    elif status == 404:
        # Not a recognised dead-model marker (see classify_error) — a
        # misconfigured endpoint / "not found for account". Short cooldown so a
        # broken model doesn't waste a routing attempt on every single request.
        ratelimit.set_cooldown(route.platform, route.model_id, route.key_id,
                               _NOT_FOUND_COOLDOWN_MS)


async def _disable_model(model_db_id: int) -> None:
    """Permanently remove a gone/invalid model from the catalog.

    We DELETE the model row (and its fallback row) rather than just flipping
    `enabled` off. Two reasons: (1) with `routing.route_all_models` on, a
    merely-disabled model still re-enters routing as an "extra" candidate, so
    disabling wouldn't actually stop retrying a dead id; (2) deletion lets
    discovery genuinely re-add it (as a fresh disabled row) if the provider
    brings the model back — the "re-discovery re-adds it" contract that a
    disabled-but-present row silently broke. Best-effort.
    """
    from sqlalchemy import delete

    from storage.db import get_session_factory
    from storage.models import LLMFallbackConfig, LLMModel

    factory = get_session_factory()
    if factory is None:
        return
    try:
        async with factory() as session:
            await session.execute(
                delete(LLMFallbackConfig)
                .where(LLMFallbackConfig.model_db_id == model_db_id)
            )
            await session.execute(
                delete(LLMModel).where(LLMModel.id == model_db_id)
            )
            await session.commit()
        log.info("routing: deleted dead model id=%s (removed from catalog)",
                 model_db_id)
    except Exception as exc:  # noqa: BLE001 — never fail a request over cleanup
        log.debug("could not delete dead model %s: %s", model_db_id, exc)


def _messages_have_images(messages: list[dict]) -> bool:
    """True if any message carries an image (OpenAI multipart `image_url`)."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _resolve_task_category(messages: list[dict], options: dict,
                           difficulty: str):
    """Capability-aware routing hint (intelligent-model-routing R2/R3). Returns
    (task_category, needs_tool, needs_json). When `capability_routing` is off →
    (None, ...) so `route_request` is byte-for-byte today's. Self-classifies from
    the last user message when the caller didn't pass an explicit category.
    Never raises."""
    task_category = options.pop("task_category", None)
    needs_tool = bool(options.pop("needs_tool", False))
    needs_json = bool(options.pop("needs_json", False))
    if task_category is None:
        try:
            from app.core.config_loader import get_config
            if bool(getattr(get_config().routing, "capability_routing", False)):
                from app.llm.task_class import classify_task
                last_user = ""
                for m in reversed(messages):
                    c = m.get("content")
                    if m.get("role") == "user" and isinstance(c, str):
                        last_user = c
                        break
                task_category = classify_task(last_user, None, difficulty)
        except Exception:  # noqa: BLE001 — fail-open to no capability hint
            task_category = None
    return task_category, needs_tool, needs_json


async def route_and_complete(
    messages: list[dict],
    options: dict,
    *,
    session_key: str | None = None,
    preferred_model_db_id: int | None = None,
) -> tuple[str, "router.RouteResult"]:
    """Non-streaming completion with fallback. Returns (text, route)."""
    est = estimate_tokens(messages)
    vision = _messages_have_images(messages)
    # Difficulty drives capability-aware routing; pop these so they aren't sent
    # to the provider as chat params.
    difficulty = options.pop("difficulty", "standard")
    difficulty = _apply_reasoning_mode(difficulty)
    _apply_operational_mode(options)
    avoid = options.pop("avoid_model_db_id", None)
    # Phase 2 semantic routing: the query embedding (from the Understanding pass)
    # keys the learned-success signal. Popped so it's never sent to a provider.
    _query_emb = options.pop("query_embedding", None)
    # G13: redact PII/secrets before anything leaves the device for a provider —
    # the single egress choke point covers every path + retry. Fail-open.
    try:
        from app.security.egress_redact import redact_messages as _redact
        messages = _redact(messages)
    except Exception:  # noqa: BLE001 — redaction must never break a call
        pass
    preferred = preferred_model_db_id or _get_sticky(session_key) \
        or _diff_pref(difficulty)
    skip: set[str] = set()
    last_err: Exception | None = None

    _task_cat, _needs_tool, _needs_json = _resolve_task_category(
        messages, options, difficulty)

    _retries = _max_retries()
    _deadline = _first_token_deadline()
    _no_route_max, _no_route_base = _no_route_retry()
    _no_route_tries = 0
    _track = _RecoveryTracker()   # failure-KB learn loop (records outcomes)
    for _attempt in range(_retries):
        try:
            route = await router.route_request(
                est, skip, preferred, require_vision=vision, min_context=est,
                difficulty=difficulty, avoid_model_db_id=avoid,
                task_category=_task_cat, needs_tool=_needs_tool,
                needs_json=_needs_json, query_embedding=_query_emb,
            )
        except router.NoRouteAvailable as exc:
            # Transient exhaustion under a concurrent burst — back off + retry
            # route selection instead of erroring straight out (see _no_route_retry).
            last_err = exc
            # Persistent no-route (no key / no provider configured) → fail fast;
            # only a TRANSIENT exhaustion (candidates exist but are all
            # momentarily rate-limited) is worth a backoff + retry.
            if not getattr(exc, "transient", False):
                _track.resolve(False)
                raise
            _no_route_tries += 1
            if (_no_route_max <= 0 or _no_route_tries > _no_route_max
                    or _attempt == _retries - 1):
                _track.resolve(False)
                raise
            _jitter = (time.monotonic() * 1000.0) % 1.0 * _no_route_base
            await asyncio.sleep(_no_route_base * _no_route_tries + _jitter)
            continue
        adapter = get_adapter(route.platform)
        _t0 = time.monotonic()   # observed completion latency (latency-aware)
        try:
            # Bound the whole non-streamed call so a hung provider can't stall
            # the turn for its full HTTP timeout — but generously: `_deadline`
            # is the FIRST-TOKEN budget (~5s), far too short for a complete
            # long answer. Use a 90s floor so a legit long generation finishes
            # (providers like ollama/minimax are specced at 120s).
            if _deadline > 0:
                text = await asyncio.wait_for(
                    adapter.complete(route.api_key, messages, route.model_id,
                                     options),
                    timeout=max(_deadline * 3, 90.0))
            else:
                text = await adapter.complete(
                    route.api_key, messages, route.model_id, options)
        except asyncio.TimeoutError:
            log.info("routing[%s]: %s/%s timed out (no completion) — falling "
                     "back", difficulty, route.platform, route.model_id)
            _ph.record(route.model_db_id, ok=False, hard_failure=True)
            router.record_rate_limit_hit(route.model_db_id)
            # Brief cooldown so the SAME slow model isn't re-picked on the very
            # next request and stalls again (skip_keys is per-request only).
            ratelimit.set_cooldown(route.platform, route.model_id,
                                   route.key_id, 60_000)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None
            last_err = ProviderError("completion timed out")
            continue
        except ProviderError as exc:
            last_err = exc
            _ph.record(route.model_db_id, ok=False, hard_failure=True)
            try:
                _fc = _failure_taxonomy.observe(exc, where="engine.route_retry")  # classify (fail-open diagnostics)
                _failure_kb.record_occurrence(_fc.id)  # learn recurring failures (Phase 7 #3)
                _plan = _plan_recovery(_fc, attempt=_attempt + 1,
                                       max_attempts=_retries)
            except Exception:  # noqa: BLE001 — bookkeeping never breaks the call
                _plan = None
            _action = _plan.effective_action if _plan else ""
            # The PLAN decides whether a retry can help; `exc.retryable` stays a
            # hard veto (a dead model/key is never worth another attempt).
            if not exc.retryable or _plan_vetoes_retry(_plan):
                _track.resolve(False)   # the recovery we'd attempted didn't work
                raise
            if exc.permanent_dead:
                await _disable_model(route.model_db_id)
            _penalize(route, exc.status, exc.permanent_dead)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None  # don't keep forcing a failing sticky model
            # …and where the plan says "go somewhere else" (transport/auth/timeout),
            # de-prioritize the failing MODEL, not just this (model,key) pair.
            if _recovery.wants_different_route(_action):
                avoid = route.model_db_id
            if _plan is not None:
                _track.note(_plan.failure_id, _action)
            _delay = _plan_backoff_s(_plan, _action)
            if _delay > 0:
                await asyncio.sleep(_delay)
            continue
        # Empty result with no error (flaky free big-model endpoints): skip this
        # model and try the next instead of returning an empty answer.
        if not (text or "").strip():
            log.info("routing[%s]: %s/%s returned EMPTY — falling back",
                     difficulty, route.platform, route.model_id)
            router.record_rate_limit_hit(route.model_db_id)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None
            last_err = ProviderError("empty response (no tokens)")
            continue
        # Degenerate output (gibberish / <unk> tokens) — skip + penalize.
        if _looks_degenerate(text):
            log.info("routing[%s]: %s/%s returned DEGENERATE output — falling "
                     "back", difficulty, route.platform, route.model_id)
            _ph.record(route.model_db_id, ok=False, hard_failure=True)
            for _ in range(3):
                router.record_rate_limit_hit(route.model_db_id)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None
            last_err = ProviderError("degenerate output")
            continue
        router.record_success(route.model_db_id)
        # Health window: success + observed latency (feeds latency-aware ranking
        # + closes any open circuit breaker for this model).
        _ph.record(route.model_db_id, ok=True,
                   latency_s=max(0.0, time.monotonic() - _t0))
        ratelimit.record_request(route.platform, route.model_id, route.key_id)
        ratelimit.record_tokens(route.platform, route.model_id, route.key_id,
                                _real_total_tokens(est, len(text)))
        _set_sticky(session_key, route.model_db_id)
        _record_last_model(session_key, route.display_name, route.model_id)
        _record_diff_success(difficulty, route.model_db_id)
        # Success — if we got here *after* a failure, the recovery we took worked.
        # This is the half of the learn loop that teaches `best_recovery` (P7 #3).
        _track.resolve(True)
        return text, route

    _track.resolve(False)   # retries exhausted — the recovery never landed
    raise router.NoRouteAvailable(
        f"Exhausted {_retries} routing attempts. Last error: {last_err}"
    )


# Tail of already-emitted text kept for the continuation seam de-dupe (must be
# >= continuation.dedupe_seam max_overlap).
_CONT_TAIL = 240


def _resilience_cfg() -> tuple[bool, int, int]:
    """(mid_stream_continuation_on, max_continuations, seam_buffer_chars)."""
    try:
        from app.core.config_loader import cfg
        r = cfg.resilience
        return (bool(getattr(r, "mid_stream_continuation", False)),
                int(getattr(r, "max_continuations", 2)),
                int(getattr(r, "seam_buffer_chars", 200)))
    except Exception:  # noqa: BLE001 — fail-open to "off"
        return (False, 0, 200)


async def stream_with_continuation(
    messages: list[dict],
    options: dict,
    *,
    session_key: str | None = None,
    preferred_model_db_id: int | None = None,
) -> AsyncGenerator[str, None]:
    """`route_and_stream` + mid-stream failover via a continuation contract (§15).

    `route_and_stream` can only fail over *before the first token*; once bytes are
    on the wire a truncation (finish_reason 'length') or a transport drop ends the
    turn with a half answer. This wrapper adds the "always finishes" guarantee: it
    detects those cut-offs and re-prompts (see `app.llm.continuation`) with the
    partial answer so far, de-duping the seam, so the user sees one continuous
    answer. Flag-gated (`resilience.mid_stream_continuation`); off → a
    byte-for-byte passthrough of `route_and_stream`.

    A caller may force the behavior per-request via `options["mid_stream_
    continuation"]` (True/False), overriding the global flag — the answer paths
    that produce long output (coding-solve, chat synthesis) opt IN so a dropped
    or length-truncated reply still finishes, while the live path keeps it OFF
    (a re-phrased continuation there reads as an echo).
    """
    _override = options.pop("mid_stream_continuation", None)
    on, max_cont, seam_buf = _resilience_cfg()
    if _override is not None:
        on = bool(_override)
    if not on:
        async for chunk in route_and_stream(
            messages, options, session_key=session_key,
            preferred_model_db_id=preferred_model_db_id,
        ):
            yield chunk
        return

    from app.llm import continuation as _cont, usage as _usage
    from app.llm.router import NoRouteAvailable

    orig = list(messages)
    partial = ""          # cumulative answer, for building the continuation prompt
    tail = ""             # rolling tail of emitted text, for seam de-dupe
    attempts = 0

    while True:
        is_cont = attempts > 0
        msgs = _cont.build_continuation_messages(orig, partial) if is_cont else orig
        seam = _cont.SeamDeduper(tail, active=is_cont, buffer=seam_buf)

        def _emit(text: str):
            # closure over the loop's partial/tail via nonlocal
            nonlocal partial, tail
            partial += text
            tail = (tail + text)[-_CONT_TAIL:]

        try:
            async for chunk in route_and_stream(
                msgs, dict(options), session_key=session_key,
                # only the first attempt honors the sticky/preferred model; a
                # continuation lets routing pick freely (next-best).
                preferred_model_db_id=preferred_model_db_id if not is_cont else None,
            ):
                out = seam.feed(chunk)
                if out:
                    _emit(out)
                    yield out
        except Exception as exc:  # noqa: BLE001 — mid-stream transport/provider drop
            out = seam.flush()
            if out:
                _emit(out)
                yield out
            # Nothing shown yet → surface the error normally (route_and_stream
            # already tried its own before-first-token failover). No routes left,
            # or out of continuation budget → stop with the partial we have.
            if not tail:
                raise
            if attempts >= max_cont or isinstance(exc, NoRouteAvailable):
                log.info("continuation: stopping with partial after error "
                         "(attempt %d): %s", attempts, exc)
                return
            attempts += 1
            log.info("continuation: mid-stream error, re-prompting (attempt %d): "
                     "%s", attempts, exc)
            continue

        out = seam.flush()
        if out:
            _emit(out)
            yield out

        # Clean end for this attempt. Continue only on a genuine length cut-off,
        # when we actually showed something and still have budget.
        if _cont.is_cutoff(_usage.finish_reason()) and tail and attempts < max_cont:
            attempts += 1
            log.info("continuation: finish_reason=length, re-prompting "
                     "(attempt %d)", attempts)
            continue
        return


async def route_and_stream(
    messages: list[dict],
    options: dict,
    *,
    session_key: str | None = None,
    preferred_model_db_id: int | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming completion with fallback before the first token."""
    est = estimate_tokens(messages)
    vision = _messages_have_images(messages)
    difficulty = options.pop("difficulty", "standard")
    difficulty = _apply_reasoning_mode(difficulty)
    _apply_operational_mode(options)
    avoid = options.pop("avoid_model_db_id", None)
    # Speculative drafting: a per-draft dict the engine writes the chosen model
    # into (see _commit_start). Popped so it's never sent to a provider.
    _route_sink = options.pop("_route_sink", None)
    # Phase 2 semantic routing: the query embedding (from the Understanding pass)
    # keys the learned-success signal. Popped so it's never sent to a provider.
    _query_emb = options.pop("query_embedding", None)
    # G13: redact PII/secrets before anything leaves the device for a provider —
    # the single egress choke point covers every path + retry. Fail-open.
    try:
        from app.security.egress_redact import redact_messages as _redact
        messages = _redact(messages)
    except Exception:  # noqa: BLE001 — redaction must never break a call
        pass
    preferred = preferred_model_db_id or _get_sticky(session_key) \
        or _diff_pref(difficulty)
    skip: set[str] = set()
    last_err: Exception | None = None

    _task_cat, _needs_tool, _needs_json = _resolve_task_category(
        messages, options, difficulty)

    _retries = _max_retries()
    _deadline = _first_token_deadline()
    _no_route_max, _no_route_base = _no_route_retry()
    _no_route_tries = 0
    _track = _RecoveryTracker()   # failure-KB learn loop (records outcomes)
    for _attempt in range(_retries):
        try:
            route = await router.route_request(
                est, skip, preferred, require_vision=vision, min_context=est,
                difficulty=difficulty, avoid_model_db_id=avoid,
                task_category=_task_cat, needs_tool=_needs_tool,
                needs_json=_needs_json, query_embedding=_query_emb,
            )
        except router.NoRouteAvailable as exc:
            # Transient exhaustion under a concurrent burst (multiple live
            # questions answered at once): every model is momentarily rate-
            # limited. Back off — recovering the 60s RPM/TPM windows and de-
            # syncing the burst with jitter — and retry route selection instead
            # of erroring straight to the UI. Re-raise once the no-route budget
            # is spent so a genuine outage still surfaces.
            last_err = exc
            # Persistent no-route (no key / no provider configured) → fail fast;
            # only a TRANSIENT exhaustion (candidates exist but are all
            # momentarily rate-limited) is worth a backoff + retry.
            if not getattr(exc, "transient", False):
                _track.resolve(False)
                raise
            _no_route_tries += 1
            if (_no_route_max <= 0 or _no_route_tries > _no_route_max
                    or _attempt == _retries - 1):
                _track.resolve(False)
                raise
            _jitter = (time.monotonic() * 1000.0) % 1.0 * _no_route_base
            await asyncio.sleep(_no_route_base * _no_route_tries + _jitter)
            continue
        log.info("routing[%s]: trying %s/%s", difficulty, route.platform,
                 route.model_id)
        adapter = get_adapter(route.platform)
        yielded = False
        produced = 0
        head: list[str] = []        # buffered until we validate the start
        head_len = 0
        validated = False
        recent = ""                 # rolling tail for mid-stream degen checks
        degenerate = False
        timed_out = False

        _t0 = time.monotonic()   # for observed time-to-first-token (latency-aware)

        def _commit_start():
            # Mark this route as the live one (once we trust its output).
            router.record_success(route.model_db_id)
            # Health window (R9.5): success + observed TTFT feeds latency-aware
            # ranking and CLOSES any open circuit breaker for this model.
            _ph.record(route.model_db_id, ok=True,
                       latency_s=max(0.0, time.monotonic() - _t0))
            ratelimit.record_request(route.platform, route.model_id, route.key_id)
            _set_sticky(session_key, route.model_db_id)
            _record_last_model(session_key, route.display_name, route.model_id)
            # Speculative drafts also stash their model in a per-draft sink, so
            # the speculation layer can RE-ASSERT the winner's model as the
            # last-model after the race (a losing draft may commit its opening
            # before being cancelled and pollute the global "answered by").
            if isinstance(_route_sink, dict):
                _route_sink["display_name"] = route.display_name
                _route_sink["model_id"] = route.model_id
            _record_diff_success(difficulty, route.model_db_id)
            log.info("routing[%s]: streaming from %s/%s", difficulty,
                     route.platform, route.model_id)

        try:
            # First-token deadline: pull chunks through an explicit iterator so
            # the wait for the FIRST byte can be bounded. A provider that
            # connects then hangs is abandoned in `_deadline` seconds instead
            # of blocking the turn for its full HTTP timeout (up to 120s).
            _aiter = adapter.stream(
                route.api_key, messages, route.model_id, options).__aiter__()
            while True:
                try:
                    # Bound only the wait for the FIRST token(s); once we've
                    # validated + started streaming, let it run (a mid-stream
                    # stall is handled by the provider's own read timeout).
                    if _deadline > 0 and not validated:
                        chunk = await asyncio.wait_for(
                            _aiter.__anext__(), timeout=_deadline)
                    else:
                        chunk = await _aiter.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    timed_out = True
                    break
                if not validated:
                    # Buffer the first slice; validate before showing anything.
                    head.append(chunk)
                    head_len += len(chunk)
                    if head_len < _DEGEN_HEAD:
                        continue
                    if _looks_degenerate("".join(head)):
                        degenerate = True
                        break  # garbage from the start — fall back cleanly
                    validated = True
                    yielded = True
                    _commit_start()
                    for b in head:
                        produced += len(b)
                        yield b
                    recent = "".join(head)[-160:]
                    head = []
                    continue
                produced += len(chunk)
                yield chunk
                # Cheap mid-stream check: a model that degenerates AFTER a clean
                # start (emits <unk>/replacement tokens) — stop the flood.
                recent = (recent + chunk)[-160:]
                if "<unk>" in recent or recent.count("\ufffd") >= 3:
                    degenerate = True
                    break
        except ProviderError as exc:
            last_err = exc
            # Hard failure → feeds the circuit breaker (a provider error, not a
            # 429). N consecutive of these open the breaker for a cooldown.
            _ph.record(route.model_db_id, ok=False, hard_failure=True)
            try:
                _fc = _failure_taxonomy.observe(exc, where="engine.stream_retry")  # classify (fail-open)
                _failure_kb.record_occurrence(_fc.id)  # learn recurring failures (Phase 7 #3)
                _plan = _plan_recovery(_fc, attempt=_attempt + 1,
                                       max_attempts=_retries)
            except Exception:  # noqa: BLE001 — bookkeeping never breaks the call
                _plan = None
            _action = _plan.effective_action if _plan else ""
            # `yielded` is an absolute veto — bytes are already on the wire, we
            # can't re-route mid-answer whatever the plan says. Otherwise the PLAN
            # decides whether a retry can help, with `exc.retryable` as a hard veto.
            if yielded or not exc.retryable or _plan_vetoes_retry(_plan):
                _track.resolve(False)
                raise
            log.info("routing[%s]: %s/%s failed before first token (%s) — "
                     "falling back", difficulty, route.platform, route.model_id,
                     exc)
            if exc.permanent_dead:
                await _disable_model(route.model_db_id)
            _penalize(route, exc.status, exc.permanent_dead)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None
            # …and where the plan says "go somewhere else" (transport/auth/timeout),
            # de-prioritize the failing MODEL, not just this (model,key) pair.
            if _recovery.wants_different_route(_action):
                avoid = route.model_db_id
            if _plan is not None:
                _track.note(_plan.failure_id, _action)
            _delay = _plan_backoff_s(_plan, _action)
            if _delay > 0:
                await asyncio.sleep(_delay)
            continue

        # First-token deadline elapsed before this model produced anything we
        # committed to — abandon it and try the next. (If we'd already yielded,
        # `_deadline` no longer applies, so timed_out implies nothing shown.)
        if timed_out and not yielded:
            log.info("routing[%s]: %s/%s slow to first token (>%.1fs) — falling "
                     "back", difficulty, route.platform, route.model_id,
                     _deadline)
            # A model that connects then hangs past the deadline is a hard
            # failure for the breaker (distinct from a 429 cooldown).
            _ph.record(route.model_db_id, ok=False, hard_failure=True)
            router.record_rate_limit_hit(route.model_db_id)
            ratelimit.set_cooldown(route.platform, route.model_id, route.key_id)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None
            last_err = ProviderError("slow to first token")
            # Best-effort: close the abandoned upstream stream.
            try:
                await _aiter.aclose()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            continue

        if degenerate:
            # Broken output (gibberish / <unk> tokens). Penalize hard so routing
            # avoids this model, and skip it.
            log.info("routing[%s]: %s/%s produced DEGENERATE output — %s",
                     difficulty, route.platform, route.model_id,
                     "stopping" if yielded else "falling back")
            _ph.record(route.model_db_id, ok=False, hard_failure=True)
            for _ in range(3):
                router.record_rate_limit_hit(route.model_db_id)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None
            last_err = ProviderError("degenerate output")
            if yielded:
                # Already streamed a (clean) prefix — can't re-route mid-answer;
                # stop the garbage flood and end the turn with what we have.
                ratelimit.record_tokens(route.platform, route.model_id,
                                        route.key_id,
                                        _real_total_tokens(est, produced))
                _track.resolve(False)   # ended on garbage — the recovery didn't land
                return
            continue  # never shown anything → try the next model

        # Short, un-validated output (< _DEGEN_HEAD): validate + flush now.
        if not validated and head_len > 0:
            buffered = "".join(head)
            if buffered.strip() and not _looks_degenerate(buffered):
                yielded = True
                _commit_start()
                produced += head_len
                for b in head:
                    yield b
            elif buffered.strip():
                degenerate = True

        if degenerate:
            for _ in range(3):
                router.record_rate_limit_hit(route.model_db_id)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None
            last_err = ProviderError("degenerate output")
            continue

        # Empty stream with NO error (common with flaky free big-model
        # endpoints): the model produced zero tokens. Don't return an empty
        # answer — treat it as a soft failure, penalize + skip this model, and
        # fall through to the next one so the turn still gets a real response.
        if not yielded:
            log.info("routing[%s]: %s/%s returned an EMPTY stream — falling "
                     "back", difficulty, route.platform, route.model_id)
            router.record_rate_limit_hit(route.model_db_id)
            skip.add(f"{route.platform}:{route.model_id}:{route.key_id}")
            preferred = None
            last_err = ProviderError("empty response (no tokens)")
            continue
        # Clean finish — record real (or estimated) tokens and stop.
        ratelimit.record_tokens(route.platform, route.model_id, route.key_id,
                                _real_total_tokens(est, produced))
        # A clean answer AFTER a failure means the recovery we took worked —
        # teach the KB (the half of the loop that makes `best_recovery` work).
        _track.resolve(True)
        return

    _track.resolve(False)   # retries exhausted — the recovery never landed
    raise router.NoRouteAvailable(
        f"Exhausted {_retries} routing attempts. Last error: {last_err}"
    )
