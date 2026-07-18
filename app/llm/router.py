"""Request router — picks the best available model+key.

Ported from freellmapi's `services/router.ts`. Models are tried in
`priority + dynamic_penalty` order, so a model that just hit a 429 sinks
below working ones and recovers as the penalty decays. Within a model,
keys are tried round-robin, skipping ones on cooldown or over their
sliding-window rate limit.

`route_request` is async only because it reads the chain/models/keys from
Postgres; the hot in-loop checks (penalty, cooldown, rate-limit windows)
are all synchronous in-memory lookups.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select

from app.llm import crypto, ratelimit
from app.llm.catalog import _size_billions, get_provider_spec
from app.llm.providers import get_adapter
from storage.db import get_session_factory
from storage.models import LLMApiKey, LLMFallbackConfig, LLMModel

log = logging.getLogger(__name__)


class NoRouteAvailable(RuntimeError):
    """All models/keys exhausted — add keys or wait for limits to reset.

    `transient` is True when candidates EXISTED but were all momentarily
    rate-limited / on cooldown (a backoff + retry may recover); False when there
    were genuinely no usable models (no key / no provider) so a retry is
    pointless and the caller should fail fast."""

    def __init__(self, *args, transient: bool = False):
        super().__init__(*args)
        self.transient = transient


@dataclass
class RouteResult:
    platform: str
    model_id: str
    model_db_id: int
    display_name: str
    api_key: str
    key_id: int


# ── Dynamic priority: 429 penalty with time decay (in-memory) ────────────
# Module defaults; the live values are read from config (llm.routing.*) via the
# helpers below so the knobs in config.yaml actually take effect.
_PENALTY_PER_429 = 3
_MAX_PENALTY = 10
_DECAY_INTERVAL_S = 120.0   # one point recovered every 2 minutes


def _penalty_per_429() -> int:
    try:
        from app.core.config_loader import cfg
        return int(cfg.llm.routing.penalty_per_429)
    except Exception:  # noqa: BLE001 — never let config break routing
        return _PENALTY_PER_429


def _decay_interval_s() -> float:
    try:
        from app.core.config_loader import cfg
        return float(cfg.llm.routing.decay_interval_s)
    except Exception:  # noqa: BLE001
        return _DECAY_INTERVAL_S


def _prefer_free() -> bool:
    """Whether to route only among free models when any is available."""
    try:
        from app.core.config_loader import get_config
        return bool(get_config().routing.prefer_free)
    except Exception:  # noqa: BLE001
        return True


def _allow_paid_tier(difficulty: str) -> bool:
    """Phase P2-1 hybrid: True when this difficulty is allowed to use a paid
    strong model (opt-in `strong_tier_for_hard` + the difficulty is listed +
    the monthly paid-request cap isn't exhausted). Default OFF → always False,
    so free-first behavior is unchanged."""
    try:
        from app.core.config_loader import get_config
        r = get_config().routing
        if not bool(getattr(r, "strong_tier_for_hard", False)):
            return False
        diffs = {str(d).strip().lower()
                 for d in (getattr(r, "strong_tier_difficulties", None)
                           or ["expert"])}
        if (difficulty or "").lower() not in diffs:
            return False
        cap = int(getattr(r, "monthly_paid_request_cap", 0) or 0)
    except Exception:  # noqa: BLE001 — never let config break routing
        return False
    try:
        from app.llm import budget
        return budget.can_use_paid(cap)
    except Exception:  # noqa: BLE001
        return True


# Platforms whose keys are inherently billed per token (no free tier). These are
# treated as PAID: excluded from the free-first pool (so they're a true last
# resort), counted against `monthly_paid_request_cap`, and dropped once the cap
# is hit. Overridable via `cfg.routing.paid_platforms`.
_DEFAULT_PAID_PLATFORMS = ("anthropic", "openai")


def _paid_platforms() -> set[str]:
    try:
        from app.core.config_loader import get_config
        vals = getattr(get_config().routing, "paid_platforms", None)
        if vals is not None:
            return {str(v).strip().lower() for v in vals}
    except Exception:  # noqa: BLE001
        pass
    return set(_DEFAULT_PAID_PLATFORMS)


def _is_free(model: "LLMModel", spec) -> bool:
    """Data-driven free check — NO hardcoded model-name list. A model is treated
    as free when its provider is an anonymous tier (no key, no cost) or — on
    OpenRouter, which mixes free + paid under one key — its id carries the
    `:free` suffix. Genuinely-paid providers (see `_paid_platforms`) are NOT
    free. Every other configured provider is used on its free/included tier, so
    all of those compete."""
    if spec is not None and getattr(spec, "allow_anonymous", False):
        return True
    if (model.platform or "").lower() in _paid_platforms():
        return False
    if model.platform == "openrouter":
        return (model.model_id or "").lower().endswith(":free")
    return True


def apply_capability_filter(pool: list, needs_tool: bool, needs_json: bool,
                            trace: list | None = None) -> list:
    """Restrict `pool` (scored candidate dicts) to models advertising the
    required tool/JSON capability, keeping the same rank-based fallback. If none
    qualify, return the full pool and record the unmet constraint in `trace`
    (intelligent-model-routing R4.2/R4.3, Property 4)."""
    if needs_tool:
        capable = [c for c in pool if c.get("supports_tools")]
        if capable:
            pool = capable
        elif trace is not None:
            trace.append({"unmet_constraint": "tool"})
    if needs_json:
        capable = [c for c in pool if c.get("supports_json")]
        if capable:
            pool = capable
        elif trace is not None:
            trace.append({"unmet_constraint": "json"})
    return pool


def _supports_flag(model: "LLMModel", which: str) -> bool:
    """Capability flag (tools/json) for a model — explicit column when present,
    else derived from the capability profile (intelligent-model-routing R4.1)."""
    try:
        attr = "supports_tools" if which == "tools" else "supports_json"
        v = getattr(model, attr, None)
        if v is not None:
            return bool(v)
        from app.llm.capabilities import profile_for
        prof = profile_for(model)
        return prof.supports_tools if which == "tools" else prof.supports_json
    except Exception:  # noqa: BLE001
        return True       # fail-open: don't exclude a model on a flag error

# ── Scoring weights (lower score = picked first) ─────────────────────────
# score = manual_order_index + _W_PENALTY*penalty + _W_HEADROOM*(1 - headroom)
#         + intel_weight*intelligence_rank + speed_weight*speed_rank
# Each recent 429 (penalty pt) moves a model ~4 places down; a model at its
# rate limit (headroom 0) drops ~20 places — enough to skip a throttled
# top model for a fresh one, without abandoning the user's order.
_W_PENALTY = 4.0
_W_HEADROOM = 20.0

# ── Proactive free-tier quota (P5 #16) ───────────────────────────────────
# `_W_HEADROOM` above is the REACTIVE signal: the sliding rate-limit window of
# THIS model+key (rpm/rpd/tpm/tpd). The quota ledger in `app.llm.quota_manager`
# is the PROACTIVE one: how much of the provider's free DAILY/monthly window is
# still left. Blending it in means a provider whose free tier is nearly drained
# sinks *before* it starts 429-ing, instead of costing the user a wasted
# round-trip + latency spike per request until the circuit breaker trips.
#   • `_W_QUOTA` weights the remaining-quota fraction (0..1, 1 = untouched),
#     exactly like the latency term: additive, `w * (1 - fraction)`.
#   • `_QUOTA_EXHAUSTED_PENALTY` buries a provider whose window is fully spent
#     (it's a guaranteed 429). Providers are also FILTERED out entirely while a
#     non-exhausted one is routable (`_quota_pool`) — the penalty is the
#     belt-and-braces for when that filter is off or everything is drained.
# Unknown provider / no ledger entry / any error → fraction 1.0, not exhausted →
# scoring is byte-for-byte what it is today (fail-open).
_W_QUOTA = 12.0
_QUOTA_EXHAUSTED_PENALTY = 250.0

# Difficulty-aware routing. The intelligence/speed weights are what actually
# decide the model now (the base capability order is only a tiebreak), so a
# trivial turn gets a FAST small model and only hard/expert turns pull in the
# big slow ones. Lower intelligence_rank = stronger; lower speed_rank = faster.
_DIFFICULTY = {
    #            intel_weight, speed_weight
    "trivial":  (0.0, 4.0),   # speed only — snappy small model for greetings/chatter
    "standard": (1.0, 2.5),   # fast, capable model for normal Q&A (not a 500B giant)
    "hard":     (4.0, 0.3),   # capability-led, tiny speed tiebreak
    "expert":   (8.0, 0.0),   # the strongest model wins outright
}

# Hard capability floor per difficulty: a hard/expert task must not fall back to
# a weak model while a capable one is routable. Candidates whose intelligence
# rank is worse (higher) than the floor are dropped — UNLESS that leaves nothing
# routable, in which case we fall back to the full pool (availability wins over
# the floor). Lower rank = stronger; None = no floor (trivial/standard).
# Kept wide enough that SEVERAL large models clear it, so hard/expert turns can
# rotate across them (diversity) rather than pinning the single top-ranked one.
_DIFFICULTY_FLOOR = {"hard": 18, "expert": 10}


def _candidate_score(penalty: int, headroom: float,
                     intel: int, speed: int, difficulty: str,
                     task_match: float = 1.0, learned: float = 1.0,
                     *, task_w: float | None = None, learn_w: float | None = None,
                     latency_factor: float = 1.0, latency_w: float = 0.0,
                     quota_headroom: float = 1.0, quota_w: float = 0.0) -> float:
    """Routing score for one model (lower = picked first). Difficulty decides:
    trivial weights SPEED (fast small model), hard/expert weight INTELLIGENCE
    (big capable model). The static base/capability order is NOT in the score —
    it's only a stable tiebreak applied when scores are equal.

    `task_match` (0..1 fit to the request's Task_Category) and `learned` (0..1
    historical success) add ADDITIVE penalties weighted by `_task_weight()` /
    `_learn_weight()`, both 0 by default → identical to today's ranking
    (intelligent-model-routing Property 2/9). `latency_factor` (0..1 observed
    speed, 1 = fast) adds a `latency_w`-weighted penalty for slow models (0 =
    off). `quota_headroom` (0..1 fraction of the PROVIDER's free-tier window
    still unspent, 1 = untouched/unknown) adds a `quota_w`-weighted penalty, so
    a nearly-drained free tier is rotated away from BEFORE it 429s (0 = off →
    today's ranking). `task_w`/`learn_w` may be PRECOMPUTED by the caller
    (hot-path: read the config once per request instead of once per candidate);
    None → read the helper, preserving byte-identical behavior for other
    callers."""
    intel_w, speed_w = _DIFFICULTY.get(difficulty, _DIFFICULTY["standard"])
    _tw = _task_weight() if task_w is None else task_w
    _lw = _learn_weight() if learn_w is None else learn_w
    return (
        _W_PENALTY * penalty
        + _W_HEADROOM * (1.0 - headroom)
        + intel_w * (intel or 100)
        + speed_w * (speed or 100)
        + _tw * (1.0 - max(0.0, min(1.0, task_match)))
        + _lw * (1.0 - max(0.0, min(1.0, learned)))
        + latency_w * (1.0 - max(0.0, min(1.0, latency_factor)))
        + quota_w * (1.0 - max(0.0, min(1.0, quota_headroom)))
    )


def _task_weight() -> float:
    """Weight of the Task_Match term (0 = off → today's ranking)."""
    try:
        from app.core.config_loader import get_config
        r = get_config().routing
        if not bool(getattr(r, "capability_routing", False)):
            return 0.0
        return float(getattr(r, "task_match_weight", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _learn_weight() -> float:
    """Weight of the learned-success term (0 = off → today's ranking). Active for
    either the category-keyed learning router OR Phase-2 semantic learning."""
    try:
        from app.core.config_loader import get_config
        r = get_config().routing
        if not (bool(getattr(r, "learning_router", False))
                or bool(getattr(r, "semantic_learning", False))):
            return 0.0
        return float(getattr(r, "learn_weight", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _quota_state() -> dict[str, tuple[float, bool]]:
    """Read the proactive free-tier ledger ONCE per request (hot path):
    ``platform -> (headroom_fraction 0..1, exhausted)``.

    Providers the ledger knows nothing about — and those with an unlimited /
    unknown window (`limit <= 0`, where `headroom()` returns None) — are simply
    ABSENT from the map, and the caller treats a miss as (1.0, False): full
    headroom, not exhausted → no ranking effect. Any error returns `{}` so the
    router scores exactly as it does today. Never let quota bookkeeping break
    routing (fail-open)."""
    try:
        from app.llm.quota_manager import quota_manager
        qm = quota_manager()
        out: dict[str, tuple[float, bool]] = {}
        for row in qm.snapshot():
            prov = str(row.get("provider") or "").strip().lower()
            limit = int(row.get("limit") or 0)
            left = row.get("headroom")
            if not prov or limit <= 0 or left is None:
                continue          # unlimited / unknown → no proactive signal
            frac = max(0.0, min(1.0, float(left) / float(limit)))
            out[prov] = (frac, bool(qm.exhausted(prov)))
        return out
    except Exception:  # noqa: BLE001 — never let quota bookkeeping break routing
        return {}


def _quota_pool(pool: list[dict]) -> list[dict]:
    """Drop candidates whose provider's free-tier window is fully SPENT — a
    request to one is a guaranteed 429 (wasted round-trip + latency spike).

    Same availability-wins fallback the capability floor / cost policy use: if
    EVERY candidate is exhausted, return the pool untouched rather than leaving
    the router with nothing to route (the score still buries them, the reactive
    429 path still catches the fallout, and the caller's emergency-paid /
    backoff behaviour is preserved). Pure + side-effect-free."""
    live = [c for c in pool if not c.get("quota_exhausted")]
    return live or pool


def _legacy_candidate_score(order_idx: int, penalty: int, headroom: float,
                     intel: int, speed: int, difficulty: str) -> float:
    """Deprecated — kept only for the old order-weighted unit test. The live
    scorer is `_candidate_score` above (difficulty-driven, order-free)."""
    intel_w, speed_w = _DIFFICULTY.get(difficulty, _DIFFICULTY["standard"])
    return (
        order_idx
        + _W_PENALTY * penalty
        + _W_HEADROOM * (1.0 - headroom)
        + intel_w * (intel or 100)
        + speed_w * (speed or 100)
    )
# model_db_id -> {count, last_hit, penalty}
_penalties: dict[int, dict] = {}

# Load-spread: among interchangeable top candidates (same rank + size) within
# this score margin, rotate by least-recently-used so several equivalent models
# share the load instead of one being hammered into its rate limit.
_SPREAD_MARGIN = 6.0
# On hard/expert turns we want DIVERSITY across the big models, not the single
# top-ranked one every time. The floor has already pruned everything below the
# strong tier, so we rotate (least-recently-used) across all floor-passing
# candidates within this wider margin — wide enough to span the tier's
# intelligence-rank spread, but tight enough to drop a heavily-throttled model.
_STRONG_SPREAD_MARGIN = 60.0
_last_used: dict[int, float] = {}  # model_db_id -> last routed at (epoch s)


def record_rate_limit_hit(model_db_id: int) -> None:
    e = _penalties.get(model_db_id)
    now = time.time()
    per = _penalty_per_429()
    if e:
        e["count"] += 1
        e["last_hit"] = now
        e["penalty"] = min(e["penalty"] + per, _MAX_PENALTY)
    else:
        _penalties[model_db_id] = {"count": 1, "last_hit": now, "penalty": per}


def record_success(model_db_id: int) -> None:
    e = _penalties.get(model_db_id)
    if e:
        e["penalty"] = max(0, e["penalty"] - 1)
        if e["penalty"] == 0:
            _penalties.pop(model_db_id, None)


def get_penalty(model_db_id: int) -> int:
    e = _penalties.get(model_db_id)
    if not e:
        return 0
    elapsed = time.time() - e["last_hit"]
    steps = int(elapsed // _decay_interval_s())
    if steps > 0:
        e["penalty"] = max(0, e["penalty"] - steps)
        e["last_hit"] = time.time()
        if e["penalty"] == 0:
            _penalties.pop(model_db_id, None)
            return 0
    return e["penalty"]


def get_all_penalties() -> list[dict]:
    out = []
    for mid, e in list(_penalties.items()):
        p = get_penalty(mid)
        if p > 0:
            out.append({"model_db_id": mid, "count": e["count"], "penalty": p})
    return sorted(out, key=lambda x: x["penalty"], reverse=True)


# ── Routing ──────────────────────────────────────────────────────────────
async def route_request(
    estimated_tokens: int = 1000,
    skip_keys: set[str] | None = None,
    preferred_model_db_id: int | None = None,
    *,
    require_vision: bool = False,
    min_context: int | None = None,
    difficulty: str = "standard",
    avoid_model_db_id: int | None = None,
    task_category: str | None = None,
    needs_tool: bool = False,
    needs_json: bool = False,
    query_embedding: list[float] | None = None,
    trace: list | None = None,
) -> RouteResult:
    """Return the highest-priority model+key that can serve this request.

    `skip_keys` holds "platform:model_id:key_id" strings that already failed
    on *this* request. `preferred_model_db_id` (sticky session) is tried first.
    `require_vision` restricts the chain to image-capable models;
    `min_context` skips models whose context window can't hold the request
    (e.g. a large inlined document). Both keep the same rank-based fallback.

    intelligent-model-routing (all additive, fail-open):
      `task_category` adds a capability Task_Match term to the score (weight 0 →
      unchanged); `needs_tool`/`needs_json` filter to capable models with the
      same fallback (empty → full pool + unmet constraint recorded in `trace`);
      `trace` (when a list is passed) collects the considered candidates.
    Raises `NoRouteAvailable` when everything is exhausted.
    """
    factory = get_session_factory()
    if factory is None:
        raise NoRouteAvailable("Database not ready — cannot route LLM requests yet.")

    # Self-heal a boot that skipped encryption-key init (Postgres up late) —
    # without a cached key every stored API key fails to decrypt and the pool
    # collapses to a misleading "all models exhausted". Best-effort.
    try:
        await crypto.ensure_initialized()
    except Exception:  # noqa: BLE001 — keyless/anonymous routes may still work
        pass

    async with factory() as session:
        # Fallback chain joined to its models (only enabled models).
        rows = (
            await session.execute(
                select(LLMFallbackConfig, LLMModel)
                .join(LLMModel, LLMFallbackConfig.model_db_id == LLMModel.id)
                .where(LLMFallbackConfig.enabled.is_(True), LLMModel.enabled.is_(True))
                .order_by(LLMFallbackConfig.priority.asc())
            )
        ).all()
        # Candidate keys grouped by platform (enabled + not-known-bad).
        key_rows = (
            await session.execute(
                select(LLMApiKey).where(
                    LLMApiKey.enabled.is_(True),
                    LLMApiKey.status.in_(("healthy", "unknown")),
                )
            )
        ).scalars().all()
        # Fold in configured-but-not-enabled models so routing can use the FULL
        # catalogue, ranked after the user's enabled chain:
        #   • vision turns always pull in every vision-capable model, and
        #   • `routing.route_all_models` pulls in EVERY model (any task).
        extra_models: list[LLMModel] = []
        # Hot-path: read the routing config ONCE per request (not once per
        # candidate — the loop below runs for every eligible model, up to the
        # whole catalogue). All flags default off → byte-identical ranking.
        _rc = None
        try:
            from app.core.config_loader import get_config
            _rc = get_config().routing
        except Exception:  # noqa: BLE001 — never let config break routing
            _rc = None

        def _rcf(name: str, default):
            try:
                v = getattr(_rc, name, default)
                return default if v is None else v
            except Exception:  # noqa: BLE001
                return default

        route_all = bool(_rcf("route_all_models", False))
        _cap_routing = bool(_rcf("capability_routing", False))
        _task_w = float(_rcf("task_match_weight", 0.0)) if _cap_routing else 0.0
        _sem_learn = bool(_rcf("semantic_learning", False))
        _learn_router = bool(_rcf("learning_router", False))
        _learn_w = (float(_rcf("learn_weight", 0.0))
                    if (_sem_learn or _learn_router) else 0.0)
        _adaptive_on = bool(_rcf("adaptive_benchmark", False))
        # Vision-scoped latency-awareness: a VISION request always steers by
        # OBSERVED latency (so the fastest CAPABLE vision model wins) even when
        # the global `latency_aware` flag is off — this keeps general routing
        # untouched while making image reads fast + accurate. Neutral on cold
        # start (no health samples → factor 1.0).
        _latency_global = bool(_rcf("latency_aware", False))
        _latency_aware = _latency_global or require_vision
        _latency_w = float(_rcf("latency_weight", 0.0)) if _latency_global else 0.0
        if require_vision:
            try:
                from app.core.config_loader import cfg as _vcfg
                _vlw = float(getattr(_vcfg.vision, "vision_latency_weight", 0.2)
                             or 0.2)
            except Exception:  # noqa: BLE001
                _vlw = 0.2
            _latency_w = max(_latency_w, _vlw)
        _latency_base = float(_rcf("latency_baseline_s", 8.0))
        _circuit_on = bool(_rcf("circuit_breaker", False))
        _circuit_thresh = int(_rcf("circuit_fail_threshold", 3))
        _circuit_cooldown = float(_rcf("circuit_cooldown_s", 30.0))
        _free_only = bool(_rcf("free_only", False))
        _free_only_emergency = bool(_rcf("free_only_emergency_paid", True))
        # Proactive free-tier quota (P5 #16). ON by default — the ledger is
        # already being written on every dispatch (`_record_quota_use`); this is
        # what finally READS it. `quota_aware` off → the terms are 0 / the map is
        # empty → byte-for-byte today's ranking.
        _quota_aware = bool(_rcf("quota_aware", True))
        _quota_w = float(_rcf("quota_weight", _W_QUOTA)) if _quota_aware else 0.0
        _quota_exh_pen = float(_rcf("quota_exhausted_penalty",
                                    _QUOTA_EXHAUSTED_PENALTY))
        _quota_skip = bool(_rcf("quota_skip_exhausted", True)) and _quota_aware
        # One ledger read per request (not per candidate). {} on any error.
        _quota_map = _quota_state() if _quota_aware else {}
        # Perceived-health tracker (observed latency + circuit-breaker state).
        # Only consulted when latency_aware / circuit_breaker are on.
        _health = None
        if _latency_aware or _circuit_on:
            try:
                from app.perceived.health import health as _health
            except Exception:  # noqa: BLE001
                _health = None
        # Hoist the per-candidate signal imports out of the loop (they were
        # re-imported for every model — up to the whole catalogue per request).
        _profile_for = _task_match = _success_for = _learned_success = None
        if _cap_routing:
            try:
                from app.llm.capabilities import profile_for as _profile_for
                from app.llm.capabilities import task_match as _task_match
            except Exception:  # noqa: BLE001
                _profile_for = _task_match = None
        if _sem_learn:
            try:
                from app.llm.semantic_routing import success_for as _success_for
            except Exception:  # noqa: BLE001
                _success_for = None
        if _learn_router:
            try:
                from app.llm.learning import learned_success as _learned_success
            except Exception:  # noqa: BLE001
                _learned_success = None
        if require_vision or route_all:
            enabled_ids = {m.id for _, m in rows}
            stmt = select(LLMModel)
            if require_vision:
                stmt = stmt.where(LLMModel.supports_vision.is_(True))
            extra_models = [
                m
                for m in (await session.execute(stmt)).scalars().all()
                if m.id not in enabled_ids
            ]

    keys_by_platform: dict[str, list[LLMApiKey]] = defaultdict(list)
    for k in key_rows:
        keys_by_platform[k.platform].append(k)

    # Base ranking = CAPABILITY first: strongest (lowest intelligence_rank) and,
    # within the same rank, the BIGGEST model first — across the enabled chain
    # AND discovered models alike, so a big discovered 235B/480B is tried before
    # a smaller enabled one. Manual fallback priority is the tiebreak among
    # equally-capable models, so reordering the chain still matters.
    entries = [{"fc": fc, "model": m} for fc, m in rows]
    entries += [{"fc": None, "model": m} for m in extra_models]

    # Operational mode (P5 #27): under offline-first, local (no-egress) models
    # sort AHEAD of cloud ones as the primary key. Default off → no change; cloud
    # is still reachable as a last resort (allow_cloud), so an offline deployment
    # with no local model still answers rather than dead-ending.
    _offline_first = False
    try:
        from app.llm.operational import offline_first as _off
        _offline_first = bool(_off())
    except Exception:  # noqa: BLE001
        _offline_first = False

    def _local_rank(m) -> int:
        if not _offline_first:
            return 0
        try:
            from app.llm.operational import is_local_platform
            return 0 if is_local_platform(getattr(m, "platform", "")) else 1
        except Exception:  # noqa: BLE001
            return 0

    def _rank_key(e: dict) -> tuple:
        m = e["model"]
        intel = getattr(m, "intelligence_rank", 100) or 100
        size = _size_billions(getattr(m, "model_id", "")) or 0.0
        enabled_first = 0 if e["fc"] is not None else 1
        prio = e["fc"].priority if e["fc"] is not None else 1_000_000
        speed = getattr(m, "speed_rank", 100) or 100
        return (_local_rank(m), intel, -size, enabled_first, prio, speed)

    base_order = sorted(entries, key=_rank_key)
    order_index = {e["model"].id: i for i, e in enumerate(base_order)}

    # A hard/expert task must not be pinned to whatever (possibly weak) model the
    # conversation was sticky to — let it route fresh to the strongest model.
    if difficulty in ("hard", "expert"):
        preferred_model_db_id = None

    skip_keys = skip_keys or set()

    # Score EVERY routable candidate and pick the best — don't just take the
    # top of the list. The manual order is the base rank; a 429 penalty and low
    # rate-limit headroom (tpm/rpm/rpd) push a model DOWN, so the router spreads
    # load and picks the best *available* model instead of hammering one.
    scored: list[dict] = []
    # Count candidates dropped because a usable key EXISTS but is momentarily
    # rate-limited / on cooldown. If `scored` ends up empty but this is > 0, the
    # exhaustion is TRANSIENT (a short backoff may recover); if it's 0, there
    # were no usable models at all → fail fast (don't retry into nothing).
    throttled = 0
    for entry in base_order:
        model: LLMModel = entry["model"]
        spec = get_provider_spec(model.platform)
        if spec is None or get_adapter(model.platform) is None:
            continue
        if require_vision and not getattr(model, "supports_vision", False):
            continue
        if min_context and model.context_window and model.context_window < min_context:
            continue
        # Circuit breaker: a model with too many consecutive hard failures is
        # skipped until its cooldown elapses (then half-opens for one probe).
        # Counted as throttled → a tripped model is a TRANSIENT drop that can
        # recover, so exhaustion here is retryable rather than a hard fail.
        if _circuit_on and _health is not None and _health.is_open(
                model.id, _circuit_thresh, _circuit_cooldown):
            throttled += 1
            continue

        limits = {
            "rpm": model.rpm_limit, "rpd": model.rpd_limit,
            "tpm": model.tpm_limit, "tpd": model.tpd_limit,
        }
        candidates = list(keys_by_platform.get(model.platform, []))
        if not candidates:
            # Anonymous-tier providers (pollinations, llm7, kilo) run keyless.
            if not spec.allow_anonymous:
                continue
            if not _anon_usable(model, limits, skip_keys, estimated_tokens):
                throttled += 1  # keyless provider, but rate-limited right now
                continue
            hr = ratelimit.headroom(model.platform, model.model_id, 0, limits)
            best_key = None  # anonymous
        else:
            best_key, hr = _best_usable_key(
                model, candidates, limits, skip_keys, estimated_tokens)
            if best_key is None:
                throttled += 1  # keys exist, but all on cooldown / over-window
                continue

        penalty = get_penalty(model.id)
        # Capability Task_Match + learned-success (additive; weights 0 → today).
        # Signal helpers + config were hoisted ABOVE the loop (read once per
        # request, not once per candidate).
        tm = 1.0
        learned = 1.0
        if task_category and _profile_for is not None and _task_match is not None:
            try:
                tm = _task_match(_profile_for(model), task_category)
            except Exception:  # noqa: BLE001
                tm = 1.0
        # Learned-success signal (independent of task_category so Phase-2
        # semantic learning works on the embedding alone). Semantic (embedding-
        # cluster) signal wins when enabled + an embedding is present.
        try:
            if _sem_learn and _success_for is not None and query_embedding:
                learned = _success_for(query_embedding, model.model_id)
            elif _learn_router and _learned_success is not None and task_category:
                learned = _learned_success(task_category, model.model_id)
        except Exception:  # noqa: BLE001
            learned = 1.0
        # Observed-latency factor (0..1, 1 = fast). Neutral (1.0) with no samples
        # or when latency-aware routing is off → no ranking effect.
        _lat_factor = 1.0
        if _latency_aware and _health is not None:
            _lat_factor = _health.latency_factor(model.id, _latency_base)
        # Proactive free-tier quota for this model's PROVIDER (0..1 left).
        # Unknown provider / unlimited window / quota off → (1.0, False) = today.
        _q_frac, _q_exhausted = _quota_map.get(
            (model.platform or "").lower(), (1.0, False))
        score = _candidate_score(
            penalty, hr,
            getattr(model, "intelligence_rank", 100),
            getattr(model, "speed_rank", 100), difficulty,
            task_match=tm, learned=learned,
            task_w=_task_w, learn_w=_learn_w,
            latency_factor=_lat_factor, latency_w=_latency_w,
            quota_headroom=_q_frac, quota_w=_quota_w,
        )
        # A provider whose free window is fully spent is a guaranteed 429 — bury
        # it. (`_quota_pool` below drops it outright while anything else is
        # routable; this keeps it last when it's all we have.)
        if _q_exhausted and _quota_aware:
            score += _quota_exh_pen
        # Adaptive benchmarking (R9): down-rank a recently-degraded model.
        if _adaptive_on:
            try:
                from app.llm.adaptive import downrank as _adapt_downrank
                score += _adapt_downrank(model.model_id)
            except Exception:  # noqa: BLE001
                pass
        scored.append({"score": score, "model": model, "key": best_key,
                       "avoid": model.id == avoid_model_db_id,
                       "intel": getattr(model, "intelligence_rank", 100),
                       "size": _size_billions(model.model_id) or 0.0,
                       "order": order_index.get(model.id, 0),
                       "task_match": tm,
                       "supports_tools": _supports_flag(model, "tools"),
                       "supports_json": _supports_flag(model, "json"),
                       "quota_headroom": _q_frac,
                       "quota_exhausted": bool(_q_exhausted),
                       "free": _is_free(model, spec)})

    if not scored:
        raise NoRouteAvailable(
            "All models exhausted. Add more API keys or wait for rate limits to reset.",
            transient=throttled > 0,
        )

    # Prefer candidates that AREN'T the model to avoid (cross-model verify) —
    # but if avoiding it leaves nothing routable, fall back to the full set
    # (a same-model check beats no check / a routing failure).
    pool = [c for c in scored if not c["avoid"]] or scored

    # Cost policy (free_only / free-first). free_only is the HARD version: route
    # to free models ONLY and, if none is routable, fail with a clear error
    # instead of silently spending on a paid model. Otherwise free-first collapses
    # to free when any is available (paid = last resort), unless the difficulty
    # was granted the paid strong tier (P2-1).
    if _free_only:
        free_pool = _cost_pool(pool, free_only=True,
                               prefer_free=False, allow_paid=False)
        if free_pool:
            pool = free_pool
        elif _free_only_emergency:
            # GRACEFUL: no free model is routable right now, but rather than fail
            # the request we fall back to the best available PAID model as a
            # one-off emergency (keeps the app working through a free-tier
            # outage). Normal traffic still spends nothing. Visible in the logs.
            log.warning(
                "routing: cost mode (free_only) — no free model routable; "
                "falling back to a PAID model this once (%d candidates). Set "
                "routing.free_only_emergency_paid=false for the strict "
                "never-spend guarantee.", len(pool),
            )
        else:
            # STRICT: never spend — fail clearly instead.
            raise NoRouteAvailable(
                "Cost mode is on (free models only) but no free model is "
                "routable right now. Add a free provider key, or wait for a "
                "free model's rate limit to reset.",
                transient=throttled > 0,
            )
    else:
        pool = _cost_pool(pool, free_only=False,
                          prefer_free=_prefer_free(),
                          allow_paid=_allow_paid_tier(difficulty))

    # Hard capability floor: on a hard/expert turn, keep only models at/above the
    # tier so the answer comes from a strong model — unless none qualify right
    # now (all throttled/absent), in which case availability wins over the floor.
    floor = _DIFFICULTY_FLOOR.get(difficulty)
    if floor is not None:
        strong = [c for c in pool if c.get("intel", 100) <= floor]
        if strong:
            pool = strong
        else:
            log.info(
                "routing: difficulty=%s wants a model at intelligence_rank<=%d "
                "but none is available right now — using the best of %d. Enable "
                "/ discover a larger model for top-tier tasks.",
                difficulty, floor, len(pool),
            )

    # Tool / structured-output capability filter (intelligent-model-routing R4):
    # restrict to flag-bearing models with the same fallback; if none qualify,
    # fall back to the full pool and record the unmet constraint (R4.3).
    pool = apply_capability_filter(pool, needs_tool, needs_json, trace)

    # Proactive quota rotation (P5 #16): skip providers whose free-tier window is
    # already SPENT — sending to one buys a guaranteed 429 and a latency spike.
    # Availability still wins: if every candidate is drained, the pool is left
    # intact (the score keeps them ordered, and the reactive 429 / emergency-paid
    # / engine-backoff paths take over) rather than manufacturing a no-route.
    if _quota_skip:
        before = len(pool)
        pool = _quota_pool(pool)
        if len(pool) < before:
            log.info(
                "routing: skipped %d candidate(s) on a free-tier quota that's "
                "already exhausted; %d left.", before - len(pool), len(pool),
            )
            if trace is not None:
                trace.append({"quota_skipped": before - len(pool)})

    # Routing explainability (R10): record the considered candidates + scores.
    if trace is not None:
        for c in sorted(pool, key=lambda c: c["score"])[:8]:
            trace.append({
                "model": c["model"].model_id,
                "score": round(c["score"], 2),
                "intel": c.get("intel"),
                "task_match": round(c.get("task_match", 1.0), 3),
                "quota": round(c.get("quota_headroom", 1.0), 3),
                "free": c.get("free", True),
            })

    # Sticky session: if the preferred model is routable right now, use it.
    if preferred_model_db_id is not None:
        for c in pool:
            if c["model"].id == preferred_model_db_id:
                res = _finalize(c["model"], c["key"], limits_for(c["model"]), skip_keys)
                if res is not None:
                    _last_used[c["model"].id] = time.time()
                    _maybe_record_paid(c)
                    return res
                break

    # Otherwise the best-scoring available candidate. Ties break by the base
    # capability order (strongest/biggest first).
    #   • standard/trivial: load-balance ONLY among INTERCHANGEABLE candidates
    #     (same rank AND same size) so we never drop to a weaker/slower model.
    #   • hard/expert: the floor already kept only the STRONG tier, so rotate
    #     (least-recently-used) across ALL of them — real model diversity on
    #     demanding turns instead of hammering the single top-ranked one.
    pool.sort(key=lambda c: (c["score"], c.get("order", 0)))
    best = pool[0]
    if difficulty in ("hard", "expert"):
        equiv = [
            c for c in pool
            if c["score"] <= best["score"] + _STRONG_SPREAD_MARGIN
        ]
    else:
        equiv = [
            c for c in pool
            if c["intel"] == best["intel"]
            and round(c.get("size", 0.0)) == round(best.get("size", 0.0))
            and c["score"] <= best["score"] + _SPREAD_MARGIN
        ]
    # Rotate by least-recently-used (capability as a stable tiebreak) so
    # successive demanding turns fan out across the big models.
    equiv.sort(key=lambda c: (_last_used.get(c["model"].id, 0.0),
                              c.get("order", 0)))
    equiv_ids = {c["model"].id for c in equiv}
    rest = [c for c in pool if c["model"].id not in equiv_ids]
    decrypt_failures = 0
    keyed_attempts = 0
    for c in equiv + rest:
        res = _finalize(c["model"], c["key"], limits_for(c["model"]), skip_keys)
        if res is not None:
            _last_used[c["model"].id] = time.time()
            _maybe_record_paid(c)
            return res
        # A non-anonymous candidate that failed to finalize = decrypt failure.
        if c.get("key") is not None:
            keyed_attempts += 1
            decrypt_failures += 1

    # Every keyed candidate failed to DECRYPT (not rate limits) — the encryption
    # key changed / is unavailable. Say so instead of "add more API keys".
    if keyed_attempts > 0 and decrypt_failures == keyed_attempts:
        raise NoRouteAvailable(
            "Stored API keys could not be decrypted — the encryption key "
            "(ZAPTHETRICK_ENCRYPTION_KEY) changed or is unavailable. Re-enter "
            "your provider keys in Settings → Providers."
        )
    raise NoRouteAvailable(
        "All models exhausted. Add more API keys or wait for rate limits to reset."
    )


def _cost_pool(pool: list[dict], *, free_only: bool,
               prefer_free: bool, allow_paid: bool) -> list[dict]:
    """Apply the cost policy to the scored candidate pool.

    - ``free_only`` → keep ONLY free candidates. The result MAY be empty (no free
      model routable) — the caller treats that as a hard no-route rather than
      spending on a paid model.
    - else free-first: collapse to free candidates when any is routable (paid is
      a last resort), UNLESS ``allow_paid`` (the difficulty was granted the paid
      strong tier), in which case the full pool competes.

    Pure + side-effect-free so the policy is unit-testable without the DB path.
    """
    if free_only:
        return [c for c in pool if c.get("free")]
    if prefer_free and not allow_paid:
        free = [c for c in pool if c.get("free")]
        return free or pool
    return pool


def _maybe_record_paid(candidate: dict) -> None:
    """Count a paid (non-free) route against the monthly budget (P2-1)."""
    if candidate.get("free", True):
        return
    try:
        from app.llm import budget
        budget.record_paid()
    except Exception:  # noqa: BLE001
        pass


def limits_for(model: LLMModel) -> dict:
    return {
        "rpm": model.rpm_limit, "rpd": model.rpd_limit,
        "tpm": model.tpm_limit, "tpd": model.tpd_limit,
    }


# Per-(model,key) last-routed timestamp — a round-robin tiebreak so keys with
# equal headroom (e.g. every no-limit discovered model reads headroom 1.0)
# actually rotate instead of always picking the first one.
_key_last_used: dict[str, float] = {}


def _est_admit(estimated_tokens: int) -> int:
    """Floor the admission estimate so a 0/None never admits onto a nearly-full
    token window; cap so a huge prompt doesn't overflow arithmetic."""
    try:
        return max(1000, min(int(estimated_tokens or 0), 2_000_000))
    except (TypeError, ValueError):
        return 1000


def _anon_usable(model: LLMModel, limits: dict, skip_keys: set[str],
                 estimated_tokens: int = 1000) -> bool:
    if f"{model.platform}:{model.model_id}:0" in skip_keys:
        return False
    if ratelimit.is_on_cooldown(model.platform, model.model_id, 0):
        return False
    if not ratelimit.can_make_request(model.platform, model.model_id, 0, limits):
        return False
    if not ratelimit.can_use_tokens(model.platform, model.model_id, 0,
                                    _est_admit(estimated_tokens), limits):
        return False
    return True


def _best_usable_key(
    model: LLMModel, keys: list[LLMApiKey], limits: dict, skip_keys: set[str],
    estimated_tokens: int = 1000,
) -> tuple[LLMApiKey | None, float]:
    """The platform key with the MOST headroom that can serve this request
    (cooldown + sliding-window checks). Balances load across multiple keys.
    Ties on headroom (common: no-limit models all read 1.0) break by
    least-recently-used so keys actually round-robin.
    Returns (key, headroom) or (None, 0.0) when none are usable."""
    est = _est_admit(estimated_tokens)
    best: LLMApiKey | None = None
    best_hr = -1.0
    best_used = float("inf")
    for key in keys:
        if f"{model.platform}:{model.model_id}:{key.id}" in skip_keys:
            continue
        if ratelimit.is_on_cooldown(model.platform, model.model_id, key.id):
            continue
        if not ratelimit.can_make_request(model.platform, model.model_id, key.id, limits):
            continue
        if not ratelimit.can_use_tokens(model.platform, model.model_id, key.id, est, limits):
            continue
        hr = ratelimit.headroom(model.platform, model.model_id, key.id, limits)
        used = _key_last_used.get(f"{model.platform}:{model.model_id}:{key.id}",
                                  0.0)
        # Prefer more headroom; on a tie prefer the least-recently-used key.
        if hr > best_hr or (hr == best_hr and used < best_used):
            best_hr, best, best_used = hr, key, used
    if best is not None:
        _key_last_used[f"{model.platform}:{model.model_id}:{best.id}"] = \
            time.time()
    return best, max(best_hr, 0.0)


def _record_quota_use(platform: str) -> None:
    """Proactive free-tier accounting (P5 #16): count a dispatch against the
    provider's window so rotation can drain quotas *before* the 429. Fail-open."""
    try:
        from app.llm.quota_manager import quota_manager
        quota_manager().record((platform or "").lower())
    except Exception:  # noqa: BLE001
        pass


def _finalize(
    model: LLMModel, key: LLMApiKey | None, limits: dict, skip_keys: set[str]
) -> RouteResult | None:
    """Turn a chosen (model, key) into a RouteResult, decrypting the key. For
    anonymous routes (`key is None`) no decryption is needed."""
    if key is None:
        _record_quota_use(model.platform)
        return RouteResult(model.platform, model.model_id, model.id, model.display_name, "", 0)
    try:
        plain = crypto.decrypt(key.encrypted_key, key.iv, key.auth_tag)
    except Exception:  # noqa: BLE001 — bad ciphertext: skip, leave for health to flag
        log.warning("could not decrypt key id=%s (platform=%s)", key.id, key.platform)
        return None
    _record_quota_use(model.platform)
    return RouteResult(model.platform, model.model_id, model.id, model.display_name, plain, key.id)


async def model_db_id_for(model_id: str) -> int | None:
    """Resolve a `model_id` string (e.g. cfg.llm.code_model) to its catalog
    row id, so the `auto` path can pin a preferred model. None if unknown."""
    if not model_id:
        return None
    factory = get_session_factory()
    if factory is None:
        return None
    async with factory() as session:
        row = (
            await session.execute(
                select(LLMModel.id).where(LLMModel.model_id == model_id, LLMModel.enabled.is_(True))
            )
        ).first()
        return row[0] if row else None
