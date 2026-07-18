"""Provider registry + curated model seed.

Ported from freellmapi's `server/src/providers/index.ts` (the provider
registry) and `server/src/db/index.ts` `seedModels()` (the model table).

Two things live here, both pure data:

  * `PROVIDERS` — every supported platform with its base URL, auth style,
    optional extra headers, and HTTP timeout. The router and the adapters
    read this; the FE catalog screen renders it.

  * `MODEL_SEED` — the hand-curated free-tier model list with the
    intelligence/speed ranks and RPM/RPD/TPM/TPD limits that drive
    routing. Copied verbatim from freellmapi so routing quality matches.

`ensure_seeded()` writes `MODEL_SEED` into `llm_models` and a default
`llm_fallback_config` (ordered by intelligence_rank) on startup. It is
idempotent — a UNIQUE(platform, model_id) makes re-runs no-ops.

Deviation from freellmapi worth noting: Google is reached through its
official OpenAI-compatible endpoint (`.../v1beta/openai`) with a bearer
key, instead of the native Gemini dialect. Same models, same routing,
far less translation code. Cloudflare keeps a dedicated adapter because
its key is `account_id:token` and the URL is account-scoped.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Auth styles ──────────────────────────────────────────────────────────
AUTH_BEARER = "bearer"          # Authorization: Bearer <key>
AUTH_CLOUDFLARE = "cloudflare"  # key = "account_id:token", URL is account-scoped


@dataclass(frozen=True)
class ProviderSpec:
    """One platform. Mirrors a `register(...)` call in providers/index.ts."""
    platform: str
    name: str
    base_url: str
    auth: str = AUTH_BEARER
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 15000
    # Some providers serve an anonymous tier (pollinations, llm7, kilo) —
    # the router will try them even with zero configured keys.
    allow_anonymous: bool = False
    # The adapter class key. "openai_compat" covers ~everything; cloudflare
    # is special. See `app/llm/providers/__init__.py`.
    adapter: str = "openai_compat"


# ── The registry — 16 providers, all free-tier, no card required ─────────
PROVIDERS: dict[str, ProviderSpec] = {
    spec.platform: spec
    for spec in [
        # Google — via the official OpenAI-compatible endpoint (bearer key).
        ProviderSpec(
            "google", "Google Gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        ProviderSpec("groq", "Groq", "https://api.groq.com/openai/v1"),
        ProviderSpec("cerebras", "Cerebras", "https://api.cerebras.ai/v1"),
        ProviderSpec("sambanova", "SambaNova", "https://api.sambanova.ai/v1"),
        ProviderSpec("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1"),
        ProviderSpec("mistral", "Mistral", "https://api.mistral.ai/v1"),
        ProviderSpec(
            "openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "https://zapthetrick.ai",
                "X-Title": "ZapTheTrick",
            },
        ),
        ProviderSpec("github", "GitHub Models", "https://models.github.ai/inference"),
        ProviderSpec("cohere", "Cohere", "https://api.cohere.ai/compatibility/v1"),
        ProviderSpec(
            "cloudflare", "Cloudflare Workers AI",
            # Account id is spliced in from the key at call time.
            "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
            auth=AUTH_CLOUDFLARE, adapter="cloudflare",
        ),
        # Z.ai (Zhipu's international platform) — direct OpenAI-compatible API
        # (Bearer key from z.ai). This is the general endpoint; the GLM Coding
        # Plan uses a separate /api/coding/paas/v4 host. Add your key under
        # Providers, then "Refresh models from provider" to pull GLM-4.6 / GLM-5.x.
        # (The earlier removal was Zhipu's Vercel AI Gateway, which needed a card;
        # this direct platform endpoint does not.)
        ProviderSpec("zai", "Z.ai (GLM)", "https://api.z.ai/api/paas/v4"),
        ProviderSpec("huggingface", "HuggingFace Router", "https://router.huggingface.co/v1"),
        # Ollama Cloud — reasoning models can take 30-90s; bump the timeout.
        ProviderSpec("ollama", "Ollama Cloud", "https://ollama.com/v1", timeout_ms=120000),
        ProviderSpec("kilo", "Kilo Gateway", "https://api.kilo.ai/api/gateway/v1", allow_anonymous=True),
        ProviderSpec("pollinations", "Pollinations", "https://text.pollinations.ai/openai/v1", allow_anonymous=True),
        ProviderSpec("llm7", "LLM7", "https://api.llm7.io/v1", allow_anonymous=True),
        # Additional OpenAI-compatible chat providers — bring your own key and
        # "Refresh models from provider" to populate the catalog. Names match
        # the Provider Atlas so the UI shows one card per provider.
        ProviderSpec("deepseek", "DeepSeek", "https://api.deepseek.com/v1"),
        ProviderSpec("xai", "xAI (Grok)", "https://api.x.ai/v1"),
        ProviderSpec("together", "Together AI", "https://api.together.xyz/v1"),
        ProviderSpec("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1"),
        ProviderSpec("deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai"),
        ProviderSpec("hyperbolic", "Hyperbolic", "https://api.hyperbolic.xyz/v1"),
        ProviderSpec("novita", "Novita", "https://api.novita.ai/v3/openai"),
        ProviderSpec("nebius", "Nebius", "https://api.studio.nebius.ai/v1"),
        ProviderSpec("moonshot", "Moonshot (Kimi)", "https://api.moonshot.ai/v1"),
        # MiniMax — international OpenAI-compatible endpoint (Bearer key from
        # platform.minimax.io). Reasoning/agentic M-series can be slow → bump
        # the timeout. Use the China host (api.minimaxi.com) if your key is there.
        ProviderSpec("minimax", "MiniMax", "https://api.minimax.io/v1",
                     timeout_ms=120000),
        ProviderSpec("qwen", "Alibaba (Qwen)",
                     "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        ProviderSpec("ai21", "AI21 Labs", "https://api.ai21.com/studio/v1"),
        ProviderSpec("featherless", "Featherless AI", "https://api.featherless.ai/v1"),
        ProviderSpec("vercel", "Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1"),
        # Anthropic exposes an OpenAI-compatible endpoint (Bearer auth).
        ProviderSpec("anthropic", "Anthropic (Claude)", "https://api.anthropic.com/v1"),
        # Bluesminds — a "New API" OpenAI-compatible gateway (verified: GET
        # /v1/models returns 401 without a key, /console/token issues sk-… keys).
        # Add your token under Providers, then "Refresh models from provider".
        ProviderSpec("bluesminds", "Bluesminds", "https://api.bluesminds.com/v1"),
    ]
}


def all_providers() -> list[ProviderSpec]:
    return list(PROVIDERS.values())


def get_provider_spec(platform: str) -> ProviderSpec | None:
    return PROVIDERS.get(platform)


# ── Curated model seed ───────────────────────────────────────────────────
# Columns mirror freellmapi's `models` table:
#   (platform, model_id, display_name, intelligence_rank, speed_rank,
#    size_label, rpm_limit, rpd_limit, tpm_limit, tpd_limit,
#    monthly_token_budget, context_window)
# Lower rank = better/faster. None = "no limit advertised on the free tier".
# Limits current as of April 2026 (copied from db/index.ts seedModels()).
_SeedRow = tuple[
    str, str, str, int, int, str,
    int | None, int | None, int | None, int | None, str, int,
]

MODEL_SEED: list[_SeedRow] = [
    # Google
    ("google", "gemini-2.5-pro", "Gemini 2.5 Pro", 1, 8, "Frontier", 5, 100, 250000, None, "~12M", 1048576),
    ("google", "gemini-2.5-flash", "Gemini 2.5 Flash", 4, 5, "Large", 10, 20, 250000, None, "~3M", 1048576),
    ("google", "gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", 8, 3, "Medium", 15, 1000, 250000, None, "~120M", 1048576),
    # OpenRouter — :free routes rotate often; the router auto-skips any that
    # 404 ("No endpoints found"), so a stale id here is self-healing.
    ("openrouter", "openai/gpt-oss-120b:free", "GPT-OSS 120B (free)", 2, 10, "Frontier", 20, 200, None, None, "~6M", 131072),
    ("openrouter", "moonshotai/kimi-k2.6:free", "Kimi K2.6 (free)", 2, 9, "Frontier", 20, 200, None, None, "~6M", 131072),
    ("openrouter", "qwen/qwen3-coder:free", "Qwen3 Coder (free)", 3, 9, "Frontier", 20, 200, None, None, "~6M", 262144),
    ("openrouter", "z-ai/glm-4.5-air:free", "GLM-4.5 Air (free)", 4, 9, "Large", 20, 200, None, None, "~6M", 131072),
    # Vision-capable FREE models (no credits) — PREFERRED for image turns so the
    # app never paywalls on a picture. :free ids rotate often; the router auto-
    # skips any that 404 ("No endpoints found"), so a stale id self-heals.
    ("openrouter", "qwen/qwen2.5-vl-72b-instruct:free", "Qwen2.5-VL 72B (free vision)", 2, 7, "Frontier", 20, 200, None, None, "~6M", 131072),
    ("openrouter", "meta-llama/llama-3.2-11b-vision-instruct:free", "Llama 3.2 11B Vision (free)", 4, 8, "Medium", 20, 200, None, None, "~6M", 131072),
    ("openrouter", "google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash exp (free vision)", 3, 9, "Large", 20, 200, None, None, "~6M", 1048576),
    ("openrouter", "mistralai/mistral-small-3.2-24b-instruct:free", "Mistral Small 3.2 (free vision)", 5, 8, "Large", 20, 200, None, None, "~6M", 131072),
    # Paid vision fallback (needs OpenRouter credits) — ranked last so it's only
    # used when every free vision model is rate-limited/unavailable.
    ("openrouter", "google/gemini-2.5-flash", "Gemini 2.5 Flash (vision, paid)", 13, 6, "Large", 20, 200, None, None, "~6M", 1048576),
    # Cerebras
    ("cerebras", "qwen-3-coder-480b", "Qwen3-Coder 480B", 2, 1, "Frontier", 30, None, 60000, 1000000, "~30M", 131072),
    ("cerebras", "llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick", 3, 1, "Frontier", 30, None, 60000, 1000000, "~30M", 131072),
    ("cerebras", "qwen3-235b", "Qwen3 235B", 3, 1, "Large", 30, None, 60000, 1000000, "~30M", 8192),
    ("cerebras", "gpt-oss-120b", "GPT-OSS 120B", 3, 1, "Large", 30, None, 60000, 1000000, "~30M", 131072),
    # GitHub Models
    ("github", "openai/gpt-5", "GPT-5 (GitHub)", 1, 7, "Frontier", 10, 50, None, None, "~18M", 128000),
    # SambaNova
    ("sambanova", "Meta-Llama-3.3-70B-Instruct", "Llama 3.3 70B", 6, 9, "Large", 20, None, None, 200000, "~6M", 8192),
    # Mistral
    ("mistral", "mistral-large-latest", "Mistral Large 3", 7, 8, "Large", 2, None, 500000, None, "~50-100M", 131072),
    ("mistral", "magistral-medium-latest", "Magistral Medium", 4, 8, "Large", 2, None, 500000, None, "~50-100M", 40000),
    ("mistral", "codestral-latest", "Codestral", 6, 6, "Medium", 2, None, 500000, None, "~50-100M", 32000),
    # Groq
    ("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B", 9, 2, "Medium", 30, 1000, 6000, 500000, "~15M", 131072),
    ("groq", "llama-4-scout-17b-16e-instruct", "Llama 4 Scout", 10, 2, "Medium", 30, 1000, 6000, 1000000, "~30M", 131072),
    # NVIDIA NIM — credit-based now, disabled by default (enabled=0 below).
    ("nvidia", "meta/llama-3.1-70b-instruct", "Llama 3.1 70B (NV)", 11, 6, "Large", 40, None, None, None, "credits-based", 131072),
    # Cohere
    ("cohere", "command-r-plus-08-2024", "Command R+ (08-2024)", 12, 11, "Large", 20, 33, None, None, "~1-2M", 131072),
    # Cloudflare
    ("cloudflare", "@cf/meta/llama-3.1-70b-instruct", "Llama 3.1 70B (CF)", 13, 11, "Medium", None, None, None, None, "~18-45M", 131072),
    # Hugging Face
    ("huggingface", "accounts/fireworks/models/llama-v3p3-70b-instruct", "Llama 3.3 70B (HF)", 14, 11, "Medium", None, None, None, None, "~1-3M", 131072),
    # MiniMax — paid (bring your own key). Long-context agentic/reasoning model.
    # If the exact id differs, add your key + "Refresh models from provider" to
    # pull MiniMax's live model list.
    ("minimax", "MiniMax-M3", "MiniMax M3", 3, 6, "Frontier", None, None, None, None, "paid", 1000000),
]

# Models that ship disabled (need credits / not truly recurring-free).
_DISABLED_BY_DEFAULT: set[tuple[str, str]] = {
    ("nvidia", "meta/llama-3.1-70b-instruct"),
}

# Curated models that accept image input — image-bearing chat turns route only
# to these. Conservative (confirmed multimodal); more can be toggled on in the
# Providers UI or via discovery.
VISION_MODELS: set[tuple[str, str]] = {
    ("google", "gemini-2.5-pro"),
    ("google", "gemini-2.5-flash"),
    ("google", "gemini-2.5-flash-lite"),
    ("github", "openai/gpt-5"),
    ("cerebras", "llama-4-maverick-17b-128e-instruct"),
    # Free OpenRouter vision (no credits needed) — preferred over the paid one.
    ("openrouter", "qwen/qwen2.5-vl-72b-instruct:free"),
    ("openrouter", "meta-llama/llama-3.2-11b-vision-instruct:free"),
    ("openrouter", "google/gemini-2.0-flash-exp:free"),
    ("openrouter", "mistralai/mistral-small-3.2-24b-instruct:free"),
    ("openrouter", "google/gemini-2.5-flash"),
}


# ── Vision capability detection (NOT a hardcoded model list) ──────────────
# Generalises to ANY provider/model so the orchestrator can route image turns
# across the user's FULL configured catalog (hundreds of discovered models),
# not just the curated few. Prefer real provider metadata; fall back to id
# markers covering the known multimodal families.
_VISION_ID_MARKERS = (
    "vl", "vision", "gpt-4o", "gpt-4.1", "gpt-5", "o4-", "chatgpt-4o",
    "gemini", "claude-3", "claude-opus-4", "claude-sonnet-4", "claude-haiku-4",
    "llava", "pixtral", "qwen2-vl", "qwen2.5-vl", "qwen3-vl",
    "llama-3.2-11b-vision", "llama-3.2-90b-vision", "llama-4", "maverick",
    "scout", "gemma-3", "internvl", "minicpm-v", "phi-3.5-vision",
    "phi-4-multimodal", "grok-2-vision", "grok-4", "mistral-small-3.1",
    "mistral-small-3.2", "mistral-medium-3", "molmo", "aria", "deepseek-vl",
    "step-1v", "glm-4v", "glm-4.1v", "kimi-vl", "ernie-4.5-vl", "dots.vlm",
)


def _markers(name: str, fallback: tuple) -> tuple:
    """Config-overridable marker list (`cfg.model_markers.<name>`), falling back
    to the in-code default. Lets a new model family be supported via config with
    no code change; fail-open to the built-in list on any error/empty override."""
    try:
        from app.core.config_loader import cfg
        vals = getattr(cfg.model_markers, name, None)
        if vals:
            return tuple(vals)
    except Exception:  # noqa: BLE001
        pass
    return fallback


def is_vision_model_id(model_id: str) -> bool:
    """Heuristic: does this model id look multimodal? Used to backfill the
    `supports_vision` flag on discovered models that carry no metadata."""
    mid = (model_id or "").lower()
    return any(m in mid for m in _markers("vision_markers", _VISION_ID_MARKERS))


def detect_vision(model_id: str, meta: dict | None = None) -> bool:
    """True if the model accepts image input. Prefers provider metadata
    (OpenRouter `architecture.input_modalities`, etc.); falls back to id markers."""
    if isinstance(meta, dict):
        arch = meta.get("architecture") if isinstance(meta.get("architecture"), dict) else {}
        mods = arch.get("input_modalities") or meta.get("input_modalities")
        if isinstance(mods, list) and any("image" in str(x).lower() for x in mods):
            return True
        modality = str(arch.get("modality") or meta.get("modality") or "").lower()
        if "image" in modality or "vision" in modality:
            return True
    return is_vision_model_id(model_id)


# ── Parameter size + Mixture-of-Experts heuristics ───────────────────────────
# OpenRouter (and most provider /models APIs) expose neither a numeric
# parameter count nor an explicit MoE flag, so we parse both from the model
# id/name. These are best-effort: a size token must be present in the id for
# `detect_param_count` to return anything, and MoE is inferred from structural
# notation (NxMB experts / aNB active params), a small set of known MoE
# families, or a "mixture of experts" mention in the description.
import re as _re

# "8x7b" / "8x22b" — expert-count × per-expert-size (always MoE).
_PARAM_MOE_RE = _re.compile(r"(?<![a-z0-9])\d+x\d+(?:\.\d+)?b(?![a-z])", _re.I)
# "8b" / "70b" / "7.5b" — dense total size.
_PARAM_DENSE_RE = _re.compile(r"(?<![a-z0-9])\d+(?:\.\d+)?b(?![a-z])", _re.I)
# "a22b" / "a3b" — active-parameter notation, a strong MoE signal.
_PARAM_ACTIVE_RE = _re.compile(r"(?<![a-z0-9])a\d+(?:\.\d+)?b(?![a-z])", _re.I)

# Known MoE families whose id carries no structural NxMB / aNB hint. Distilled
# variants (e.g. deepseek-r1-distill-*) are dense, so they're excluded below.
_MOE_MARKERS = (
    "mixtral", "deepseek-v2", "deepseek-v3", "deepseek-chat", "deepseek-r1",
    "dbrx", "jamba", "arctic", "llama-4", "gpt-oss", "minimax", "olmoe",
    "-moe", "moe-",
)


def detect_param_count(model_id: str, meta: dict | None = None) -> str | None:
    """Best-effort parameter-size label ("8B", "8x7B", "235B") parsed from the
    model id/name. Returns None when no size token is present (e.g. closed
    models like gpt-4o whose size is undisclosed)."""
    name = ""
    if isinstance(meta, dict):
        name = str(meta.get("name") or "")
    hay = f"{model_id or ''} {name}".lower()
    m = _PARAM_MOE_RE.search(hay) or _PARAM_DENSE_RE.search(hay)
    if not m:
        return None
    # Normalise to "8x22B" / "70B": upper-case, then keep the 'x' lower.
    return m.group(0).upper().replace("X", "x")


def detect_moe(model_id: str, meta: dict | None = None) -> bool:
    """Best-effort: is this a Mixture-of-Experts model? Inferred from id/name
    structure (NxMB / aNB notation), known MoE families, or a "mixture of
    experts" mention in the OpenRouter description (when metadata is given)."""
    name = desc = ""
    if isinstance(meta, dict):
        name = str(meta.get("name") or "")
        desc = str(meta.get("description") or "")
    struct = f"{model_id or ''} {name}".lower()
    if _PARAM_MOE_RE.search(struct) or _PARAM_ACTIVE_RE.search(struct):
        return True
    if "distill" not in struct and any(
            mark in struct for mark in _markers("moe_markers", _MOE_MARKERS)):
        return True
    d = desc.lower()
    return "mixture-of-experts" in d or "mixture of experts" in d


# ── Capability ranking for DISCOVERED models ─────────────────────────────────
# Discovered models are imported at intelligence_rank=100 (worst), so the router
# can never escalate to them — even a 480B model scores dead last. We derive a
# real rank PURELY from the model's advertised size + MoE parsed from its id —
# no hardcoded list of "good" model names, so every provider ranks the same way.
# Lower rank = stronger.


def _size_billions(model_id: str) -> float | None:
    """Approx total parameter size in billions parsed from the id, or None.
    For MoE "NxMB" notation returns N*M (a rough total-parameter figure)."""
    label = detect_param_count(model_id)
    if not label:
        return None
    s = label.lower().rstrip("b")
    try:
        if "x" in s:
            a, b = s.split("x", 1)
            return float(a) * float(b)
        return float(s)
    except ValueError:
        return None


def rank_from_id(model_id: str, meta: dict | None = None) -> tuple[int, int]:
    """Best-effort (intelligence_rank, speed_rank) for a discovered model, from
    its advertised parameter size + MoE only — provider-agnostic and with no
    curated name list, so every provider's big models rank the same way. The
    `:free` variants get a tiny bonus so a free duplicate sorts before its paid
    twin. Lower = stronger / faster."""
    mid = (model_id or "").lower()
    free_bonus = -1 if mid.endswith(":free") else 0

    gb = _size_billions(model_id)
    if gb is not None:
        if gb >= 300:
            intel = 2
        elif gb >= 150:
            intel = 3
        elif gb >= 100:
            intel = 4
        elif gb >= 60:
            intel = 6
        elif gb >= 30:
            intel = 9
        elif gb >= 13:
            intel = 13
        elif gb >= 7:
            intel = 17
        else:
            intel = 22
        if detect_moe(model_id, meta):
            intel = max(1, intel - 1)  # MoE punches above its dense weight
        # Speed tracks ACTIVE params for MoE (a 235B-a22b answers like a ~22B
        # model), so big MoE models aren't wrongly treated as slow.
        eff = gb
        am = _PARAM_ACTIVE_RE.search(mid)
        if am:
            try:
                eff = float(am.group(0).lower().lstrip("a").rstrip("b"))
            except ValueError:
                pass
        speed = 2 if eff < 13 else (6 if eff < 70 else 10)
        return max(1, intel + free_bonus), speed
    # Unknown size (no token in the id) — a neutral mid rank: reachable and
    # ranked below models with a known large size, but never excluded.
    return max(1, 30 + free_bonus), 8


async def backfill_discovered_ranks() -> int:
    """(Re)assign intelligence/speed ranks to every DISCOVERED model from the
    current size heuristic. Runs each startup; curated seed models (the ones in
    MODEL_SEED, which carry hand-tuned ranks + free-tier rate limits) are left
    untouched. Idempotent — re-deriving gives the same ranks — and self-healing
    if the heuristic changes. Without it, discovered 235B/480B models keep the
    import default (100) and are never routed even with route_all_models on."""
    from sqlalchemy import select

    from storage.db import get_session_factory
    from storage.models import LLMModel

    factory = get_session_factory()
    if factory is None:
        return 0
    curated = {(r[0], r[1]) for r in MODEL_SEED}  # (platform, model_id)
    async with factory() as session:
        rows = (await session.execute(select(LLMModel))).scalars().all()
        n = 0
        for m in rows:
            if (m.platform, m.model_id) in curated:
                continue  # never override curated/seed ranks
            intel, speed = rank_from_id(m.model_id)
            if m.intelligence_rank != intel or m.speed_rank != speed:
                m.intelligence_rank = intel
                m.speed_rank = speed
                n += 1
        if n:
            await session.commit()
        return n


async def backfill_vision_flags() -> int:
    """Flip `supports_vision` True on existing models whose id looks multimodal,
    so the user's DISCOVERED vision models (not just curated ones) become usable
    for image turns — no hardcoded list. Idempotent; only flips False→True."""
    from sqlalchemy import select, update

    from storage.db import get_session_factory
    from storage.models import LLMModel

    factory = get_session_factory()
    if factory is None:
        return 0
    async with factory() as session:
        rows = (
            await session.execute(
                select(LLMModel.id, LLMModel.model_id).where(
                    LLMModel.supports_vision.is_(False))
            )
        ).all()
        ids = [mid for (mid, model_id) in rows if is_vision_model_id(model_id)]
        if not ids:
            return 0
        await session.execute(
            update(LLMModel).where(LLMModel.id.in_(ids)).values(supports_vision=True)
        )
        await session.commit()
        return len(ids)


def seed_rows() -> list[dict]:
    """MODEL_SEED as a list of insert-ready dicts (skips unknown platforms)."""
    rows: list[dict] = []
    for r in MODEL_SEED:
        platform = r[0]
        if platform not in PROVIDERS:
            continue  # provider was dropped; don't seed an unroutable model
        rows.append(
            {
                "platform": platform,
                "model_id": r[1],
                "display_name": r[2],
                "intelligence_rank": r[3],
                "speed_rank": r[4],
                "size_label": r[5],
                "rpm_limit": r[6],
                "rpd_limit": r[7],
                "tpm_limit": r[8],
                "tpd_limit": r[9],
                "monthly_token_budget": r[10],
                "context_window": r[11],
                "enabled": (platform, r[1]) not in _DISABLED_BY_DEFAULT,
                "supports_vision": (platform, r[1]) in VISION_MODELS,
            }
        )
    return rows


async def seed_provider(platform: str) -> int:
    """Idempotently seed ONE provider's curated models (enabled) + fallback rows.

    Called when a key is added for `platform`, so the curated free-tier models
    (with their rate-limit metadata + intelligence ranks) become routable
    immediately. Discovery then layers the provider's full /models list on top
    (disabled). Returns the number of curated models added.

    Fallback priority for curated models = their intelligence_rank, so the
    default chain is sensibly ordered regardless of which provider was keyed
    first. No-op if Postgres isn't ready or the provider has no curated rows.
    """
    from sqlalchemy import select

    from storage.db import get_session_factory
    from storage.models import LLMFallbackConfig, LLMModel

    rows = [r for r in seed_rows() if r["platform"] == platform]
    if not rows:
        return 0
    factory = get_session_factory()
    if factory is None:
        return 0

    async with factory() as session:
        existing = {
            m.model_id
            for m in (
                await session.execute(select(LLMModel).where(LLMModel.platform == platform))
            ).scalars().all()
        }
        added = [r for r in rows if r["model_id"] not in existing]
        for row in added:
            session.add(LLMModel(**row))
        await session.flush()  # assign ids

        configured = {
            fc.model_db_id
            for fc in (await session.execute(select(LLMFallbackConfig))).scalars().all()
        }
        models = (
            await session.execute(select(LLMModel).where(LLMModel.platform == platform))
        ).scalars().all()
        for m in models:
            if m.id in configured:
                continue
            session.add(
                LLMFallbackConfig(
                    model_db_id=m.id, priority=m.intelligence_rank, enabled=m.enabled
                )
            )
        await session.commit()
        return len(added)


async def reseed_keyed_providers() -> int:
    """Re-seed curated models for every provider that has an enabled key, so new
    free models added to MODEL_SEED appear on restart WITHOUT re-adding the key
    (e.g. the free vision models). Idempotent — `seed_provider` only inserts
    models that don't already exist, and adds their fallback-chain rows."""
    from sqlalchemy import select

    from storage.db import get_session_factory
    from storage.models import LLMApiKey

    factory = get_session_factory()
    if factory is None:
        return 0
    async with factory() as session:
        keyed = {
            p
            for (p,) in (
                await session.execute(
                    select(LLMApiKey.platform).where(LLMApiKey.enabled.is_(True))
                )
            ).all()
        }
    total = 0
    for platform in sorted(keyed):
        try:
            total += await seed_provider(platform)
        except Exception:  # noqa: BLE001 — one bad provider must not block others
            pass
    return total


async def prune_unknown_providers() -> int:
    """Delete DB rows (models, keys; fallback cascades) for any platform no
    longer in the catalogue — so a removed provider (e.g. Zhipu) fully
    disappears from the UI and routing on the next restart."""
    from sqlalchemy import delete

    from storage.db import get_session_factory
    from storage.models import LLMApiKey, LLMModel

    factory = get_session_factory()
    if factory is None:
        return 0
    known = set(PROVIDERS.keys())
    if not known:
        return 0
    async with factory() as session:
        result = await session.execute(
            delete(LLMModel).where(LLMModel.platform.notin_(known))
        )
        await session.execute(
            delete(LLMApiKey).where(LLMApiKey.platform.notin_(known))
        )
        await session.commit()
        return result.rowcount or 0


async def prune_keyless_providers() -> int:
    """Delete catalog models (+ cascaded fallback rows) for every provider that
    has no enabled key, so the UI shows nothing until a key is added.

    Runs on startup and after a key is removed. Anonymous-tier providers
    (llm7 / pollinations / kilo) are EXEMPT — they route keyless, so pruning
    them made them unroutable after every restart (and wiped any models the
    user had enabled) until a manual "Refresh models". Returns the number of
    model rows deleted.
    """
    from sqlalchemy import delete, select

    from storage.db import get_session_factory
    from storage.models import LLMApiKey, LLMModel

    factory = get_session_factory()
    if factory is None:
        return 0

    # Anonymous providers are legitimately keyless but routable — keep them.
    try:
        anon = {s.platform for s in all_providers()
                if getattr(s, "allow_anonymous", False)}
    except Exception:  # noqa: BLE001
        anon = set()

    async with factory() as session:
        keyed = {
            p
            for (p,) in (
                await session.execute(
                    select(LLMApiKey.platform).where(LLMApiKey.enabled.is_(True))
                )
            ).all()
        }
        keep = keyed | anon
        stmt = delete(LLMModel)
        if keep:
            stmt = stmt.where(LLMModel.platform.notin_(keep))
        result = await session.execute(stmt)  # fallback rows cascade (FK ondelete)
        await session.commit()
        return result.rowcount or 0
