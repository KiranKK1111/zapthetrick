"""
FastAPI application entry point.

Mounts the three Phase-1 routers (chat, resume, settings) and loads the
application configuration from config.yaml on startup. Later phases add
more routers (live audio WebSocket, code solver, ...) — they go in
`app/api/routes_*.py` and get included here.

Run with:
    uvicorn app.main:app --reload
or:
    python -m app.main
"""
# Kill Chroma's anonymous PostHog telemetry BEFORE any module loads
# `chromadb`. The bundled posthog has a signature mismatch with
# `chromadb.telemetry.product.posthog` and spams the log on every
# startup, collection-create, AND query. Doing this first thing in
# `app.main` guarantees the env vars are set before any of our import
# chain can transitively pull chromadb in.
import os as _os

_os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
_os.environ.setdefault(
    "CHROMA_TELEMETRY_IMPL",
    "chromadb.telemetry.product.NoopProductTelemetryClient",
)

# Prefer OFFLINE HuggingFace loading when the models are already downloaded.
# `from_pretrained` otherwise fires an online "is there a newer revision?" HEAD
# request per model on startup; on a slow or blocked connection to
# huggingface.co that hangs (10s timeout x 5 retries ≈ 30s of ReadTimeout noise
# PER model) before falling back to the cache — the slow/noisy startup the user
# sees. `local_files_only=True` is NOT enough (transformers 4.57 still makes the
# HEAD call); only HF_HUB_OFFLINE fully stops it, and it must be set before
# huggingface_hub is first imported — hence here, at the very top.
#
# We enable it automatically ONLY when the cache already holds models (so a
# populated install is fast/silent), and leave a FRESH install online so it can
# still download. Explicit user setting always wins. To fetch a NEW model later,
# run with HF_HUB_OFFLINE=0.
if _os.environ.get("HF_HUB_OFFLINE") is None:
    try:
        import pathlib as _pl

        _hub = _os.environ.get("HF_HUB_CACHE")
        if not _hub:
            _home = _os.environ.get("HF_HOME")
            _hub = (_pl.Path(_home) / "hub") if _home else (
                _pl.Path.home() / ".cache" / "huggingface" / "hub")
        _hubp = _pl.Path(_hub)
        _cached = _hubp.is_dir() and any(
            d.name.startswith("models--") for d in _hubp.iterdir())
        if _cached:
            _os.environ["HF_HUB_OFFLINE"] = "1"
            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    except Exception:  # noqa: BLE001 — never block startup over a cache probe
        pass

# Give the app's loggers a stdout handler. uvicorn only configures its own
# `uvicorn.*` loggers, so without this every app log below WARNING (including
# "migration: starting/complete/failed" from storage.bootstrap) is silently
# dropped — which once hid a migration failure in production for days.
# basicConfig is a no-op when the root logger already has handlers, so an
# embedding host (tests, the frozen desktop app) that configures logging
# first is unaffected.
import logging as _logging

_logging.basicConfig(
    level=_logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.update_check import APP_VERSION
from app.middleware.selective_gzip import SelectiveGZipMiddleware

from app.api.routes_agent_approvals import router as agent_approvals_router
from app.api.routes_agents import router as agents_router
from app.api.routes_attachments import router as attachments_router
from app.api.routes_blob import router as blob_router
from app.api.routes_chat import router as chat_router
from app.api.routes_chat_agent import router as chat_agent_router
from app.api.routes_documents import router as documents_router
from app.api.routes_eval import router as eval_router
from app.api.routes_mcp import router as mcp_router
from app.api.routes_messages import router as messages_router
from app.api.routes_prefetch import router as prefetch_router
from app.api.routes_projects import router as projects_router
from app.api.routes_wsgroup import router as wsgroup_router
from app.api.routes_providers import router as providers_router
from app.api.routes_resume import router as resume_router
from app.api.routes_settings import router as settings_router
from app.api.routes_setup import router as setup_router
from app.api.routes_solve import router as solve_router
from app.api.routes_stt import router as stt_router
from app.api.routes_vision import router as vision_router
from app.api.routes_sandbox import router as sandbox_router
from app.api.routes_jobs import router as jobs_router
from app.api.routes_jobs import health_router as health_router
from app.api.routes_workspace import router as workspace_router
from app.api.routes_ws import router as ws_router
from app.api.routes_live import router as live_router
from app.core.config_loader import cfg, get_config
import asyncio

from storage import bootstrap_storage, shutdown_storage
from storage.bootstrap import run_migrations_in_background


async def _bootstrap_llm_routing(migration_task: "asyncio.Task") -> None:
    """Wait for migrations, then seed the LLM catalog + start health checks.

    Every step is best-effort: a Postgres outage or a missing encryption key
    must not crash the app — only the `auto` routing path degrades, with a
    clear 503 from the providers API.
    """
    import logging as _logging

    log = _logging.getLogger(__name__)
    try:
        await migration_task
    except Exception:  # noqa: BLE001 — migration errors are logged elsewhere
        pass

    from storage import bootstrap as _bs

    if not getattr(_bs, "POSTGRES_READY", False):
        log.info("llm routing: Postgres not ready — skipping catalog seed/health.")
        return

    try:
        from app.llm.crypto import init_encryption_key

        await init_encryption_key()
    except Exception as exc:  # noqa: BLE001 — bad env key; key mgmt degrades but
        # the rest of the catalog still works, so continue rather than return.
        log.warning("llm routing: encryption key init failed (%s)", exc)

    try:
        # Models are seeded per-provider when a key is added — here we just
        # clear any models for providers that have no key, so the catalog
        # starts empty until the user supplies keys.
        from app.llm.catalog import (
            prune_keyless_providers,
            prune_unknown_providers,
        )

        # Drop providers removed from the catalogue (e.g. Zhipu) entirely.
        gone = await prune_unknown_providers()
        if gone:
            log.info("llm routing: removed %d models for retired providers", gone)
        pruned = await prune_keyless_providers()
        if pruned:
            log.info("llm routing: pruned %d models for keyless providers", pruned)
        # Re-seed keyed providers so newly-curated free models (e.g. the free
        # vision models) reach an existing DB on restart, without re-adding keys.
        from app.llm.catalog import (
            backfill_discovered_ranks,
            backfill_vision_flags,
            reseed_keyed_providers,
        )

        added = await reseed_keyed_providers()
        if added:
            log.info("llm routing: seeded %d new curated free models", added)
        # Flag discovered multimodal models so image turns can route across the
        # user's FULL configured catalog, not just the curated vision models.
        vis = await backfill_vision_flags()
        if vis:
            log.info("llm routing: flagged %d discovered models as vision", vis)
        # Rank discovered models by parameter size / MoE / family so the router
        # can escalate hard/expert work to the big models (235B/480B/Claude),
        # not just the curated small ones. Idempotent (only 100/100 rows).
        ranked = await backfill_discovered_ranks()
        if ranked:
            log.info("llm routing: ranked %d discovered models by capability", ranked)
    except Exception:  # noqa: BLE001
        log.exception("llm routing: catalog prune/reseed failed")

    try:
        from app.llm.ratelimit import (
            load_cooldowns_from_db,
            load_usage_from_db,
        )

        await load_cooldowns_from_db()
        # Also rehydrate the sliding rate-limit windows so a restart doesn't
        # silently re-grant every provider's daily budget.
        await load_usage_from_db()
    except Exception:  # noqa: BLE001
        log.debug("llm routing: cooldown rehydrate failed", exc_info=True)

    try:
        from app.llm.health import start_health_loop

        start_health_loop()
    except Exception:  # noqa: BLE001
        log.exception("llm routing: health loop failed to start")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force config load on startup so any YAML error fails loudly here
    # rather than on the first request.
    get_config()
    # Wire the live-config event bus so subsystems rebuild themselves
    # in-place when the user saves Settings. No-op if already done.
    try:
        from app.settings.subscribers import register_default_subscribers

        register_default_subscribers()
    except Exception:  # noqa: BLE001 — bus failures must not block startup
        import logging as _logging

        _logging.getLogger(__name__).exception("config bus init failed")
    # Architecture.md §"MCP": load any servers declared in
    # cfg.mcp.servers into the registry so the Tools screen has
    # something to render on first launch.
    try:
        from app.core.config_loader import cfg as _cfg
        from app.mcp.registry import registry as _mcp

        if getattr(_cfg, "mcp", None) and _cfg.mcp.servers:
            _mcp.bootstrap_from_config(_cfg.mcp.servers)
    except Exception:  # noqa: BLE001
        import logging as _logging

        _logging.getLogger(__name__).exception("mcp bootstrap failed")
    # Bring up the full storage stack — Postgres pool, Qdrant client,
    # cache, blob store, optional graph. Each backend is lazy past this
    # point, so a Qdrant outage doesn't sink chat routes.
    await bootstrap_storage()
    # Migrations run in the background so uvicorn's startup completes
    # immediately — the app accepts requests right away. Data routes
    # 503 with a clear message until `MIGRATION_STATE == "ready"`.
    migration_task = asyncio.create_task(
        run_migrations_in_background(), name="lifespan-migrations"
    )
    # After migrations land, bring up the multi-provider LLM routing engine:
    # resolve the encryption key, seed the curated catalog, rehydrate
    # cooldowns, and start the periodic key health checker. Runs in the
    # background so startup stays instant; routing 503s with a clear message
    # until it's ready.
    llm_task = asyncio.create_task(
        _bootstrap_llm_routing(migration_task), name="lifespan-llm-routing"
    )
    # Zero-touch provisioning: if Docker is available, bring up the bundled
    # database stack, enable pgvector, and run migrations automatically — so a
    # fresh machine goes straight to the app with no setup clicks. No-op when
    # already provisioned; leaves a clear status (surfaced via /api/setup/checks)
    # if Docker is missing/stopped. Passed `migration_task` so it waits for the
    # startup migration before running its own (no concurrent alembic).
    from app.api.routes_setup import auto_provision

    provision_task = asyncio.create_task(
        auto_provision(migration_task), name="lifespan-auto-provision"
    )
    # ── Warm the local models + report status for the UI modal ──────────
    # The app warms its STT chain + the sentence-transformers embedder in
    # daemon threads at boot; on the FIRST deploy these download from
    # HuggingFace (~3 GB). Each model's download/load stage is reported to
    # app.models_warmup so the client can show a "downloading models" modal
    # (GET /api/models/warmup) until everything is ready. Request paths also
    # fail open (regex fallback) while the embedder loads. Fail-open: a warm-up
    # error never blocks boot.
    try:
        import threading

        from app import models_warmup as _mw
        from app.core.config_loader import cfg as _wcfg

        _STT_LABELS = {
            "parakeet": "Speech recognition (Parakeet)",
            "qwen_asr": "Speech recognition (Qwen3-ASR)",
            "faster_whisper": "Speech recognition (Whisper)",
        }
        _stt_chain = [_wcfg.stt.provider] + list(
            getattr(_wcfg.stt, "fallback_providers", None) or [])

        def _stt_repo(provider: str) -> str | None:
            """HF repo id backing an STT provider, for the cache probe."""
            if provider == "qwen_asr":
                return getattr(_wcfg.stt, "qwen_model", "Qwen/Qwen3-ASR-1.7B")
            if provider == "parakeet":
                # onnx-asr aliases "nemo-parakeet-tdt-0.6b-vX" to the
                # istupakov ONNX export repos on the hub.
                alias = getattr(_wcfg.stt, "parakeet_model",
                                "nemo-parakeet-tdt-0.6b-v2")
                if "/" in alias:
                    return alias
                if alias.startswith("nemo-"):
                    return f"istupakov/{alias.removeprefix('nemo-')}-onnx"
                return None
            if provider == "faster_whisper":
                size = str(getattr(_wcfg.stt, "model", "base.en"))
                return f"Systran/faster-whisper-{size}"
            return None  # unknown provider — treated as cached

        # Declare everything up-front so the modal renders the full checklist.
        # `cached` marks whether the weights are already on disk — the client
        # only shows the blocking "Preparing models" screen when something
        # actually needs DOWNLOADING (first run), not on every warm start.
        _mw.register("embedder", "Language understanding (bge-m3)",
                     cached=_mw.hf_repo_cached(_wcfg.embeddings.model))
        for _p in _stt_chain:
            _mw.register(f"stt:{_p}",
                         _STT_LABELS.get(_p, f"Speech recognition ({_p})"),
                         cached=_mw.hf_repo_cached(_stt_repo(_p)))

        def _sampler(key: str, repo: str | None,
                     stop: "threading.Event") -> None:
            """Poll bytes-on-disk in the model's HF cache while it downloads and
            report byte-level progress (drives the per-model progress bar)."""
            if not repo:
                return
            try:
                from app import model_sizes as _ms
            except Exception:  # noqa: BLE001
                return
            while not stop.is_set():
                try:
                    _done, _total = _ms.progress_for(repo)
                    _mw.set_progress(key, _done, _total)
                except Exception:  # noqa: BLE001
                    pass
                stop.wait(1.0)

        def _warm_embedder() -> None:
            _repo = str(getattr(_wcfg.embeddings, "model", "") or "")
            _mw.set_stage("embedder", _mw.STAGE_LOADING,
                          "Downloading + loading (first run may take a minute)…")
            _stop = threading.Event()
            threading.Thread(target=_sampler, args=("embedder", _repo, _stop),
                             name="embedder-progress", daemon=True).start()
            try:
                from app.rag.embedder import embed
                embed(["warmup"])           # triggers download + load
                _mw.set_stage("embedder", _mw.STAGE_READY, "Ready")
            except Exception as _e:  # noqa: BLE001
                _mw.set_stage("embedder", _mw.STAGE_ERROR, str(_e)[:140])
            finally:
                _stop.set()

        def _warm_stt() -> None:
            import numpy as _np
            # One second of silence: loading alone is NOT enough — the first
            # real inference pays kernel/CUDA warmup. A dummy transcribe moves
            # that cost to startup so the first real transcript is instant.
            dummy = _np.zeros(16_000, dtype=_np.float32)
            for _idx, _name in enumerate(_stt_chain):
                _key = f"stt:{_name}"
                _is_primary = _idx == 0
                _mw.set_stage(_key, _mw.STAGE_LOADING,
                              "Downloading + loading…")
                _stop = threading.Event()
                threading.Thread(
                    target=_sampler, args=(_key, _stt_repo(_name), _stop),
                    name=f"stt-progress-{_name}", daemon=True).start()
                try:
                    if _name == "qwen_asr":
                        from app.stt import qwen_asr_stt
                        qwen_asr_stt.transcribe(dummy)
                    elif _name == "parakeet":
                        from app.stt import parakeet_stt
                        parakeet_stt.transcribe(dummy)
                    elif _name == "faster_whisper":
                        from app.stt import whisper_stt
                        whisper_stt.transcribe(dummy)
                    else:
                        _mw.set_stage(_key, _mw.STAGE_SKIPPED, "not applicable")
                        continue
                    _mw.set_stage(_key, _mw.STAGE_READY, "Ready")
                except Exception as _e:  # noqa: BLE001 — warmup is best-effort
                    # A FALLBACK that can't load is not a failure — the primary
                    # transcribes. Show it muted (skipped), not an alarming red
                    # error. Only the PRIMARY failing is a real error.
                    if _is_primary:
                        _mw.set_stage(_key, _mw.STAGE_ERROR, str(_e)[:140])
                    else:
                        _mw.set_stage(_key, _mw.STAGE_SKIPPED,
                                      "Optional fallback unavailable")
                finally:
                    _stop.set()

        def _warm_vision() -> None:
            # Pre-load the active LOCAL vision model so the first real
            # screenshot/document parse doesn't pay the ~15s cold load — the
            # cost the user feels as "why is reading this image so slow". Safe
            # to warm on GPU: STT runs on CPU here, so VRAM is free. Best-effort;
            # a too-big model is refused by memcheck (fail-open) and warms on CPU
            # or not at all, never crashing. Uses its own event loop (thread).
            try:
                import asyncio as _asyncio
                from app.core.config_loader import cfg as _cfg
                if not getattr(_cfg.vision, "enabled", True):
                    return
                from app.vision import factory as _vf
                _asyncio.run(_vf.warm_active())
            except Exception:  # noqa: BLE001 — warmup must never break startup
                pass

        def _start_model_warmups() -> None:
            threading.Thread(target=_warm_embedder, name="embedder-warmup",
                             daemon=True).start()
            threading.Thread(target=_warm_stt, name="stt-warmup",
                             daemon=True).start()
            threading.Thread(target=_warm_vision, name="vision-warmup",
                             daemon=True).start()

        # LITE build: the AI runtime pack (torch/onnxruntime) may still need
        # its one-time download — do that FIRST (with its own progress row on
        # the same first-run screen), then warm the models. On full builds /
        # dev (torch importable) this starts the warm-ups immediately.
        from app import runtime_pack as _rp
        _rp.ensure_async_then(_start_model_warmups)
    except Exception:  # noqa: BLE001 — warmup must never block startup
        pass
    # Background maintenance loop (Phase 7 #10/#11/#13 + Phase 1 #24): the single
    # periodic trigger that runs memory consolidation, self-healing diagnostics,
    # the self-benchmark trend snapshot, and the update check on a timer. Each
    # task is fail-open and runs off the request path; disabled via
    # `obs.maintenance: false`.
    try:
        from app.obs.maintenance import start_maintenance_loop
        start_maintenance_loop()
    except Exception:  # noqa: BLE001 — maintenance must never block startup
        import logging as _logging
        _logging.getLogger(__name__).debug("maintenance loop start failed",
                                            exc_info=True)
    try:
        yield
    finally:
        try:
            from app.obs.maintenance import stop_maintenance_loop
            stop_maintenance_loop()
        except Exception:  # noqa: BLE001
            pass
        try:
            from app.llm.health import stop_health_loop

            stop_health_loop()
        except Exception:  # noqa: BLE001
            pass
        for t in (migration_task, llm_task, provision_task):
            if not t.done():
                t.cancel()
        # Tear down MCP subprocesses cleanly so a `Ctrl-C` doesn't
        # leave a tree of zombie servers hanging around.
        try:
            from app.mcp.transport import shutdown_all as _mcp_shutdown

            await _mcp_shutdown()
        except Exception:  # noqa: BLE001
            pass
        await shutdown_storage()


app = FastAPI(
    title="ZapTheTrick Backend",
    description=(
        "Config-driven, provider-agnostic interview-copilot backend. "
        "All knobs live in config.yaml (or POST /api/settings)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Never combine credentialed CORS with a wildcard origin: it's the classic
# unsafe combo (any site could ride a user's credentials) AND browsers reject
# `Access-Control-Allow-Origin: *` with credentials anyway. Credentials are
# enabled only when an explicit origin allow-list is configured in
# `cfg.server.cors_origins`; the permissive `["*"]` default (convenient for the
# local desktop/mobile bundle) runs WITHOUT credentials.
_cors_origins = cfg.server.cors_origins
_cors_wildcard = "*" in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# R37: shrink ordinary JSON/text API payloads on the wire. SSE responses
# (text/event-stream) are passed through unbuffered so token streaming stays
# smooth — see app/middleware/selective_gzip.py.
app.add_middleware(SelectiveGZipMiddleware, minimum_size=500)

app.include_router(chat_router)
app.include_router(chat_agent_router)
app.include_router(attachments_router)
app.include_router(blob_router, prefix="/api")
app.include_router(resume_router)
app.include_router(settings_router)
app.include_router(solve_router)
app.include_router(stt_router)
app.include_router(vision_router)
app.include_router(sandbox_router)
app.include_router(jobs_router)
app.include_router(health_router)
app.include_router(ws_router)
app.include_router(live_router)
app.include_router(agents_router)
app.include_router(projects_router)
app.include_router(mcp_router)
app.include_router(workspace_router)
app.include_router(providers_router)
app.include_router(messages_router)
app.include_router(documents_router)
app.include_router(setup_router, prefix="/api/setup")
app.include_router(agent_approvals_router)
app.include_router(eval_router)
app.include_router(prefetch_router)
app.include_router(wsgroup_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Friendly root so hitting the base URL is not a 404."""
    return {
        "name": cfg.app.name,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
        "settings": "/api/settings",
    }


@app.get("/api/models/warmup")
async def models_warmup_status() -> dict:
    """Live status of the startup model warm-up (STT + embedder download/load),
    so the client can show a 'downloading models' modal until they're ready.
    `all_ready` flips true once every model reaches a terminal state."""
    from app import models_warmup as _mw
    return _mw.snapshot()


@app.get("/api/capabilities")
async def capabilities() -> dict:
    """Runtime capability snapshot (Phase 2 — capability discovery): which
    document formats can be rendered, sandbox/GPU/model/tool availability.
    Consumed by the UI and by capability negotiation."""
    from app.capabilities import capability_snapshot
    return capability_snapshot()


@app.get("/api/obs/decisions")
async def decision_metrics() -> dict:
    """Central decision metrics (Phase 6): pre-gate decision rates (clarify /
    answer / defer), which policy rules fired, and artifact-validation
    outcomes (validated / repaired / degraded / failed)."""
    from app.obs.decision_metrics import snapshot
    return snapshot()


if __name__ == "__main__":
    import uvicorn

    # R37: prefer uvloop on Unix for a faster streaming event loop; it's not
    # available on Windows, so fall back to asyncio's default loop there.
    loop = "auto"
    if _os.name != "nt":
        try:
            import uvloop  # noqa: F401

            loop = "uvloop"
        except Exception:  # noqa: BLE001
            loop = "auto"

    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=True,
        loop=loop,
    )
