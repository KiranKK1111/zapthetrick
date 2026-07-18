"""Default config-bus subscribers — registered at app startup.

Each subscriber is a thin shim: when its section's diff lands it
rebuilds the relevant in-process singleton (LLM client, embedder,
reranker, vector store, …). Heavy lifting lives in the subsystem
modules; this file is just plumbing.

Wire-up:
    from app.settings.subscribers import register_default_subscribers
    register_default_subscribers()    # called once from app.main on startup.
"""
from __future__ import annotations

import logging

from .bus import bus


log = logging.getLogger(__name__)


# ---- handlers ------------------------------------------------------------
async def _on_llm(section: str, diff: dict, full_cfg: dict) -> None:
    """LLM client reads `cfg` lazily per call, so the new model/provider is
    picked up on the next request. The one stateful piece is the shared pooled
    HTTP client (perceived-speed R2) — dispose it so the next request rebuilds
    it with the current timeout/limits."""
    log.info("config: LLM section changed → %s", _summary(diff))
    try:
        from app.core.http_pool import dispose_http_client
        await dispose_http_client()
    except Exception as exc:  # noqa: BLE001 — never let a reload break routing
        log.warning("pooled HTTP client dispose failed: %s", exc)


async def _on_embeddings(section: str, diff: dict, full_cfg: dict) -> None:
    """Reload the sentence-transformer model in place.

    `app.advanced_rag` has never existed in this tree — the RAG package is
    `app.rag` — so until 2026-07-14 this handler raised ImportError into the
    except below on every embeddings change and logged a warning. Net effect:
    switching `embeddings.model` in Settings kept serving vectors from the OLD
    model until the process restarted.

    The embedder has no `reload()`; its state is the `lru_cache(1)` around
    `_model` plus the single-string vector cache. Dropping both makes the next
    embed load the newly configured model.
    """
    log.info("config: embeddings changed → %s", _summary(diff))
    try:
        from app.rag import embedder as _e

        if hasattr(_e, "reload"):
            await _maybe_await(_e.reload())
            return
        _e._model.cache_clear()      # noqa: SLF001 — drop the resident model
        _e._ONE_CACHE.clear()        # noqa: SLF001 — vectors from the OLD model
        _e._LOAD_STARTED = False     # noqa: SLF001 — re-arm background warm-up
    except Exception as exc:  # noqa: BLE001 — a reload must never break the app
        log.warning("embedder reload failed: %s", exc)


async def _on_reranker(section: str, diff: dict, full_cfg: dict) -> None:
    """Drop the cached cross-encoder so a `reranker.model` change takes effect.

    Same latent bug as `_on_embeddings`: this imported `app.advanced_rag.
    reranker`, a package that does not exist. (`app.rag.reranker` is a
    pass-through STUB nobody imports; the live cross-encoder is
    `app.rag.rerank._cross_encoder`, with a second one in
    `app.rag.retriever._reranker` for the resume path.) `reranker.enabled` was
    always honoured live — both call sites re-read cfg per call — but a MODEL
    change stuck to the lru_cached model until restart. Clear both caches.
    """
    log.info("config: reranker changed → %s", _summary(diff))
    try:
        from app.rag import rerank as _r

        _r._cross_encoder.cache_clear()   # noqa: SLF001 — chat-doc reranker
    except Exception as exc:  # noqa: BLE001
        log.warning("reranker reload failed (app.rag.rerank): %s", exc)
    try:
        from app.rag import retriever as _rt

        _rt._reranker.cache_clear()       # noqa: SLF001 — resume reranker
    except Exception as exc:  # noqa: BLE001
        log.warning("reranker reload failed (app.rag.retriever): %s", exc)


async def _on_vector_store(section: str, diff: dict, full_cfg: dict) -> None:
    """Swap the active vector store provider (chroma ↔ qdrant)."""
    log.info("config: vector_store changed → %s", _summary(diff))
    try:
        from storage.vectors import factory as _f

        if hasattr(_f, "reset"):
            _f.reset()
    except Exception as exc:  # noqa: BLE001
        log.warning("vector store reset failed: %s", exc)


async def _on_audio(section: str, diff: dict, full_cfg: dict) -> None:
    """Audio source / VAD change — restart the capture pipeline."""
    log.info("config: audio changed → %s", _summary(diff))
    try:
        from app.audio import capture as _a

        if hasattr(_a, "restart"):
            await _maybe_await(_a.restart())
    except Exception as exc:  # noqa: BLE001
        log.warning("audio restart failed: %s", exc)


async def _on_stt(section: str, diff: dict, full_cfg: dict) -> None:
    """Apply an STT settings change immediately: the single-engine provider
    chain re-reads cfg per utterance already, but the DUAL-engine singleton
    (and any per-model caches keyed at first load) must be reset so a
    provider/model switch takes effect without a restart (2026-07-11, the
    old `app.stt.client` import here never existed — this was a no-op)."""
    log.info("config: stt changed → %s", _summary(diff))
    try:
        # ONE model resident at a time (2026-07-12): the tracked switch frees
        # every engine, then downloads/warms the selected one — its state
        # feeds the Settings popup via GET /api/stt/status. Single-flight:
        # a concurrent /select for the same target joins this switch.
        from app.core.config_loader import cfg as _cfg
        from app.stt import switch as _switch
        provider = str(getattr(_cfg.stt, "provider", "parakeet"))
        target = (f"faster_whisper::{getattr(_cfg.stt, 'model', 'base.en')}"
                  if provider == "faster_whisper" else provider)
        await _switch.start_switch(target)
    except Exception as exc:  # noqa: BLE001
        log.warning("stt reset failed: %s", exc)


async def _on_themes(section: str, diff: dict, full_cfg: dict) -> None:
    """Theme changes are UI-side — backend just logs."""
    log.info("config: themes changed → %s", _summary(diff))


# ---- helpers -------------------------------------------------------------
async def _maybe_await(value):
    """Tolerant `await`: accepts None, sync return, or coroutine."""
    import inspect as _i

    if _i.isawaitable(value):
        await value


def _summary(diff: dict) -> str:
    """Single-line summary of a diff for log lines."""
    if not isinstance(diff, dict):
        return repr(diff)[:120]
    return ", ".join(sorted(diff.keys()))[:200]


# ---- registration --------------------------------------------------------
def register_default_subscribers() -> None:
    """Wire the built-in subscribers. Idempotent (clears existing first)."""
    # Re-register cleanly — the bus is a singleton so during reload we
    # don't want to double-fire.
    for section in (
        "llm",
        "embeddings",
        "reranker",
        "vector_store",
        "audio",
        "stt",
        "themes",
    ):
        bus._subs.pop(section, None)  # noqa: SLF001 — internal reset

    bus.subscribe("llm", _on_llm)
    bus.subscribe("embeddings", _on_embeddings)
    bus.subscribe("reranker", _on_reranker)
    bus.subscribe("vector_store", _on_vector_store)
    bus.subscribe("audio", _on_audio)
    bus.subscribe("stt", _on_stt)
    bus.subscribe("themes", _on_themes)


__all__ = ["register_default_subscribers"]
