"""
Sentence-transformers embedder.

Wraps the configured model in a thin singleton so the model is loaded
once per process (loading bge-small takes ~1s on first call). Native
deps (torch, sentence-transformers) are imported lazily so the app
boots even when those packages are not installed — the embedder only
fails when something actually asks it for an embedding.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config_loader import cfg


class EmbedderError(RuntimeError):
    """Raised when the embeddings model cannot be loaded or used."""


@lru_cache(maxsize=1)
def _model():
    """Load the sentence-transformers model lazily and cache it forever."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbedderError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc

    # `device: auto` → GPU when present (bge-m3 embeds a whole resume in <1s
    # on CUDA vs tens of seconds on CPU — the "resume upload is very slow"
    # report). Explicit `cpu`/`cuda` values are honored unchanged, so the
    # CPU-only VPS keeps its behavior.
    device = (cfg.embeddings.device or "cpu").lower()
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    # local-first: a cached model skips the online "newer revision?" HEAD check
    # (which hangs on a slow/blocked connection); un-cached → downloads.
    try:
        return SentenceTransformer(cfg.embeddings.model, device=device,
                                   local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached yet → fetch
        return SentenceTransformer(cfg.embeddings.model, device=device)


# ---- Cold-start protection -------------------------------------------------
# Loading bge-m3 takes tens of seconds on CPU (minutes on first download). A
# synchronous load inside a request FREEZES the event loop — no SSE keepalives,
# no tokens — until the client watchdog kills the connection. Request-path
# callers must therefore check `is_ready()` and fail open (regex fallback)
# while `ensure_loading_in_background()` warms the model in a thread.
import threading as _threading

_LOAD_STARTED = False
_LOAD_LOCK = _threading.Lock()


def is_ready() -> bool:
    """True once the model is loaded (an embed call will be fast)."""
    try:
        return _model.cache_info().currsize > 0
    except Exception:  # noqa: BLE001
        return False


def ensure_loading_in_background() -> None:
    """Kick a ONE-TIME daemon thread that loads + warms the model. Returns
    immediately; safe to call on every request."""
    global _LOAD_STARTED
    with _LOAD_LOCK:
        if _LOAD_STARTED:
            return
        _LOAD_STARTED = True

    def _load() -> None:
        try:
            embed(["warmup"])
        except Exception:  # noqa: BLE001 — unavailable → callers stay on regex
            pass

    _threading.Thread(target=_load, name="embedder-warmup",
                      daemon=True).start()


# G7: a small LRU cache for SINGLE-string embeds so the same query embedded more
# than once in a turn (Understanding pass, intent classifier, RAG retrieval) costs
# one forward pass, not three. Returns copies so callers can't corrupt the cache.
from collections import OrderedDict as _OrderedDict

_ONE_CACHE: "_OrderedDict[str, list[float]]" = _OrderedDict()
_ONE_MAX = 256
# Disabled by the test harness (conftest) so a test's fake embedder can't leak a
# vector into a later test via this process-global cache. On in production.
_CACHE_ENABLED = True


def embed(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors, one per input text.

    Normalises vectors to unit length — required for cosine similarity to
    behave correctly with Chroma's default L2 metric and for the
    follow-up similarity check downstream. Single-string calls are LRU-cached
    (G7) so a query embedded several times in one turn is computed once.
    """
    if not texts:
        return []
    if len(texts) == 1 and _CACHE_ENABLED:
        key = texts[0]
        hit = _ONE_CACHE.get(key)
        if hit is not None:
            _ONE_CACHE.move_to_end(key)
            return [list(hit)]
        vec = _model().encode(texts, normalize_embeddings=True).tolist()[0]
        _ONE_CACHE[key] = vec
        if len(_ONE_CACHE) > _ONE_MAX:
            _ONE_CACHE.popitem(last=False)
        return [list(vec)]
    model = _model()
    arr = model.encode(texts, normalize_embeddings=True)
    return arr.tolist()


def embed_one(text: str) -> list[float]:
    """Convenience wrapper for single-string callers (queries)."""
    return embed([text])[0]


def dimensions() -> int:
    """The output dimension of the configured model (used to validate stores)."""
    return _model().get_sentence_embedding_dimension()
