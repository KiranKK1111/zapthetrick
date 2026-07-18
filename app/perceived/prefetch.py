"""Intent prediction + prefetch while the user types (perceived-speed R1, R2).

While the user is composing, predict the likely shape of the request LOCALLY
(no remote generation — R1.5) and warm the path: ensure the shared pooled HTTP
client exists (so the first real request reuses a live keep-alive connection)
and best-effort pre-open a connection to the active provider. Work is keyed by a
`prefetch_token`; `reuse(token)` consumes it on a matching submit and
`discard(token)` drops it on a mismatch (R1.3, R1.4). Stale tokens are evicted.

ALL of this is gated by the SpeculationBudget, so with `speculation_enabled=False`
nothing runs (R19.4) and the prefetch endpoint is a cheap no-op.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from app.chat.difficulty import _TECH_RE, is_build_request
from app.perceived.budget import budget as _default_budget

log = logging.getLogger(__name__)

# Tokens older than this with no submit are dropped (R1.4 housekeeping).
_TOKEN_TTL_S = 30.0
_MAX_TOKENS = 64
# Predictive context prefetch (P5 #3): don't spend an embed on a half-typed
# fragment — wait until the partial looks like a real query.
_MIN_PREFETCH_CHARS = 12
# Reuse across a CHANGED prompt (P5 #5): how similar the submitted request must
# be to the warmed partial before the warmed retrieval is reused as a warm start.
_REUSE_SIMILARITY = 0.6


def predictive_prefetch_enabled() -> bool:
    """`cfg.perceived.predictive_prefetch` — enabling default True. When on,
    `warm()` also precomputes the query embedding (and best-effort context) so
    the first real request reuses it, not just the warmed socket."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "perceived", None),
                            "predictive_prefetch", True))
    except Exception:  # noqa: BLE001
        return True


@dataclass
class Prediction:
    """A LOCAL, model-free guess at the composing request's shape."""
    topic: str = "general"        # "coding" | "general"
    complexity: str = "standard"  # "trivial" | "standard"
    is_build: bool = False


class IntentPredictor:
    """Deterministic, model-free predictor — safe to run on every keystroke."""

    def predict(self, partial: str) -> Prediction:
        t = (partial or "").strip()
        if not t:
            return Prediction()
        coding = bool(_TECH_RE.search(t.lower()))
        complexity = "trivial" if len(t) < 12 else "standard"
        return Prediction(
            topic="coding" if coding else "general",
            complexity=complexity,
            is_build=is_build_request(t),
        )


@dataclass
class _Warm:
    token: str
    prediction: Prediction
    created: float = field(default_factory=time.monotonic)
    partial: str = ""                       # the text warmed (for reuse matching)
    embedding: list[float] | None = None    # precomputed query embedding (P5 #3)
    retrieval: list | None = None           # precomputed context snippets (P5 #3)


@dataclass
class ReusedContext:
    """What a submit was able to reuse from warmed work (P5 #5)."""
    exact: bool                    # request identical to the warmed partial
    similarity: float              # 0..1 similarity to the warmed partial
    embedding: list[float] | None  # reusable iff the query text is unchanged
    retrieval: list | None         # reusable as a warm start when similar enough

    @property
    def any(self) -> bool:
        return self.embedding is not None or self.retrieval is not None


class PrefetchManager:
    """Warms connections + predicted handles before submit; never generates."""

    def __init__(self, budget=None) -> None:
        self._warm: dict[str, _Warm] = {}
        self._budget = budget or _default_budget

    async def warm(self, partial: str, *,
                   retrieve: "Callable[[str], object] | None" = None) -> str | None:
        """Predict + warm. Returns a `prefetch_token`, or None when speculation
        is disabled / over budget. NEVER starts answer generation (R1.5).

        Beyond warming the HTTP socket, when `predictive_prefetch` is on this
        also precomputes the query EMBEDDING (the expensive GPU/remote op) and,
        if `retrieve` is supplied, best-effort context — so the first real
        request reuses real work, not just a live connection (P5 #3)."""
        if not self._budget.allow(kind="prefetch"):
            return None
        pred = IntentPredictor().predict(partial)
        self._budget.account(1)
        # Ensure the pooled client exists so the first real request reuses it.
        try:
            from app.core.http_pool import get_http_client
            get_http_client()
        except Exception:  # noqa: BLE001 — warming must never raise
            pass
        await self._warm_connection(pred)
        token = uuid.uuid4().hex
        entry = _Warm(token=token, prediction=pred, partial=(partial or "").strip())
        await self._prefetch_context(entry, retrieve)
        self._warm[token] = entry
        self._evict_stale()
        return token

    async def _prefetch_context(
        self, entry: "_Warm", retrieve: "Callable[[str], object] | None"
    ) -> None:
        """Precompute the query embedding + optional context for `entry.partial`.
        Fully guarded — any failure leaves the entry with just a warm socket."""
        text = entry.partial
        if not predictive_prefetch_enabled() or len(text) < _MIN_PREFETCH_CHARS:
            return
        # Embedding: warms the embedder's own LRU cache too, so even if the
        # submit text differs slightly the model is hot.
        try:
            from app.rag import embedder
            entry.embedding = await asyncio.to_thread(embedder.embed_one, text)
        except Exception:  # noqa: BLE001 — embedding is best-effort
            entry.embedding = None
        if retrieve is not None:
            try:
                res = retrieve(text)
                if asyncio.iscoroutine(res) or asyncio.isfuture(res):
                    res = await res
                entry.retrieval = list(res) if res else None
            except Exception:  # noqa: BLE001 — retrieval prefetch is best-effort
                entry.retrieval = None

    async def _warm_connection(self, pred: Prediction) -> None:
        """Best-effort: pre-open a connection to the active provider so TLS +
        keep-alive are ready. Fully guarded — any failure is ignored."""
        try:
            from app.core.config_loader import cfg
            from app.core.http_pool import get_http_client
            base = (getattr(cfg.llm, "base_url", "") or "").rstrip("/")
            # Only warm a concrete single-provider base_url; the "auto" router
            # spans many providers — warming all of them per keystroke is
            # impractical, and the shared pool already removes most setup cost.
            if cfg.llm.provider != "auto" and base:
                client = get_http_client()
                try:
                    await client.get(base, timeout=2.0)
                except Exception:  # noqa: BLE001 — a 401/404 still warms the socket
                    pass
        except Exception:  # noqa: BLE001
            pass

    def reuse(self, token: str | None, request: str = "") -> bool:
        """Consume a token on submit; True when warmed work was reused (R1.3)."""
        if not token:
            return False
        return self._warm.pop(token, None) is not None

    def reuse_artifacts(self, token: str | None, request: str = "") -> ReusedContext:
        """Consume a token AND return the reusable warmed artifacts (P5 #5).

        Unlike `reuse` (bool), this salvages the precomputed embedding/context
        even when the submitted `request` DIFFERS from the warmed partial — the
        common case where the user kept typing after the prefetch fired:

          * identical text  → reuse the embedding AND the retrieval;
          * similar text (≥ `_REUSE_SIMILARITY`) → reuse the retrieval as a warm
            start (the embedding is text-exact, so it's dropped);
          * unrelated text  → reuse nothing.

        Always pops the token. Fail-open: any error → an empty ReusedContext."""
        empty = ReusedContext(exact=False, similarity=0.0, embedding=None,
                              retrieval=None)
        if not token:
            return empty
        warm = self._warm.pop(token, None)
        if warm is None:
            return empty
        try:
            req = (request or "").strip()
            if not req or req == warm.partial:
                return ReusedContext(True, 1.0, warm.embedding, warm.retrieval)
            sim = difflib.SequenceMatcher(None, warm.partial, req).ratio()
            if sim >= _REUSE_SIMILARITY:
                # Retrieval for the earlier prefix is a valid warm start; the
                # embedding is text-specific, so only reuse it on an exact match.
                return ReusedContext(False, round(sim, 3), None, warm.retrieval)
            return ReusedContext(False, round(sim, 3), None, None)
        except Exception:  # noqa: BLE001
            return empty

    def discard(self, token: str | None) -> None:
        """Drop a token's warmed work (mismatch / cancel — R1.4)."""
        if token:
            self._warm.pop(token, None)

    def _evict_stale(self) -> None:
        now = time.monotonic()
        for tok in [t for t, w in self._warm.items()
                    if now - w.created > _TOKEN_TTL_S]:
            self._warm.pop(tok, None)
        # Hard cap so a flood of tokens can't grow unbounded.
        if len(self._warm) > _MAX_TOKENS:
            for tok in sorted(self._warm, key=lambda t: self._warm[t].created)[
                : len(self._warm) - _MAX_TOKENS
            ]:
                self._warm.pop(tok, None)

    @property
    def pending(self) -> int:
        return len(self._warm)


# Process-wide manager shared by the prefetch endpoint + the chat route.
manager = PrefetchManager()


async def warm_live_provider() -> None:
    """Best-effort: pre-open a TLS connection to the provider hosting the
    pinned live answer model (cfg.llm.live_model), so the first live answer
    of a turn doesn't pay DNS + TLS setup. The pool's keepalive expiry is
    ~60s and interview questions are routinely further apart, so without
    this the connection is cold exactly when latency matters most.

    Called from the live audio path when speech STARTS — the handshake
    completes while the speaker is still talking. Never generates anything,
    never raises, and respects the speculation kill-switch."""
    try:
        if not _default_budget.allow(kind="prefetch"):
            return
        from app.core.config_loader import cfg
        from app.core.http_pool import get_http_client
        model = (getattr(cfg.llm, "live_model", "") or "").strip()
        base = ""
        if model:
            from app.llm.catalog import MODEL_SEED, get_provider_spec
            for row in MODEL_SEED:
                if row[1] == model:
                    spec = get_provider_spec(row[0])
                    if spec is not None and "{" not in spec.base_url:
                        base = spec.base_url.rstrip("/")
                    break
        if not base and cfg.llm.provider != "auto":
            base = (getattr(cfg.llm, "base_url", "") or "").rstrip("/")
        if not base:
            return
        client = get_http_client()
        try:
            await client.get(base, timeout=2.0)
        except Exception:  # noqa: BLE001 — a 401/404 still warms the socket
            pass
    except Exception:  # noqa: BLE001 — warming must never raise
        pass


__all__ = ["Prediction", "IntentPredictor", "PrefetchManager", "manager",
           "warm_live_provider", "ReusedContext", "predictive_prefetch_enabled"]
