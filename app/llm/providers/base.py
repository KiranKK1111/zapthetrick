"""Base provider adapter — shared httpx plumbing + the typed error.

Ported from freellmapi's `providers/base.ts`. Each adapter exposes:

  * `complete(...)  -> str`              — non-streaming, returns the text.
  * `stream(...)    -> AsyncGenerator`   — yields text deltas.
  * `validate_key(...) -> bool`          — False only on a confirmed 401/403.

Messages arrive already in OpenAI format (the caller converts our internal
Ollama-style image messages via `LLMClient._to_openai_messages`). Errors
are raised as `ProviderError` carrying the HTTP status so the router can
decide whether to fall through to the next model.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx

from app.core.http_pool import get_http_client
from app.llm.catalog import ProviderSpec


@asynccontextmanager
async def pooled_client():
    """Yield the shared pooled HTTP client without closing it on exit
    (perceived-speed R2) — drop-in for ``async with httpx.AsyncClient(...)``.
    Per-request timeouts move onto the actual request call."""
    yield get_http_client()


# Transient HTTP statuses — always worth trying the next model/key.
# 402 = out of credits, 403 = forbidden / "add a credit card" / customer
# verification: retrying the SAME provider is futile, but a DIFFERENT one may
# well work, so we fall through (the engine cools the walled one down for a day
# so it stops getting picked).
RETRYABLE_STATUSES = {402, 403, 404, 408, 409, 413, 425, 429, 500, 502, 503, 504}

# Message fragments that mean "this model id is gone/invalid — stop using it".
# Providers signal this inconsistently: OpenRouter returns 404 "No endpoints
# found for <id>" OR 400 "<id> is not a valid model ID"; others say "model not
# found" / "does not exist". A model matching these is permanently demoted
# (disabled), not just cooled down.
_DEAD_MODEL_MARKERS = (
    "no endpoints found", "not a valid model", "is not a valid model",
    "model not found", "model_not_found", "does not exist", "unknown model",
    "no allowed providers",
    # NOTE: deliberately NOT the bare "not found" — it matched transient
    # gateway/CDN 404 bodies ("resource not found", "page not found") and
    # permanently disabled healthy models. Keep only model-specific phrasings.
)

# Message fragments (beyond the dead-model ones) that warrant a retry on the
# next model — mirrors freellmapi's `isRetryableError`. "api error 400/422"
# matches our adapter's "<Provider> API error 400: ..." format: a 400 from one
# provider (bad param / unsupported model) may succeed on another.
_RETRYABLE_MARKERS = _DEAD_MODEL_MARKERS + (
    "rate limit", "too many requests", "quota", "resource_exhausted",
    "aborted", "timeout", "etimedout", "econnrefused", "econnreset",
    "unavailable", "internal server error",
    "payload too large", "request entity too large", "content too large",
    "api error 400", "api error 422",
    # Credit / billing / verification walls — fall through to a different
    # provider rather than surfacing a paywall as the answer.
    "more credits", "insufficient credit", "insufficient_quota",
    "upgrade to a paid", "payment required", "billing", "fewer max_tokens",
    "credit card", "customer_verification", "verification_required",
    "unlock your free credits", "add a card", "requires a valid credit",
    "access denied", "forbidden", "permission",
)


def classify_error(status: int | None, message: str | None) -> tuple[bool, bool]:
    """Return (retryable, permanent_dead) for a provider failure.

    `retryable` → fall through to the next model in the chain.
    `permanent_dead` → the model id itself is gone/invalid; disable it so it
    leaves the routing chain entirely (not just a temporary cooldown).
    """
    msg = (message or "").lower()
    # A model is "permanently dead" ONLY when the message explicitly says the id
    # is gone/invalid. A bare 404 (with no dead marker) is treated as retryable
    # but NOT dead — a one-off gateway/CDN 404 must not disable a good model.
    dead = any(k in msg for k in _DEAD_MODEL_MARKERS)
    retryable = (
        dead
        or status is None  # transport error (timeout / connreset)
        or status in RETRYABLE_STATUSES
        or any(k in msg for k in _RETRYABLE_MARKERS)
    )
    return retryable, dead


class ProviderError(RuntimeError):
    """An upstream provider error. `retryable` drives router fallback;
    `permanent_dead` marks a gone/invalid model id for removal."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.status = status
        auto_retry, dead = classify_error(status, message)
        self.retryable = auto_retry if retryable is None else retryable
        self.permanent_dead = dead


class BaseAdapter:
    def __init__(self, spec: ProviderSpec):
        self.spec = spec
        self.platform = spec.platform
        self.name = spec.name
        self._timeout = spec.timeout_ms / 1000.0

    # ---- to be implemented by subclasses ----------------------------
    async def complete(self, api_key: str, messages: list[dict], model_id: str, options: dict) -> str:
        raise NotImplementedError

    async def stream(
        self, api_key: str, messages: list[dict], model_id: str, options: dict
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator

    async def validate_key(self, api_key: str) -> bool:
        raise NotImplementedError

    # ---- shared helpers ---------------------------------------------
    def _payload(self, messages: list[dict], model_id: str, options: dict, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
        }
        if options.get("temperature") is not None:
            payload["temperature"] = options["temperature"]
        if options.get("max_tokens") is not None:
            payload["max_tokens"] = options["max_tokens"]
        if options.get("top_p") is not None:
            payload["top_p"] = options["top_p"]
        # Reproducible mode (P5 #27): a fixed seed for providers that honour it.
        if options.get("seed") is not None:
            payload["seed"] = options["seed"]
        if options.get("response_format_json"):
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _fold_reasoning(message: dict) -> str:
        """Normalize a choice's message into plain text.

        Mirrors freellmapi `normalizeChoices`: flatten array content
        (Mistral magistral) and fold `reasoning_content`/`reasoning` into
        content when content is empty (Z.ai, Ollama reasoning models).
        """
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                seg if isinstance(seg, str) else (seg.get("text") or "")
                for seg in content
            )
        if not content:
            content = message.get("reasoning_content") or message.get("reasoning") or ""
        return content or ""
