"""Real provider usage + finish_reason (Architecture §14 / gap G6.1).

Providers return authoritative token counts (`usage`) and a true stop reason
(`finish_reason`); the engine historically estimated tokens as `chars//4`, which
mis-counts rate-limit windows. Adapters record the last completion's usage here
(task-local via a ContextVar, so concurrent requests never race); the engine
reads it for accurate `ratelimit.record_tokens`, falling back to the estimate
when a provider omits usage.
"""
from __future__ import annotations

from contextvars import ContextVar

_last: ContextVar[dict | None] = ContextVar("_llm_last_completion", default=None)
# Rate-limit headers from the last response (task-local), for §2.7 quota
# header-correction. Kept separate from usage so a 4xx path (which has no usage
# but DOES carry Retry-After) can still record them.
_headers: ContextVar[dict | None] = ContextVar("_llm_last_headers", default=None)


def record(usage: dict | None, finish_reason: str | None = None) -> None:
    """Called by an adapter after a completion with the provider's usage frame."""
    _last.set({"usage": usage or {}, "finish_reason": finish_reason})


# Only these headers matter for quota correction — keep the sink tiny.
_RL_KEYS = (
    "retry-after",
    "x-ratelimit-remaining", "x-ratelimit-remaining-requests",
    "x-ratelimit-reset", "x-ratelimit-reset-requests",
)


def record_headers(headers) -> None:
    """Called by an adapter with a response's headers (any mapping). Keeps only
    the rate-limit keys, task-local. Fail-safe: any error clears to None."""
    try:
        h = {str(k).lower(): v for k, v in dict(headers or {}).items()}
        kept = {k: h[k] for k in _RL_KEYS if k in h}
        _headers.set(kept or None)
    except Exception:  # noqa: BLE001
        _headers.set(None)


def rate_limit_headers() -> dict | None:
    return _headers.get()


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def tokens() -> tuple[int | None, int | None, int | None]:
    """(prompt_tokens, completion_tokens, total_tokens) from the last completion,
    each None when the provider didn't report it."""
    u = (_last.get() or {}).get("usage") or {}
    return (_int(u.get("prompt_tokens")),
            _int(u.get("completion_tokens")),
            _int(u.get("total_tokens")))


def finish_reason() -> str | None:
    return (_last.get() or {}).get("finish_reason")


def reset() -> None:
    _last.set(None)
    _headers.set(None)


__all__ = ["record", "tokens", "finish_reason", "reset",
           "record_headers", "rate_limit_headers"]
