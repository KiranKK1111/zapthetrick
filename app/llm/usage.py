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


def record(usage: dict | None, finish_reason: str | None = None) -> None:
    """Called by an adapter after a completion with the provider's usage frame."""
    _last.set({"usage": usage or {}, "finish_reason": finish_reason})


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


__all__ = ["record", "tokens", "finish_reason", "reset"]
