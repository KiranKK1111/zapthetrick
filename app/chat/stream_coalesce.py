"""Token coalescing for smoother streaming (smooth-streaming-rendering R16).

One SSE frame per model token can be bursty; coalescing a few tokens into a
single frame reduces frame count under fast providers without hurting perceived
latency (the frontend's frame-synced reveal smooths the rest). The coalescer is
size-based and OFF by default (`threshold <= 0` emits every token immediately),
so the SSE contract is unchanged unless explicitly enabled via config.

Pure + synchronous so it is unit-testable without the route/LLM stack.
"""
from __future__ import annotations

from typing import Iterable


class TokenCoalescer:
    """Accumulates token text and emits a combined chunk once it reaches
    `threshold` characters. `threshold <= 0` disables coalescing (passthrough)."""

    def __init__(self, threshold: int) -> None:
        self.threshold = int(threshold or 0)
        self._buf: list[str] = []
        self._len = 0

    def push(self, text: str) -> str | None:
        """Feed one token. Returns a chunk to emit now, or None to keep buffering."""
        if not text:
            return None
        if self.threshold <= 0:
            return text
        self._buf.append(text)
        self._len += len(text)
        if self._len >= self.threshold:
            return self.flush()
        return None

    def flush(self) -> str | None:
        """Emit and clear whatever is buffered (call before the stream ends)."""
        if not self._buf:
            return None
        out = "".join(self._buf)
        self._buf.clear()
        self._len = 0
        return out


def coalesce_tokens(tokens: Iterable[str], threshold: int) -> list[str]:
    """Convenience: run a whole token sequence through a [TokenCoalescer] and
    return the emitted chunks (used in tests). Concatenation is preserved."""
    c = TokenCoalescer(threshold)
    out: list[str] = []
    for t in tokens:
        chunk = c.push(t)
        if chunk:
            out.append(chunk)
    rem = c.flush()
    if rem:
        out.append(rem)
    return out


def effective_threshold(base: int, client_load: float | None) -> int:
    """Adaptive chunk sizing (R46): scale the base chunk threshold up with the
    client's rendering load (0..1). Off (0) stays off; no load → base."""
    if base <= 0:
        return 0
    if client_load is None:
        return base
    try:
        cl = float(client_load)
    except (TypeError, ValueError):
        return base
    if cl <= 0:
        return base
    cl = max(0.0, min(1.0, cl))
    return int(base * (1.0 + cl * 3.0))
