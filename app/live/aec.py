"""
Acoustic echo cancellation (AEC) seam (Gap 5).

The voice surface already has SEMANTIC barge-in + a text echo-reference guard
(app/live/echo.py) and a first-real-word gate, which stop most self-triggering.
The remaining piece — true speaker barge-in, where the app's OWN spoken output
leaks into the mic and is cancelled ACOUSTICALLY so the user can interrupt over
the speakers — needs a real echo canceller (WebRTC APM / a Windows AEC APO / an
NLMS adaptive filter fed the played-audio reference). That is a NATIVE audio
module + on-hardware tuning, out of scope for pure-Python here.

This module is the typed SEAM so such a processor drops in without touching the
pipeline: `get_aec()` returns the configured processor (a no-op passthrough by
default). A native build registers its implementation via `register_aec(...)`.
`process(mic, reference)` returns echo-reduced mic audio; the default returns the
mic unchanged (today's behaviour, byte-identical). Fail-open everywhere.
"""
from __future__ import annotations

from typing import Protocol

from app.core.config_loader import cfg


class AecProcessor(Protocol):
    """Contract a (native) echo canceller implements."""

    def process(self, mic, reference):  # -> mic-like
        """Return `mic` with the `reference` (played audio) echo removed."""
        ...


class _NoopAec:
    """Passthrough — no cancellation. The default until a native AEC is wired."""

    def process(self, mic, reference=None):
        return mic


_registered: "AecProcessor | None" = None


def register_aec(processor: "AecProcessor") -> None:
    """A native build calls this at startup to install its echo canceller."""
    global _registered
    _registered = processor


def enabled() -> bool:
    return bool(getattr(cfg.voice, "native_aec", False))


def get_aec() -> "AecProcessor":
    """The active AEC processor. The registered native one when enabled + present;
    otherwise the no-op passthrough (today's behaviour). Never raises."""
    try:
        if enabled() and _registered is not None:
            return _registered
    except Exception:  # noqa: BLE001
        pass
    return _NoopAec()


def process(mic, reference=None):
    """Convenience: run the active AEC. Fail-open → returns `mic` unchanged."""
    try:
        return get_aec().process(mic, reference)
    except Exception:  # noqa: BLE001
        return mic


__all__ = ["AecProcessor", "get_aec", "register_aec", "process", "enabled"]
