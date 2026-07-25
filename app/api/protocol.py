"""FE↔BE protocol versioning (vNext §11.6).

The frame vocabulary (meta / token / done / stage / assumption / audio / screen /
dictation / preview …) is now large and both halves deploy independently. The
contract: the connect handshake carries ``protocol_version``; the backend emits
at the HIGHEST version the client can understand, with **one-version-back
compatibility guaranteed**. Unknown frame types are ignored-with-log on both
sides (the §5.3 unknown-stage rule, generalized), so a newer peer can add frames
without breaking an older one.

A version bump is a deliberate act (raise ``PROTOCOL_VERSION`` + a migration
note); ``MIN_SUPPORTED_VERSION`` is what "one back" means.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Highest protocol version this backend speaks, and the oldest it still supports
# (one back). Bump PROTOCOL_VERSION on a breaking frame change; move
# MIN_SUPPORTED_VERSION forward only when dropping support for an old client.
PROTOCOL_VERSION = 1
MIN_SUPPORTED_VERSION = 1

# The frame types this version knows. A frame NOT in here is ignored-with-log by
# the receiver (forward-compatible). This is documentation + the unknown-frame
# gate; it is intentionally generous.
KNOWN_FRAMES = frozenset({
    "meta", "token", "done", "error", "stage", "assumption", "citation",
    "tool", "tool_call", "tool_result", "verdict", "hot_swap", "patch",
    "audio", "screen", "dictation", "preview", "duplicate", "keepalive",
    "clarify", "plan", "reasoning",
})


def _clamp_client(client_version) -> int:
    return client_version if isinstance(client_version, int) and \
        client_version >= 1 else 1


def negotiate(client_version, *, server_max: int | None = None,
              min_supported: int | None = None) -> int:
    """The version the backend will EMIT at for a client declaring
    ``client_version`` (None/absent = a legacy pre-versioning client → 1).

    Capped at the server's max (the backend can't speak a version it doesn't
    know) and floored at ``min_supported`` (it won't emit older than it
    supports). Result is always ``<= client_version`` when the client is within
    the supported window, so the client can parse everything it receives."""
    smax = PROTOCOL_VERSION if server_max is None else server_max
    smin = MIN_SUPPORTED_VERSION if min_supported is None else min_supported
    return max(smin, min(smax, _clamp_client(client_version)))


def is_compatible(client_version, *, min_supported: int | None = None) -> bool:
    """Whether the client is within the supported window — i.e. it can understand
    the version the backend will emit. An older-than-``min_supported`` client is
    incompatible (the handshake should surface a version-mismatch note)."""
    smin = MIN_SUPPORTED_VERSION if min_supported is None else min_supported
    return _clamp_client(client_version) >= smin


def is_known_frame(frame_type: str) -> bool:
    return frame_type in KNOWN_FRAMES


def handle_unknown_frame(frame_type: str) -> None:
    """Ignore-with-log: a version bump can add frames; an older peer drops the
    ones it doesn't know rather than erroring. Never raises."""
    log.debug("protocol: ignoring unknown frame type %r", frame_type)


def handshake_info(client_version) -> dict:
    """The version block the backend echoes on the connect/meta frame so the
    client knows what it will receive."""
    return {
        "protocol_version": negotiate(client_version),
        "server_max": PROTOCOL_VERSION,
        "server_min": MIN_SUPPORTED_VERSION,
        "compatible": is_compatible(client_version),
    }


__all__ = [
    "PROTOCOL_VERSION", "MIN_SUPPORTED_VERSION", "KNOWN_FRAMES",
    "negotiate", "is_compatible", "is_known_frame", "handle_unknown_frame",
    "handshake_info",
]
