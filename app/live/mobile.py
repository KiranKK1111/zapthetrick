"""
Mobile runtime constraints (live-conversational-intelligence R26).

Deterministic helpers for the Android app running alongside a call on the same
device: classify a mic-contention / audio-routing error into a clear state
(never a silent failure), expose whether an overlay/PiP surface is offered, and
decide when sustained battery/thermal pressure should degrade the live path to a
lower-cost (fast) answer. Composes with the adaptive-latency path. Fail-open;
desktop/disabled → today's behavior.
"""
from __future__ import annotations

# Battery %/thermal thresholds below/above which we prefer the fast path.
_LOW_BATTERY_PCT = 20.0
_HOT_THERMAL = {"serious", "critical", "emergency", "hot"}


def classify_audio_error(detail: str) -> dict:
    """Map a mobile audio error into a clear, surfaceable state (no silent
    failure). Returns an additive frame dict."""
    d = (detail or "").lower()
    if "permission" in d:
        state = "mic_permission"
    elif "in use" in d or "busy" in d or "contention" in d or "another app" in d:
        state = "mic_contention"
    elif "rout" in d or "device" in d or "output" in d:
        state = "audio_routing"
    else:
        state = "audio_error"
    return {"type": "audio_status", "state": state, "detail": detail or ""}


def overlay_supported() -> bool:
    """Whether the overlay / picture-in-picture surface is offered (config-
    gated; the FE decides if the platform actually supports it)."""
    from app.core.config_loader import cfg
    return bool(getattr(cfg.live, "mobile_runtime", False))


def degrade_for_pressure(battery_pct: float | None = None,
                         thermal: str | None = None) -> bool:
    """True when sustained battery/thermal pressure warrants the fast/cheaper
    answer path. Unknown signals → no degrade (today's behavior)."""
    try:
        if battery_pct is not None and float(battery_pct) <= _LOW_BATTERY_PCT:
            return True
        if thermal is not None and str(thermal).strip().lower() in _HOT_THERMAL:
            return True
        return False
    except Exception:  # noqa: BLE001
        return False
