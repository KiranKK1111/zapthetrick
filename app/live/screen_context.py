"""Continuous screen context (vNext §9.4, Stage 10 Component E).

Ambient awareness of the shared screen — WITHOUT flooding a VLM or the network.
The client samples at ~1 fps and runs a luma-diff gate: a STATIC screen (an
unchanged IDE) produces no upstream traffic at all; only a genuinely changed
frame is downscaled (≤1280 px), JPEG'd, and sent (role byte 0x02) to the RESIDENT
VLM on a background lane, whose structured read folds into a rolling `ScreenState`
in the session tracker.

Two hard invariants:
  * **privacy** — ambient frames go ONLY to the local VLM; cloud vision REFUSES
    ambient (an always-on capture must never leave the device);
  * **honesty** — a live-coding delta hint is verified against the merged snippet
    before it's trusted.

This module owns the deterministic gate + state + guards; the VLM read is the
injected seam. Pure + fail-open. Flag-gated (`live.screen_context`, default OFF →
no ambient capture).
"""
from __future__ import annotations

from dataclasses import dataclass, field

_DEFAULT_DIFF = 0.06          # min normalized luma change to send a frame
_MAX_EDGE = 1280              # downscale target (px, longest edge)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "screen_context", False))
    except Exception:  # noqa: BLE001
        return False


def _diff_threshold() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.live, "screen_diff_threshold", _DEFAULT_DIFF)
                     or _DEFAULT_DIFF)
    except Exception:  # noqa: BLE001
        return _DEFAULT_DIFF


# --------------------------------------------------------------------------- #
# Luma-diff gate (the "static IDE costs zero upstream" gate)
# --------------------------------------------------------------------------- #
@dataclass
class DiffDecision:
    changed: bool
    delta: float                  # normalized 0..1 luma change
    reason: str = ""

    def to_dict(self) -> dict:
        return {"changed": self.changed, "delta": round(self.delta, 4),
                "reason": self.reason}


def _norm_hist(luma) -> "list[float]":
    """Normalize a luma summary (a small histogram or a scalar mean) to a
    distribution that sums to 1. Accepts a list/tuple or a single number."""
    if isinstance(luma, (int, float)):
        return [float(luma)]
    vals = [float(x) for x in (luma or [])]
    s = sum(vals)
    return [v / s for v in vals] if s > 0 else vals


def luma_delta(luma_now, luma_prev) -> float:
    """Normalized difference between two luma summaries (0 = identical, 1 = max).
    L1 distance over the normalized histograms / scalars. Never raises → 1.0
    (treat an error as 'changed', the safe over-send)."""
    try:
        a = _norm_hist(luma_now)
        b = _norm_hist(luma_prev)
        if not a or not b:
            return 1.0
        n = max(len(a), len(b))
        a += [0.0] * (n - len(a))
        b += [0.0] * (n - len(b))
        if len(a) == 1:              # scalar means → absolute normalized diff
            return min(1.0, abs(a[0] - b[0]) / (max(a[0], b[0], 1e-9)))
        return min(1.0, 0.5 * sum(abs(x - y) for x, y in zip(a, b)))
    except Exception:  # noqa: BLE001
        return 1.0


def should_process(luma_now, luma_prev=None, *,
                   threshold: float | None = None) -> DiffDecision:
    """The gate: send a frame to the VLM only when it CHANGED enough. The first
    frame (no prev) always processes. A static screen → changed=False (zero
    upstream). Disabled → never process. Never raises."""
    try:
        if not enabled():
            return DiffDecision(False, 0.0, "disabled")
        if luma_prev is None:
            return DiffDecision(True, 1.0, "first frame")
        thr = _diff_threshold() if threshold is None else threshold
        d = luma_delta(luma_now, luma_prev)
        if d >= thr:
            return DiffDecision(True, d, f"changed (delta {d:.3f} >= {thr:.3f})")
        return DiffDecision(False, d, f"static (delta {d:.3f} < {thr:.3f})")
    except Exception:  # noqa: BLE001
        return DiffDecision(True, 1.0, "error → send")


def downscale_target(width: int, height: int, *, max_edge: int = _MAX_EDGE
                     ) -> "tuple[int, int]":
    """Target (w, h) for a frame downscaled so its longest edge ≤ max_edge,
    preserving aspect. Never upscales. Never raises."""
    try:
        w, h = int(width), int(height)
        longest = max(w, h)
        if longest <= max_edge or longest <= 0:
            return w, h
        scale = max_edge / longest
        return max(1, int(w * scale)), max(1, int(h * scale))
    except Exception:  # noqa: BLE001
        return int(width or 0), int(height or 0)


# --------------------------------------------------------------------------- #
# Privacy invariant — cloud refuses ambient
# --------------------------------------------------------------------------- #
def allow_vision_target(target: str, *, ambient: bool) -> bool:
    """Whether a vision request may go to `target` ('local' | 'cloud'). An AMBIENT
    (continuous screen) frame may ONLY go to the local VLM — cloud refuses it, so
    an always-on capture never leaves the device. A user-initiated (non-ambient)
    upload may use either. Never raises → False (refuse on doubt)."""
    try:
        t = (target or "").strip().lower()
        if ambient:
            return t == "local"
        return t in ("local", "cloud")
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Rolling ScreenState
# --------------------------------------------------------------------------- #
@dataclass
class ScreenRead:
    """One VLM read of a frame (structured schema)."""
    app: str = ""                 # e.g. "vscode", "browser", "terminal"
    summary: str = ""
    code_language: str = ""
    has_error: bool = False
    frame_ts: float = 0.0


@dataclass
class ScreenState:
    """Rolling ambient screen state per session. The tracker holds one of these;
    each fresh VLM read updates it."""
    current: "ScreenRead | None" = None
    updated_ts: float = 0.0
    history: list = field(default_factory=list)   # recent ScreenReads (bounded)

    def update(self, read: ScreenRead) -> None:
        """Fold a new VLM read into the rolling state. No-op on a None read."""
        try:
            if read is None:
                return
            self.current = read
            self.updated_ts = read.frame_ts or self.updated_ts
            self.history.append(read)
            if len(self.history) > 20:
                self.history.pop(0)
        except Exception:  # noqa: BLE001
            pass

    def is_stale(self, *, now: float, max_age_s: float = 30.0) -> bool:
        """Whether the state is older than `max_age_s` (nothing seen recently)."""
        try:
            if self.current is None:
                return True
            return (now - self.updated_ts) > max_age_s
        except Exception:  # noqa: BLE001
            return True

    def context_hint(self) -> str:
        """A short directive the answer path can fold in ('' when empty/stale)."""
        try:
            r = self.current
            if r is None:
                return ""
            bits = []
            if r.app:
                bits.append(f"on screen: {r.app}")
            if r.code_language:
                bits.append(f"editing {r.code_language}")
            if r.has_error:
                bits.append("an error is visible")
            return ("The candidate's screen shows " + ", ".join(bits) + "."
                    ) if bits else ""
        except Exception:  # noqa: BLE001
            return ""

    def to_dict(self) -> dict:
        r = self.current
        return {"has_state": r is not None,
                "app": getattr(r, "app", ""),
                "code_language": getattr(r, "code_language", ""),
                "has_error": getattr(r, "has_error", False),
                "updated_ts": self.updated_ts}


def for_tracker(tracker) -> ScreenState:
    """One ScreenState per session, stashed on the tracker."""
    s = getattr(tracker, "_screen_state", None)
    if s is None:
        s = ScreenState()
        try:
            setattr(tracker, "_screen_state", s)
        except Exception:  # noqa: BLE001
            pass
    return s


# --------------------------------------------------------------------------- #
# Live-coding delta verification
# --------------------------------------------------------------------------- #
# Filler that carries no verifiable code content (an NL delta hint is prose).
_HINT_STOP = frozenset((
    "the", "and", "for", "with", "added", "adds", "new", "now", "has",
    "have", "that", "this", "into", "from", "was", "are", "changed", "change",
    "code", "some", "line", "lines", "block", "here", "there", "your", "you"))


def verify_delta_hint(hint: str, merged_snippet: str, *, verify_fn=None) -> bool:
    """A live-coding DELTA hint (what the VLM thinks changed on screen) is trusted
    only if it's reflected in the merged snippet the candidate has — a hint about
    code that isn't there is a hallucination → reject. `verify_fn(hint, snippet)
    -> bool` is an optional INJECTED semantic verifier; the deterministic fallback
    requires the hint's meaningful (non-filler) tokens to overlap the snippet.
    Never raises → False (reject on doubt)."""
    try:
        h = (hint or "").strip().lower()
        snip = (merged_snippet or "").lower()
        if not h or not snip:
            return False
        if verify_fn is not None:
            try:
                return bool(verify_fn(hint, merged_snippet))
            except Exception:  # noqa: BLE001
                pass
        import re
        tokens = [t for t in re.findall(r"\w+", h)
                  if len(t) > 2 and t not in _HINT_STOP]
        if not tokens:
            return False
        present = sum(1 for t in tokens if t in snip)
        # A pure hallucination (no meaningful token present) is rejected; a real
        # delta (its technical terms appear) is accepted.
        return present / len(tokens) >= 0.5
    except Exception:  # noqa: BLE001
        return False


__all__ = ["enabled", "DiffDecision", "luma_delta", "should_process",
           "downscale_target", "allow_vision_target", "ScreenRead",
           "ScreenState", "for_tracker", "verify_delta_hint"]
