"""Per-category capability registry (intelligent-model-routing R1/R4).

Gives every model a usable ``CapabilityProfile`` — explicit when a model row
carries ``capability_json`` (additive, optional), else **derived** from the
existing ``intelligence_rank`` / ``speed_rank`` / ``rank_from_id`` + id markers
so no model is left without a profile (R1.2) and no destructive migration is
required (R11.5). The existing ``intelligence_rank`` / ``speed_rank`` /
``context_window`` / ``supports_vision`` meanings are untouched (R1.3).

``task_match(profile, category) -> 0..1`` is the fit term consumed by the
router score, and ``supports_tools`` / ``supports_json`` are derived capability
flags (mirroring the existing ``supports_vision``) used by the capability
filters (R4).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.llm.catalog import detect_moe, is_vision_model_id, rank_from_id

TASK_CATEGORIES = (
    "coding", "architecture", "research", "writing", "reasoning", "math",
    "vision", "conversation", "agentic", "general",
)

# id/name markers that signal a model is strong at a category (best-effort,
# provider-agnostic — never a curated allow-list). In-code fallback; the live
# lists come from `cfg.model_markers.*` (config-overridable, see below).
_CATEGORY_MARKERS = {
    "coding": ("coder", "code", "codestral", "deepseek-coder", "qwen3-coder",
               "starcoder", "codegemma"),
    "math": ("math", "deepseek-math", "qwen2.5-math", "wizardmath"),
    "reasoning": ("reason", "-r1", "o1", "o3", "o4", "thinking", "qwq",
                  "magistral", "deepseek-r1"),
    "vision": ("vl", "vision", "-vl-", "multimodal"),
}
_NO_CAP_MARKERS = ("base", "embed", "rerank", "-1b", "-2b", "tiny")


def _category_markers() -> dict:
    """Config-overridable category→markers map; fail-open to the in-code default."""
    try:
        from app.core.config_loader import cfg
        vals = cfg.model_markers.category_markers
        if vals:
            return vals
    except Exception:  # noqa: BLE001
        pass
    return _CATEGORY_MARKERS


def _no_cap_markers() -> tuple:
    """Config-overridable 'no tools/json' markers; fail-open to the in-code list."""
    try:
        from app.core.config_loader import cfg
        vals = cfg.model_markers.no_capability_markers
        if vals:
            return tuple(vals)
    except Exception:  # noqa: BLE001
        pass
    return _NO_CAP_MARKERS


@dataclass
class CapabilityProfile:
    scores: dict[str, int] = field(default_factory=dict)   # category -> 0..100
    supports_tools: bool = False
    supports_json: bool = False
    supports_vision: bool = False
    derived: bool = True                                   # False when explicit

    def score_for(self, category: str) -> int:
        if category in self.scores:
            return int(self.scores[category])
        return int(self.scores.get("general", 50))


def _rank_to_score(intel_rank: int | None) -> int:
    """Map an intelligence_rank (1=best .. 100=worst) to a 0..100 capability
    score (higher = stronger). Rank 1 → ~99, rank 30 → ~55, rank 100 → ~5."""
    r = int(intel_rank or 100)
    r = max(1, min(100, r))
    return int(round(max(5.0, 100.0 - (r - 1) * 0.95)))


def _speed_to_score(speed_rank: int | None) -> int:
    s = int(speed_rank or 100)
    s = max(1, min(100, s))
    return int(round(max(5.0, 100.0 - (s - 1) * 5.0)))


def _derive_flags(model_id: str, meta: dict | None) -> tuple[bool, bool]:
    """Derive (supports_tools, supports_json). Most modern instruct/chat models
    on these providers support both; tiny/base models are conservative."""
    mid = (model_id or "").lower()
    # Base/embedding/older tiny models: assume no tools/json.
    if any(m in mid for m in _no_cap_markers()):
        return False, False
    # Provider metadata wins when present.
    if isinstance(meta, dict):
        st = meta.get("supports_tools")
        sj = meta.get("supports_json") or meta.get("supports_response_format")
        if st is not None or sj is not None:
            return bool(st), bool(sj)
    # Default for instruct/chat models on OpenAI-compatible providers.
    return True, True


def profile_for(model, meta: dict | None = None) -> CapabilityProfile:
    """Build a CapabilityProfile for a model row (or any object exposing
    `model_id`/`intelligence_rank`/`speed_rank`/`supports_vision`). Explicit
    `capability_json` wins; otherwise derive. Never raises."""
    try:
        return _profile_for(model, meta)
    except Exception:  # noqa: BLE001 — fail-open to a neutral profile
        return CapabilityProfile(scores={"general": 50}, derived=True)


def _profile_for(model, meta: dict | None) -> CapabilityProfile:
    model_id = getattr(model, "model_id", "") or getattr(model, "id", "") or ""
    vision = bool(getattr(model, "supports_vision", False)) or is_vision_model_id(model_id)

    # 1) Explicit capability_json (additive optional column / attr).
    explicit = getattr(model, "capability_json", None)
    if explicit:
        try:
            data = explicit if isinstance(explicit, dict) else json.loads(explicit)
            scores = {k: int(v) for k, v in (data.get("scores") or {}).items()
                      if k in TASK_CATEGORIES}
            if scores:
                return CapabilityProfile(
                    scores=scores,
                    supports_tools=bool(data.get("supports_tools", False)),
                    supports_json=bool(data.get("supports_json", False)),
                    supports_vision=vision or bool(data.get("supports_vision", False)),
                    derived=False,
                )
        except Exception:  # noqa: BLE001 — bad json → fall through to derived
            pass

    # 2) Derive. A base general score from intelligence_rank, then per-category
    #    bumps for models whose id signals a specialty.
    intel = getattr(model, "intelligence_rank", None)
    speed = getattr(model, "speed_rank", None)
    if intel is None:
        intel, speed = rank_from_id(model_id, meta)
    base = _rank_to_score(intel)
    scores: dict[str, int] = {c: base for c in TASK_CATEGORIES}
    # Conversation leans on speed; trivial chatter wants a snappy model.
    scores["conversation"] = max(base, _speed_to_score(speed))
    mid = (model_id or "").lower()
    for cat, markers in _category_markers().items():
        if any(m in mid for m in markers):
            scores[cat] = min(100, base + 18)
    if detect_moe(model_id, meta):
        # MoE models tend to be strong reasoners for their active size.
        scores["reasoning"] = min(100, scores["reasoning"] + 6)
    scores["vision"] = max(scores.get("vision", base), 80 if vision else 10)

    tools, jsonf = _derive_flags(model_id, meta)
    return CapabilityProfile(scores=scores, supports_tools=tools,
                             supports_json=jsonf, supports_vision=vision,
                             derived=True)


def task_match(profile: CapabilityProfile, category: str) -> float:
    """0..1 fit of a model's profile to a task category (R3.1). A model strong
    in the category scores near 1.0; a weak one near 0.0. Unknown category →
    neutral 0.5 so it never distorts the score."""
    try:
        if not category or category not in TASK_CATEGORIES:
            return 0.5
        return max(0.0, min(1.0, profile.score_for(category) / 100.0))
    except Exception:  # noqa: BLE001
        return 0.5


__all__ = ["TASK_CATEGORIES", "CapabilityProfile", "profile_for", "task_match"]
