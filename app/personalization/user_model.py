"""User model (personalization-and-governance R1).

`UserModel` (expertise / verbosity_pref / comm_style / frustration) is inferred
deterministically from observed interaction signals and persisted in the
existing `User.preferences` JSONB (no new schema). It defaults to neutral when
signal is insufficient (today's behavior, R1.3) and REUSES the existing
answer-depth preference rather than a parallel store (R1.4). User-scoped + honors
data-clear (R6.1/R6.2). Pure; never raises.
"""
from __future__ import annotations

from dataclasses import dataclass

UNKNOWN = "unknown"

# Map a consistent depth preference (already tracked elsewhere) → verbosity.
_DEPTH_TO_VERBOSITY = {
    "tldr": "concise", "concise": "concise",
    "standard": "balanced", "balanced": "balanced",
    "deeper": "detailed", "exhaustive": "detailed", "detailed": "detailed",
}


@dataclass
class UserModel:
    expertise: str = UNKNOWN        # beginner | intermediate | senior | expert | unknown
    verbosity_pref: str = UNKNOWN   # concise | balanced | detailed | unknown
    comm_style: str = UNKNOWN       # bullet | prose | technical | unknown
    frustration: float = 0.0        # 0..1

    @property
    def is_neutral(self) -> bool:
        return (self.expertise == UNKNOWN and self.verbosity_pref == UNKNOWN
                and self.comm_style == UNKNOWN and self.frustration < 0.5)

    def to_dict(self) -> dict:
        return {"expertise": self.expertise, "verbosity_pref": self.verbosity_pref,
                "comm_style": self.comm_style, "frustration": round(self.frustration, 3)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "UserModel":
        d = d or {}
        return cls(
            expertise=d.get("expertise", UNKNOWN),
            verbosity_pref=d.get("verbosity_pref", UNKNOWN),
            comm_style=d.get("comm_style", UNKNOWN),
            frustration=float(d.get("frustration", 0.0) or 0.0),
        )


# Expertise cues (deterministic).
_EXPERT_CUES = ("optimize", "race condition", "time complexity", "big-o",
                "idempotent", "concurrency", "throughput", "p99", "kernel",
                "assembly", "simd", "lock-free", "memory barrier")
_BEGINNER_CUES = ("what is", "how do i", "explain like", "eli5", "i'm new",
                  "beginner", "step by step", "don't understand", "what does")


def infer(signals: dict | None) -> UserModel:
    """Infer a UserModel from interaction signals. Never raises; insufficient
    signal → neutral (Property 1).

    Recognized signals (all optional): `depth_pref` (the existing answer-depth
    preference), `recent_user_texts` (list[str]), `bullet_ratio` (0..1),
    `expertise_hint`, `frustration` (0..1)."""
    try:
        return _infer(signals or {})
    except Exception:  # noqa: BLE001
        return UserModel()


def _infer(s: dict) -> UserModel:
    m = UserModel()

    # 1) Verbosity reuses the existing answer-depth preference (R1.4).
    dp = str(s.get("depth_pref", "")).lower()
    if dp in _DEPTH_TO_VERBOSITY:
        m.verbosity_pref = _DEPTH_TO_VERBOSITY[dp]

    # 2) Expertise from explicit hint or lexical cues across recent turns.
    hint = str(s.get("expertise_hint", "")).lower()
    if hint in ("beginner", "intermediate", "senior", "expert"):
        m.expertise = hint
    else:
        blob = " ".join(str(t) for t in (s.get("recent_user_texts") or [])).lower()
        if blob:
            if any(c in blob for c in _EXPERT_CUES):
                m.expertise = "senior"
            elif any(c in blob for c in _BEGINNER_CUES):
                m.expertise = "beginner"

    # 3) Communication style from observed bullet usage.
    br = s.get("bullet_ratio")
    if isinstance(br, (int, float)):
        if br >= 0.5:
            m.comm_style = "bullet"
        elif br > 0:
            m.comm_style = "prose"
    if m.comm_style == UNKNOWN and m.expertise in ("senior", "expert"):
        m.comm_style = "technical"

    # 4) Frustration carried in (updated by the detector).
    fr = s.get("frustration")
    if isinstance(fr, (int, float)):
        m.frustration = max(0.0, min(1.0, float(fr)))

    return m


# ── persistence (User.preferences JSONB; no new schema) ──────────────────────
_KEY = "user_model"


def load_user_model(prefs: dict | None) -> UserModel:
    try:
        return UserModel.from_dict((prefs or {}).get(_KEY))
    except Exception:  # noqa: BLE001
        return UserModel()


def save_user_model(prefs: dict, model: UserModel) -> None:
    try:
        if isinstance(prefs, dict):
            prefs[_KEY] = model.to_dict()
    except Exception:  # noqa: BLE001
        pass


def clear_user_model(prefs: dict) -> None:
    """Data-clear (R6.2)."""
    try:
        if isinstance(prefs, dict):
            prefs.pop(_KEY, None)
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "UserModel", "infer", "load_user_model", "save_user_model",
    "clear_user_model", "UNKNOWN",
]
