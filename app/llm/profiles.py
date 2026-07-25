"""Task-profile scoring matrix (vNext §2.6).

Each call site declares a **task profile** — `live_answer`, `chat_code`,
`dsa_reasoning`, `extraction_json`, `doc_script`, `speculation_draft` — and a
profile is DATA: a set of weights + floors over *measured* stats. The router
then scores candidates as `Σ wᵢ·norm(statᵢ) − penalties`, so criteria shape
ranking (never existence — floors relax to penalties at T2, keeping the
never-empty invariant).

**Semantic-first (standing directive):** the profile of a turn is chosen
SEMANTICALLY — `classify()` runs the exemplar-embedding gate as the AUTHORITY and
only falls back to the task-category map when the embedder is cold. The gate
function is INJECTED (`gate_classify=`), so this module (in `llm`) needs no
`semantics` edge — the call site (chat/live, which already reaches semantics)
passes `gates.classify` in and declares the result via `options["task_profile"]`.
Profile weights/exemplars are DATA, not intent.
"""
from __future__ import annotations

from dataclasses import dataclass

# Weight scale (higher = matters more): highest=3, high=2, med=1, low=0.5.
_HIGHEST, _HIGH, _MED, _LOW = 3.0, 2.0, 1.0, 0.5


@dataclass(frozen=True)
class Profile:
    name: str
    quality_basis: str      # which measured stat is the quality floor/basis
    ttft_wt: float
    tps_wt: float
    json_floor: float | None
    context_need: str       # "small" | "med" | "large"


# The §2.6 matrix, as data.
PROFILES: dict[str, Profile] = {
    "live_answer": Profile("live_answer", "instruction", _HIGH, _HIGH, None, "small"),
    "chat_code": Profile("chat_code", "verify_pass", _MED, _HIGH, None, "med"),
    "live_code": Profile("live_code", "verify_pass", _MED, _HIGH, None, "med"),
    "dsa_reasoning": Profile("dsa_reasoning", "reasoning", _LOW, _LOW, None, "med"),
    "extraction_json": Profile("extraction_json", "accuracy", _HIGH, _MED, 0.99, "small"),
    "classify": Profile("classify", "accuracy", _HIGH, _MED, 0.99, "small"),
    "doc_script": Profile("doc_script", "verify_pass", _LOW, _MED, None, "large"),
    "speculation_draft": Profile("speculation_draft", "instruction", _HIGHEST, _HIGH, None, "small"),
}

PROFILE_NAMES: frozenset[str] = frozenset(PROFILES)

# Exemplars for the SEMANTIC profile gate (DATA). Only the chat-reachable
# profiles need exemplars; live_answer/speculation_draft are declared directly by
# the Live/speculation lanes, not classified from user text.
PROFILE_EXEMPLARS: dict[str, list[str]] = {
    "chat_code": [
        "write a function to reverse a string",
        "implement a rest api for users",
        "give me the python code for a login form",
        "code a script that reads a csv",
        "build a class to represent a bank account",
    ],
    "dsa_reasoning": [
        "solve this leetcode two sum problem",
        "what's the optimal algorithm for this",
        "find the time complexity and optimize it",
        "design an efficient solution for this data structure problem",
        "what's the best approach, dynamic programming or greedy",
    ],
    "extraction_json": [
        "extract the fields from this into json",
        "classify this text into a category",
        "parse this and return structured data",
        "pull out the entities as a json object",
    ],
    "doc_script": [
        "make me a resume as a pdf",
        "generate a report document about this",
        "create a spreadsheet of the results",
        "build a presentation on this topic",
    ],
}

# Cold-start fallback: the existing task-category → a profile.
_CATEGORY_TO_PROFILE: dict[str, str] = {
    "code_generation": "chat_code",
    "code": "chat_code",
    "debugging": "chat_code",
    "test_generation": "chat_code",
    "dsa": "dsa_reasoning",
    "algorithm": "dsa_reasoning",
    "reasoning": "dsa_reasoning",
    "extraction": "extraction_json",
    "classification": "classify",
    "document": "doc_script",
    "doc": "doc_script",
}


def profile(name: str | None) -> Profile | None:
    return PROFILES.get((name or "").strip()) if name else None


def classify(text: str, *, gate_classify=None, fallback_category: str | None = None,
             threshold: float = 0.6) -> str | None:
    """The turn's task profile, SEMANTIC-first. `gate_classify` is the injected
    `app.semantics.gates.classify(query, classes, *, threshold, cache_key)` (the
    call site passes it — keeps `llm` free of a `semantics` edge). Returns a
    profile name or None. Never raises."""
    t = (text or "").strip()
    if not t:
        return None
    if gate_classify is not None:
        try:
            cls = gate_classify(t, PROFILE_EXEMPLARS, threshold=threshold,
                                cache_key="task_profile")
            if cls in PROFILES:
                return cls
        except Exception:  # noqa: BLE001 — fail-open to the fallback
            pass
    # Cold-start fallback: map the deterministic task category → a profile.
    if fallback_category:
        return _CATEGORY_TO_PROFILE.get(fallback_category.strip().lower())
    return None


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.routing, "task_profiles", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = ["Profile", "PROFILES", "PROFILE_NAMES", "PROFILE_EXEMPLARS",
           "profile", "classify", "enabled"]
