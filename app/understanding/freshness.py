"""Freshness layer (vNext §9.8, Stage 9 Component C).

How time-sensitive is a question's answer? That decides whether the interleaved
tool loop (§9.2) needs to hit the web:

  * **STABLE** — definitions, maths, closed history, language semantics. The
    answer never goes stale → answer DIRECTLY, no search.
  * **SLOW** — framework versions, best practices, API surfaces. Drifts over
    months → answer from knowledge, then run ONE verification search.
  * **VOLATILE** — prices, news, weather, scores, standings, "latest"/"current"/
    "today". Changes by the hour → search FIRST, answer cited.

Semantic-first per the project rule: `app/semantics/gates.classify` over per-tier
exemplars (generalizes to paraphrase), the cue lists a cold-start FALLBACK only.
The distilled §9.7 freshness head replaces the LLM judge later — same seam. For
LIVE, VOLATILE means: give the knowledge answer immediately + a "verifying" chip +
a correction footnote (never make the candidate wait on a search).

Fail-open: disabled OR unsure → STABLE (answer directly — today's behaviour).
Flag-gated (`freshness.classifier`, default OFF).
"""
from __future__ import annotations

from dataclasses import dataclass

STABLE = "stable"
SLOW = "slow"
VOLATILE = "volatile"
_TIERS = (STABLE, SLOW, VOLATILE)

# Strategy modes (how the interleaved loop treats the turn).
DIRECT = "direct"                 # answer from knowledge, no search
VERIFY = "verify"                # answer, then one verification search
SEARCH_FIRST = "search_first"     # search first, answer cited

# Semantic exemplars per tier — the PRIMARY signal (embedded + matched).
_EXEMPLARS: dict[str, list[str]] = {
    STABLE: [
        "what is a hash map",
        "explain how TCP handshake works",
        "prove the pythagorean theorem",
        "who wrote hamlet",
        "difference between a list and a tuple",
    ],
    SLOW: [
        "what is the latest stable version of react",
        "current best practices for python packaging",
        "recommended way to deploy a fastapi app",
        "which node lts should I use",
        "how do people structure django projects these days",
    ],
    VOLATILE: [
        "what is the price of bitcoin right now",
        "who won the game last night",
        "what is the weather today",
        "latest news about the election",
        "current stock price of apple",
    ],
}

# Cold-start FALLBACK cues only (semantic path is authoritative).
_VOLATILE_CUES = (
    "right now", "today", "currently", "latest", "current price", "as of",
    "this morning", "breaking", "live score", "who won", "stock price",
    "weather", "exchange rate", "trending", "just announced", "this week",
)
_SLOW_CUES = (
    "latest version", "current version", "best practice", "recommended",
    "these days", "nowadays", "up to date", "modern way", "which version",
    "lts", "deprecated",
)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.freshness, "classifier", False))
    except Exception:  # noqa: BLE001
        return False


def _threshold() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.freshness, "threshold", 0.55) or 0.55)
    except Exception:  # noqa: BLE001
        return 0.55


@dataclass
class FreshnessRead:
    tier: str = STABLE
    confidence: float = 0.0
    source: str = "default"       # "semantic" | "fallback" | "default"

    def to_dict(self) -> dict:
        return {"tier": self.tier, "confidence": round(self.confidence, 3),
                "source": self.source}


def _fallback(query: str) -> "tuple[str, float]":
    t = (query or "").lower()
    if not t.strip():
        return STABLE, 0.0
    # SLOW cues are checked FIRST because they're more specific ("latest version"
    # is SLOW, not VOLATILE even though it contains "latest").
    if any(c in t for c in _SLOW_CUES):
        return SLOW, 0.65
    if any(c in t for c in _VOLATILE_CUES):
        return VOLATILE, 0.7
    return STABLE, 0.0


def classify_freshness(query: str, *, classify_fn=None) -> FreshnessRead:
    """Classify a query's freshness tier. SEMANTIC first (`gates.classify` over
    the exemplars — injectable `classify_fn` for tests), cue fallback when the
    embedder is unavailable. Fail-open: disabled OR nothing matches → STABLE
    (answer directly). Never raises."""
    try:
        if not enabled():
            return FreshnessRead(STABLE, 0.0, "default")
        text = (query or "").strip()
        if not text:
            return FreshnessRead(STABLE, 0.0, "default")
        # 1) Semantic (primary).
        cls = None
        try:
            fn = classify_fn
            if fn is None:
                from app.semantics import gates
                fn = lambda q: gates.classify(  # noqa: E731
                    q, _EXEMPLARS, threshold=_threshold(),
                    cache_key="freshness")
            cls = fn(text)
        except Exception:  # noqa: BLE001
            cls = None
        if cls in _TIERS:
            return FreshnessRead(cls, 0.8, "semantic")
        # 2) Cue fallback.
        tier, conf = _fallback(text)
        if tier != STABLE:
            return FreshnessRead(tier, conf, "fallback")
        return FreshnessRead(STABLE, 0.0, "default")
    except Exception:  # noqa: BLE001
        return FreshnessRead(STABLE, 0.0, "default")


@dataclass
class FreshnessStrategy:
    tier: str
    mode: str                     # direct | verify | search_first
    needs_search: bool
    directive: str = ""

    def to_dict(self) -> dict:
        return {"tier": self.tier, "mode": self.mode,
                "needs_search": self.needs_search, "directive": self.directive}


def strategy_for(read: "FreshnessRead | str") -> FreshnessStrategy:
    """Map a freshness tier to an interleaved-loop strategy. Never raises."""
    try:
        tier = read.tier if isinstance(read, FreshnessRead) else str(read or STABLE)
        if tier == VOLATILE:
            return FreshnessStrategy(
                VOLATILE, SEARCH_FIRST, True,
                "This is time-sensitive — search for the current value FIRST, "
                "then answer and CITE the source with its date.")
        if tier == SLOW:
            return FreshnessStrategy(
                SLOW, VERIFY, True,
                "This may have changed recently — answer from what you know, then "
                "run ONE verification search and correct/confirm with a citation.")
        return FreshnessStrategy(
            STABLE, DIRECT, False,
            "This is stable knowledge — answer directly; no search needed.")
    except Exception:  # noqa: BLE001
        return FreshnessStrategy(STABLE, DIRECT, False, "")


def live_directive(read: "FreshnessRead | str") -> str:
    """The LIVE-mode directive: never make the candidate wait. A VOLATILE tier →
    give the knowledge answer immediately, flag a 'verifying' chip, and add a
    correction footnote if the search disagrees. '' for STABLE. Never raises."""
    try:
        tier = read.tier if isinstance(read, FreshnessRead) else str(read or STABLE)
        if tier == VOLATILE:
            return ("Answer now from your knowledge (do NOT wait on a search); a "
                    "background check is running — if it disagrees, a short "
                    "correction footnote follows.")
        if tier == SLOW:
            return ("Answer now; a quick verification runs in the background and "
                    "may append a correction.")
        return ""
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["STABLE", "SLOW", "VOLATILE", "DIRECT", "VERIFY", "SEARCH_FIRST",
           "enabled", "FreshnessRead", "classify_freshness", "FreshnessStrategy",
           "strategy_for", "live_directive"]
