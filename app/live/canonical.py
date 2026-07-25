"""Live canonicalization + said-state (vNext §4.8, Stage 7 Component G).

A spoken interview question arrives noisy: disfluencies ("um, so, like…"),
anaphora ("what about that?"), and — the one that silently breaks answers —
MULTIPLE questions in one breath ("what is Kafka and how does it scale?"). §4.8
canonicalizes the utterance ONCE (fast tier, ~150 ms) so every downstream stage
consumes the same clean form, and — critically — SPLITS a multi-question into its
ordered parts so BOTH get answered, in order (the §0 acceptance bar).

It also maintains a per-session **claims ledger** ("build, don't re-introduce"):
the claims the candidate has already made this interview, so a follow-up BUILDS
on them ("as I mentioned, the pipeline…") instead of repeating them — which
hard-feeds the consistency stage.

Deterministic + fail-open (a split failure just yields the whole utterance as one
question — never drops it). Flag-gated (`live.canonicalize` / `live.said_state`,
default OFF). The disfluency/anaphora heuristics are conservative; the LLM
canonicalizer (`live/interpret.py`) still handles the ambiguous cases.
"""
from __future__ import annotations

import re
import threading

_LOCK = threading.RLock()

# Filler words/phrases stripped from spoken input (whole-word / phrase).
_FILLERS = (
    "um", "uh", "erm", "ah", "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "literally", "like",
)
_FILLER_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(f) for f in _FILLERS) + r")(?!\w)", re.I)
_LEAD_RE = re.compile(r"^\s*(?:so|ok|okay|well|yeah|right|now)\b[\s,]+", re.I)
_WH = r"(?:what|why|how|when|where|which|who|whom|whose|can|could|would|do|does|" \
      r"did|is|are|was|were|explain|describe|tell|walk|give|write|implement|" \
      r"design)"
_WH_RE = re.compile(r"\b" + _WH + r"\b", re.I)


def canonicalize_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "canonicalize", False))
    except Exception:  # noqa: BLE001
        return False


def said_state_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "said_state", False))
    except Exception:  # noqa: BLE001
        return False


def canonicalize(text: str) -> str:
    """Strip disfluencies + a leading filler, collapse whitespace. Conservative —
    never changes the substantive words. Never raises."""
    try:
        t = (text or "").strip()
        if not t:
            return ""
        t = _LEAD_RE.sub("", t, count=1)
        t = _FILLER_RE.sub(" ", t)
        # Tidy the punctuation the filler removal left dangling.
        t = re.sub(r"\s+([,?.!;])", r"\1", t)        # space before punct
        t = re.sub(r"(?:,\s*)+,", ",", t)            # collapse comma runs
        t = re.sub(r",+(?=[?.!])", "", t)            # comma before a terminal
        return re.sub(r"\s+", " ", t).strip(" ,")
    except Exception:  # noqa: BLE001
        return (text or "").strip()


def _looks_like_question(seg: str) -> bool:
    s = (seg or "").strip()
    return bool(s) and (s.endswith("?") or bool(_WH_RE.match(s.lower())))


# A coordinating joiner between two questions: an optional preceding `. , ;`,
# then and / also / then / and-also / and-then, an optional trailing comma, and
# a following WH word (so "a list and a tuple" — "a" isn't WH — never splits).
_JOINER = re.compile(
    r"\s*[.,;]?\s+(?:and\s+(?:also|then)|also|and|then)\s*,?\s+(?=" + _WH
    + r"\b)", re.I)


def split_questions(text: str) -> list[str]:
    """Split a multi-question utterance into ordered sub-questions. Splits on
    strong boundaries (`?` / `;` / newlines) and on a coordinating joiner
    (`and`/`also`/`then`) that precedes another WH question. Returns the WHOLE
    utterance as a single item when it isn't multi (never drops content). Never
    raises."""
    try:
        t = canonicalize(text)
        if not t:
            return []
        # 1) Strong boundaries — keep a trailing '?' on each question.
        parts = [c.strip() for c in re.split(r"(?<=[?;])\s+|\n+", t) if c.strip()]
        # 2) Within a part, split on a coordinating joiner before a WH question,
        #    but only accept it when it yields 2+ question-like clauses.
        out: list[str] = []
        for p in parts:
            sub = [s.strip() for s in _JOINER.split(p) if s.strip()]
            if len(sub) > 1 and sum(_looks_like_question(s) for s in sub) >= 2:
                out.extend(sub)
            else:
                out.append(p)
        qs = [s for s in out if _looks_like_question(s)]
        if len(qs) >= 2:
            return [_ensure_q(s) for s in out if s.strip()]
        return [t]
    except Exception:  # noqa: BLE001
        return [canonicalize(text) or (text or "").strip()]


def _ensure_q(seg: str) -> str:
    s = seg.strip()
    if s and _WH_RE.match(s.lower()) and not s.endswith(("?", ".", "!")):
        return s + "?"
    return s


def is_multi_question(text: str) -> bool:
    return len(split_questions(text)) > 1


# --------------------------------------------------------------------------- #
# Claims ledger — "build, don't re-introduce"
# --------------------------------------------------------------------------- #
_NON_WORD = re.compile(r"[^0-9a-z]+")
_claims: dict[str, list[str]] = {}          # session -> [claim, ...]
_claim_keys: dict[str, set] = {}            # session -> {normalized_key, ...}


def _norm_claim(claim: str) -> str:
    return _NON_WORD.sub(" ", (claim or "").lower()).strip()


def is_new_claim(session_id: str, claim: str) -> bool:
    """Whether `claim` hasn't already been established this session."""
    key = _norm_claim(claim)
    if not key:
        return False
    with _LOCK:
        return key not in _claim_keys.get(session_id or "", set())


def record_claim(session_id: str, claim: str) -> bool:
    """Record a claim the candidate has now MADE. Returns True if it was new
    (recorded), False if already established. No-op when said-state is off."""
    if not said_state_enabled():
        return False
    key = _norm_claim(claim)
    if not key:
        return False
    with _LOCK:
        keys = _claim_keys.setdefault(session_id or "", set())
        if key in keys:
            return False
        keys.add(key)
        _claims.setdefault(session_id or "", []).append((claim or "").strip())
        return True


def claims(session_id: str) -> list[str]:
    with _LOCK:
        return list(_claims.get(session_id or "", ()))


def build_directive(session_id: str) -> str:
    """The consistency directive: build on what's already established, don't
    re-introduce it. '' when nothing's been claimed yet / said-state off."""
    if not said_state_enabled():
        return ""
    prior = claims(session_id)
    if not prior:
        return ""
    listed = "; ".join(prior[-6:])
    return ("You have already established this in the interview: " + listed
            + ". BUILD on these — reference them briefly, do not re-introduce or "
            "re-explain them from scratch.")


def forget_session(session_id: str) -> None:
    with _LOCK:
        _claims.pop(session_id or "", None)
        _claim_keys.pop(session_id or "", None)


def reset_for_tests() -> None:
    with _LOCK:
        _claims.clear()
        _claim_keys.clear()


__all__ = ["canonicalize_enabled", "said_state_enabled", "canonicalize",
           "split_questions", "is_multi_question", "is_new_claim",
           "record_claim", "claims", "build_directive", "forget_session",
           "reset_for_tests"]
