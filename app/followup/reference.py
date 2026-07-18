"""Reference resolution (followup-context-engine R3/R4).

`resolve(turn, state) -> Resolution` replaces pronouns, selection references
(ordinals), and entity mentions with concrete antecedents drawn from the
``ConversationState`` (Entity_Registry + last enumerations + goal). Deterministic;
no LLM call.

Confidence gating (R3.3 / Property 4): below the configured
``resolution_confidence_threshold`` the turn should defer to the
Clarification_System for ONE disambiguation question rather than guessing
(``Resolution.needs_clarification`` is set). No antecedent at all → fall back to
prompt handling (R3.4): an empty resolution with ``confidence == 0`` so the
caller leaves the turn untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ORDINALS = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1, "previous": -1,
}
_PRONOUNS = ("it", "that", "this", "those", "these", "them", "they", "same",
             "the above", "above", "the previous", "the last one")

_SELECTION_RE = re.compile(
    r"\b(?:the\s+)?(first|second|third|fourth|fifth|last|previous|"
    r"1st|2nd|3rd|4th|5th)\b(?:\s+one)?", re.IGNORECASE)
_OPTION_RE = re.compile(r"\boption\s+([a-z0-9])\b", re.IGNORECASE)


@dataclass
class Resolution:
    """The outcome of resolving references in a turn."""
    refs: list[str] = field(default_factory=list)        # the reference tokens found
    antecedents: list[str] = field(default_factory=list)  # resolved concrete targets
    confidence: float = 0.0                               # resolution_confidence
    needs_clarification: bool = False                     # below-threshold → ask one Q

    @property
    def resolved(self) -> bool:
        return bool(self.antecedents) and self.confidence > 0.0


def _threshold() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.followup, "resolution_confidence_threshold", 0.6))
    except Exception:  # noqa: BLE001
        return 0.6


def resolve(turn: str, state) -> Resolution:
    """Resolve references in `turn` against `state`. Deterministic; fail-open."""
    try:
        return _resolve(turn, state)
    except Exception:  # noqa: BLE001 — never break a turn (R3.4)
        return Resolution()


def _resolve(turn: str, state) -> Resolution:
    t = " ".join((turn or "").strip().split())
    if not t:
        return Resolution()
    low = t.lower()

    try:
        enumerations = state.enumerations() or []
    except Exception:  # noqa: BLE001
        enumerations = []
    try:
        entities = state.entities() or []
    except Exception:  # noqa: BLE001
        entities = []

    refs: list[str] = []
    antecedents: list[str] = []
    confidences: list[float] = []

    # 1) Selection references (ordinals / option letters) → enumerated options.
    sel = _SELECTION_RE.search(low)
    opt = _OPTION_RE.search(low)
    if sel or opt:
        if opt:
            refs.append(opt.group(0))
            idx = _letter_index(opt.group(1))
        else:
            refs.append(sel.group(0))
            idx = _ORDINALS.get(sel.group(1).lower())
        ant = _pick_enumeration(enumerations, idx)
        if ant is not None:
            antecedents.append(ant)
            # High when we have the options to point at; low otherwise → clarify.
            confidences.append(0.85 if enumerations else 0.3)
        else:
            # Named an ordinal but we have no options to resolve against.
            confidences.append(0.3)

    # 2) Entity mentions → Entity_Registry (exact, case-insensitive token match).
    for ent in entities:
        if re.search(rf"\b{re.escape(ent.lower())}\b", low):
            refs.append(ent)
            antecedents.append(ent)
            confidences.append(0.9)

    # 3) Pronouns → the most-salient antecedent (last entity / goal).
    if _has_pronoun(low) and not antecedents:
        refs.append(_first_pronoun(low))
        salient = entities[-1] if entities else _safe_goal(state)
        if salient:
            antecedents.append(salient)
            # Single salient antecedent → confident; ambiguous (many) → lower.
            confidences.append(0.8 if len(entities) <= 1 else 0.55)
        else:
            confidences.append(0.2)   # nothing to point at

    if not refs:
        return Resolution()           # no references → leave the turn untouched

    confidence = min(confidences) if confidences else 0.0
    res = Resolution(refs=refs, antecedents=antecedents, confidence=confidence)
    # Below threshold with at least one reference → defer to the clarifier (R3.3).
    if confidence < _threshold():
        res.needs_clarification = True
    return res


def _pick_enumeration(options: list[str], idx):
    if not options or idx is None:
        return None
    if idx == -1:
        return options[-1]
    if 0 <= idx < len(options):
        return options[idx]
    return None


def _letter_index(ch: str):
    ch = (ch or "").lower()
    if ch.isdigit():
        return int(ch) - 1
    if "a" <= ch <= "z":
        return ord(ch) - ord("a")
    return None


def _has_pronoun(low: str) -> bool:
    toks = set(re.findall(r"[a-z']+", low))
    return any((" " not in p and p in toks) or (" " in p and p in low)
               for p in _PRONOUNS)


def _first_pronoun(low: str) -> str:
    for p in _PRONOUNS:
        if (" " in p and p in low) or (" " not in p and
                                       re.search(rf"\b{re.escape(p)}\b", low)):
            return p
    return "it"


def _safe_goal(state):
    try:
        return state.goal()
    except Exception:  # noqa: BLE001
        return None


__all__ = ["resolve", "Resolution"]
