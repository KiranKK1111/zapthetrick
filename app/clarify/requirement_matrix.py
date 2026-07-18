"""Structured Requirement Matrix (SeveralFeatures.md — Step 5 / blackboard §).

Upgrades the pre-gate's flat `missing_required` / `missing_optional` lists into
a per-slot evidence table: every slot that matters for the detected intent is a
[SlotFact] carrying WHERE its value came from (prompt | history | preference |
attachment | inference | default) and HOW SURE we are of it. This is the
"Required vs Available" comparison the design doc calls the layer most
assistants get wrong — now with provenance, so downstream policies can treat
"language=python said this turn" differently from "language=python assumed".

Behavior contract: building the matrix NEVER changes the pre-gate decision by
itself — `missing_required()`/`missing_optional()` reproduce the flat lists the
pipeline already computes, so `assess()` stays byte-for-byte compatible while
gaining `.matrix` for policies, tracing, and Phase-2 attachment slot-filling.

Fail-open: every public function swallows its own errors (the clarifier gate
must never break on an enrichment).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Evidence sources, strongest-first. Kept as plain strings (JSON-friendly for
# traces + the SSE meta channel).
SOURCE_PROMPT = "prompt"          # named in the current turn
SOURCE_HISTORY = "history"        # named earlier in the conversation window
SOURCE_PREFERENCE = "preference"  # durable/known user preference or ledger slot
SOURCE_ATTACHMENT = "attachment"  # detected inside an uploaded file/project
SOURCE_INFERENCE = "inference"    # inferred (assumption) — not user-stated
SOURCE_DEFAULT = "default"        # safe default applied

# Default confidence per source — how much a value from that source should be
# trusted. Prompt beats history beats preference; inference/default are the
# weakest (they are what "proceed with stated assumptions" leans on).
_SOURCE_CONFIDENCE = {
    SOURCE_PROMPT: 0.95,
    SOURCE_ATTACHMENT: 0.90,
    SOURCE_PREFERENCE: 0.90,
    SOURCE_HISTORY: 0.80,
    SOURCE_INFERENCE: 0.60,
    SOURCE_DEFAULT: 0.50,
}

REQUIRED = "required"
OPTIONAL = "optional"
INFO = "info"                     # extracted but not load-bearing for the intent


@dataclass
class SlotFact:
    """One requirement-matrix row: a slot, its value (if any), and evidence."""
    name: str
    level: str = INFO                    # required | optional | info
    value: str | None = None
    source: str | None = None            # one of SOURCE_* (None when unfilled)
    confidence: float = 0.0              # 0..1 trust in the value

    @property
    def filled(self) -> bool:
        return bool(self.value)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "level": self.level, "value": self.value,
            "source": self.source, "confidence": round(self.confidence, 3),
        }


@dataclass
class RequirementMatrix:
    """Required-vs-available table for one turn's detected intent."""
    intent: str
    facts: dict[str, SlotFact] = field(default_factory=dict)

    # ---- writes -----------------------------------------------------------
    def add(self, name: str, level: str = INFO, value: str | None = None,
            source: str | None = None, confidence: float | None = None) -> None:
        try:
            conf = (confidence if confidence is not None
                    else (_SOURCE_CONFIDENCE.get(source or "", 0.0) if value else 0.0))
            self.facts[name] = SlotFact(name=name, level=level, value=value,
                                        source=source if value else None,
                                        confidence=conf if value else 0.0)
        except Exception:  # noqa: BLE001 — enrichment must never break the gate
            pass

    def fill(self, name: str, value: str, source: str,
             confidence: float | None = None) -> bool:
        """Fill a slot from new evidence (e.g. a StackProfile from an uploaded
        project). Only upgrades: an existing HIGHER-confidence value wins.
        Returns True when the value was applied."""
        try:
            if not value:
                return False
            conf = (confidence if confidence is not None
                    else _SOURCE_CONFIDENCE.get(source, 0.5))
            cur = self.facts.get(name)
            if cur is None:
                self.facts[name] = SlotFact(name=name, level=INFO, value=value,
                                            source=source, confidence=conf)
                return True
            if cur.filled and cur.confidence >= conf:
                return False
            cur.value, cur.source, cur.confidence = value, source, conf
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- reads ------------------------------------------------------------
    def missing_required(self) -> list[str]:
        return [f.name for f in self.facts.values()
                if f.level == REQUIRED and not f.filled]

    def missing_optional(self) -> list[str]:
        return [f.name for f in self.facts.values()
                if f.level == OPTIONAL and not f.filled]

    def available(self) -> dict[str, str]:
        return {f.name: f.value for f in self.facts.values() if f.filled}

    def evidence_score(self) -> float:
        """Mean confidence over the load-bearing (required+optional) filled
        slots — a matrix-level "how well-evidenced is this request" signal.
        1.0 when nothing load-bearing exists (nothing to be unsure about)."""
        rows = [f for f in self.facts.values() if f.level in (REQUIRED, OPTIONAL)]
        if not rows:
            return 1.0
        return sum(f.confidence for f in rows) / len(rows)

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "facts": [f.as_dict() for f in self.facts.values()],
            "missing_required": self.missing_required(),
            "missing_optional": self.missing_optional(),
            "evidence_score": round(self.evidence_score(), 3),
        }


def _source_for(name: str, value: str | None, text_slots: dict,
                known: dict) -> str | None:
    """Attribute WHERE a slot value came from, strongest evidence first."""
    if not value:
        return None
    if (text_slots.get(name) or "") == value:
        return SOURCE_PROMPT
    if (str(known.get(name) or "").lower() == str(value).lower()):
        return SOURCE_PREFERENCE
    return SOURCE_HISTORY          # blended text+recent extraction found it


def build_matrix(intent: str, slots: dict, missing_req: list[str],
                 missing_opt: list[str], *, text_slots: dict | None = None,
                 known_prefs: dict | None = None) -> RequirementMatrix:
    """Construct the matrix from the pre-gate's existing outputs.

    `slots` is the blended (text+recent) extraction the pipeline already made;
    `text_slots` is the same extraction over the CURRENT turn only, used purely
    for provenance (prompt vs history). `missing_req`/`missing_opt` are the
    pipeline's own lists — the matrix mirrors them 1:1 so it can never disagree
    with the decision that was made.
    """
    m = RequirementMatrix(intent=intent)
    try:
        tslots = text_slots or {}
        known = {k: v for k, v in (known_prefs or {}).items() if v}
        # Load-bearing rows first: everything the pipeline flagged missing.
        for name in missing_req:
            m.add(name, level=REQUIRED)
        for name in missing_opt:
            m.add(name, level=OPTIONAL)
        # Filled evidence rows for every extracted slot.
        for name, value in (slots or {}).items():
            if name in ("constraints", "has_tech") or not value:
                continue
            level = m.facts[name].level if name in m.facts else INFO
            src = _source_for(name, value, tslots, known)
            m.add(name, level=level, value=str(value), source=src)
        # The composite PROJECT_BUILD requirement is satisfied by either slot.
        if "language_or_framework" in m.facts and (
                slots.get("language") or slots.get("framework")):
            v = slots.get("language") or slots.get("framework")
            m.fill("language_or_framework", str(v),
                   _source_for("language", slots.get("language"), tslots, known)
                   or _source_for("framework", slots.get("framework"), tslots,
                                  known) or SOURCE_HISTORY)
        # Constraints as info rows (they inform generation, never block).
        for c in (slots or {}).get("constraints") or []:
            m.add(f"constraint:{c}", level=INFO, value=c, source=SOURCE_PROMPT)
    except Exception:  # noqa: BLE001 — a partial matrix is still useful
        pass
    return m


__all__ = ["RequirementMatrix", "SlotFact", "build_matrix",
           "REQUIRED", "OPTIONAL", "INFO",
           "SOURCE_PROMPT", "SOURCE_HISTORY", "SOURCE_PREFERENCE",
           "SOURCE_ATTACHMENT", "SOURCE_INFERENCE", "SOURCE_DEFAULT"]
