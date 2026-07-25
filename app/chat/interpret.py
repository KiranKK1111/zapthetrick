"""Chat interpretation layer — the structured brief (vNext §3.13, Stage 7 A).

"Understood once, understood consistently everywhere." A turn is CANONICALIZED
(typo / shorthand / anaphora) and then interpreted into ONE typed **structured
brief** that every downstream decision reads — clarify (§3.7), the deliverable
engine (§3.8), the task profile (§2.6), the effort dial (§8.2) — so a vague
prompt is disambiguated ONCE at the top of the turn instead of re-guessed at each
stage (the drift that makes "understood it here, forgot it there" bugs).

    StructuredBrief = { goal, deliverable_class, constraints[], tone,
                        referents[], missing_slots[], contradictions[] }

The deterministic `canonicalize` pass is conservative — whitespace + a tiny SAFE
whole-word shorthand map + a leading-filler strip — and NEVER changes meaning;
the semantic work (typo/anaphora/mixed-language, the brief fields) is a
schema-enforced (§8.7) structured call whose function is INJECTED, so the logic
is unit-tested with no LLM. Fully fail-open: any error yields a minimal brief so
the pipeline always has one. Flag-gated (`chat.interpretation`, default OFF).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "chat", None), "interpretation", False))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Canonicalize — deterministic + conservative (never changes meaning)
# --------------------------------------------------------------------------- #
# Whole-word shorthand → expansion. Kept tiny + unambiguous so it can't corrupt
# a real word (e.g. "u" as a word → "you", but "u" inside "ubuntu" is untouched).
_SHORTHAND = {
    "u": "you", "ur": "your", "pls": "please", "plz": "please",
    "thx": "thanks", "w/": "with", "w/o": "without", "b/c": "because",
    "abt": "about", "rn": "right now", "tmrw": "tomorrow",
}
# A RUN of leading filler openers ("ok so ", "well um ") — repeated so "ok so do
# it" → "do it". Conservative: only strips at the very start of the turn.
_LEADING_FILLER = re.compile(
    r"^\s*(?:(?:so|ok|okay|well|um|uh|like|hey|yeah)\b[\s,]+)+", re.I)


def canonicalize(text: str) -> str:
    """Normalize a turn WITHOUT changing its meaning: collapse whitespace, strip a
    leading filler, and expand a small set of safe whole-word shorthands. Never
    raises → returns the input on error."""
    try:
        t = (text or "").strip()
        if not t:
            return ""
        t = _LEADING_FILLER.sub("", t, count=1)

        def _expand(m: re.Match) -> str:
            w = m.group(0)
            rep = _SHORTHAND.get(w.lower())
            if rep is None:
                return w
            return rep.capitalize() if w[:1].isupper() else rep

        # Match shorthand tokens (incl. the `w/` forms) as whole units.
        t = re.sub(r"(?<!\w)(?:w/o|w/|b/c|[A-Za-z]+)(?!\w)", _expand, t)
        return re.sub(r"\s+", " ", t).strip()
    except Exception:  # noqa: BLE001
        return (text or "").strip()


# --------------------------------------------------------------------------- #
# The structured brief
# --------------------------------------------------------------------------- #
BRIEF_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "deliverable_class": {"type": "string"},   # chat | answer_and_artifact | artifact_only | unknown
        "constraints": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
        "referents": {"type": "array", "items": {"type": "string"}},
        "missing_slots": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["goal"],
}


@dataclass
class StructuredBrief:
    goal: str = ""
    deliverable_class: str = "chat"
    constraints: list[str] = field(default_factory=list)
    tone: str = ""
    referents: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.missing_slots or self.contradictions)

    def as_dict(self) -> dict:
        return {"goal": self.goal, "deliverable_class": self.deliverable_class,
                "constraints": list(self.constraints), "tone": self.tone,
                "referents": list(self.referents),
                "missing_slots": list(self.missing_slots),
                "contradictions": list(self.contradictions)}

    @classmethod
    def from_obj(cls, obj: dict, *, canonical: str = "") -> "StructuredBrief":
        def _list(v):
            return [str(x) for x in v] if isinstance(v, (list, tuple)) else []
        o = obj or {}
        return cls(
            goal=str(o.get("goal") or canonical).strip(),
            deliverable_class=str(o.get("deliverable_class") or "chat").strip().lower(),
            constraints=_list(o.get("constraints")),
            tone=str(o.get("tone") or "").strip(),
            referents=_list(o.get("referents")),
            missing_slots=_list(o.get("missing_slots")),
            contradictions=_list(o.get("contradictions")))


_SYS = (
    "You interpret one user turn into a compact JSON brief the assistant's "
    "downstream stages all read. Fill: goal (one line), deliverable_class "
    "(chat | answer_and_artifact | artifact_only | unknown), constraints (hard "
    "requirements stated), tone, referents (what pronouns like 'it/that/the "
    "file' point to, resolved from the history), missing_slots (information "
    "NEEDED to answer well but ABSENT — the smallest set), contradictions "
    "(mutually incompatible asks). Empty arrays when none. Do NOT invent "
    "requirements the user didn't state.")


async def build_brief(text: str, *, history: "list[dict] | None" = None,
                      structured_fn=None) -> StructuredBrief:
    """Canonicalize `text` and interpret it into a `StructuredBrief`. The
    schema-enforced structured call is INJECTED (`structured_fn`; default the
    `core.structured` facade), so this is unit-tested with no LLM. Fail-open: any
    error → a minimal brief with `goal` = the canonical text, so the pipeline
    always has a brief. No-op-ish when disabled (still returns a minimal brief so
    callers can rely on the shape)."""
    canonical = canonicalize(text)
    minimal = StructuredBrief(goal=canonical)
    if not enabled() or not canonical:
        return minimal
    try:
        if structured_fn is None:
            from app.core.structured import structured as structured_fn
        msgs = [{"role": "system", "content": _SYS}]
        if history:
            _h = "\n".join(f"{m.get('role')}: {m.get('content','')[:300]}"
                           for m in history[-4:] if isinstance(m, dict))
            if _h:
                msgs.append({"role": "user",
                             "content": f"Recent history:\n{_h}"})
        msgs.append({"role": "user", "content": f"Turn: {canonical}"})
        res = await structured_fn(BRIEF_SCHEMA, msgs, tier="fast", name="brief")
        obj = getattr(res, "obj", None)
        if obj and isinstance(obj, dict):
            return StructuredBrief.from_obj(obj, canonical=canonical)
        return minimal
    except Exception as exc:  # noqa: BLE001
        log.info("interpret.build_brief failed: %s", exc)
        return minimal


__all__ = ["enabled", "canonicalize", "BRIEF_SCHEMA", "StructuredBrief",
           "build_brief"]
