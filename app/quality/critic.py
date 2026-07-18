"""Response quality critic (evaluation-and-reliability R7).

`review(answer, asked_items, decisions) -> CriticReport` deterministically checks
two things after an answer is produced (non-blocking, post-stream):

  • **requirement coverage** — were the explicitly asked items addressed in the
    answer (R7.1);
  • **decision consistency** — does the answer contradict a decision recorded in
    the follow-up ConversationState (e.g. decided database=postgres but the
    answer pushes mysql).

No LLM call (Property 8/10). Missing data → an empty, "skipped" report (R7.4).
Material findings are surfaced as optional additive `quality` meta (R7.3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CriticReport:
    covered: bool = True
    gaps: list[str] = field(default_factory=list)          # asked items not addressed
    contradictions: list[str] = field(default_factory=list)  # vs recorded decisions
    skipped: bool = False

    @property
    def has_findings(self) -> bool:
        return bool(self.gaps or self.contradictions)

    def to_dict(self) -> dict:
        return {
            "covered": self.covered,
            "gaps": self.gaps,
            "contradictions": self.contradictions,
            "skipped": self.skipped,
        }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def review(answer: str, asked_items: list[str] | None = None,
           decisions: dict[str, str] | None = None) -> CriticReport:
    """Deterministic coverage + consistency check. Fail-safe: any error or no
    inputs → a skipped report (R7.4 / Property 8)."""
    try:
        return _review(answer, asked_items, decisions)
    except Exception:  # noqa: BLE001
        return CriticReport(skipped=True)


def _review(answer: str, asked_items, decisions) -> CriticReport:
    ans = (answer or "").strip()
    asked = [a for a in (asked_items or []) if a and a.strip()]
    decided = {k: v for k, v in (decisions or {}).items() if v}

    if not ans or (not asked and not decided):
        return CriticReport(skipped=True)

    ans_tokens = _tokens(ans)
    ans_low = ans.lower()

    # 1) Requirement coverage: an asked item is "covered" when most of its
    #    content tokens appear in the answer.
    gaps: list[str] = []
    for item in asked:
        toks = {t for t in _tokens(item) if len(t) > 2}
        if not toks:
            continue
        hit = len(toks & ans_tokens)
        if hit / len(toks) < 0.5:        # majority of the item's terms missing
            gaps.append(item)

    # 2) Decision consistency: the answer should not advocate a value that
    #    contradicts a recorded decision for the same key.
    contradictions: list[str] = []
    for key, value in decided.items():
        v = str(value).strip().lower()
        if not v:
            continue
        # If the decided value is entirely absent from the answer but the key's
        # topic is clearly discussed, that's a soft signal — only flag a HARD
        # contradiction: the answer names a competing value for a known slot.
        if key in ("language", "framework", "database", "platform", "choice"):
            alt = _competing_value(key, v, ans_low)
            if alt:
                contradictions.append(f"decided {key}={value} but answer uses {alt}")

    covered = not gaps
    return CriticReport(covered=covered, gaps=gaps,
                        contradictions=contradictions, skipped=False)


# Small known competitor sets so a contradiction is a real, named conflict —
# never a guess. Only flags when the decided value is ABSENT and a competitor is
# PRESENT.
_COMPETITORS = {
    "database": {"postgres", "postgresql", "mysql", "mongodb", "mongo",
                 "sqlite", "redis", "mariadb"},
    "language": {"python", "javascript", "typescript", "java", "go", "rust",
                 "c++", "c#", "ruby", "php", "kotlin", "swift"},
    "framework": {"react", "vue", "angular", "svelte", "flutter", "django",
                  "flask", "fastapi", "express", "spring", "rails"},
    "platform": {"web", "android", "ios", "desktop", "mobile"},
}


def _canon(v: str) -> str:
    """Canonicalize synonyms so postgres≡postgresql, mongo≡mongodb, etc."""
    v = (v or "").strip().lower().replace(" ", "")
    _SYN = {
        "postgresql": "postgres", "postgre": "postgres",
        "mongodb": "mongo",
        "js": "javascript", "ts": "typescript",
        "golang": "go",
    }
    return _SYN.get(v, v)


def _competing_value(key: str, decided: str, answer_low: str) -> str | None:
    pool = _COMPETITORS.get(key)
    if not pool:
        return None
    decided_canon = _canon(decided)
    # The decided value (or a synonym) present anywhere → no contradiction.
    for token in re.findall(r"[a-z0-9+#]+", answer_low):
        if _canon(token) == decided_canon:
            return None
    # Otherwise, a DIFFERENT known competitor present → a real contradiction.
    for cand in pool:
        if _canon(cand) == decided_canon:
            continue
        if re.search(rf"\b{re.escape(cand)}\b", answer_low):
            return cand
    return None


__all__ = ["CriticReport", "review"]
