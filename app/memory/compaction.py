"""Auto-compaction → structured L4 digest (vNext §8.4, Stage 7 Component D).

A long thread can't send its whole history every turn. `app/chat/history.py`
already windows the recent turns and folds the aged ones into a flat-prose
rolling summary. §8.4 sharpens that in two ways this module owns:

  * a **window trigger** — compact when the live context fills past a ratio
    (~70%), not only when a message-count batch ages out; and
  * a **STRUCTURED digest** ("L4") — the aged turns become a TYPED
    `StructuredDigest{decisions, facts, entities, open_threads, goals,
    artifacts}` rather than an opaque paragraph, so it is searchable (Component
    D's `conversation_search`), durable, and losslessly re-injectable.

The digest is produced by a schema-enforced (§8.7) structured call whose function
is INJECTED (default `app.core.structured.structured`), so the whole module is
unit-tested with no LLM. Fail-open throughout: disabled OR any error → the caller
keeps today's flat-prose summary; nothing here can break a turn. Flag-gated
(`memory.auto_compaction`, default OFF).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.memory, "auto_compaction", False))
    except Exception:  # noqa: BLE001
        return False


def _window_ratio() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.memory, "compaction_window_ratio", 0.70) or 0.70)
    except Exception:  # noqa: BLE001
        return 0.70


# --------------------------------------------------------------------------- #
# Window trigger
# --------------------------------------------------------------------------- #
@dataclass
class CompactionDecision:
    compact: bool
    ratio: float                  # used / window
    threshold: float
    headroom_tokens: int          # tokens left before the window is full

    def to_dict(self) -> dict:
        return {"compact": self.compact, "ratio": round(self.ratio, 3),
                "threshold": round(self.threshold, 3),
                "headroom_tokens": self.headroom_tokens}


def should_compact(used_tokens: int, window_tokens: int, *,
                   threshold: float | None = None) -> CompactionDecision:
    """Decide whether to compact: True once the live window is `threshold`-full
    (default from config, ~0.70). Never raises → a no-compact decision on bad
    input (the safe default keeps everything verbatim)."""
    try:
        thr = _window_ratio() if threshold is None else float(threshold)
        w = int(window_tokens or 0)
        u = max(0, int(used_tokens or 0))
        if w <= 0:
            return CompactionDecision(False, 0.0, thr, 0)
        ratio = u / w
        return CompactionDecision(ratio >= thr, ratio, thr, max(0, w - u))
    except Exception:  # noqa: BLE001
        return CompactionDecision(False, 0.0, threshold or 0.70, 0)


# --------------------------------------------------------------------------- #
# Structured L4 digest
# --------------------------------------------------------------------------- #
DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "open_threads": {"type": "array", "items": {"type": "string"}},
        "goals": {"type": "array", "items": {"type": "string"}},
        "artifacts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decisions", "facts", "entities", "open_threads", "goals",
                 "artifacts"],
    "additionalProperties": False,
}

_DIGEST_INSTRUCTION = (
    "You compact the aged part of a conversation into a durable, SEARCHABLE "
    "digest so the thread can continue past the model's context window. From the "
    "messages, extract: decisions (choices made + why), facts (concrete stable "
    "facts, numbers, code/file references), entities (people, systems, files, "
    "libraries named), open_threads (unresolved questions / TODOs), goals (the "
    "user's stated objectives + preferences), artifacts (files, snippets, docs "
    "produced or referenced). Be concise and specific; drop pleasantries; never "
    "invent. Each item one short line.")


@dataclass
class StructuredDigest:
    decisions: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any((self.decisions, self.facts, self.entities,
                        self.open_threads, self.goals, self.artifacts))

    def to_dict(self) -> dict:
        return {"decisions": list(self.decisions), "facts": list(self.facts),
                "entities": list(self.entities),
                "open_threads": list(self.open_threads),
                "goals": list(self.goals), "artifacts": list(self.artifacts)}


def _coerce_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


async def build_digest(messages, *, structured_fn=None) -> StructuredDigest:
    """Fold aged messages into a typed L4 digest via a schema-enforced structured
    call (INJECTED `structured_fn`, default the `core.structured` facade — same
    seam as `chat.interpret.build_brief`), so this is unit-tested with no LLM.
    Fully fail-open: disabled OR any error → an empty digest (the caller then
    keeps the flat-prose summary). `messages` is [{role, content}] chronological."""
    if not messages or not enabled():
        return StructuredDigest()
    try:
        block = "\n".join(
            f"{m.get('role', '?')}: {(m.get('content') or '').strip()[:1200]}"
            for m in messages if (m.get("content") or "").strip())
        if not block.strip():
            return StructuredDigest()
        fn = structured_fn
        if fn is None:
            from app.core.structured import structured as fn  # type: ignore
        msgs = [{"role": "system", "content": _DIGEST_INSTRUCTION},
                {"role": "user", "content": f"MESSAGES:\n{block[:12000]}"}]
        res = await fn(DIGEST_SCHEMA, msgs, tier="standard", name="digest")
        obj = getattr(res, "obj", None)
        if not isinstance(obj, dict):
            return StructuredDigest()
        return StructuredDigest(
            decisions=_coerce_list(obj.get("decisions")),
            facts=_coerce_list(obj.get("facts")),
            entities=_coerce_list(obj.get("entities")),
            open_threads=_coerce_list(obj.get("open_threads")),
            goals=_coerce_list(obj.get("goals")),
            artifacts=_coerce_list(obj.get("artifacts")))
    except Exception:  # noqa: BLE001 — fail-open to empty (keep prose summary)
        return StructuredDigest()


def digest_to_text(digest: StructuredDigest) -> str:
    """Render the L4 digest as compact labelled prose for the system prompt (the
    re-injected long-range context). '' when the digest is empty."""
    try:
        if digest is None or digest.is_empty():
            return ""
        sections = [
            ("Goals", digest.goals), ("Decisions", digest.decisions),
            ("Facts", digest.facts), ("Entities", digest.entities),
            ("Open threads", digest.open_threads), ("Artifacts", digest.artifacts),
        ]
        parts = [f"{label}: " + "; ".join(items)
                 for label, items in sections if items]
        return "\n".join(parts).strip()
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["enabled", "CompactionDecision", "should_compact", "DIGEST_SCHEMA",
           "StructuredDigest", "build_digest", "digest_to_text"]
