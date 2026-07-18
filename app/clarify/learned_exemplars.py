"""Learned intent exemplars — self-improving intent (Architecture roadmap #12).

The semantic classifier (`intent_semantic.py`) matches a turn against seed
exemplar phrasings per intent. This store lets those exemplars GROW from user
feedback:

  * a 👍 (or a confirmed-correct turn) adds the phrasing as a **positive**
    exemplar for its intent — reinforcing that phrasing for next time;
  * a 👎 where the intent was wrong adds it as a **negative** exemplar for the
    (mis)classified intent — so a near-identical phrase is penalized and the
    classifier defers to the next-best intent / the regex fallback.

Device-local (one user per install), so this is a single global store persisted
to `~/.zapthetrick/learned_exemplars.json` — no user threading through the hot
classification path. Bounded per intent, deduped, length-capped. Everything is
fail-open: a read/write error degrades to the seed exemplars only.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import threading

log = logging.getLogger(__name__)

# Bounds so feedback can't grow the matrix without limit.
MAX_PER_INTENT = 50
MAX_PHRASE_CHARS = 200

_LOCK = threading.Lock()
_POS: dict[str, list[str]] = {}
_NEG: dict[str, list[str]] = {}
# G2: parallel stores for the Understanding pass's OTHER classifiers, so
# difficulty + task self-improve from feedback too. Keyed by space →
# {label: [phrases]}. Intent stays in _POS/_NEG (100% unchanged).
_POS_SPACES: dict[str, dict[str, list[str]]] = {}
_NEG_SPACES: dict[str, dict[str, list[str]]] = {}
_LOADED = False
# Bumped on every mutation so the classifier cache knows to rebuild.
_VERSION = 0


def _bucket(space: str, negative: bool) -> dict[str, list[str]]:
    """The intent/label → phrases dict for a (space, polarity)."""
    if space == "intent":
        return _NEG if negative else _POS
    store = _NEG_SPACES if negative else _POS_SPACES
    return store.setdefault(space, {})


def _store_path() -> pathlib.Path:
    override = os.environ.get("ZAPTHETRICK_LEARNED_EXEMPLARS")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".zapthetrick" / "learned_exemplars.json"


def enabled() -> bool:
    """Master switch (`semantic_intent.learn_exemplars`, default off)."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.semantic_intent, "learn_exemplars", False))
    except Exception:  # noqa: BLE001 — fail-open to off
        return False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        try:
            p = _store_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                _POS.update({k: list(v) for k, v in
                             (data.get("positive") or {}).items()})
                _NEG.update({k: list(v) for k, v in
                             (data.get("negative") or {}).items()})
                for sp, m in (data.get("positive_spaces") or {}).items():
                    _POS_SPACES[sp] = {k: list(v) for k, v in m.items()}
                for sp, m in (data.get("negative_spaces") or {}).items():
                    _NEG_SPACES[sp] = {k: list(v) for k, v in m.items()}
        except Exception as exc:  # noqa: BLE001 — corrupt/missing file → empty
            log.info("learned exemplars: load failed (%s); starting empty", exc)
        _LOADED = True


def _persist() -> None:
    try:
        p = _store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"positive": _POS, "negative": _NEG,
                        "positive_spaces": _POS_SPACES,
                        "negative_spaces": _NEG_SPACES}, ensure_ascii=False),
            encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort
        log.info("learned exemplars: persist failed (%s)", exc)


def _norm(phrase: str) -> str:
    return " ".join((phrase or "").split())[:MAX_PHRASE_CHARS]


def add(intent: str, phrase: str, *, negative: bool = False,
        persist: bool = True, space: str = "intent") -> bool:
    """Record a learned exemplar for a classifier `space` (intent | difficulty |
    task). Returns True if newly added (deduped, case-insensitive, per-label cap).
    Never raises."""
    global _VERSION
    intent = (intent or "").strip()
    text = _norm(phrase)
    if not intent or len(text) < 3:
        return False
    _ensure_loaded()
    with _LOCK:
        cur = _bucket(space, negative).setdefault(intent, [])
        low = text.lower()
        if any(e.lower() == low for e in cur):
            return False
        cur.append(text)
        if len(cur) > MAX_PER_INTENT:
            del cur[0:len(cur) - MAX_PER_INTENT]   # drop oldest
        _VERSION += 1
        if persist:
            _persist()
    # Invalidate the classifier's exemplar matrix so the new phrase is used.
    try:
        from app.clarify import intent_semantic
        intent_semantic.reset_cache()
    except Exception:  # noqa: BLE001
        pass
    return True


def positives(space: str = "intent") -> dict[str, list[str]]:
    """Learned positive exemplars for a space (label → phrases). Empty when
    disabled."""
    if not enabled():
        return {}
    _ensure_loaded()
    with _LOCK:
        src = _POS if space == "intent" else _POS_SPACES.get(space, {})
        return {k: list(v) for k, v in src.items() if v}


def negatives(space: str = "intent") -> dict[str, list[str]]:
    """Learned negative exemplars for a space (label → phrases). Empty when
    disabled."""
    if not enabled():
        return {}
    _ensure_loaded()
    with _LOCK:
        src = _NEG if space == "intent" else _NEG_SPACES.get(space, {})
        return {k: list(v) for k, v in src.items() if v}


def version() -> int:
    """Monotonic mutation counter — the classifier cache keys on this."""
    return _VERSION


def stats() -> dict:
    _ensure_loaded()
    return {
        "positive": sum(len(v) for v in _POS.values()),
        "negative": sum(len(v) for v in _NEG.values()),
        "intents": sorted(set(_POS) | set(_NEG)),
    }


def clear(*, persist: bool = True) -> None:
    """Forget all learned exemplars (privacy control)."""
    global _VERSION
    with _LOCK:
        _POS.clear()
        _NEG.clear()
        _POS_SPACES.clear()
        _NEG_SPACES.clear()
        _VERSION += 1
        if persist:
            _persist()
    try:
        from app.clarify import intent_semantic
        intent_semantic.reset_cache()
    except Exception:  # noqa: BLE001
        pass


def learn_from_feedback(kind: str, question: str | None,
                        intent: str | None,
                        corrected_intent: str | None = None,
                        difficulty: str | None = None,
                        task: str | None = None) -> dict:
    """Map a 👍/👎 feedback event to exemplar updates (#12). No-op when learning
    is off or the question/intent aren't supplied. Never raises.

      • 👍 (up)   → the question is a POSITIVE exemplar for `intent` (reinforce).
      • 👎 (down) → the question is a NEGATIVE exemplar for `intent` (demote); if
        `corrected_intent` is given, also a POSITIVE for that.

    Returns ``{"added": [ {intent, negative}... ]}``.
    """
    added: list[dict] = []
    try:
        if not enabled() or not question or not intent:
            return {"added": added}
        k = (kind or "").strip().lower()
        if k == "up":
            if add(intent, question):
                added.append({"intent": intent, "negative": False})
            # G2: a good answer means the WHOLE read was right → reinforce the
            # difficulty + task classifiers too (not just intent).
            if difficulty and add(difficulty, question, space="difficulty"):
                added.append({"space": "difficulty", "label": difficulty})
            if task and add(task, question, space="task"):
                added.append({"space": "task", "label": task})
        elif k == "down":
            if add(intent, question, negative=True):
                added.append({"intent": intent, "negative": True})
            if corrected_intent and corrected_intent != intent:
                if add(corrected_intent, question):
                    added.append({"intent": corrected_intent, "negative": False})
    except Exception as exc:  # noqa: BLE001 — feedback learning is best-effort
        log.info("learn_from_feedback failed: %s", exc)
    return {"added": added}


def _reset_for_test() -> None:
    """Test helper: wipe in-memory state without touching disk."""
    global _LOADED, _VERSION
    with _LOCK:
        _POS.clear()
        _NEG.clear()
        _POS_SPACES.clear()
        _NEG_SPACES.clear()
        _LOADED = True          # skip disk load
        _VERSION += 1


__all__ = [
    "add", "positives", "negatives", "version", "stats", "clear", "enabled",
    "learn_from_feedback", "MAX_PER_INTENT", "MAX_PHRASE_CHARS",
]
