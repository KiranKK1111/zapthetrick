"""Unified Understanding pass — the semantic 'brain' of a turn.

Instead of a triage LLM call + a semantic-intent embed + a regex `task_class` +
a keyword topic-shift check, this computes ONE query embedding and derives every
routing signal from it by nearest-exemplar similarity (bge-m3, already loaded):

    intent · difficulty · task_category · topic_shift · ambiguity ·
    capabilities · output_complexity

The exemplars are DATA (like intent's) and grow from feedback via
`learned_exemplars`. Every field is semantic-first with a deterministic
fail-open, so a missing/again-loading embedder degrades to today's behavior —
never an error. Injectable `embed_fn` makes the whole pass unit-testable without
the model.

This is the "brain"; `app.llm.router` is the "traffic controller" that turns an
`Understanding` into a concrete free model. Off by default
(`understanding.enabled`) — when off, callers keep their existing per-signal
paths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

log = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

# ── Difficulty exemplars: complexity of answering WELL, not topic. ──────────
DIFFICULTY_EXEMPLARS: dict[str, list[str]] = {
    "trivial": [
        "hi", "hello", "thanks", "what time is it", "who are you",
        "good morning", "ok cool", "what does TODO mean",
    ],
    "standard": [
        "what is a hash map", "explain this function", "write a function to "
        "reverse a string", "how do I read a file in Python",
        "summarize this paragraph", "fix this small bug",
    ],
    "hard": [
        "debug this concurrency race condition", "design a rate limiter",
        "optimize this query that scans millions of rows",
        "explain the tradeoffs between microservices and a monolith",
        "implement binary search with all edge cases handled",
        "walk through this multi-step algorithm and prove its complexity",
    ],
    "expert": [
        "prove this algorithm is optimal", "design a distributed consensus "
        "protocol", "build a complete production web app end to end",
        "derive the time complexity and prove correctness of this scheduler",
        "architect a fault-tolerant multi-region system with failover",
        "refactor this large codebase across many files safely",
    ],
}

# ── Task-category exemplars: what KIND of work, for capability routing. ─────
TASK_EXEMPLARS: dict[str, list[str]] = {
    "coding": [
        "write a function", "implement this feature", "fix this bug in my code",
        "refactor this module", "add tests for this", "debug this stack trace",
    ],
    "math": [
        "solve this equation", "compute the derivative", "prove this theorem",
        "what is the probability of", "calculate the complexity",
    ],
    "reasoning": [
        "think step by step about", "what are the tradeoffs", "why does this "
        "happen", "analyze the pros and cons", "reason about the best approach",
    ],
    "writing": [
        "write a professional email", "draft a resume", "make this persuasive",
        "improve the tone of this", "write a cover letter", "summarize this "
        "for an executive audience",
    ],
    "architecture": [
        "design the system architecture", "propose a database schema",
        "how should I structure this project", "high level system design",
    ],
    "agentic": [
        "build me a full app", "scaffold a new service", "create a project "
        "from scratch", "set up the whole repo with auth and tests",
    ],
    "research": [
        "what's the latest on", "look up recent information about",
        "find sources on", "what happened with", "current best practices for",
    ],
    "general": [
        "what is", "explain", "tell me about", "how does this work",
    ],
}

# Which capabilities a task category implies (fed to the router).
_TASK_CAPS: dict[str, tuple[str, ...]] = {
    "coding": ("code",),
    "math": ("reasoning",),
    "reasoning": ("reasoning",),
    "writing": ("writing",),
    "architecture": ("reasoning", "code"),
    "agentic": ("code", "tools"),
    "research": ("tools",),
    "general": (),
}

# Output size by intent (large = long build/doc; small = a fact/greeting).
_LARGE_INTENTS = frozenset({"project_build", "documentation"})
_MEDIUM_INTENTS = frozenset({"code_generation", "design", "comparison",
                             "test_generation", "debugging"})

_DIFF_ORDER = {"trivial": 0, "standard": 1, "hard": 2, "expert": 3}


@dataclass
class Understanding:
    """One coherent, semantic read of a turn (the router's input)."""
    intent: str = "general"
    intent_confidence: float = 0.0
    difficulty: str = "standard"
    difficulty_confidence: float = 0.0
    task_category: str = "general"
    topic_shift: bool = False
    ambiguity: float = 0.0
    capabilities: tuple[str, ...] = ()
    output_complexity: str = "medium"      # small | medium | large
    needs_fresh: bool = False              # time-sensitive → wants a web lookup
    source: str = "semantic"               # semantic | degraded
    embedding: list[float] | None = None   # cached query vector (for the router)

    def as_meta(self) -> dict:
        """Compact, JSON-safe view for the envelope/trace (no raw embedding)."""
        return {
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 3),
            "difficulty": self.difficulty,
            "task_category": self.task_category,
            "topic_shift": self.topic_shift,
            "ambiguity": round(self.ambiguity, 3),
            "capabilities": list(self.capabilities),
            "output_complexity": self.output_complexity,
            "needs_fresh": self.needs_fresh,
            "source": self.source,
        }


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.understanding, "enabled", False))
    except Exception:  # noqa: BLE001
        return False


def _default_embed(texts: Sequence[str]) -> list[list[float]]:
    from app.rag.embedder import embed
    return embed(list(texts))


# Cached exemplar matrices for the real embedder, keyed on the learned-exemplar
# version so feedback (G2) rebuilds them: {name: (version, (labels, matrix))}.
_MATRIX_CACHE: dict[str, tuple[int, tuple[list[str], object]]] = {}


def _merged(name: str, exemplars: dict[str, list[str]]) -> dict[str, list[str]]:
    """Seed exemplars + learned positives for this classifier space (G2). The
    space name matches `name` (difficulty | task). Never mutates the seed."""
    out = {k: list(v) for k, v in exemplars.items()}
    try:
        from app.clarify import learned_exemplars as _le
        for label, phrases in _le.positives(space=name).items():
            out.setdefault(label, [])
            out[label] = out[label] + list(phrases)
    except Exception:  # noqa: BLE001
        pass
    return out


def _build_matrix(name: str, exemplars: dict[str, list[str]], embed_fn: EmbedFn):
    import numpy as np
    labels, flat = [], []
    for label, phrases in _merged(name, exemplars).items():
        for p in phrases:
            labels.append(label)
            flat.append(p)
    vecs = np.asarray(embed_fn(flat), dtype="float32")
    return labels, vecs


def _learned_version() -> int:
    try:
        from app.clarify import learned_exemplars as _le
        return _le.version() if _le.enabled() else 0
    except Exception:  # noqa: BLE001
        return 0


def _matrix(name: str, exemplars: dict, embed_fn: EmbedFn | None):
    if embed_fn is not None:                     # injected (tests) → never cache
        return _build_matrix(name, exemplars, embed_fn)
    ver = _learned_version()
    cached = _MATRIX_CACHE.get(name)
    if cached is None or cached[0] != ver:       # rebuild when learning changes
        _MATRIX_CACHE[name] = (ver, _build_matrix(name, exemplars, _default_embed))
    return _MATRIX_CACHE[name][1]


def reset_cache() -> None:
    _MATRIX_CACHE.clear()


# Last turn's query embedding per conversation — the reference point for the
# next turn's implicit topic-shift check. Bounded so it can't grow unbounded.
_LAST_EMBEDDING: dict[str, list[float]] = {}
_LAST_MAX = 512


def last_embedding(conversation_id: str | None) -> list[float] | None:
    if not conversation_id:
        return None
    return _LAST_EMBEDDING.get(str(conversation_id))


def remember_embedding(conversation_id: str | None, vec: list[float] | None) -> None:
    if not conversation_id or not vec:
        return
    if len(_LAST_EMBEDDING) >= _LAST_MAX:
        _LAST_EMBEDDING.pop(next(iter(_LAST_EMBEDDING)), None)   # drop oldest
    _LAST_EMBEDDING[str(conversation_id)] = list(vec)


# episode_id -> (difficulty, task, intent_confidence) so a later 👍 can reinforce
# those classifiers (G2) + calibrate the intent threshold (G1), without the FE
# echoing them back.
_TURN_META: dict[str, tuple[str, str, float]] = {}


def remember_turn_meta(episode_id, difficulty: str, task: str,
                       intent_confidence: float = 0.0) -> None:
    if not episode_id:
        return
    if len(_TURN_META) >= _LAST_MAX:
        _TURN_META.pop(next(iter(_TURN_META)), None)
    _TURN_META[str(episode_id)] = (difficulty or "", task or "",
                                   float(intent_confidence or 0.0))


def turn_meta(episode_id) -> tuple[str, str, float] | None:
    return _TURN_META.get(str(episode_id)) if episode_id else None


def _classify(qvec, name: str, exemplars: dict, embed_fn: EmbedFn | None):
    """(label, cosine) for the nearest exemplar of `exemplars`."""
    import numpy as np
    labels, mat = _matrix(name, exemplars, embed_fn)
    sims = mat @ qvec
    idx = int(np.argmax(sims))
    return labels[idx], float(sims[idx])


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def understand(
    text: str,
    *,
    prev_embedding: list[float] | None = None,
    has_image: bool = False,
    needs_json: bool = False,
    caption: str | None = None,
    embed_fn: EmbedFn | None = None,
) -> Understanding:
    """The unified semantic read of a turn. §12/G10: when an image `caption` is
    supplied (from the vision model), it's folded into the embedded text so
    intent/difficulty/task reflect the picture, not just the words. Never raises —
    on any embedder failure it returns a degraded `Understanding`."""
    t = (text or "").strip()
    u = Understanding()
    if not t and not (caption or "").strip():
        return u
    # G10: embed text + image caption together (multimodal understanding).
    embed_text = t if not caption else (f"{t}\n{caption.strip()}" if t
                                        else caption.strip())
    try:
        import numpy as np
        ef = embed_fn or _default_embed
        qvec = np.asarray(ef([embed_text])[0], dtype="float32")
        u.embedding = qvec.tolist()
        if caption:
            has_image = True

        # Intent — reuse the intent classifier (keeps learned exemplars +
        # negative penalty). Classify the caption-folded text so intent reflects
        # the image too (G10).
        try:
            from app.clarify import intent_semantic
            hit = intent_semantic.classify(embed_text, embed_fn=embed_fn)
            if hit:
                u.intent, u.intent_confidence = hit
        except Exception:  # noqa: BLE001
            pass

        u.difficulty, u.difficulty_confidence = _classify(
            qvec, "difficulty", DIFFICULTY_EXEMPLARS, embed_fn)
        u.task_category, _tc_conf = _classify(
            qvec, "task", TASK_EXEMPLARS, embed_fn)

        # Implicit topic-shift: this turn sits far from the previous one in
        # embedding space (catches subject changes with no "new topic" cue).
        _explicit = _explicit_shift(t)
        _semantic_shift = False
        if prev_embedding is not None:
            _semantic_shift = _cosine(qvec, prev_embedding) < _shift_threshold()
        u.topic_shift = bool(_explicit or _semantic_shift)

        # Ambiguity: low intent confidence + short/vague turn → more ambiguous.
        u.ambiguity = _ambiguity(u.intent_confidence, t)

        # Capabilities + output size (derived — cheap, deterministic).
        caps = set(_TASK_CAPS.get(u.task_category, ()))
        if has_image:
            caps.add("vision")
        if needs_json:
            caps.add("json")
        # G6: a research/time-sensitive turn wants fresh external info → a `web`
        # capability that forces the tool loop (web_search) regardless of level.
        u.needs_fresh = (u.task_category == "research")
        if u.needs_fresh:
            caps.add("web")
        u.capabilities = tuple(sorted(caps))
        u.output_complexity = _complexity(u.intent, u.difficulty)
        u.source = "semantic"
        return u
    except Exception as exc:  # noqa: BLE001 — degrade, never fail a turn
        log.info("understanding degraded (%s)", exc)
        u.source = "degraded"
        u.topic_shift = _explicit_shift(t)
        return u


def _explicit_shift(text: str) -> bool:
    try:
        from app.followup.acts import is_topic_shift
        return is_topic_shift(text)
    except Exception:  # noqa: BLE001
        return False


def _shift_threshold() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.understanding, "topic_shift_similarity", 0.35))
    except Exception:  # noqa: BLE001
        return 0.35


def _ambiguity(intent_conf: float, text: str) -> float:
    # Farther below the semantic-intent threshold → more ambiguous; a very short
    # turn adds a little. Bounded [0,1].
    base = max(0.0, 0.6 - intent_conf)
    if len(text.split()) <= 3:
        base += 0.15
    return max(0.0, min(1.0, base))


def _complexity(intent: str, difficulty: str) -> str:
    if intent in _LARGE_INTENTS or difficulty == "expert":
        return "large"
    if intent in _MEDIUM_INTENTS or difficulty == "hard":
        return "medium"
    return "small"


__all__ = [
    "Understanding", "understand", "enabled", "reset_cache",
    "DIFFICULTY_EXEMPLARS", "TASK_EXEMPLARS",
]
