"""Predictive pre-answering (vNext §9.5, Stage 10 Component D).

The lowest-latency answer is the one already computed. §9.5: during a silence
(≥4 s) while the GPU is idle, pre-generate the top-2 PREDICTED questions
(`live.predict.predict_next`) in full — then, when the interviewer's real
question lands and MATCHES a prediction (≥0.92 cosine), flush the cached answer
instantly (perceived TTFT ≈ 0). A pre-answer goes STALE if a context dependency
it was built on has changed. Pre-answer is the first of three stacked
latency-hiding layers (pre-answer → speculation → hedge).

This module owns the deterministic orchestration:
  * `should_pregenerate` — the trigger (silence + idle GPU + capacity);
  * `PreAnswerCache` — store predicted (question → answer + embedding + deps),
    `match` (semantic flush at ≥ threshold, INJECTED embedder), `invalidate`
    (staleness by dependency);
  * a ledger hook records `pre_answered` / `flushed` / `stale`.

The generation + the embedder are injected seams; the trigger/match/staleness
logic is pure. Fail-open. Flag-gated (`live.pre_answering`, default OFF).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "pre_answering", False))
    except Exception:  # noqa: BLE001
        return False


def _silence_floor() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.live, "pre_answer_silence_ms", 4000.0) or 4000.0)
    except Exception:  # noqa: BLE001
        return 4000.0


def _match_threshold() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.live, "pre_answer_match_threshold", 0.92) or 0.92)
    except Exception:  # noqa: BLE001
        return 0.92


# --------------------------------------------------------------------------- #
# Trigger
# --------------------------------------------------------------------------- #
@dataclass
class PregenDecision:
    pregenerate: bool
    slots: int = 0                # how many predictions to pre-answer (≤ top_n)
    reason: str = ""

    def to_dict(self) -> dict:
        return {"pregenerate": self.pregenerate, "slots": self.slots,
                "reason": self.reason}


def should_pregenerate(*, silence_ms: float, gpu_idle: bool,
                       already_pregen: int = 0, top_n: int = 2,
                       min_silence_ms: float | None = None) -> PregenDecision:
    """Whether to spend idle-GPU cycles pre-generating. Fires only on a real
    silence (≥ floor) with an idle GPU and spare pre-answer slots. Disabled →
    never. Never raises."""
    try:
        if not enabled():
            return PregenDecision(False, 0, "disabled")
        floor = _silence_floor() if min_silence_ms is None else min_silence_ms
        if silence_ms < floor:
            return PregenDecision(False, 0, f"silence {silence_ms:.0f}ms < {floor:.0f}ms")
        if not gpu_idle:
            return PregenDecision(False, 0, "gpu busy")
        free = max(0, top_n - already_pregen)
        if free == 0:
            return PregenDecision(False, 0, "slots full")
        return PregenDecision(True, free, "idle + silent -> pre-generate")
    except Exception:  # noqa: BLE001
        return PregenDecision(False, 0, "error")


# --------------------------------------------------------------------------- #
# Pre-answer cache
# --------------------------------------------------------------------------- #
@dataclass
class PreAnswer:
    question: str
    answer: str
    embedding: list = field(default_factory=list)
    deps: frozenset = field(default_factory=frozenset)  # context deps (staleness)
    stale: bool = False


@dataclass
class PreAnswerHit:
    answer: str
    question: str
    similarity: float

    def to_dict(self) -> dict:
        return {"question": self.question, "similarity": round(self.similarity, 4)}


def _cosine(a, b) -> float:
    try:
        import math
        num = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return num / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


@dataclass
class PreAnswerCache:
    items: list = field(default_factory=list)      # list[PreAnswer]

    def store(self, question: str, answer: str, *, embedding=None,
              deps=None) -> None:
        """Record a pre-generated answer for a predicted question. No-op on empty."""
        try:
            if not (question or "").strip() or not (answer or "").strip():
                return
            self.items.append(PreAnswer(
                question=question.strip(), answer=answer,
                embedding=list(embedding or []),
                deps=frozenset(deps or ())))
        except Exception:  # noqa: BLE001
            pass

    def match(self, question: str, *, embed_fn=None,
              threshold: float | None = None) -> "PreAnswerHit | None":
        """Flush a cached pre-answer if `question` matches a fresh prediction at ≥
        threshold cosine. `embed_fn(text) -> vec` is INJECTED. Returns the hit or
        None (miss / disabled / all stale / no embedder). Never raises."""
        try:
            if not enabled() or not self.items or embed_fn is None:
                return None
            thr = _match_threshold() if threshold is None else threshold
            qv = embed_fn(question or "")
            if not qv:
                return None
            best, best_sim = None, -1.0
            for it in self.items:
                if it.stale or not it.embedding:
                    continue
                sim = _cosine(qv, it.embedding)
                if sim > best_sim:
                    best_sim, best = sim, it
            if best is not None and best_sim >= thr:
                return PreAnswerHit(answer=best.answer, question=best.question,
                                    similarity=best_sim)
            return None
        except Exception:  # noqa: BLE001
            return None

    def invalidate(self, dep) -> int:
        """Mark every pre-answer that depended on `dep` STALE (a context change).
        Returns how many were invalidated. Never raises."""
        try:
            n = 0
            for it in self.items:
                if not it.stale and dep in it.deps:
                    it.stale = True
                    n += 1
            return n
        except Exception:  # noqa: BLE001
            return 0

    def fresh_count(self) -> int:
        return sum(1 for it in self.items if not it.stale)

    def clear(self) -> None:
        self.items.clear()


__all__ = ["enabled", "PregenDecision", "should_pregenerate", "PreAnswer",
           "PreAnswerHit", "PreAnswerCache"]
