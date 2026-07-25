"""Distilled decision heads — the runtime side (vNext §9.7, Stage 11 Component B).

`distillation.py` exports traces into a training set; §9.7 is the RUNTIME: a
nightly-trained SetFit/LoRA head on the resident bge-m3 replaces a
heuristic/remote-fast decision with a ~5 ms local call — but ONLY once it has
proven it's at least as good ("never worse"). A promoted head sits in FRONT of
the old path: if it's confident, it decides; otherwise the turn falls through to
the existing logic. Every distilled decision logs its head VERSION.

Targets, in the §9.7 promotion order: SKIP/ANSWER → intent → difficulty →
freshness → semantic gates.

This module owns the deterministic dispatch:
  * `DistillHead` — a versioned head; `predict(embedding, model_fn)` (the head is
    the INJECTED seam — trained on-pod);
  * `HeadRegistry` — the ACTIVE promoted head per target; `consider(head, verdict)`
    promotes only on a PROMOTE canary verdict (never worse); `rollback`;
  * `decide(target, …)` — front-of-path: confident head → distilled decision +
    version log; else the injected `fallback_fn` (the old path);
  * `build_training_set` — labeled examples for a target from the golden set.

Pure dispatch/gating; the head + fallback are injected. Fail-open (any doubt →
the old path). Flag-gated (`distillation.enabled`, default OFF).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Distillation targets, in promotion order (§9.7).
SKIP_ANSWER = "skip_answer"
INTENT = "intent"
DIFFICULTY = "difficulty"
FRESHNESS = "freshness"
GATES = "gates"
TARGET_ORDER = (SKIP_ANSWER, INTENT, DIFFICULTY, FRESHNESS, GATES)

DISTILLED = "distilled"
FALLBACK = "fallback"


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.distillation, "enabled", False))
    except Exception:  # noqa: BLE001
        return False


def _min_conf() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.distillation, "min_confidence", 0.70) or 0.70)
    except Exception:  # noqa: BLE001
        return 0.70


@dataclass
class HeadPrediction:
    label: str
    confidence: float
    version: str = ""

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3),
                "version": self.version}


@dataclass
class DistillHead:
    target: str
    version: str
    labels: tuple = ()            # the valid output classes

    def predict(self, embedding, *, model_fn) -> "HeadPrediction | None":
        """Run the head over a bge-m3 `embedding`. `model_fn(embedding) ->
        {label: prob}` is the INJECTED trained head. Returns the top prediction or
        None (no model / bad output). Never raises."""
        try:
            if model_fn is None:
                return None
            dist = model_fn(embedding) or {}
            dist = {k: float(v) for k, v in dist.items()
                    if (not self.labels or k in self.labels) and float(v) >= 0}
            if not dist:
                return None
            top = max(dist, key=dist.get)
            total = sum(dist.values()) or 1.0
            return HeadPrediction(label=top, confidence=dist[top] / total,
                                  version=self.version)
        except Exception:  # noqa: BLE001
            return None


@dataclass
class DecisionResult:
    label: str
    source: str                   # distilled | fallback
    version: str = ""
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {"label": self.label, "source": self.source,
                "version": self.version, "confidence": round(self.confidence, 3),
                "reason": self.reason}


@dataclass
class HeadRegistry:
    """The ACTIVE promoted head per target + a per-decision version log."""
    active: dict = field(default_factory=dict)      # target -> DistillHead
    log: list = field(default_factory=list)         # recent (target, version)

    def consider(self, head: DistillHead, *, verdict: str) -> bool:
        """Promote `head` to active for its target ONLY on a PROMOTE canary
        verdict ('never worse'). `verdict` is the string from
        `eval.flywheel.CanaryController.evaluate` (== "promote") — compared as a
        literal so `llm` takes no dependency on `eval`. Never raises."""
        try:
            if verdict != "promote" or head is None or head.target not in TARGET_ORDER:
                return False
            self.active[head.target] = head
            return True
        except Exception:  # noqa: BLE001
            return False

    def rollback(self, target: str) -> None:
        """Drop the active head for a target (a regression alarm) → the old path."""
        self.active.pop(target, None)

    def active_version(self, target: str) -> str:
        h = self.active.get(target)
        return h.version if h is not None else ""

    def decide(self, target: str, embedding, *, model_fn=None,
               fallback_fn=None, min_confidence: float | None = None
               ) -> DecisionResult:
        """Front-of-path decision. A confident promoted head decides (~5 ms) and
        the version is logged; otherwise fall through to `fallback_fn(embedding)`
        (the old path). Disabled / no head / low-confidence / no model → fallback.
        Never raises — falls back on any error (never worse)."""
        try:
            def _fallback(reason: str) -> DecisionResult:
                lbl = ""
                if fallback_fn is not None:
                    try:
                        lbl = fallback_fn(embedding)
                    except Exception:  # noqa: BLE001
                        lbl = ""
                return DecisionResult(lbl or "", FALLBACK, "", 0.0, reason)

            if not enabled():
                return _fallback("distillation disabled")
            head = self.active.get(target)
            if head is None:
                return _fallback("no promoted head")
            pred = head.predict(embedding, model_fn=model_fn)
            if pred is None:
                return _fallback("head produced no prediction")
            thr = _min_conf() if min_confidence is None else min_confidence
            if pred.confidence < thr:
                return _fallback(f"head unsure ({pred.confidence:.2f} < {thr:.2f})")
            self.log.append((target, pred.version))
            if len(self.log) > 200:
                self.log.pop(0)
            return DecisionResult(pred.label, DISTILLED, pred.version,
                                  pred.confidence, "distilled head")
        except Exception:  # noqa: BLE001
            return DecisionResult("", FALLBACK, "", 0.0, "error → fallback")


def build_training_set(golden_set, target: str, *, label_fn) -> "list[dict]":
    """Extract labeled training examples for a `target` from the golden set.
    `label_fn(case) -> label | None` derives the target label from a case; a case
    with no label for this target is skipped. Never raises → []."""
    try:
        cases = getattr(golden_set, "cases", None)
        items = list(cases.values()) if isinstance(cases, dict) else list(golden_set or [])
        out: list[dict] = []
        for case in items:
            try:
                label = label_fn(case)
            except Exception:  # noqa: BLE001
                label = None
            if label is None:
                continue
            out.append({"target": target,
                        "prompt": getattr(case, "prompt", "") or "",
                        "label": label})
        return out
    except Exception:  # noqa: BLE001
        return []


__all__ = ["SKIP_ANSWER", "INTENT", "DIFFICULTY", "FRESHNESS", "GATES",
           "TARGET_ORDER", "DISTILLED", "FALLBACK", "enabled", "HeadPrediction",
           "DistillHead", "DecisionResult", "HeadRegistry", "build_training_set"]
