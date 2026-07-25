"""Panel diarization (vNext §4.16, Stage 7 Component K).

A panel interview has 2–4 interviewers who take turns and hand off. The existing
`app/live/diarize.py` attributes by capture SOURCE + textual hand-off cues; §4.16
adds the missing acoustic layer — cluster the FOREIGN (interviewer) voice by a
speaker embedding (ECAPA-class) into distinct speaker SLOTS **P1/P2/P3**, so the
copilot knows WHO is speaking even mid-turn, and tracks a per-speaker role +
situation + attribution.

The speaker embedding is model-bound (ECAPA runs on-pod, on the audio). So this
module owns only the deterministic ONLINE CLUSTERING over embeddings that are
INJECTED — a new utterance's embedding is assigned to the nearest existing
speaker centroid above a merge threshold, else it spawns a new speaker (capped);
the centroid is a running mean. That logic is fully unit-testable with synthetic
vectors on the dev box; the real embedder simply feeds it in production.

**Fail-soft** is the whole contract: no embedding, an embedder error, or the flag
off → a single `P1` speaker (today's single-interviewer behaviour). Never raises.
Flag-gated (`live.panel_diarization`, default OFF).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_DEFAULT_MERGE = 0.70          # ECAPA cosine: same speaker typically > 0.7
_DEFAULT_MAX_SPEAKERS = 4


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "panel_diarization", False))
    except Exception:  # noqa: BLE001
        return False


def _cfg_float(name: str, default: float) -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.live, name, default) or default)
    except Exception:  # noqa: BLE001
        return default


def _cfg_int(name: str, default: int) -> int:
    try:
        from app.core.config_loader import cfg
        return int(getattr(cfg.live, name, default) or default)
    except Exception:  # noqa: BLE001
        return default


def _normalize(vec) -> "list[float] | None":
    try:
        v = [float(x) for x in vec]
        if not v:
            return None
        n = math.sqrt(sum(x * x for x in v))
        if n <= 1e-9:
            return None
        return [x / n for x in v]
    except Exception:  # noqa: BLE001
        return None


def _cosine(a: "list[float]", b: "list[float]") -> float:
    # Both already L2-normalized → cosine is the dot product.
    try:
        return sum(x * y for x, y in zip(a, b))
    except Exception:  # noqa: BLE001
        return 0.0


@dataclass
class SpeakerSlot:
    id: str                                  # "P1" / "P2" / "P3" …
    centroid: list[float] = field(default_factory=list)
    count: int = 0
    role: str = ""                           # inferred (from diarize cues)
    situation: str = ""                      # last per-speaker situation
    turns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "count": self.count, "role": self.role,
                "situation": self.situation, "turns": len(self.turns)}


@dataclass
class PanelDiarizer:
    """Online voice-embedding clustering into P1/P2/P3 speaker slots."""
    merge_threshold: float = 0.0             # 0 → read from config lazily
    max_speakers: int = 0
    slots: list[SpeakerSlot] = field(default_factory=list)

    def _threshold(self) -> float:
        return self.merge_threshold or _cfg_float("panel_merge_threshold", _DEFAULT_MERGE)

    def _max(self) -> int:
        return self.max_speakers or _cfg_int("panel_max_speakers", _DEFAULT_MAX_SPEAKERS)

    def _single(self, text: str) -> SpeakerSlot:
        """Fail-soft: ensure a single P1 slot exists and return it."""
        if not self.slots:
            self.slots.append(SpeakerSlot(id="P1"))
        slot = self.slots[0]
        if text:
            slot.turns.append(text.strip())
            if len(slot.turns) > 50:
                slot.turns.pop(0)
        return slot

    def assign(self, embedding=None, *, text: str = "", role: str = "",
               situation: str = "") -> SpeakerSlot:
        """Assign an interviewer utterance to a speaker slot by nearest voice
        centroid. No/blank embedding → fail-soft single P1. Never raises."""
        try:
            v = _normalize(embedding) if embedding is not None else None
            if v is None:
                slot = self._single(text)
            else:
                thr = self._threshold()
                best, best_sim = None, -1.0
                for s in self.slots:
                    if not s.centroid:
                        continue
                    sim = _cosine(v, s.centroid)
                    if sim > best_sim:
                        best_sim, best = sim, s
                if best is not None and best_sim >= thr:
                    slot = best
                    self._update_centroid(slot, v)
                elif len(self.slots) < self._max():
                    slot = SpeakerSlot(id=f"P{len(self.slots) + 1}",
                                       centroid=list(v), count=1)
                    self.slots.append(slot)
                elif best is not None:
                    # Cap reached — attach to the nearest rather than spawn more.
                    slot = best
                    self._update_centroid(slot, v)
                else:
                    slot = SpeakerSlot(id="P1", centroid=list(v), count=1)
                    self.slots.append(slot)
                if text:
                    slot.turns.append(text.strip())
                    if len(slot.turns) > 50:
                        slot.turns.pop(0)
            if role:
                slot.role = role
            if situation:
                slot.situation = situation
            return slot
        except Exception:  # noqa: BLE001 — fail-soft to a single interviewer
            return self._single(text)

    def _update_centroid(self, slot: SpeakerSlot, v: "list[float]") -> None:
        """Incremental running-mean centroid, re-normalized."""
        c = slot.centroid
        n = slot.count
        if not c:
            slot.centroid = list(v)
            slot.count = 1
            return
        merged = [(c[i] * n + v[i]) / (n + 1) for i in range(min(len(c), len(v)))]
        slot.centroid = _normalize(merged) or c
        slot.count = n + 1

    def panel_size(self) -> int:
        return len(self.slots) or 1

    def is_panel(self) -> bool:
        return len(self.slots) > 1

    def describe(self) -> list[dict]:
        return [s.to_dict() for s in self.slots]


def for_tracker(tracker) -> PanelDiarizer:
    """One PanelDiarizer per live session, stashed on its sequence tracker."""
    d = getattr(tracker, "_panel_diarizer", None)
    if d is None:
        d = PanelDiarizer()
        try:
            setattr(tracker, "_panel_diarizer", d)
        except Exception:  # noqa: BLE001
            pass
    return d


__all__ = ["enabled", "SpeakerSlot", "PanelDiarizer", "for_tracker"]
