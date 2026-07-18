"""Evidence graph + per-source trust for the CHAT turn (roadmap Phase 3 #7).

The live pipeline has `live/evidence.py`; the chat side surfaced sources but
carried NO per-source trust — a recalled memory, a cross-session episode and a
knowledge-graph neighbor were all presented flat. This module assembles the
chat turn's evidence into typed `EvidenceSource` records, each with a calibrated
trust weight, and rolls them up into a compact `grounding` block for the
response envelope.

Trust is a deterministic prior per source KIND, refined by the signal the source
already carries (a retrieval hit's relevance, a memory object's importance, a KG
relation's support). No LLM, no embeddings — it reads fields already computed.

Consumed in `routes_agents.py` at envelope-build time:
  * `grounding` envelope block (sources + aggregate trust + any conflicts).
  * feeds the aggregate-confidence retrieval signal.

Fail-open: any error → an empty evidence set (today's behavior).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Trust priors per evidence kind (0..1). Retrieval from the user's own uploaded
# documents is the most trustworthy; a cross-session episode the least (it is a
# hint about a *related* past turn, not grounding for THIS one).
_KIND_PRIOR = {
    "document": 0.85,     # retrieved chunk from an uploaded doc / RAG store
    "kg": 0.75,           # knowledge-graph relation extracted from real content
    "memory": 0.65,       # durable memory object (decision/preference/entity)
    "episode": 0.45,      # a related prior conversation (weak, associative)
}
_DEFAULT_PRIOR = 0.5


@dataclass
class EvidenceSource:
    kind: str                       # document | kg | memory | episode
    ref: str                        # short human-readable pointer
    trust: float = 0.0              # 0..1 calibrated trust
    detail: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "ref": self.ref,
                "trust": round(float(self.trust), 3), "detail": self.detail}


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _blend(prior: float, signal: float | None) -> float:
    """Blend the kind prior with the source's own signal (when present)."""
    if signal is None:
        return _clip(prior)
    return _clip(0.5 * prior + 0.5 * _clip(signal))


def from_memory(items) -> list[EvidenceSource]:
    out: list[EvidenceSource] = []
    for m in (items or []):
        content = str(getattr(m, "content", "") or (m.get("content") if isinstance(m, dict) else ""))
        if not content.strip():
            continue
        imp = getattr(m, "importance", None)
        if imp is None and isinstance(m, dict):
            imp = m.get("importance")
        out.append(EvidenceSource(
            "memory", content[:80], _blend(_KIND_PRIOR["memory"],
                                           float(imp) if imp is not None else None),
            detail=content[:160]))
    return out


def from_episodes(items) -> list[EvidenceSource]:
    out: list[EvidenceSource] = []
    for e in (items or []):
        q = str((e.get("question") if isinstance(e, dict) else getattr(e, "question", "")) or "")
        if not q.strip():
            continue
        out.append(EvidenceSource("episode", q[:80], _KIND_PRIOR["episode"],
                                  detail=q[:160]))
    return out


def from_kg(names, relations=None) -> list[EvidenceSource]:
    out: list[EvidenceSource] = []
    rel_list = list(relations or [])
    for n in (names or []):
        name = str(n)
        if not name.strip():
            continue
        # A neighbor backed by an explicit relation is trusted more.
        supported = any(name.lower() in str(r).lower() for r in rel_list)
        out.append(EvidenceSource(
            "kg", name[:80],
            _blend(_KIND_PRIOR["kg"], 0.9 if supported else None),
            detail=name[:160]))
    return out


def from_retrieval(hits) -> list[EvidenceSource]:
    out: list[EvidenceSource] = []
    for h in (hits or []):
        if isinstance(h, dict):
            text = str(h.get("content") or h.get("text") or "")
            score = h.get("rerank_score", h.get("score"))
        else:
            text = str(getattr(h, "text", "") or getattr(h, "content", "") or "")
            score = getattr(h, "rerank_score", None)
            if score is None:
                score = getattr(h, "score", None)
        if not text.strip():
            continue
        sig = None
        if isinstance(score, (int, float)):
            # rerank logits are unbounded; squash. RRF scores are already small.
            sig = _clip(float(score)) if 0 <= float(score) <= 1 else 0.7
        out.append(EvidenceSource("document", text[:80],
                                  _blend(_KIND_PRIOR["document"], sig),
                                  detail=text[:160]))
    return out


def assemble(*, memory=None, episodes=None, kg_names=None, kg_relations=None,
             retrieval=None) -> list[EvidenceSource]:
    """Collect all available chat evidence into trust-scored sources.
    Never raises; a failing sub-source is simply skipped."""
    sources: list[EvidenceSource] = []
    for fn, arg in ((from_retrieval, retrieval), (from_memory, memory),
                    (from_episodes, episodes)):
        try:
            sources.extend(fn(arg))
        except Exception:  # noqa: BLE001
            pass
    try:
        sources.extend(from_kg(kg_names, kg_relations))
    except Exception:  # noqa: BLE001
        pass
    return sources


def aggregate_trust(sources) -> float:
    """Trust-weighted mean trust of the evidence set (0 when empty)."""
    vals = [s.trust for s in (sources or [])]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def grounding_block(sources, *, conflicts=None, limit: int = 8) -> dict | None:
    """A compact `grounding` envelope block, or None when there is no evidence."""
    srcs = list(sources or [])
    if not srcs:
        return None
    srcs.sort(key=lambda s: s.trust, reverse=True)
    block = {
        "sources": [s.as_dict() for s in srcs[:limit]],
        "count": len(srcs),
        "trust": aggregate_trust(srcs),
    }
    if conflicts:
        block["conflicts"] = list(conflicts)
    return block


__all__ = [
    "EvidenceSource", "assemble", "aggregate_trust", "grounding_block",
    "from_memory", "from_episodes", "from_kg", "from_retrieval",
]
