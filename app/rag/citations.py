"""Span citations — grounding claims to sources (vNext §8.3, Stage 7 C).

Retrieval v2's contextual chunking (`contextual.py`) and the bge cross-encoder
rerank (`rerank.py`) already surface the RIGHT chunks; this closes the loop by
grounding the ANSWER to them. For each claim (sentence) in the answer, it finds
the retrieved chunk + the exact QUOTE SPAN that supports it, and emits a
`{claim_span, doc, chunk, quote_span}` citation into `envelope.grounding.
citations` — so the FE can render tappable ¹²³ marks that open a source card with
the supporting quote highlighted, and a reader (or the model itself) can tell a
grounded claim from an ungrounded one.

Deterministic span alignment (difflib longest-matching-block over token
sequences) — no model, so it's fast + unit-tested with no network. A claim with
no chunk above `span_citation_min_score` is simply left un-cited (never a false
citation). Pure + fail-open. Flag-gated (`advanced_rag.span_citations`, OFF).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_SENT = re.compile(r"[^.!?\n]+[.!?]?", re.S)
_WORD = re.compile(r"\w+")


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.advanced_rag, "span_citations", False))
    except Exception:  # noqa: BLE001
        return False


def _min_score() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.advanced_rag, "span_citation_min_score", 0.4))
    except Exception:  # noqa: BLE001
        return 0.4


@dataclass
class SpanCitation:
    index: int                       # the ¹²³ number
    claim_span: tuple[int, int]      # (start, end) char range in the ANSWER
    doc: str                         # source doc id / title
    chunk: str                       # chunk id
    quote: str                       # the supporting text
    quote_span: tuple[int, int]      # (start, end) char range in the CHUNK
    score: float

    def as_dict(self) -> dict:
        return {"index": self.index,
                "claim_span": list(self.claim_span), "doc": self.doc,
                "chunk": self.chunk, "quote": self.quote,
                "quote_span": list(self.quote_span), "score": round(self.score, 3)}


def _claims(answer: str) -> "list[tuple[str, int, int]]":
    """Split the answer into claim sentences with their char offsets."""
    out = []
    for m in _SENT.finditer(answer or ""):
        s = m.group(0).strip()
        if len(_WORD.findall(s)) >= 4:      # skip fragments / headers
            start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
            out.append((s, start, start + len(s)))
    return out


def _best_quote(claim: str, chunk_text: str) -> "tuple[str, int, int, float]":
    """The span of `chunk_text` that best supports `claim` + a 0..1 score. Uses
    difflib's longest matching block over token sequences, mapped back to char
    offsets. ('', 0, 0, 0.0) when there's no meaningful overlap."""
    ct = chunk_text or ""
    claim_toks = [t.lower() for t in _WORD.findall(claim)]
    # Token spans in the chunk (word -> its char range).
    spans = [(m.group(0).lower(), m.start(), m.end())
             for m in _WORD.finditer(ct)]
    chunk_toks = [s[0] for s in spans]
    if not claim_toks or not chunk_toks:
        return ("", 0, 0, 0.0)
    sm = difflib.SequenceMatcher(a=claim_toks, b=chunk_toks, autojunk=False)
    block = sm.find_longest_match(0, len(claim_toks), 0, len(chunk_toks))
    if block.size == 0:
        return ("", 0, 0, 0.0)
    # Score = matched claim tokens / total claim tokens (how much of the claim
    # the quote covers).
    score = block.size / max(1, len(claim_toks))
    q_start = spans[block.b][1]
    q_end = spans[block.b + block.size - 1][2]
    return (ct[q_start:q_end], q_start, q_end, score)


def build_citations(answer: str, chunks: "list[dict]") -> "list[SpanCitation]":
    """Ground each claim in `answer` to its best supporting chunk. `chunks` = the
    retrieved+reranked hits (each `{content|text, doc|source, chunk_id|id}`).
    Returns sequential `SpanCitation`s (only claims that clear the min score are
    cited). Never raises → []."""
    try:
        if not enabled() or not (answer or "").strip() or not chunks:
            return []
        thr = _min_score()
        cites: list[SpanCitation] = []
        idx = 0
        for claim, c_start, c_end in _claims(answer):
            best = None
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                text = str(ch.get("content") or ch.get("text") or "")
                if not text.strip():
                    continue
                quote, q_start, q_end, score = _best_quote(claim, text)
                if quote and (best is None or score > best[0]):
                    best = (score, ch, quote, q_start, q_end)
            if best is None or best[0] < thr:
                continue
            score, ch, quote, q_start, q_end = best
            idx += 1
            cites.append(SpanCitation(
                index=idx, claim_span=(c_start, c_end),
                doc=str(ch.get("doc") or ch.get("source") or ch.get("title") or ""),
                chunk=str(ch.get("chunk_id") or ch.get("id") or ""),
                quote=quote, quote_span=(q_start, q_end), score=score))
        return cites
    except Exception:  # noqa: BLE001
        return []


def grounding_citations(cites: "list[SpanCitation]") -> dict:
    """Envelope shape — `envelope.grounding.citations`. Empty dict when none."""
    if not cites:
        return {}
    return {"citations": [c.as_dict() for c in cites]}


__all__ = ["enabled", "SpanCitation", "build_citations", "grounding_citations"]
