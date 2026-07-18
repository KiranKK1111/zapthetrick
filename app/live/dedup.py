"""Near-duplicate question guard for the live pipeline (user report
2026-07-08: "one interviewer question gets framed and answered multiple
times").

Root causes (audit): endpoint splits finalize one spoken question as two
utterances; the continuation-merge re-answer races an already-finished first
answer; speculative/final mismatches re-answer. The dead QidRegistry.should_process_final hook was removed 2026-07-09.

[QuestionDeduper] is the live guard: per-session, remembers the questions
answered in the last N seconds and flags a new one as a duplicate when its
normalized text is near-identical (difflib ratio ≥ threshold — deterministic,
no model). Deliberate bypasses stay with the CALLER: continuation merges and
verifier regenerations legitimately re-answer and must not consult the guard.

A repeat AFTER the window (interviewer genuinely re-asks) answers normally.
Fail-open: any internal error reports "not a duplicate".
"""
from __future__ import annotations

import re
import time
from collections import deque
from difflib import SequenceMatcher

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
# Filler the ASR/interviewer varies between takes ("so", "okay", "um") — a
# re-transcription differing only in filler is the same question.
_FILLER_RE = re.compile(
    r"\b(um+|uh+|so|okay|ok|well|right|like|you know|alright)\b")


def normalize(text: str) -> str:
    t = (text or "").lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _FILLER_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _embed_ready() -> bool:
    try:
        from app.rag import embedder as _emb
        return _emb.is_ready()
    except Exception:  # noqa: BLE001
        return False


def _embed_one(text: str):
    try:
        from app.rag.embedder import embed
        import numpy as np
        return np.asarray(embed([text])[0], dtype="float32")
    except Exception:  # noqa: BLE001
        return None


class QuestionDeduper:
    """Per-session sliding window of recently ANSWERED questions.

    Two similarity layers (2026-07-09): the char-ratio fast-path catches
    re-transcriptions; the SEMANTIC layer (embedding cosine, when the shared
    embedder is warm) catches PARAPHRASED re-asks the char ratio misses
    ("tell me about yourself" ≈ "walk me through your background"). Both are
    per-turn windowed and fail-open."""

    def __init__(self, *, window_s: float = 20.0,
                 similarity: float = 0.87, max_items: int = 12,
                 semantic: bool = True,
                 semantic_similarity: float = 0.90) -> None:
        self.window_s = window_s
        self.similarity = similarity
        self.semantic = semantic
        self.semantic_similarity = semantic_similarity
        # (normalized_text, ts, vector-or-None)
        self._recent: deque[tuple[str, float, object]] = deque(
            maxlen=max_items)

    def _prune(self, now: float) -> None:
        while self._recent and (now - self._recent[0][1]) > self.window_s:
            self._recent.popleft()

    def is_duplicate(self, question: str, *, now: float | None = None) -> bool:
        """True when a near-identical question was answered inside the
        window. Does NOT record — call [note_answered] when actually
        answering, so skipped/failed attempts never poison the window."""
        try:
            now = now if now is not None else time.monotonic()
            self._prune(now)
            q = normalize(question)
            if len(q) < 8:            # too short to judge — never suppress
                return False
            v = (_embed_one(q) if self.semantic and self._recent
                 and _embed_ready() else None)
            for prev, _ts, pvec in self._recent:
                if q == prev:
                    return True
                # A strict SUPERSET extends the earlier fragment ("how would
                # you scale kafka" ⊃ "how would you") — that's the continued
                # question, not a repeat: the caller's merge path handles it.
                if q.startswith(prev) or prev.startswith(q):
                    continue
                if SequenceMatcher(None, q, prev).ratio() >= self.similarity:
                    return True
                if v is not None and pvec is not None:
                    try:
                        if float(v @ pvec) >= self.semantic_similarity:
                            return True
                    except Exception:  # noqa: BLE001
                        pass
            return False
        except Exception:  # noqa: BLE001 — fail-open: never block an answer
            return False

    def note_answered(self, question: str, *, now: float | None = None) -> None:
        try:
            now = now if now is not None else time.monotonic()
            self._prune(now)
            q = normalize(question)
            if q:
                v = (_embed_one(q) if self.semantic and _embed_ready()
                     else None)
                self._recent.append((q, now, v))
        except Exception:  # noqa: BLE001
            pass


__all__ = ["QuestionDeduper", "normalize"]
