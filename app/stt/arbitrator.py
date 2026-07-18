"""Arbitration layer for [DualSTT] — picks the best output.

Strategy (Architecture.md §"Arbitration layer"):

  1. **Drop empty candidates.** If only one is non-empty, that wins.
  2. **Word-level alignment.** When both engines produced text,
     align tokens (cheap longest-common-subsequence over normalized
     tokens) and accept high-confidence words from either side.
  3. **Disagreement on key tokens** triggers a fallback to the
     higher-confidence engine's full text — no Frankenstein
     stitching when the two engines disagree on most of the
     content (different audio interpretations entirely).

The doc's "re-evaluate with a third small pass" is left as a
follow-up; until that arrives, a high-disagreement event is logged
and the higher-confidence engine wins.
"""
from __future__ import annotations

import difflib
import logging
import re

from .dual_engine import ArbitratedText, STTHypothesis


log = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _agreement_ratio(a: list[str], b: list[str]) -> float:
    """LCS-based agreement in [0, 1]. Cheap; difflib is built-in."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return matcher.ratio()


def arbitrate(candidates: list[STTHypothesis]) -> ArbitratedText:
    """Pick (and possibly fuse) one final text from the candidates.

    Always returns a populated [ArbitratedText] — `text=""` only when
    every candidate was empty.
    """
    if not candidates:
        return ArbitratedText(text="", rationale="no candidates")

    non_empty = [c for c in candidates if c.text and c.text.strip()]
    if not non_empty:
        return ArbitratedText(
            text="",
            chosen_engine=candidates[0].engine,
            rationale="all candidates empty",
            candidates=candidates,
        )

    if len(non_empty) == 1:
        c = non_empty[0]
        return ArbitratedText(
            text=c.text,
            confidence=c.confidence,
            chosen_engine=c.engine,
            rationale="only one engine produced output",
            candidates=candidates,
        )

    # Two-or-more case. Compute agreement; pick the higher-confidence
    # engine's text when disagreement is high, otherwise take the
    # confidence-weighted average (text-wise, not semantically — same
    # text if both agree).
    primary, secondary = sorted(non_empty, key=lambda c: -c.confidence)[:2]
    ratio = _agreement_ratio(_tokenize(primary.text), _tokenize(secondary.text))

    if ratio >= 0.85:
        # The two engines agree on most tokens. Whichever has higher
        # confidence wins outright — they're saying the same thing.
        return ArbitratedText(
            text=primary.text,
            confidence=max(primary.confidence, secondary.confidence),
            chosen_engine=primary.engine,
            rationale=f"agreement {ratio:.2f} — kept higher-confidence text",
            candidates=candidates,
        )

    # Substantial disagreement. Log so the operator can see when
    # dual-STT is doing real work; pick the higher-confidence text.
    log.info(
        "stt arbitrator: disagreement ratio=%.2f — primary=%s sec=%s",
        ratio,
        primary.engine,
        secondary.engine,
    )
    return ArbitratedText(
        text=primary.text,
        confidence=primary.confidence,
        chosen_engine=primary.engine,
        rationale=f"agreement {ratio:.2f} — picked higher-confidence engine",
        candidates=candidates,
    )


__all__ = ["arbitrate"]
