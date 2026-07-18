"""Self-improvement on hard turns (P2-11, report_2 §P2-11).

A free-model quality squeeze: instead of one draft from one model, generate N
candidates from DIFFERENT free models and pick the best by combining two signals

  1. self-consistency — if a majority of the models converge on the same answer
     (normalized), trust that (cheap, no judge),
  2. an LLM judge (`council.best_of_n`) — otherwise have a judge pick the best.

`generation_council` returns the chosen text + how it was chosen. It's N× the
cost of a single call, so it's budget-gated (expert turns only, opt-in) by the
caller. `self_consistency` exposes signal (1) on its own for critical yes/no
determinations. `reflect` turns a finished run into a one-line lesson for the
project brain.

Provider-agnostic, never raises (degrades to a single completion), so it can't
break a run.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass

from app.core.llm_client import LLMError, llm

log = logging.getLogger(__name__)


@dataclass
class CouncilResult:
    text: str = ""
    n: int = 0                 # candidates actually generated
    method: str = "single"     # single | consistency | judge
    agreement: float = 0.0     # top-cluster fraction (self-consistency)
    why: str = ""


def _normalize(text: str) -> str:
    """A canonical form for clustering: parse JSON actions to a sorted dump so
    formatting noise doesn't split identical answers; else collapsed text."""
    s = (text or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(s[i:j + 1])
            return json.dumps(obj, sort_keys=True, separators=(",", ":"))
        except Exception:  # noqa: BLE001
            pass
    return re.sub(r"\s+", " ", s.lower())


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
    return ""


async def _generate(messages: list[dict], n: int, options: dict) -> list[str]:
    """N candidates, each forced onto a different model (rotate avoid_model)."""
    cands: list[str] = []
    avoid: int | None = None
    base = dict(options or {})
    base.setdefault("temperature", 0.4)   # a little diversity across drafts
    for _ in range(max(1, n)):
        opts = dict(base)
        if avoid:
            opts["avoid_model_db_id"] = avoid
        try:
            text, mid = await llm.complete_routed(
                messages, options=opts)
        except (LLMError, Exception) as exc:  # noqa: BLE001
            log.info("generation council draft failed: %s", exc)
            continue
        if (text or "").strip():
            cands.append(text)
        if mid:
            avoid = mid
    return cands


def _consistency_pick(candidates: list[str]) -> tuple[int, float]:
    """(index of a representative of the largest cluster, agreement fraction)."""
    norms = [_normalize(c) for c in candidates]
    counts = Counter(norms)
    top_norm, top_n = counts.most_common(1)[0]
    agreement = top_n / len(candidates)
    idx = norms.index(top_norm)
    return idx, agreement


async def generation_council(
    messages: list[dict],
    *,
    n: int = 3,
    options: dict | None = None,
    consistency_threshold: float = 0.5,
) -> CouncilResult:
    """Best-of-N across different models for one (expert) completion.

    Generates N drafts; if a majority converge (self-consistency ≥ threshold and
    ≥2 agree) returns that, else an LLM judge picks. Falls back to a single
    completion when fewer than 2 drafts come back."""
    opts = dict(options or {})
    cands = await _generate(messages, n, opts)
    if not cands:
        # total failure → one plain completion (no diversity temp)
        text = await llm.complete(messages, options=options)
        return CouncilResult(text=text, n=0, method="single")
    if len(cands) == 1:
        return CouncilResult(text=cands[0], n=1, method="single")

    idx, agreement = _consistency_pick(cands)
    top_n = round(agreement * len(cands))
    if top_n >= 2 and agreement >= consistency_threshold:
        return CouncilResult(text=cands[idx], n=len(cands),
                             method="consistency", agreement=agreement,
                             why=f"{top_n}/{len(cands)} models agreed")

    from app.chat.council import best_of_n
    best_idx, why = await best_of_n(_last_user(messages), cands)
    return CouncilResult(text=cands[best_idx], n=len(cands), method="judge",
                         agreement=agreement, why=why or "judge selected")


async def self_consistency(
    prompt: str,
    *,
    n: int = 3,
    options: dict | None = None,
) -> tuple[str, float]:
    """Run a single prompt N times, return (majority answer, agreement).

    For critical short determinations (a verdict, a small edit) where the most
    self-consistent answer is the most trustworthy."""
    msgs = [{"role": "user", "content": prompt}]
    cands = await _generate(msgs, n, options or {})
    if not cands:
        return "", 0.0
    idx, agreement = _consistency_pick(cands)
    return cands[idx], agreement


def reflect(task: str, *, success: bool, verify_ok: bool | None = None,
            rounds: int = 1, issues: list[str] | None = None) -> str:
    """A one-line lesson from a finished run, for the project brain."""
    bits = ["succeeded" if success else "did not fully succeed"]
    if verify_ok is True:
        bits.append("build/tests passed")
    elif verify_ok is False:
        bits.append("build/tests failing")
    if rounds and rounds > 1:
        bits.append(f"{rounds} repair round(s)")
    note = f"On \"{(task or '').strip()[:70]}\": " + ", ".join(bits) + "."
    real_issues = [i for i in (issues or []) if str(i).strip()]
    if real_issues:
        note += " Watch: " + "; ".join(str(i) for i in real_issues[:3])
    return note


__all__ = ["CouncilResult", "generation_council", "self_consistency", "reflect"]
