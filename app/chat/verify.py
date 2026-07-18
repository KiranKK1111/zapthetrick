"""Self-refine for demanding turns: draft → verify → revise.

For expert (configurable) turns we (1) draft a full answer on the strong model,
(2) have a DIFFERENT model critically check it (cross-model verification —
catches the drafter's own blind spots), and (3) revise if the check found real
problems — then the route streams the verified result.

Reliability:
- the verdict is STRUCTURED JSON ({correct, problems}), not string-matching;
- every LLM step has a TIMEOUT; if verify/revise is slow or fails we return the
  DRAFT we already have (never wasted), and only fall back to plain streaming
  when even the draft didn't materialise;
- a revision only replaces the draft if it's substantive.
"""
from __future__ import annotations

import asyncio
import json
import logging

log = logging.getLogger(__name__)

def _timeouts() -> tuple[float, float]:
    """(draft, step) ceilings from cfg, with safe fallbacks."""
    try:
        from app.core.config_loader import cfg
        return (float(cfg.advanced_rag.verify_draft_timeout),
                float(cfg.advanced_rag.verify_step_timeout))
    except Exception:  # noqa: BLE001
        return (90.0, 45.0)

_VERIFY = (
    "You are rigorously reviewing a draft answer for CORRECTNESS against the "
    "user's request: factual/logical errors, wrong calculations, broken or buggy "
    "code, missed constraints or edge cases, unsupported claims. Be a tough but "
    "fair reviewer — do NOT invent problems that aren't there, and do NOT rewrite "
    "the answer.\n\n"
    "Reply with ONLY compact JSON:\n"
    '{"correct": true|false, "problems": ["<specific problem>", ...]}\n'
    "Set correct=true with an empty list when the draft is fully correct and "
    "complete.\n\n"
    "User request:\n{q}\n\nDraft answer:\n{a}"
)

_REVISE = (
    "Produce the FINAL answer to the user's request, fixing every problem listed "
    "below. Keep what was already correct; make it complete, correct, and "
    "elegant. Output ONLY the final answer — no preamble, no mention of this "
    "review.\n\n"
    "User request:\n{q}\n\nDraft answer:\n{a}\n\nProblems to fix:\n{c}"
)


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _parse_verdict(raw: str) -> tuple[bool, list[str]]:
    """(correct, problems) from the reviewer's JSON. On unparseable output we
    default to correct=True — we won't revise (and risk degrading) a good draft
    just because the reviewer's formatting slipped."""
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        return True, []
    if not isinstance(obj, dict):
        return True, []
    problems = [str(p).strip() for p in (obj.get("problems") or []) if str(p).strip()]
    correct = bool(obj.get("correct", True)) or not problems
    return correct, problems


def _applies(difficulty: str) -> bool:
    try:
        from app.core.config_loader import cfg
        return (cfg.advanced_rag.self_refine
                and difficulty in (cfg.advanced_rag.verify_levels or []))
    except Exception:  # noqa: BLE001
        return False


def _rounds_for(difficulty: str) -> int:
    """How many verify→revise rounds to run. Expert keeps looping (config
    `refine_rounds`); hard does a single pass."""
    if difficulty != "expert":
        return 1
    try:
        from app.core.config_loader import cfg
        return max(1, int(getattr(cfg.advanced_rag, "refine_rounds", 2)))
    except Exception:  # noqa: BLE001
        return 2


async def verified_answer(messages: list[dict], *, difficulty: str) -> str | None:
    """Draft → (cross-model) verify → revise. Returns the final text, or None
    when self-refine doesn't apply or no draft could be produced (caller then
    streams normally)."""
    if not _applies(difficulty):
        return None
    from app.core.llm_client import LLMError, llm
    from app.core.prompt import fill
    from app.core.config_loader import cfg

    draft_timeout, step_timeout = _timeouts()
    opts = {"difficulty": difficulty}

    # 1) Draft on the strong model. If even this fails/times out, fall back to
    #    normal streaming (return None) rather than risk a dead turn.
    try:
        draft, draft_model = await asyncio.wait_for(
            llm.complete_routed(messages, options=opts), timeout=draft_timeout)
    except (asyncio.TimeoutError, LLMError, Exception) as exc:  # noqa: BLE001
        log.info("self-refine draft failed (%s); streaming normally", exc)
        return None
    draft = (draft or "").strip()
    if not draft:
        return None

    # 2+3) Iteratively verify on a DIFFERENT model and revise — up to N rounds
    #      (hard = 1, expert = cfg.refine_rounds). Each round re-checks the
    #      LATEST draft on another model (avoid_model_db_id) and only keeps a
    #      substantive revision, so a hard problem keeps getting re-examined and
    #      improved across multiple models instead of a single shot.
    q = _last_user(messages)
    last_model = draft_model
    for _ in range(_rounds_for(difficulty)):
        vopts = dict(opts)
        if last_model:
            vopts["avoid_model_db_id"] = last_model
        try:
            # A slightly warmer verify gives an independent perspective even when
            # only one model is routable (cross-model `avoid` handles the ≥2 case).
            critique, crit_model = await asyncio.wait_for(
                llm.complete_routed(
                    [{"role": "user",
                      "content": fill(_VERIFY, q=q[:8000], a=draft[:12000])}],
                    options={**vopts, "temperature": cfg.temperature.verify}),
                timeout=step_timeout)
            correct, problems = _parse_verdict(critique)
            if correct:
                return draft
            revised, rev_model = await asyncio.wait_for(
                llm.complete_routed(
                    [{"role": "user", "content": fill(
                        _REVISE, q=q[:8000], a=draft[:12000],
                        c="\n".join(f"- {p}" for p in problems)[:4000])}],
                    options=vopts),
                timeout=step_timeout)
            revised = (revised or "").strip()
            # Only accept a substantive revision (guard against a truncated/empty
            # one replacing a complete draft); otherwise stop with what we have.
            if not (revised and len(revised) >= 0.5 * len(draft)):
                return draft
            draft = revised
            last_model = rev_model or crit_model or last_model
        except (asyncio.TimeoutError, LLMError, Exception) as exc:  # noqa: BLE001
            log.info("self-refine round failed (%s); using current draft", exc)
            return draft
    return draft


def assess_partial_sections(text: str, *, expect_refusal_ok: bool = False):
    """Per-SECTION incremental verification while streaming (roadmap Phase 6 #9).

    `quality.stream_controller.assess_partial` scores a partial answer as one
    blob; a long multi-section answer can hide a bad section inside a good
    average. This splits the in-progress text into logical blocks
    (`response_arch.blocks`) and assesses each independently, returning:

        {"action", "score", "sections": [{"index","type","action","score",
         "reasons"}, ...]}

    ``action``/``score`` are the WORST section's — one degenerate/refusal
    section flags the whole turn even when the rest looks fine. Fail-open: any
    error degrades to a single whole-text assessment.
    """
    from app.quality.stream_controller import (
        CONTINUE, FLAG, REGENERATE, assess_partial)

    _rank = {CONTINUE: 0, FLAG: 1, REGENERATE: 2}
    try:
        from app.response_arch.blocks import BlockAssembler
        asm = BlockAssembler()
        blocks = asm.feed(text or "")
        blocks += asm.flush()
        chunks = [b.text for b in blocks if b.text.strip()]
    except Exception:  # noqa: BLE001
        chunks = []
    if not chunks:
        v = assess_partial(text or "", expect_refusal_ok=expect_refusal_ok)
        return {"action": v.action, "score": v.score,
                "sections": [{"index": 0, "type": "whole", "action": v.action,
                              "score": v.score, "reasons": v.reasons}]}

    sections = []
    worst = None
    for i, (chunk, blk) in enumerate(zip(chunks, blocks)):
        # A code block legitimately trips the error/keyword heuristics — never
        # flag it for "error"/"exception" words that are just source tokens.
        if getattr(blk, "type", "") == "code":
            sections.append({"index": i, "type": "code", "action": CONTINUE,
                             "score": 1.0, "reasons": []})
            continue
        v = assess_partial(chunk, expect_refusal_ok=expect_refusal_ok)
        sections.append({"index": i, "type": blk.type, "action": v.action,
                         "score": v.score, "reasons": v.reasons})
        if worst is None or _rank[v.action] > _rank[worst.action] or (
                _rank[v.action] == _rank[worst.action] and v.score < worst.score):
            worst = v
    if worst is None:
        return {"action": CONTINUE, "score": 1.0, "sections": sections}
    return {"action": worst.action, "score": worst.score, "sections": sections}


def chunk_text(text: str, size: int = 160):
    """Yield `text` in modest pieces so a pre-computed answer still streams to
    the UI smoothly (prefers to break on a newline near the boundary)."""
    i, n = 0, len(text)
    while i < n:
        end = min(i + size, n)
        nl = text.rfind("\n", i, end)
        if nl > i + size // 2:
            end = nl + 1
        yield text[i:end]
        i = end


__all__ = ["verified_answer", "chunk_text", "assess_partial_sections"]
