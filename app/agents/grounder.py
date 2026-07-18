"""Grounder — verifies claims in the draft against the evidence.

P0 — runs in parallel with the Persona stream. Architecture.md §5: when it flags
an unverified claim, the supervisor surfaces it and Persona is responsible for
any revision.

LLM-driven (no keyword/stopword/overlap heuristics): one fast JSON call reads
the evidence + draft and returns the factual claims in the draft that the
evidence does NOT support. It runs concurrently with the stream, so the verdict
simply appears when ready. Any failure (or no evidence) yields empty grounding —
we never flag a claim we couldn't actually check.
"""
from __future__ import annotations

import json
import logging

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

from ..blackboard.board import Blackboard
from ..blackboard.schema import KEY_DRAFTS, KEY_EVIDENCE, KEY_GROUNDING, Grounding
from ..blackboard.scheduler import P0
from .base import Agent

log = logging.getLogger(__name__)

# Bounds on what we feed the verifier (the evidence can be many chunks).
_MAX_EVIDENCE_CHARS = 8000
_MAX_DRAFT_CHARS = 4000

_PROMPT = (
    "You fact-check a draft answer against the EVIDENCE below. Identify factual "
    "claims in the DRAFT that are NOT supported by the evidence (possible "
    "hallucinations). Ignore opinions, hedges, generic filler, questions, and "
    "the assistant's own reasoning — only flag concrete, checkable statements "
    "(names, numbers, facts, relationships) that the evidence does not back up. "
    "If everything checkable is supported, or there are no checkable claims, "
    "return an empty list.\n\n"
    "Reply with ONLY compact JSON and nothing else:\n"
    "{\"unverified\": [\"<verbatim claim sentence>\", ...]}\n\n"
    "EVIDENCE:\n{evidence}\n\nDRAFT:\n{draft}\n"
)


def _parse_unverified(raw: str) -> list[str]:
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        return []
    items = obj.get("unverified") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if str(x).strip()]


class GrounderAgent(Agent):
    name = "grounder"
    priority = P0
    expected_latency_ms = 600
    reads = frozenset({KEY_DRAFTS, KEY_EVIDENCE})
    writes = frozenset({KEY_GROUNDING})

    async def run(self, board: Blackboard) -> None:
        draft = (board.get("drafts_current") or "").strip()
        evidence = board.get(KEY_EVIDENCE)
        chunks = list(getattr(evidence, "chunks", None) or [])

        # Nothing to verify, or nothing to verify against → empty grounding
        # (don't flag claims we can't actually check).
        if not draft or not chunks:
            board.write(KEY_GROUNDING, Grounding(), agent=self.name)
            return

        from app.core.prompt import fill
        evidence_text = "\n\n".join(c.text for c in chunks)[:_MAX_EVIDENCE_CHARS]
        prompt = fill(_PROMPT, evidence=evidence_text, draft=draft[:_MAX_DRAFT_CHARS])
        try:
            raw = await llm.complete(
                [{"role": "user", "content": prompt}],
                model=(cfg.llm.classifier_model or cfg.llm.model),
                options={"temperature": cfg.temperature.classifier,
                         "num_predict": cfg.output_tokens.verdict},
            )
            unverified = _parse_unverified(raw)
        except (LLMError, Exception) as exc:  # noqa: BLE001 — never block on this
            log.info("grounding check failed (treating as grounded): %s", exc)
            unverified = []

        board.write(
            KEY_GROUNDING, Grounding(unverified=unverified), agent=self.name
        )
