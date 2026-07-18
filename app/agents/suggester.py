"""Suggester — proactive next-move recommendations from recent turns.

P1 — surfaces as dismissible chips in the right rail
([widgets/suggestions_rail.dart]).

LLM-driven (no keyword/stopword/template rules): it shows the model a compact
view of the user's recent turns (questions + any 👍/👎) and asks for up to three
brief, specific next-step suggestions — or none. Runs after the persona has
started (declares a read of the draft slot) so the chips arrive with the answer;
any failure or thin history yields no suggestions.
"""
from __future__ import annotations

import json
import logging

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

from ..blackboard.board import Blackboard
from ..blackboard.schema import KEY_DRAFTS, KEY_SUGGESTIONS, Suggestions
from ..blackboard.scheduler import P1
from .base import Agent

log = logging.getLogger(__name__)

_RECENT_WINDOW = 8

_PROMPT = (
    "You are a concise coach watching a user's recent assistant turns. Propose "
    "AT MOST 3 short, specific, actionable next-step suggestions tailored to "
    "what they've been doing — each one a single sentence the user could click "
    "to act on. Only suggest something genuinely useful; if nothing clearly "
    "helps, return an empty list. Do not restate their questions or invent "
    "facts.\n\n"
    "Reply with ONLY compact JSON and nothing else:\n"
    "{\"suggestions\": [\"...\", \"...\"]}\n\n"
    "Recent turns (oldest first; [down]/[up] = the user's feedback):\n{turns}\n"
)


def _parse_suggestions(raw: str) -> list[str]:
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        return []
    items = obj.get("suggestions") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if str(x).strip()][:3]


class SuggesterAgent(Agent):
    name = "suggester"
    priority = P1
    expected_latency_ms = 600
    # We don't actually need the draft, but declaring a read of it ensures the
    # scheduler runs us *after* the persona has at least started — so our chips
    # reach the UI alongside the answer.
    reads = frozenset({KEY_DRAFTS})
    writes = frozenset({KEY_SUGGESTIONS})

    async def run(self, board: Blackboard) -> None:
        extras = board.get("extras", {}) or {}
        db_session = extras.get("db_session")
        session_id = extras.get("session_id")

        if db_session is None or not session_id:
            board.write(KEY_SUGGESTIONS, Suggestions(), agent=self.name)
            return

        from ..memory.episodic import recent_episodes

        try:
            episodes = await recent_episodes(
                db_session, session_id=session_id, limit=_RECENT_WINDOW
            )
        except Exception:  # noqa: BLE001
            board.write(KEY_SUGGESTIONS, Suggestions(), agent=self.name)
            return

        # Need a little history before a suggestion is meaningful.
        if len(episodes) < 2:
            board.write(KEY_SUGGESTIONS, Suggestions(), agent=self.name)
            return

        lines = []
        for ep in episodes:
            q = (ep.question or "").strip().replace("\n", " ")[:200]
            if not q:
                continue
            fb = {"down": " [down]", "up": " [up]"}.get(
                getattr(ep, "feedback", None) or "", ""
            )
            lines.append(f"- {q}{fb}")
        if not lines:
            board.write(KEY_SUGGESTIONS, Suggestions(), agent=self.name)
            return

        try:
            from app.core.prompt import fill
            raw = await llm.complete(
                [{"role": "user",
                  "content": fill(_PROMPT, turns="\n".join(lines))}],
                model=(cfg.llm.classifier_model or cfg.llm.model),
                options={"temperature": cfg.temperature.creative,
                         "num_predict": cfg.output_tokens.short_json},
            )
            proactive = _parse_suggestions(raw)
        except (LLMError, Exception) as exc:  # noqa: BLE001 — non-critical
            log.info("suggester LLM failed (no chips): %s", exc)
            proactive = []

        board.write(
            KEY_SUGGESTIONS, Suggestions(proactive=proactive), agent=self.name
        )
