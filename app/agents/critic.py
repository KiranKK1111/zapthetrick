"""Critic — heuristic 4-axis review of the draft.

P1 — runs alongside the response stream. Today this is heuristic, not
an LLM judge:

  - **Length**: too short to be substantive, or too long to read.
  - **AI-tells**: phrases the user explicitly doesn't want to hear from
    a candidate ("As an AI", "I cannot", "I don't have access to…").
  - **First-person consistency**: behavioral answers should use first
    person; "the candidate" / "this person" leaks the AI's POV.
  - **Structure**: behavioral intent expects STAR; coding expects an
    "approach → code → complexity" shape — we flag missing sections.

Issues are surfaced on the blackboard as plain strings the supervisor
can use mid-flight (Architecture.md §5: "incorporate Critic's notes
mid-flight if confident, time_left > 1s").

TODO: gate a small LLM-judge call behind `cfg.learning.online_critique`
to pick up the issues this heuristic misses (factual coherence, tone
mismatch, repetitive phrasing).
"""
from __future__ import annotations

import re

from ..blackboard.board import Blackboard
from ..blackboard.schema import KEY_CRITIQUES, KEY_DRAFTS, KEY_INTENT, Critiques
from ..blackboard.scheduler import P1
from .base import Agent


_MIN_REASONABLE_LEN = 60
_MAX_REASONABLE_LEN = 4500

_AI_TELLS = (
    "as an ai",
    "as a language model",
    "i cannot",
    "i'm unable to",
    "i don't have access",
    "i do not have access",
    "as a chatbot",
    "i am an ai",
)

_THIRD_PERSON_LEAKS = (
    "the candidate",
    "this person",
    "the user",
    "the interviewee",
)


class CriticAgent(Agent):
    name = "critic"
    priority = P1
    expected_latency_ms = 200
    reads = frozenset({KEY_DRAFTS, KEY_INTENT})
    writes = frozenset({KEY_CRITIQUES})

    async def run(self, board: Blackboard) -> None:
        draft = (board.get("drafts_current") or "").strip()
        intent = board.get(KEY_INTENT)
        intent_type = getattr(intent, "type", "general")

        issues: list[str] = []
        suggestions: list[str] = []

        if not draft:
            issues.append("Draft is empty.")
            board.write(
                KEY_CRITIQUES,
                Critiques(issues=issues, suggestions=suggestions),
                agent=self.name,
            )
            return

        lower = draft.lower()

        # ---- length ---------------------------------------------------
        if len(draft) < _MIN_REASONABLE_LEN:
            issues.append("Answer is too short to be useful.")
            suggestions.append("Expand with a concrete example or one detail.")
        elif len(draft) > _MAX_REASONABLE_LEN:
            issues.append("Answer is long — consider trimming.")
            suggestions.append("Cut filler; lead with the most important point.")

        # ---- AI-tells -------------------------------------------------
        for phrase in _AI_TELLS:
            if phrase in lower:
                issues.append(f"AI-tell detected: '{phrase}'.")
                suggestions.append(
                    "Rephrase from the candidate's first-person voice."
                )
                break

        # ---- first-person consistency for behavioral --------------------
        if intent_type == "behavioral":
            for leak in _THIRD_PERSON_LEAKS:
                if leak in lower:
                    issues.append(
                        f"Third-person leak ('{leak}') in a behavioral answer."
                    )
                    suggestions.append("Switch to first person ('I', 'we', 'my team').")
                    break

        # ---- structure ------------------------------------------------
        if intent_type == "behavioral":
            star_markers = ("situation", "task", "action", "result")
            present = sum(1 for m in star_markers if m in lower)
            if present < 2:
                suggestions.append(
                    "Behavioral answers land harder when they follow STAR "
                    "(Situation → Task → Action → Result)."
                )
        elif intent_type == "coding":
            if "complexity" not in lower and "o(" not in lower:
                suggestions.append(
                    "Add the time + space complexity — interviewers expect it."
                )
            if "```" not in draft:
                suggestions.append(
                    "Wrap the code in a fenced block so the UI highlights it."
                )

        board.write(
            KEY_CRITIQUES,
            Critiques(issues=issues, suggestions=suggestions),
            agent=self.name,
        )
