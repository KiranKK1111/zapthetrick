"""Coder — solves coding / algorithm problems with full DSA structure.

Activated when [PlannerAgent] classifies intent as `coding`. Delegates
the heavy lifting to `app.dsa.pipeline.solve()`, which runs the
extract → classify → generate → verify → repair → beautify chain from
Architecture2.md §4.

`run` writes structured stage data (pattern, complexity, verify
counts) into the blackboard `coder_*` slots so other agents
(Critic, Suggester) can read them. `stream` yields the streamed
markdown to the persona/UI.
"""
from __future__ import annotations

from typing import AsyncIterator

from ..blackboard.board import Blackboard
from ..blackboard.schema import KEY_EVIDENCE, KEY_INTENT, KEY_QUESTION
from ..blackboard.scheduler import P0
from .base import Agent


# Blackboard slots the Coder publishes. Other agents pick these up.
KEY_CODER_PATTERN = "coder_pattern"
KEY_CODER_COMPLEXITY = "coder_complexity"
KEY_CODER_VERIFY = "coder_verify"
KEY_CODER_VIZ = "coder_viz"
KEY_CODER_MARKDOWN = "coder_markdown"


class CoderAgent(Agent):
    name = "coder"
    priority = P0
    expected_latency_ms = 3_500
    reads = frozenset({KEY_QUESTION, KEY_INTENT, KEY_EVIDENCE})
    writes = frozenset(
        {
            "drafts_current",
            KEY_CODER_PATTERN,
            KEY_CODER_COMPLEXITY,
            KEY_CODER_VERIFY,
            KEY_CODER_VIZ,
            KEY_CODER_MARKDOWN,
        }
    )

    async def run(self, board: Blackboard) -> None:
        """Non-streaming entry: writes the final markdown + structured
        slots to the blackboard. Used when a downstream agent needs the
        DSA output to reason over (Critic, Suggester) without re-running."""
        if board.has(KEY_CODER_MARKDOWN):
            return
        question = board.get(KEY_QUESTION) or ""
        if not question.strip():
            return

        final_md = ""
        async for evt in _solve_events(question):
            self._record_event(board, evt)
            if evt.get("kind") == "markdown":
                final_md = str(evt.get("text") or "")
        if final_md:
            board.write("drafts_current", final_md, agent=self.name)

    async def stream(self, board: Blackboard) -> AsyncIterator[str]:
        """Stream the beautified markdown chunk-by-chunk.

        The DSA pipeline produces the full markdown in one beautifier
        pass at the end; we chunk it into ~120-char windows so the UI
        renders progressively instead of all-at-once. Stage events
        keep landing on the blackboard so the tool-chip rail still
        animates as work progresses.
        """
        question = board.get(KEY_QUESTION) or ""
        if not question.strip():
            yield "(no question to solve)"
            return

        final_md = ""
        async for evt in _solve_events(question):
            self._record_event(board, evt)
            kind = evt.get("kind")
            if kind == "markdown":
                final_md = str(evt.get("text") or "")
            elif kind == "verify":
                # Emit a one-liner so the user sees verification
                # progress before the full markdown lands.
                passed = evt.get("passed", 0)
                failed = evt.get("failed", 0)
                if passed or failed:
                    yield f"\n_verify: {passed} passed, {failed} failed_\n"

        if not final_md:
            yield "(coder pipeline produced no answer — check logs)"
            return

        # Chunk for progressive rendering. The persona path streams
        # tokens; we mimic that with paragraph-sized blocks so markdown
        # parsers don't choke on half-rendered fenced blocks.
        for block in final_md.split("\n\n"):
            yield block + "\n\n"
        board.write("drafts_current", final_md, agent=self.name)

    # ---- helpers ------------------------------------------------------
    def _record_event(self, board: Blackboard, evt: dict) -> None:
        """Project pipeline events into the matching blackboard slots."""
        kind = evt.get("kind")
        if kind == "stage":
            name = evt.get("name")
            data = evt.get("data") or {}
            if name == "classifier":
                board.write(KEY_CODER_PATTERN, data, agent=self.name)
            elif name == "complexity":
                board.write(KEY_CODER_COMPLEXITY, data, agent=self.name)
        elif kind == "verify":
            board.write(
                KEY_CODER_VERIFY,
                {
                    "passed": evt.get("passed", 0),
                    "failed": evt.get("failed", 0),
                    "errors": evt.get("errors", []),
                },
                agent=self.name,
            )
        elif kind == "viz":
            board.write(
                KEY_CODER_VIZ,
                {
                    "viz_type": evt.get("viz_type"),
                    "frames": evt.get("frames"),
                    "legend": evt.get("legend"),
                },
                agent=self.name,
            )
        elif kind == "markdown":
            board.write(KEY_CODER_MARKDOWN, evt.get("text") or "", agent=self.name)


# ---- pipeline routing ----------------------------------------------------
async def _solve_events(question: str):
    """Pick the right pipeline and forward its events.

    Routing rule:
      - Run the cheap technical-pipeline domain classifier.
      - If domain is 'generic' (no specialised tech-domain matched),
        we assume this is a DSA / coding problem and send it through
        the DSA pipeline.
      - Otherwise (system_design / databases / devops / cloud /
        frontend / …) hand off to the domain-specific pipeline.

    Either way the event stream conforms to the shape the route layer
    expects: stage / markdown / verify / viz / done.
    """
    from app.core.config_loader import cfg as _cfg

    forced = _cfg.technical_pipeline.force_domain if _cfg.technical_pipeline.enabled else None

    if forced and forced != "generic":
        from app.technical_pipeline import dispatch as _dispatch

        async for evt in _dispatch(question, hint=forced):
            yield evt
        return

    # Auto-classify.
    from app.technical_pipeline.dispatcher import classify_domain

    domain = forced or classify_domain(question)
    if domain == "generic":
        from app import dsa as _dsa

        async for evt in _dsa.solve(question):
            yield evt
    else:
        from app.technical_pipeline import dispatch as _dispatch

        async for evt in _dispatch(question, hint=domain):
            yield evt
