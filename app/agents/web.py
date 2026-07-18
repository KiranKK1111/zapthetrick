"""Web — optional external lookup when the user enables it.

Disabled by default in config.yaml. The Suggester / Critic don't need
this; it's only fired when Planner's intent says we need fresh facts.

TODO: wire to [config.yaml].web_search provider (DuckDuckGo today).
"""
from __future__ import annotations

from ..blackboard.board import Blackboard
from ..blackboard.scheduler import P1
from .base import Agent


class WebAgent(Agent):
    name = "web"
    priority = P1
    expected_latency_ms = 1_000
    reads = frozenset({"question"})
    writes = frozenset({"web_hits"})

    async def run(self, board: Blackboard) -> None:
        # TODO: ddgs.text(...) once enabled.
        board.write("web_hits", [], agent=self.name)
