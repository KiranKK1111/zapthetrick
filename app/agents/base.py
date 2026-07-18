"""Common interface every agent on the mesh implements.

Agents declare *what they read from the blackboard* and *what they
write* — the [PriorityScheduler] uses those sets to figure out when an
agent is "ready" (all reads present) and which agents are blocking each
other (so it can boost a producer's priority).

An agent's actual model call lives in `.run(board)`. Don't subclass and
add I/O elsewhere — the blackboard is the only side-effecting surface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from ..blackboard.board import Blackboard


class Agent(ABC):
    """Base class for every agent on the multi-agent mesh."""

    # Subclasses override these. `name` is used for logging + UI chips.
    name: str = "agent"

    # Priority lane: 0=P0 realtime, 1=P1 improvement, 2=P2 background.
    priority: int = 1

    # Used by the scheduler to compute a priority score.
    expected_latency_ms: int = 500

    # Slots this agent depends on (must exist on the board before `run`).
    reads: frozenset[str] = frozenset()

    # Slots this agent writes to.
    writes: frozenset[str] = frozenset()

    @abstractmethod
    async def run(self, board: "Blackboard") -> None:
        """Do this agent's work and publish any results to `board`."""

    # Streaming agents (Persona, Coder) override this to yield tokens.
    async def stream(self, board: "Blackboard") -> AsyncIterator[str]:
        """Optional: yield tokens for the user-facing response.

        Non-streaming agents leave this as the default no-op generator.
        """
        if False:  # pragma: no cover — marker so this stays an async generator
            yield ""


class AgentRegistry:
    """Holds the active set of agents for one session.

    Construct from the `agents.enabled` block in config.yaml; the
    supervisor reads from here when picking who to schedule.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def all(self) -> list[Agent]:
        return list(self._agents.values())

    def by_priority(self, lane: int) -> list[Agent]:
        return [a for a in self._agents.values() if a.priority == lane]
