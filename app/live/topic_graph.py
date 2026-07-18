"""
Per-session topic graph + drift detection
(live-conversational-intelligence R5).

Widens the existing per-session `question_detection.context_tracker` with a tree
of topics and sub-topics (Microservices -> Kafka -> Consumer Groups -> Offsets)
so follow-ups attach to the right node and "go back to partitions" resolves
against the graph rather than only the last utterance. Stored ON the tracker
object (no new registry, no DB). Deterministic + fail-open: disabled or on error
the live path uses today's linear recent-questions context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import time

# Below this topic-similarity the new topic is treated as a drift (a switch),
# not a sub-topic of the current one.
DRIFT_THRESHOLD = 0.35


@dataclass
class TopicNode:
    name: str
    parent: str | None = None
    turns: list[int] = field(default_factory=list)
    last_seen: float = field(default_factory=time)


class TopicGraph:
    """A tree of topics for one live session. `current()` is the active node;
    `previous()` the one before a drift."""

    def __init__(self) -> None:
        self._nodes: dict[str, TopicNode] = {}
        self._current: str | None = None
        self._previous: str | None = None

    @staticmethod
    def _norm(topic: str | None) -> str:
        return (topic or "").strip().lower()

    def add_topic(self, topic: str, parent: str | None = None) -> TopicNode | None:
        """Add (or touch) a topic node and make it current. When `parent` is
        given the node is linked under it (a sub-topic)."""
        name = self._norm(topic)
        if not name:
            return None
        node = self._nodes.get(name)
        if node is None:
            node = TopicNode(name=name, parent=self._norm(parent) or None)
            self._nodes[name] = node
        elif parent and not node.parent:
            node.parent = self._norm(parent)
        node.last_seen = time()
        if name != self._current:
            self._previous = self._current
        self._current = name
        return node

    def attach_followup(self, turn_idx: int, topic: str | None = None) -> None:
        """Attach a turn to a topic node (default: the current node)."""
        name = self._norm(topic) if topic else self._current
        if not name:
            return
        node = self._nodes.get(name) or self.add_topic(name)
        if node is not None:
            node.turns.append(int(turn_idx))
            node.last_seen = time()

    @staticmethod
    def _related(a: str, b: str) -> bool:
        if not a or not b:
            return False
        return a == b or a in b or b in a

    def detect_drift(self, new_topic: str, similarity: float | None = None) -> bool:
        """True when `new_topic` is a switch away from the current topic (not a
        sub-topic). Uses string relatedness first; when an embedding
        `similarity` is supplied it drifts below DRIFT_THRESHOLD."""
        name = self._norm(new_topic)
        if not name or self._current is None:
            return False
        if self._related(name, self._current):
            return False
        if similarity is not None:
            return similarity < DRIFT_THRESHOLD
        return name != self._current

    def observe(self, topic: str, turn_idx: int | None = None,
                similarity: float | None = None) -> bool:
        """Convenience: record a topic for a turn, linking it as a sub-topic of
        the current node unless it is a drift. Returns True on a detected drift."""
        name = self._norm(topic)
        if not name:
            return False
        drift = self.detect_drift(name, similarity)
        parent = None if (drift or self._current is None) else self._current
        self.add_topic(name, parent=parent)
        if turn_idx is not None:
            self.attach_followup(turn_idx, name)
        return drift

    def resolve_reference(self, text: str) -> str | None:
        """Resolve an earlier-topic reference ("back to partitions", "how is
        that different from Redis") to a known node name, or None."""
        t = (text or "").lower()
        if not t:
            return None
        m = re.search(r"(?:back to|return to|go back to)\s+([a-z0-9 ._-]{2,40})", t)
        if m:
            frag = m.group(1).strip()
            for name in self._nodes:
                if name and (name in frag or frag in name):
                    return name
        # Any known topic mentioned (most-recently-seen first), excluding current.
        for node in sorted(
            (n for n in self._nodes.values() if n.name != self._current),
            key=lambda n: n.last_seen, reverse=True,
        ):
            if re.search(r"\b" + re.escape(node.name) + r"\b", t):
                return node.name
        return None

    def current(self) -> str | None:
        return self._current

    def previous(self) -> str | None:
        return self._previous

    def topics(self) -> list[str]:
        return list(self._nodes.keys())

    def node(self, name: str) -> TopicNode | None:
        return self._nodes.get(self._norm(name))


def for_tracker(tracker) -> TopicGraph:
    """Return the TopicGraph attached to a per-session context tracker, creating
    it lazily (stored on the tracker object — no new registry, no DB)."""
    g = getattr(tracker, "_live_topic_graph", None)
    if g is None:
        g = TopicGraph()
        try:
            setattr(tracker, "_live_topic_graph", g)
        except Exception:  # noqa: BLE001
            pass
    return g
