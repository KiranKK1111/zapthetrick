"""Cross-conversation continuity — Architecture.md §"Conversation link graph".

Three pieces working together:

    link_graph.py            — explicit graph of session-to-session links
    topic_threads.py         — clusters sessions that share a topic
    continuation_detector.py — detects when a new user turn is a follow-up

The graph is persisted in a `session_links` table; topic membership
in `session_topics`. Both are populated by the continuation detector
as a side-effect of each new turn — no batch job required.
"""
from .continuation_detector import detect_continuation
from .link_graph import LinkKind, LinkGraphRepo
from .topic_threads import TopicThreadRepo


__all__ = [
    "detect_continuation",
    "LinkKind",
    "LinkGraphRepo",
    "TopicThreadRepo",
]
