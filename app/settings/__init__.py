"""Live-config event bus — Architecture2.md §"Event-driven config".

When the Settings UI saves changes, the route layer publishes one
[ConfigEvent] per top-level section that changed. Subsystems
(`llm`, `embeddings`, `reranker`, `audio`, `vector_store`, `themes`,
…) register a subscriber and rebuild themselves in-place — no
process restart, no manual cache clears.

Public surface:
    bus.subscribe(section, handler)   register an async/sync callback
    bus.publish(section, diff, full)  fire all subscribers for `section`
    bus.diff_paths(old, new)          compute top-level sections that changed
"""
from .bus import (
    ConfigBus,
    ConfigEvent,
    bus,
    diff_paths,
)


__all__ = ["ConfigBus", "ConfigEvent", "bus", "diff_paths"]
