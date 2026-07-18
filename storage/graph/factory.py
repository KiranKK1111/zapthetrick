"""Build the [GraphStore]. Currently Apache AGE only; Kùzu / Neo4j
adapters slot in alongside [AgeStore] when needed."""
from __future__ import annotations

from app.core.config_loader import cfg

from .age_store import AgeStore
from .base import GraphStore


_singleton: GraphStore | None = None


def get_graph() -> GraphStore | None:
    global _singleton
    if _singleton is not None:
        return _singleton
    if not cfg.database.postgres.enable_age:
        return None
    _singleton = AgeStore()
    return _singleton


async def close_graph() -> None:
    global _singleton
    if _singleton is not None:
        await _singleton.close()
    _singleton = None
