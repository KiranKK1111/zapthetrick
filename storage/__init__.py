"""Storage layer — the source-of-truth data stack.

Per DataBaseArchitecture.md, the stack is:
  - Postgres 16 + Apache AGE + pg_search (relational + FTS + graph)
  - Qdrant                                (vectors)
  - DragonflyDB / Redis                   (cache, pub/sub)
  - Filesystem / MinIO                    (blobs)

Nothing in [app.*] imports psycopg / qdrant_client / redis directly.
Callers use the repositories ([app.storage.repos.*]) and the
backend-agnostic [VectorStore], [Cache], [BlobStore], [GraphStore]
interfaces.

Bootstrap is via [bootstrap_storage] in [app.main]'s lifespan.
"""
from .bootstrap import bootstrap_storage, shutdown_storage
from .db import (
    SessionFactory,
    create_engine,
    dispose_engine,
    ensure_schema_exists,
    get_session,
    get_session_factory,
    reinit_engine,
    test_connection,
)

__all__ = [
    "SessionFactory",
    "get_session_factory",
    "create_engine",
    "dispose_engine",
    "reinit_engine",
    "ensure_schema_exists",
    "test_connection",
    "get_session",
    "bootstrap_storage",
    "shutdown_storage",
]
