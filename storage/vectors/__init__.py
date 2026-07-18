"""Vector store abstraction — pgvector (Postgres) is default, Chroma is the
alternative embedded path.

Pick the backend via `cfg.vector_store.provider` + the factory.
"""
from .base import Hit, VectorStore
from .factory import get_vector_store

__all__ = ["Hit", "VectorStore", "get_vector_store"]
