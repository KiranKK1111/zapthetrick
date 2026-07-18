"""Property-graph layer for the knowledge graph.

Per DataBaseArchitecture.md, the default backend is Apache AGE (Cypher
inside Postgres). The interface keeps room for Kùzu / Neo4j later.
"""
from .base import GraphStore, Node, Edge
from .factory import close_graph, get_graph

__all__ = ["GraphStore", "Node", "Edge", "get_graph", "close_graph"]
