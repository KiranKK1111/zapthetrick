"""Code knowledge graph — build a graph of a project's symbols and their
relationships from an uploaded archive, modelled on `codegraph`.

Pipeline: enumerate source files → tree-sitter parse → extract nodes (files,
classes, functions, methods, imports) + edges (contains, calls, imports,
extends) → resolve references by name/import → persist + summarise + query.

Public surface:
    build_code_graph(files)        -> CodeGraph        (app/codegraph/builder.py)
    summarize_graph(graph)         -> str              (app/codegraph/summary.py)
    find_symbol/callers/callees/…  (app/codegraph/query.py)
"""
from .model import EDGE_KINDS, NODE_KINDS, CodeGraph, Edge, Node

__all__ = ["CodeGraph", "Node", "Edge", "NODE_KINDS", "EDGE_KINDS"]
