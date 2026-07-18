"""Repository intelligence for Code-In (code-intelligence spec).

AST-based symbol index, dependency + call graphs, symbol search, and a code-aware
context builder over the sandboxed `app/agent_workspace` tree — reusing the
existing `app/codegraph` extractor (Python `ast` + tree-sitter) and augmenting
`app/chat/context_builder.rank_files`. Deterministic (no LLM call); gated by
`cfg.advanced_rag.use_code_knowledge_graph`; fail-open to today's `rank_files`.
"""
from .index import CodeIndex, build_index, build_index_from_files, get_index
from .graph import dependency_graph, call_graph, dependents_of, usages_of
from .search import find
from .context import select

__all__ = [
    "CodeIndex", "build_index", "build_index_from_files", "get_index",
    "dependency_graph", "call_graph", "dependents_of", "usages_of",
    "find", "select",
]
