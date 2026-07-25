"""Core-level facade for schema-enforced structured output (vNext §8.7).

`app.core.llm_client` is the app's single sanctioned LLM integration boundary, so
call sites depend on `app.core` rather than reaching into `app.llm` directly
(architecture rule #13 — no hidden coupling). This module re-exports the §8.7
helpers through that boundary: existing edges `codeintel/live/... -> core` and
`core -> llm` carry it, no new cross-package coupling is introduced.

Use:
    from app.core.structured import structured, parse_with_repair
    res = await structured(schema, messages)      # full router-backed ladder
    obj, errs = parse_with_repair(text, schema)   # deterministic parse only
"""
from __future__ import annotations

from app.llm.structured import (
    StructuredResult,
    parse_with_repair,
    schema_to_gbnf,
    structured,
)

__all__ = ["structured", "StructuredResult", "parse_with_repair",
           "schema_to_gbnf"]
