"""Prompt Compiler (roadmap Phase 8A #5).

Compiles a final prompt from STRUCTURED components (system rules, intent, context,
evidence, constraints, style) rather than ad-hoc string concatenation — so every
prompt is assembled the same way, sections are ordered and de-duplicated, and the
result is inspectable/testable. Complements `core/prompt.py` (existing prompt
helpers) by giving a declarative spec → deterministic render.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptSpec:
    system: str = ""
    intent: str = ""
    context: list[str] = field(default_factory=list)      # background snippets
    evidence: list[str] = field(default_factory=list)     # grounded facts (resume/docs)
    constraints: list[str] = field(default_factory=list)  # output constraints
    style: str = ""                                       # tone/format directive
    task: str = ""                                        # the actual request


# Section order is fixed so prompts are consistent and diff-able.
_ORDER = [
    ("system", "SYSTEM"),
    ("intent", "INTENT"),
    ("context", "CONTEXT"),
    ("evidence", "EVIDENCE"),
    ("constraints", "CONSTRAINTS"),
    ("style", "STYLE"),
    ("task", "TASK"),
]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        s = (x or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def compile_prompt(spec: PromptSpec) -> str:
    """Render a PromptSpec to a final prompt string. Empty sections are omitted;
    list sections are de-duplicated and bulleted. Deterministic + fail-open."""
    try:
        blocks: list[str] = []
        for attr, label in _ORDER:
            val = getattr(spec, attr, None)
            if isinstance(val, list):
                items = _dedupe(val)
                if items:
                    body = "\n".join(f"- {i}" for i in items)
                    blocks.append(f"# {label}\n{body}")
            else:
                s = (val or "").strip()
                if s:
                    blocks.append(f"# {label}\n{s}")
        return "\n\n".join(blocks)
    except Exception:  # noqa: BLE001
        # Fail-open: at least return the raw task so a turn can proceed.
        return (getattr(spec, "task", "") or "").strip()


__all__ = ["PromptSpec", "compile_prompt"]
