"""
Tool registry for the orchestrator.

Each tool is registered with:
  - name           (str, unique)
  - description    (str, fed to the LLM)
  - input_schema   (JSON schema dict; matches Ollama / Anthropic tool format)
  - handler        (async callable returning str or dict)

The orchestrator either:
  (a) hands the tool list to the LLM and lets it call tools via its
      native tool-use API (Ollama supports this for Qwen 2.5 / Llama 3.1+), or
  (b) picks tools heuristically based on question type and runs them
      itself before composing the final answer (Phase 1 default — more
      reliable across all local models).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class Tool:
    """One tool the orchestrator can invoke."""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]

    def to_ollama_format(self) -> dict[str, Any]:
        """Render this tool in the format Ollama / OpenAI tool calling expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


_registry: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    """Add a tool to the global registry. Overwrites by name."""
    _registry[tool.name] = tool


def get(name: str) -> Tool | None:
    return _registry.get(name)


def all_tools() -> list[Tool]:
    return list(_registry.values())


def names() -> list[str]:
    return list(_registry.keys())


def by_names(names_list: list[str]) -> list[Tool]:
    """Return tools matching the provided names, preserving order."""
    return [t for t in (_registry.get(n) for n in names_list) if t is not None]
