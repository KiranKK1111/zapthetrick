"""Per-model tokenizer registry.

Models register their tokenizer once at startup; callers ask for the
tokenizer by model id. Falls back to a whitespace-approximation
tokenizer when nothing is registered — useful for tests but logs a
warning so prod misconfigs surface.

TODO: wire `tiktoken` for OpenAI-compatible models and Hugging Face
`AutoTokenizer` for the rest. The deps are already in the venv.
"""
from __future__ import annotations

import hashlib
from typing import Callable

from .cache import TokenCountCache


TokenizerFn = Callable[[str], list[int | str]]


_REGISTRY: dict[str, TokenizerFn] = {}
_CACHE = TokenCountCache()


def _whitespace_tokenizer(text: str) -> list[str]:
    return text.split()


def register_tokenizer(model_id: str, fn: TokenizerFn) -> None:
    _REGISTRY[model_id] = fn


def get_tokenizer(model_id: str) -> TokenizerFn:
    return _REGISTRY.get(model_id, _whitespace_tokenizer)


def count_tokens(text: str, *, model_id: str) -> int:
    """Tokenize `text` for `model_id`, with memoization."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = _CACHE.get(model_id, h)
    if cached is not None:
        return cached
    n = len(get_tokenizer(model_id)(text))
    _CACHE.put(model_id, h, n)
    return n
