"""Tokenizer-aware utilities.

Per Architecture.md §3, chunks are measured in *real tokens* (via the
target model's tokenizer), never character approximations. We register
one tokenizer per model id and cache (model_id, text_hash) → count so
we don't tokenize the same string twice.
"""
from .tokenizers import get_tokenizer, register_tokenizer, count_tokens
from .cache import TokenCountCache

__all__ = [
    "get_tokenizer",
    "register_tokenizer",
    "count_tokens",
    "TokenCountCache",
]
