"""Provider HTTP adapters.

`get_adapter(platform)` returns the adapter instance for a platform. Almost
everything speaks the OpenAI Chat Completions dialect, so `OpenAICompat`
covers 15 of 16 providers (including Google via its `/v1beta/openai`
endpoint and Cohere via its compatibility endpoint). Cloudflare is the
lone special case — its key is `account_id:token` and the URL is
account-scoped.
"""
from __future__ import annotations

from app.llm.catalog import ProviderSpec, get_provider_spec
from app.llm.providers.base import BaseAdapter, ProviderError
from app.llm.providers.cloudflare import CloudflareAdapter
from app.llm.providers.openai_compat import OpenAICompatAdapter


_CACHE: dict[str, BaseAdapter] = {}


def get_adapter(platform: str) -> BaseAdapter | None:
    """Return (and cache) the adapter for `platform`, or None if unknown."""
    if platform in _CACHE:
        return _CACHE[platform]
    spec: ProviderSpec | None = get_provider_spec(platform)
    if spec is None:
        return None
    adapter: BaseAdapter
    if spec.adapter == "cloudflare":
        adapter = CloudflareAdapter(spec)
    else:
        adapter = OpenAICompatAdapter(spec)
    _CACHE[platform] = adapter
    return adapter


__all__ = ["get_adapter", "BaseAdapter", "ProviderError"]
