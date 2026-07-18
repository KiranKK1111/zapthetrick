"""OpenAI-compatible adapter — the workhorse for 15 of 16 providers.

Ported from freellmapi's `providers/openai-compat.ts`. Handles bearer auth,
per-provider extra headers + timeout, SSE streaming, and the reasoning/array
content normalization. Google (via its `/v1beta/openai` endpoint) and Cohere
(via its compatibility endpoint) ride this same adapter.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

from app.llm.providers.base import BaseAdapter, ProviderError, pooled_client


class OpenAICompatAdapter(BaseAdapter):
    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.spec.extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.spec.base_url.rstrip('/')}{path}"

    async def complete(self, api_key: str, messages: list[dict], model_id: str, options: dict) -> str:
        payload = self._payload(messages, model_id, options, stream=False)
        try:
            async with pooled_client() as client:
                resp = await client.post(
                    self._url("/chat/completions"), json=payload,
                    headers=self._headers(api_key), timeout=self._timeout,
                )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: {exc}", retryable=True) from exc
        if resp.status_code != 200:
            raise ProviderError(
                f"{self.name} API error {resp.status_code}: {resp.text[:300]}",
                status=resp.status_code,
            )
        body = resp.json()
        choices = body.get("choices") or []
        # G6.1: record the provider's authoritative usage + stop reason so the
        # engine can account real tokens (not chars//4). Task-local, fail-safe.
        try:
            from app.llm import usage as _usage
            _usage.record(body.get("usage"),
                          (choices[0].get("finish_reason") if choices else None))
        except Exception:  # noqa: BLE001
            pass
        if not choices:
            return ""
        return self._fold_reasoning(choices[0].get("message") or {})

    async def stream(
        self, api_key: str, messages: list[dict], model_id: str, options: dict
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(messages, model_id, options, stream=True)
        try:
            async with pooled_client() as client:
                async with client.stream(
                    "POST", self._url("/chat/completions"), json=payload,
                    headers=self._headers(api_key), timeout=self._timeout,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise ProviderError(
                            f"{self.name} API error {resp.status_code}: "
                            f"{body.decode('utf-8', 'replace')[:300]}",
                            status=resp.status_code,
                        )
                    # Reasoning models (GPT-OSS, DeepSeek-R1, …) stream their
                    # chain-of-thought in `reasoning`/`reasoning_content` and
                    # the *answer* in `content`. We must NOT surface the CoT —
                    # that's the "The user is asking…" preamble users were
                    # seeing. Yield `content` only; buffer reasoning and emit
                    # it as a last resort solely when the model produced no
                    # content at all (some providers misfile the answer there).
                    content_seen = False
                    reasoning_buf: list[str] = []
                    _usage_frame = None
                    _finish = None
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        # G6.1: a usage frame (some providers, esp. with
                        # include_usage) carries `usage` + empty choices — capture
                        # it before the empty-choices skip below.
                        if obj.get("usage"):
                            _usage_frame = obj.get("usage")
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        if choices[0].get("finish_reason"):
                            _finish = choices[0].get("finish_reason")
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            content_seen = True
                            yield content
                            continue
                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                        )
                        if reasoning:
                            reasoning_buf.append(reasoning)
                    if not content_seen and reasoning_buf:
                        # Reasoning-only model (the whole answer arrived in
                        # the reasoning field): clean it before yielding — a
                        # raw CoT dump is the "messy response" users see.
                        _blob = "".join(reasoning_buf)
                        try:
                            from app.response_arch.sanitize import (
                                strip_reasoning)
                            _blob = strip_reasoning(_blob)
                        except Exception:  # noqa: BLE001
                            pass
                        yield _blob
                    # G6.1: record whatever usage/stop-reason the stream carried.
                    try:
                        from app.llm import usage as _usage
                        _usage.record(_usage_frame, _finish)
                    except Exception:  # noqa: BLE001
                        pass
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: {exc}", retryable=True) from exc

    async def validate_key(self, api_key: str) -> bool:
        try:
            async with pooled_client() as client:
                resp = await client.get(
                    self._url("/models"), headers=self._headers(api_key), timeout=10.0
                )
        except httpx.HTTPError as exc:
            # Transport error — propagate so health.py marks 'error' without
            # counting toward auto-disable.
            raise ProviderError(f"{self.name}: {exc}", retryable=True) from exc
        # 403 is NOT a definitive bad key — a WAF / geo-block / rate-limit on
        # /models returns 403 for a perfectly valid key. Treat it as
        # inconclusive (transport-like) so a transient 403 can't mark a working
        # key invalid. Only a 401 is an authoritative "bad key".
        if resp.status_code == 403:
            raise ProviderError(f"{self.name}: 403 on /models (inconclusive)",
                                retryable=True)
        return resp.status_code != 401
