"""Cloudflare Workers AI adapter.

Ported from freellmapi's `providers/cloudflare.ts`. The key is stored as
`account_id:token`; the account id is spliced into the account-scoped URL
and the token goes in the bearer header. Array/None content is flattened
to a string because the CF endpoint rejects the OpenAI array envelope.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

from app.llm.providers.base import BaseAdapter, ProviderError, pooled_client


class CloudflareAdapter(BaseAdapter):
    @staticmethod
    def _parse_key(api_key: str) -> tuple[str, str]:
        sep = api_key.find(":")
        if sep == -1:
            raise ProviderError(
                'Cloudflare key must be "account_id:api_token"', retryable=False
            )
        return api_key[:sep], api_key[sep + 1:]

    def _url(self, account_id: str) -> str:
        base = self.spec.base_url.format(account_id=account_id).rstrip("/")
        return f"{base}/chat/completions"

    @staticmethod
    def _normalize(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(
                    seg if isinstance(seg, str) else (seg.get("text") or "")
                    for seg in content
                )
            out.append({**m, "content": content or ""})
        return out

    async def complete(self, api_key: str, messages: list[dict], model_id: str, options: dict) -> str:
        account_id, token = self._parse_key(api_key)
        payload = self._payload(self._normalize(messages), model_id, options, stream=False)
        try:
            async with pooled_client() as client:
                resp = await client.post(
                    self._url(account_id),
                    json=payload,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    timeout=self._timeout,
                )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cloudflare: {exc}", retryable=True) from exc
        if resp.status_code != 200:
            raise ProviderError(
                f"Cloudflare API error {resp.status_code}: {resp.text[:300]}",
                status=resp.status_code,
            )
        choices = resp.json().get("choices") or []
        return self._fold_reasoning(choices[0].get("message") or {}) if choices else ""

    async def stream(
        self, api_key: str, messages: list[dict], model_id: str, options: dict
    ) -> AsyncGenerator[str, None]:
        account_id, token = self._parse_key(api_key)
        payload = self._payload(self._normalize(messages), model_id, options, stream=True)
        try:
            async with pooled_client() as client:
                async with client.stream(
                    "POST",
                    self._url(account_id),
                    json=payload,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    timeout=self._timeout,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise ProviderError(
                            f"Cloudflare API error {resp.status_code}: "
                            f"{body.decode('utf-8', 'replace')[:300]}",
                            status=resp.status_code,
                        )
                    # Yield the answer (`content`) only; never surface the
                    # reasoning/CoT channel. Buffer it and emit only if the
                    # model produced no content at all. See openai_compat.py.
                    content_seen = False
                    reasoning_buf: list[str] = []
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
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            content_seen = True
                            yield content
                            continue
                        reasoning = delta.get("reasoning")
                        if reasoning:
                            reasoning_buf.append(reasoning)
                    if not content_seen and reasoning_buf:
                        _blob = "".join(reasoning_buf)
                        try:
                            from app.response_arch.sanitize import (
                                strip_reasoning)
                            _blob = strip_reasoning(_blob)
                        except Exception:  # noqa: BLE001
                            pass
                        yield _blob
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cloudflare: {exc}", retryable=True) from exc

    async def validate_key(self, api_key: str) -> bool:
        _, token = self._parse_key(api_key)
        try:
            async with pooled_client() as client:
                resp = await client.get(
                    "https://api.cloudflare.com/client/v4/user/tokens/verify",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0,
                )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cloudflare: {exc}", retryable=True) from exc
        if resp.status_code in (401, 403):
            return False
        if resp.status_code != 200:
            return True  # unexpected non-auth error — don't disable
        data = resp.json()
        return bool(data.get("success")) and (data.get("result") or {}).get("status") == "active"
