"""
Provider-agnostic LLM client.

The single entry point every other module uses to talk to an LLM. It
dispatches to a provider adapter based on `cfg.llm.provider`. Three
providers are wired up:

  - "ollama"     — local, via the Ollama HTTP API (default).
  - "openrouter" — cloud, OpenAI-compatible (chat/completions + /models).
  - "nvidia"     — cloud, NVIDIA NIM (also OpenAI-compatible).

OpenRouter and NVIDIA share the same OpenAI-compatible adapter — they
differ only in `base_url` and API key. Adding more OpenAI-compatible
providers later (Groq, OpenAI, DeepSeek, ...) is a one-line dispatch
change plus credentials in config.
"""
import contextlib
import json
import logging
import re
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncGenerator

import httpx

from app.core.config_loader import cfg
# Shared pooled HTTP client (perceived-speed R2). Re-exported so callers can
# `from app.core.llm_client import get_http_client` as well.
from app.core.http_pool import dispose_http_client, get_http_client


@asynccontextmanager
async def _pooled():
    """Yield the shared pooled client WITHOUT closing it on exit (so the
    keep-alive pool survives across requests). Drop-in for the old
    ``async with httpx.AsyncClient(...) as client:`` blocks — the per-request
    timeout moves onto the actual request call."""
    yield get_http_client()


class LLMError(RuntimeError):
    """Raised when the configured LLM provider is unreachable or errors out."""


_OPENAI_COMPAT_PROVIDERS = ("openrouter", "nvidia")


def _sniff_image_mime(b64: str) -> str:
    """Best-effort image mime from a base64 string's leading bytes."""
    head = b64[:16]
    if head.startswith("/9j/"):
        return "image/jpeg"
    if head.startswith("iVBOR"):
        return "image/png"
    if head.startswith("R0lGOD"):
        return "image/gif"
    if head.startswith("UklGR"):
        return "image/webp"
    return "image/png"


# ---- Mid-stream quality control (app/quality/stream_controller) ----------
# The stream guard (app/llm/stream_guard) already kills *runaway* output:
# sentence-level loops and a hard char ceiling. The quality controller adds the
# failures it cannot see — refusal leakage, error/apology spikes, emptiness, and
# unpunctuated token-level degeneration (word salad the sentence splitter never
# sees). It is sampled on a cadence, never per token.
_QC_MIN_CHARS = 240        # don't judge an answer before it has a shape
_QC_SAMPLE_CHARS = 400     # …then re-assess every N visible chars
_QC_WINDOW_CHARS = 1200    # assess a bounded trailing window, not the whole answer
_QC_KILL_REPETITION = 0.85  # only an unmistakable loop may stop a stream
_QC_KILL_STRIKES = 2       # …and only if two consecutive windows agree

_QC_REP_REASON = re.compile(r"degenerate_repetition\(([0-9.]+)\)")

# Last mid-stream verdict for the current task — async generators run in the
# caller's context, so the route/verifier consuming the stream can read this
# after the fact without any new plumbing.
_qc_verdict: ContextVar[object | None] = ContextVar("dtt_stream_verdict", default=None)


def last_stream_verdict():
    """The most recent mid-stream `QualityVerdict` for this task, or None.

    Set by `LLMClient._guarded_stream`; a hook for the post-`done` verifier /
    routes to see that an answer already looked degraded while it streamed.
    """
    return _qc_verdict.get()


def _qc_kill_worthy(verdict) -> bool:
    """Is this verdict decisive enough to STOP a stream mid-flight?

    Deliberately narrow. A false-positive kill truncates a good answer, so only
    an unmistakable degenerate loop qualifies. A refusal or an error/apology
    spike is *flagged*, never killed: refusals are sometimes legitimate, and
    "error/exception/null" are ordinary words in a technical answer — acting on
    either would burn a good response.
    """
    for reason in getattr(verdict, "reasons", None) or []:
        m = _QC_REP_REASON.search(str(reason))
        if m and float(m.group(1)) >= _QC_KILL_REPETITION:
            return True
    return False


class LLMClient:
    # ---- Public API --------------------------------------------------
    async def stream_chat(
        self,
        messages: list[dict],
        model: str | None = None,
        session_key: str | None = None,
        options: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text chunks from the configured provider.

        When `cfg.llm.provider == "auto"`, the request is routed through the
        multi-provider fallback engine (`app.llm`): it picks the best
        available model+key, and falls through the priority chain on rate
        limits / outages before the first token is emitted. `session_key`
        (e.g. a conversation id) keeps a thread on one model (sticky session).
        """
        provider = cfg.llm.provider
        if provider == "auto":
            inner = self._auto_stream(messages, model, session_key, options)
        elif provider == "ollama":
            inner = self._ollama_stream(messages, model)
        elif provider in _OPENAI_COMPAT_PROVIDERS:
            inner = self._openai_compat_stream(messages, model)
        else:
            raise LLMError(
                f"Provider '{provider}' is not implemented. "
                f"Supported providers: auto, ollama, openrouter, nvidia."
            )
        async for chunk in self._guarded_stream(inner):
            yield chunk

    async def _guarded_stream(
        self, agen: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Wrap every visible token stream with the mid-stream output guards:
        incremental <think>/harmony scrubbing + a repetition/ceiling kill
        switch (user report 2026-07-09: "unwanted, messy, never-ending
        responses from some models"). Fail-open via the config flag."""
        if not bool(getattr(cfg.llm, "stream_guard", True)):
            async for chunk in agen:
                yield chunk
            return
        from app.llm.stream_guard import RepetitionGuard, StreamScrubber
        scrub = StreamScrubber()
        rep = RepetitionGuard(
            max_repeats=int(getattr(cfg.llm, "repetition_max_repeats", 3)),
            max_chars=int(getattr(cfg.llm, "stream_max_chars", 120_000)),
        )
        # Real-time quality controller — sampled, additive to the guard above.
        qc_on = bool(getattr(cfg.quality, "stream_control", True))
        qc_window = ""      # trailing slice of the VISIBLE answer
        qc_seen = 0         # visible chars streamed so far
        qc_next = _QC_MIN_CHARS
        qc_strikes = 0
        killed = False
        try:
            async for raw in agen:
                clean = scrub.feed(raw)
                if not clean:
                    continue
                if rep.feed(clean):
                    yield clean
                    yield ("\n\n*…output stopped — the model began "
                           "repeating itself.*")
                    logging.getLogger("zapthetrick.llm").warning(
                        "stream guard: runaway output stopped")
                    killed = True
                    break
                yield clean
                if not qc_on:
                    continue
                qc_window = (qc_window + clean)[-_QC_WINDOW_CHARS:]
                qc_seen += len(clean)
                if qc_seen < qc_next:
                    continue
                qc_next = qc_seen + _QC_SAMPLE_CHARS
                verdict = self._assess_partial(qc_window)
                if verdict is None:          # controller misbehaved → fail open
                    qc_on = False
                    continue
                if _qc_kill_worthy(verdict) and verdict.action == "regenerate":
                    qc_strikes += 1
                    if qc_strikes >= _QC_KILL_STRIKES:
                        yield ("\n\n*…output stopped — the response was "
                               "degenerating.*")
                        logging.getLogger("zapthetrick.quality").warning(
                            "stream control: stopped degenerate output "
                            "(score=%s reasons=%s)",
                            verdict.score, verdict.reasons)
                        killed = True
                        break
                else:
                    qc_strikes = 0
            if not killed:
                tail = scrub.flush()
                if tail:
                    yield tail
                    qc_window = (qc_window + tail)[-_QC_WINDOW_CHARS:]
                    qc_seen += len(tail)
                if qc_on:
                    # Final read on short answers (never sampled) and on a
                    # stream that produced no visible text at all.
                    self._assess_partial(qc_window)
        finally:
            with contextlib.suppress(BaseException):
                await agen.aclose()

    @staticmethod
    def _assess_partial(text: str):
        """Run the mid-stream quality controller on a window of visible text.

        Records the verdict (readable via `last_stream_verdict()`) and logs
        anything that is not a clean "continue". Returns None when the
        controller itself failed — the caller then switches it off for the rest
        of the stream, so a bug in the monitor can never break an answer.
        """
        try:
            from app.quality.stream_controller import assess_partial
            verdict = assess_partial(text)
            _qc_verdict.set(verdict)
            if verdict.action != "continue":
                logging.getLogger("zapthetrick.quality").warning(
                    "stream control: %s (score=%s reasons=%s)",
                    verdict.action, verdict.score, verdict.reasons)
            return verdict
        except Exception:  # noqa: BLE001 — fail open, always
            logging.getLogger("zapthetrick.quality").debug(
                "stream control disabled for this stream (controller error)",
                exc_info=True)
            return None

    async def chat_json(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> str:
        """One-shot call asking the provider for a JSON-formatted response.

        Callers parse the returned string with the lenient JSON helper
        in `persona/profile_builder.py` and `question_detection/classifier.py`
        since providers vary in how strictly they obey JSON hints.
        """
        provider = cfg.llm.provider
        if provider == "auto":
            text, _route = await self._auto_complete(
                messages, model, options={"response_format_json": True}
            )
            return text
        if provider == "ollama":
            return await self._ollama_json(messages, model)
        if provider in _OPENAI_COMPAT_PROVIDERS:
            return await self._openai_compat_complete(
                messages, model, options={"response_format_json": True}
            )
        raise LLMError(
            f"Provider '{provider}' is not implemented for chat_json."
        )

    async def health(self) -> dict:
        """Probe the configured provider. Returns a status dict for /api/health."""
        provider = cfg.llm.provider
        if provider == "auto":
            return await self._auto_health()
        if provider == "ollama":
            return await self._ollama_health()
        if provider in _OPENAI_COMPAT_PROVIDERS:
            return await self._openai_compat_health()
        return {
            "status": "unknown_provider",
            "provider": provider,
            "model": cfg.llm.model,
        }

    # ---- Auto (multi-provider fallback) adapter ---------------------
    def _auto_options(self, options: dict | None) -> dict:
        """Translate per-call options into the engine's option dict.

        Maps Ollama's `num_predict` to `max_tokens` and falls back to the
        global cfg.llm knobs. `response_format_json` flips JSON mode on.
        """
        options = options or {}
        out: dict = {
            "temperature": options.get("temperature", cfg.llm.temperature),
            "max_tokens": options.get("num_predict", options.get("max_tokens", cfg.llm.max_tokens)),
        }
        if options.get("response_format_json"):
            out["response_format_json"] = True
        # Capability-aware routing hints (consumed + stripped by the engine).
        if options.get("difficulty"):
            out["difficulty"] = options["difficulty"]
        if options.get("avoid_model_db_id") is not None:
            out["avoid_model_db_id"] = options["avoid_model_db_id"]
        # Forward the intelligent-model-routing hints the caller set (persona /
        # understanding pass). Dropping these made semantic routing write-only
        # and left tool/JSON capability filtering inactive.
        for _k in ("task_category", "needs_tool", "needs_json",
                   "query_embedding"):
            if options.get(_k) is not None:
                out[_k] = options[_k]
        # Per-call mid-stream continuation override (engine pops it): answer
        # paths opt in so a long reply that drops / hits the token ceiling still
        # finishes seamlessly, without flipping the global (live-safe) default.
        if options.get("mid_stream_continuation") is not None:
            out["mid_stream_continuation"] = options["mid_stream_continuation"]
        return out

    async def _auto_preferred(self, model: str | None) -> int | None:
        from app.llm import router as _router

        if not model:
            return None
        try:
            return await _router.model_db_id_for(model)
        except Exception:  # noqa: BLE001 — never block a call on this lookup
            return None

    async def _auto_stream(
        self,
        messages: list[dict],
        model: str | None,
        session_key: str | None,
        options: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        from app.llm import engine
        from app.llm.providers import ProviderError
        from app.llm.router import NoRouteAvailable

        oai_messages = self._to_openai_messages(messages)
        preferred = await self._auto_preferred(model)
        try:
            # Phase 2 — speculative multi-model drafting (R4). Flag-gated +
            # budget-gated; off → today's single-route stream. A call that opts
            # into mid-stream continuation ("always finishes") skips speculation
            # — the two are mutually exclusive (speculation races short drafts
            # and bypasses the continuation wrapper), and a long answer that must
            # not drop halfway prefers the guaranteed-finish path over latency.
            from app.perceived import speculative as _spec
            _force_cont = bool((options or {}).get("mid_stream_continuation"))
            if model is None and not _force_cont and _spec.should_speculate():
                async for chunk in _spec.speculative_auto_stream(
                    oai_messages, self._auto_options(options), session_key,
                ):
                    yield chunk
                return
            # §15 resilience: mid-stream failover via a continuation contract.
            # Passthrough of route_and_stream when the flag is off.
            async for chunk in engine.stream_with_continuation(
                oai_messages, self._auto_options(options),
                session_key=session_key, preferred_model_db_id=preferred,
            ):
                yield chunk
        except NoRouteAvailable as exc:
            raise LLMError(
                "No LLM route available right now. Add a provider key in "
                f"Settings → Providers, or wait for rate limits to reset. ({exc})"
            ) from exc
        except ProviderError as exc:
            raise LLMError(f"Provider error: {exc}") from exc

    async def _auto_complete(
        self,
        messages: list[dict],
        model: str | None,
        options: dict | None,
    ) -> tuple[str, object]:
        from app.llm import engine
        from app.llm.providers import ProviderError
        from app.llm.router import NoRouteAvailable

        oai_messages = self._to_openai_messages(messages)
        preferred = await self._auto_preferred(model)
        try:
            return await engine.route_and_complete(
                oai_messages, self._auto_options(options),
                preferred_model_db_id=preferred,
            )
        except NoRouteAvailable as exc:
            raise LLMError(
                "No LLM route available right now. Add a provider key in "
                f"Settings → Providers, or wait for rate limits to reset. ({exc})"
            ) from exc
        except ProviderError as exc:
            raise LLMError(f"Provider error: {exc}") from exc

    async def _auto_health(self) -> dict:
        """Report whether the router can serve a request right now."""
        from app.llm import router as _router

        try:
            route = await _router.route_request(estimated_tokens=10)
            return {"status": "ok", "provider": "auto", "model": route.display_name}
        except _router.NoRouteAvailable:
            return {
                "status": "unconfigured",
                "provider": "auto",
                "model": cfg.llm.model,
                "error": "No usable provider key. Add one in Settings → Providers.",
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "provider": "auto", "model": cfg.llm.model, "error": str(exc)}

    async def _auto_list_models(self) -> list[dict]:
        """List enabled catalog models as {name, size} for the model picker."""
        from sqlalchemy import select

        from storage.db import get_session_factory
        from storage.models import LLMModel

        factory = get_session_factory()
        if factory is None:
            return []
        async with factory() as session:
            rows = (
                await session.execute(
                    select(LLMModel).where(LLMModel.enabled.is_(True)).order_by(
                        LLMModel.intelligence_rank.asc()
                    )
                )
            ).scalars().all()
        return [{"name": r.model_id, "size": 0} for r in rows]

    # ---- Ollama adapter ---------------------------------------------
    async def _ollama_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None,
    ) -> AsyncGenerator[str, None]:
        model = model or cfg.llm.model
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": cfg.llm.temperature,
                "num_predict": cfg.llm.max_tokens,
            },
        }
        try:
            async with _pooled() as client:
                async with client.stream(
                    "POST", f"{cfg.llm.base_url}/api/chat", json=payload,
                    timeout=cfg.llm.timeout_seconds,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise LLMError(
                            f"Ollama returned {resp.status_code}: "
                            f"{body.decode('utf-8', 'replace')[:300]}"
                        )
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            # Ollama emits one JSON object per line; skip
                            # malformed lines rather than crashing the stream.
                            continue
                        chunk = obj.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                        if obj.get("done"):
                            break
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Ollama at {cfg.llm.base_url}. "
                f"Is it running? Original error: {exc}"
            ) from exc

    async def _ollama_json(
        self,
        messages: list[dict[str, str]],
        model: str | None,
    ) -> str:
        model = model or cfg.llm.model
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": cfg.llm.temperature},
        }
        try:
            async with _pooled() as client:
                resp = await client.post(
                    f"{cfg.llm.base_url}/api/chat", json=payload,
                    timeout=cfg.llm.timeout_seconds,
                )
                if resp.status_code != 200:
                    raise LLMError(
                        f"Ollama returned {resp.status_code}: {resp.text[:300]}"
                    )
                return resp.json().get("message", {}).get("content", "")
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Ollama at {cfg.llm.base_url}. "
                f"Is it running? Original error: {exc}"
            ) from exc

    async def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        options: dict | None = None,
    ) -> str:
        """Non-streaming plain-text completion. Used by the OCR step of solve.

        Unlike `chat_json`, no `format: json` flag is set, so callers get
        free-form text. `options` overrides per-call values like temperature
        and num_predict — useful for OCR (temperature 0, smaller token cap).
        """
        text, _ = await self.complete_routed(messages, model, options)
        return text

    async def complete_routed(
        self,
        messages: list[dict],
        model: str | None = None,
        options: dict | None = None,
    ) -> tuple[str, int | None]:
        """Like `complete`, but also returns the routed model's db id (or None
        for non-auto providers) — lets the self-refine step verify on a
        DIFFERENT model than it drafted on."""
        # Cognitive cache (P2-10): an identical low-temperature request reuses
        # the stored answer instead of paying latency/quota again.
        from app.llm import cache as _cache
        _ck = _cache.maybe_key(messages, options or {}, model=model)
        if _ck:
            _hit = _cache.get(_ck)
            if _hit is not None:
                return _hit, None
        provider = cfg.llm.provider
        if provider == "auto":
            text, route = await self._auto_complete(messages, model, options or {})
            if _ck:
                _cache.put(_ck, text)
            return text, getattr(route, "model_db_id", None)
        if provider == "ollama":
            text = await self._ollama_complete(messages, model, options or {})
            if _ck:
                _cache.put(_ck, text)
            return text, None
        if provider in _OPENAI_COMPAT_PROVIDERS:
            text = await self._openai_compat_complete(messages, model, options or {})
            if _ck:
                _cache.put(_ck, text)
            return text, None
        raise LLMError(
            f"Provider '{provider}' is not implemented for complete()."
        )

    async def _ollama_complete(
        self,
        messages: list[dict],
        model: str | None,
        options: dict,
    ) -> str:
        model = model or cfg.llm.model
        merged_options = {
            "temperature": cfg.llm.temperature,
            **options,
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": merged_options,
        }
        try:
            async with _pooled() as client:
                resp = await client.post(
                    f"{cfg.llm.base_url}/api/chat", json=payload,
                    timeout=cfg.llm.timeout_seconds,
                )
                if resp.status_code != 200:
                    raise LLMError(
                        f"Ollama returned {resp.status_code}: {resp.text[:300]}"
                    )
                return resp.json().get("message", {}).get("content", "")
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Ollama at {cfg.llm.base_url}. "
                f"Is it running? Original error: {exc}"
            ) from exc

    async def _ollama_health(self) -> dict:
        try:
            async with _pooled() as client:
                resp = await client.get(f"{cfg.llm.base_url}/api/tags", timeout=5.0)
                ok = resp.status_code == 200
                return {
                    "status": "ok" if ok else "unreachable",
                    "provider": "ollama",
                    "model": cfg.llm.model,
                }
        except (httpx.HTTPError, OSError) as exc:
            return {
                "status": "unreachable",
                "provider": "ollama",
                "model": cfg.llm.model,
                "error": str(exc),
            }

    async def list_models(self) -> list[dict]:
        """Return the list of models the configured provider has available.

        For Ollama this is `/api/tags` — each entry is at least
        {"name": "<model>", "size": <int>, "digest": "...", "modified_at": ...}.
        The UI uses it to populate per-tool model pickers (Solve screen).
        """
        provider = cfg.llm.provider
        if provider == "auto":
            return await self._auto_list_models()
        if provider == "ollama":
            return await self._ollama_list_models()
        if provider in _OPENAI_COMPAT_PROVIDERS:
            return await self._openai_compat_list_models()
        raise LLMError(
            f"Provider '{provider}' is not implemented for list_models()."
        )

    async def _ollama_list_models(self) -> list[dict]:
        try:
            async with _pooled() as client:
                resp = await client.get(f"{cfg.llm.base_url}/api/tags", timeout=10.0)
                if resp.status_code != 200:
                    raise LLMError(
                        f"Ollama /api/tags returned {resp.status_code}: "
                        f"{resp.text[:200]}"
                    )
                body = resp.json()
                models = body.get("models", [])
                # Sort alphabetically so the dropdown is stable.
                models.sort(key=lambda m: m.get("name", ""))
                return models
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Ollama at {cfg.llm.base_url}: {exc}"
            ) from exc

    # ---- OpenAI-compatible adapter (OpenRouter + NVIDIA NIM) --------
    def _openai_compat_endpoint(self) -> tuple[str, str, dict]:
        """Resolve (base_url, api_key, extra_headers) for the active provider.

        Both providers speak the OpenAI Chat Completions dialect; only the
        endpoint, API key, and a handful of optional headers differ.
        OpenRouter recommends an `HTTP-Referer` + `X-Title` for analytics
        but they are not required.
        """
        provider = cfg.llm.provider
        if provider == "openrouter":
            api_key = (cfg.llm.openrouter_api_key or "").strip()
            base_url = (cfg.llm.openrouter_base_url or "").rstrip("/")
            if not api_key:
                raise LLMError(
                    "OpenRouter API key is not configured. Open Settings → "
                    "LLM Provider and paste your OpenRouter key."
                )
            return base_url, api_key, {
                "HTTP-Referer": "https://zapthetrick.ai",
                "X-Title": cfg.app.name,
            }
        if provider == "nvidia":
            api_key = (cfg.llm.nvidia_api_key or "").strip()
            base_url = (cfg.llm.nvidia_base_url or "").rstrip("/")
            if not api_key:
                raise LLMError(
                    "NVIDIA NIM API key is not configured. Open Settings → "
                    "LLM Provider and paste your NVIDIA API key."
                )
            return base_url, api_key, {}
        raise LLMError(f"Provider '{provider}' is not OpenAI-compatible.")

    def _to_openai_messages(self, messages: list[dict]) -> list[dict]:
        """Translate our Ollama-style messages into OpenAI's multipart format.

        Internally we represent images Ollama-style: a `images: [base64]`
        field on a user message. OpenAI providers expect a multipart
        `content` array with `{type: image_url, image_url: {url: data:...}}`
        entries instead. This converter walks the message list and reshapes
        only messages that carry images.
        """
        out: list[dict] = []
        for m in messages:
            images = m.get("images") if isinstance(m, dict) else None
            if not images:
                out.append({"role": m["role"], "content": m.get("content", "")})
                continue
            content: list[dict] = []
            text = (m.get("content") or "").strip()
            if text:
                content.append({"type": "text", "text": text})
            for img in images:
                # Base64 payload; the data URL mime tells the provider how to
                # decode. Sniff jpeg/png/gif/webp from the base64 prefix so
                # chat-uploaded JPEGs aren't mislabeled as PNG (some providers
                # validate the declared type).
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{_sniff_image_mime(img)};base64,{img}"},
                })
            out.append({"role": m["role"], "content": content})
        return out

    def _openai_compat_payload(
        self,
        messages: list[dict],
        model: str | None,
        options: dict,
        stream: bool,
    ) -> dict:
        """Build the JSON body for /chat/completions.

        Honours per-call options: `temperature`, `num_predict` (mapped to
        `max_tokens`), and `response_format_json` to flip JSON mode on.
        """
        merged_temp = options.get("temperature", cfg.llm.temperature)
        merged_max = options.get("num_predict", cfg.llm.max_tokens)
        payload: dict = {
            "model": model or cfg.llm.model,
            "messages": self._to_openai_messages(messages),
            "stream": stream,
            "temperature": merged_temp,
            "max_tokens": merged_max,
        }
        if options.get("response_format_json"):
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _openai_compat_stream(
        self,
        messages: list[dict],
        model: str | None,
    ) -> AsyncGenerator[str, None]:
        base_url, api_key, extra_headers = self._openai_compat_endpoint()
        payload = self._openai_compat_payload(
            messages, model, options={}, stream=True
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **extra_headers,
        }
        url = f"{base_url}/chat/completions"
        try:
            async with _pooled() as client:
                async with client.stream(
                    "POST", url, json=payload, headers=headers,
                    timeout=cfg.llm.timeout_seconds,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise LLMError(
                            f"{cfg.llm.provider} returned {resp.status_code}: "
                            f"{body.decode('utf-8', 'replace')[:300]}"
                        )
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
                        text = delta.get("content")
                        if text:
                            yield text
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach {cfg.llm.provider} at {base_url}: {exc}"
            ) from exc

    async def _openai_compat_complete(
        self,
        messages: list[dict],
        model: str | None,
        options: dict,
    ) -> str:
        """Non-streaming chat completion against the configured cloud provider."""
        base_url, api_key, extra_headers = self._openai_compat_endpoint()
        payload = self._openai_compat_payload(
            messages, model, options=options, stream=False
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **extra_headers,
        }
        url = f"{base_url}/chat/completions"
        try:
            async with _pooled() as client:
                resp = await client.post(
                    url, json=payload, headers=headers,
                    timeout=cfg.llm.timeout_seconds,
                )
                if resp.status_code != 200:
                    raise LLMError(
                        f"{cfg.llm.provider} returned {resp.status_code}: "
                        f"{resp.text[:300]}"
                    )
                body = resp.json()
                choices = body.get("choices") or []
                if not choices:
                    return ""
                msg = choices[0].get("message") or {}
                return msg.get("content", "") or ""
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach {cfg.llm.provider} at {base_url}: {exc}"
            ) from exc

    async def _openai_compat_health(self) -> dict:
        provider = cfg.llm.provider
        try:
            base_url, api_key, extra_headers = self._openai_compat_endpoint()
        except LLMError as exc:
            return {
                "status": "unconfigured",
                "provider": provider,
                "model": cfg.llm.model,
                "error": str(exc),
            }
        try:
            async with _pooled() as client:
                resp = await client.get(
                    f"{base_url}/models",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        **extra_headers,
                    },
                    timeout=8.0,
                )
                ok = resp.status_code == 200
                return {
                    "status": "ok" if ok else f"unreachable ({resp.status_code})",
                    "provider": provider,
                    "model": cfg.llm.model,
                }
        except (httpx.HTTPError, OSError) as exc:
            return {
                "status": "unreachable",
                "provider": provider,
                "model": cfg.llm.model,
                "error": str(exc),
            }

    async def _openai_compat_list_models(self) -> list[dict]:
        """List the cloud provider's models via the OpenAI-style /models endpoint."""
        base_url, api_key, extra_headers = self._openai_compat_endpoint()
        try:
            async with _pooled() as client:
                resp = await client.get(
                    f"{base_url}/models",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        **extra_headers,
                    },
                    timeout=15.0,
                )
                if resp.status_code != 200:
                    raise LLMError(
                        f"{cfg.llm.provider} /models returned {resp.status_code}: "
                        f"{resp.text[:200]}"
                    )
                body = resp.json()
                data = body.get("data") or body.get("models") or []
                # Normalise to the same shape Ollama uses: {name, size}.
                # OpenAI-compat /models payloads use `id`; ignore the rest.
                models = [
                    {"name": m.get("id") or m.get("name") or "", "size": 0}
                    for m in data
                    if (m.get("id") or m.get("name"))
                ]
                models.sort(key=lambda m: m["name"])
                return models
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach {cfg.llm.provider} at {base_url}: {exc}"
            ) from exc


# Module-level singleton — every importer shares the same instance.
llm = LLMClient()
