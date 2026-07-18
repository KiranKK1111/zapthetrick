"""Multi-provider LLM routing + fallback subsystem.

A faithful Python port of the `freellmapi` reference engine. The pieces:

  - `catalog`   — static registry of the supported providers and the
                  curated free-tier model seed (ranks + rate limits).
  - `crypto`    — AES-256-GCM at-rest encryption for API keys.
  - `keys`      — async repo over the `llm_api_keys` table (multi-key).
  - `ratelimit` — sliding-window RPM/RPD/TPM/TPD + escalating cooldowns.
  - `router`    — picks the best available model+key by priority+penalty.
  - `health`    — periodic key validation, auto-disable on repeated 401/403.
  - `providers` — OpenAI-compatible + Cloudflare HTTP adapters.

The single integration point is `app.core.llm_client.LLMClient`: when
`cfg.llm.provider == "auto"` every call routes through this subsystem and
falls back across providers/models automatically. Every other call site
(chat route, the 13 agents, the code solver) is untouched.
"""
