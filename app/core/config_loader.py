"""
Config loader: reads `config.yaml` into a typed Pydantic model.

Exposes `cfg` — a module-level proxy that always reflects the current
in-memory config. Updates flow through `update_config(dict)`, which merges
partial updates, re-validates, persists back to disk, and refreshes the
singleton. No file-system watching: callers (the /api/settings endpoint)
trigger reloads explicitly.

Architectural intent: this is the single source of truth. Any module that
needs an LLM model, a provider name, a server port — anything — reads from
`cfg.<section>.<field>` instead of taking constructor args or env vars.
That keeps later phases (RAG, STT, agent orchestrator) reconfigurable
without code changes.
"""
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# C1 control chars (U+0080..U+009F). PyYAML rejects control chars outright, so a
# single mangled byte — a clipboard paste, or a tool that re-encoded the file as
# cp1252 — would otherwise crash startup. We strip them defensively on read.
_C1_CONTROLS = re.compile("[\x80-\x9f]")


BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
# Honour ZAPTHETRICK_CONFIG_PATH so the installed/frozen app can keep its config
# in a writable per-user dir (%APPDATA%\ZapTheTrick\config.yaml) — Program Files
# is read-only. Falls back to the repo's config.yaml when run from source.
_CONFIG_ENV = os.environ.get("ZAPTHETRICK_CONFIG_PATH")
CONFIG_PATH = Path(_CONFIG_ENV) if _CONFIG_ENV else (BASE_DIR / "config.yaml")


# ---- Section models ----------------------------------------------------
class AppSection(BaseModel):
    name: str = "InterviewCopilot"
    theme_default: str = "dark"
    language: str = "en"


class EngineRoutingSection(BaseModel):
    """Knobs for the multi-provider fallback engine (provider == "auto").

    Mirrors the freellmapi router. `sticky_sessions` keeps a conversation
    on one model for 30 min; `max_retries` bounds the fallback loop;
    `penalty_per_429` / `decay_interval_s` tune how fast a rate-limited
    model sinks and recovers in the priority chain.
    """
    enabled: bool = True
    sticky_sessions: bool = True
    max_retries: int = 6
    penalty_per_429: int = 3
    decay_interval_s: int = 120
    # First-token deadline (seconds): abandon a routed model that hasn't
    # streamed its first token within this budget and fall to the next. 0
    # disables (unbounded, legacy). This is the primary "stuck" guard.
    first_token_deadline_s: float = 7.0


class LLMSection(BaseModel):
    """LLM provider + model selection.

    Providers: `auto` (multi-provider fallback engine — routes to the best
    available model across every configured provider), `ollama` (local),
    `openrouter` and `nvidia` (cloud, OpenAI-compatible). Under `auto`,
    keys live in the encrypted DB keystore (Settings → Providers), not the
    legacy per-provider fields below. Per-provider auth fields stay for the
    single-provider modes; the active selection is driven by `provider`.
    """
    provider: str = "ollama"

    # Multi-provider routing config (used when provider == "auto").
    routing: EngineRoutingSection = Field(default_factory=EngineRoutingSection)

    # Active model selections — used by the LLMClient for the configured
    # provider. When switching provider, also update these to model names
    # the target provider recognises (e.g. `anthropic/claude-3.5-sonnet`
    # for OpenRouter, `meta/llama-3.1-70b-instruct` for NVIDIA).
    model: str = "llama3.1:latest"
    code_model: str | None = None
    vision_model: str | None = None
    classifier_model: str | None = None
    # Live Listen answer model. Set to a FAST inference provider's model id
    # (e.g. Groq "llama-3.3-70b-versatile" or Cerebras
    # "llama-4-maverick-17b-128e-instruct") so interview answers start
    # streaming in well under a second. The auto-router tries this model
    # first and falls back to the normal chain if it has no usable key.
    live_model: str | None = None
    # Live Listen latency guards. Interview answers must be FAST and focused,
    # not 10k-token essays — and a stalled/rate-limited free model must never
    # hang the turn at "Thinking". `live_max_tokens` caps the answer length;
    # `live_first_token_timeout` aborts (with a clear error) if no first token
    # arrives in time, instead of waiting out the full provider timeout (120s).
    live_max_tokens: int = 4000
    # 12s let a rate-limited free model stall the whole answer (live-latency
    # report 2026-07-08); 6s fails over to the next model twice as fast while
    # still tolerating normal provider jitter.
    live_first_token_timeout: float = 6.0
    # Use the full chat-quality answer prompt on the live path (detailed
    # answers) instead of the terse real-time one. Length is still bounded by
    # live_max_tokens / forced_depth, but structure + depth match chat.
    live_detailed: bool = True

    # Ollama-specific endpoint. Used only when provider == "ollama".
    base_url: str = "http://localhost:11434"

    # OpenRouter (https://openrouter.ai). OpenAI-compatible. Free tier
    # exists for some models; most need credits.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # NVIDIA NIM (https://build.nvidia.com). OpenAI-compatible chat
    # completions endpoint. API key from the NVIDIA developer portal.
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    temperature: float = 0.3
    max_tokens: int = 10000
    timeout_seconds: float = 120.0
    fallback_model: str | None = None
    # Mid-stream output guards (2026-07-09): incremental <think>/harmony
    # scrubbing + repetition/ceiling kill switch on every visible stream.
    stream_guard: bool = True
    stream_max_chars: int = 120_000
    repetition_max_repeats: int = 3
    # Wall-clock ceiling for one chat turn's SSE stream (0 = unlimited).
    chat_stream_budget_s: float = 300.0


class EmbeddingsSection(BaseModel):
    provider: str = "sentence_transformers"
    model: str = "BAAI/bge-m3"  # 1024-dim; must match pgvector column dimension
    device: str = "cpu"


class VectorStoreSection(BaseModel):
    provider: str = "pgvector"
    collection: str = "resumes"
    persist_dir: str = "./data/vectors"


class RerankerSection(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"


class RAGSection(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    hybrid_search: bool = True


class STTSection(BaseModel):
    # "local" (Parakeet/Qwen-ASR chain) or "cloud" (Groq Whisper API). The
    # Settings toggle flips this; local enables the model dropdown.
    mode: str = "local"
    cloud_model: str = "whisper-large-v3-turbo"
    # DEFAULT is Qwen3-ASR — high-accuracy multilingual local STT for the GPU
    # deploy. On a small machine switch to "parakeet" (fast, tiny) via Settings.
    provider: str = "qwen_asr"
    # Local fallback chain: providers tried in order when the primary fails
    # (load error, OOM, mid-session crash). Keeps a fully-local STT deploy
    # alive with no cloud/API engines. e.g. ["parakeet"].
    fallback_providers: list[str] = []
    model: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5
    cpu_threads: int = 4   # faster-whisper CPU threads (4 is optimal here)
    # Qwen3-ASR (provider "qwen_asr") — local multilingual STT (30 languages),
    # GPU-preferring (bf16 on CUDA, fp32 on CPU). qwen_language null = autodetect.
    qwen_model: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_language: str | None = None
    # Parakeet TDT v3 (provider "parakeet") — local multilingual STT
    # (25 languages) via onnx-asr; int8 CPU inference runs ~8x realtime.
    parakeet_model: str = "nemo-parakeet-tdt-0.6b-v3"
    parakeet_quantization: str | None = "int8"
    # GPU-first Parakeet: load fp32 on onnxruntime's CUDAExecutionProvider
    # when available (tens of ms per utterance vs seconds on CPU); any GPU
    # failure falls back to the int8 CPU model transparently.
    parakeet_use_gpu: bool = True
    # STREAMING partials: while the speaker is still talking, the growing
    # utterance is transcribed by this FAST local provider and interim text
    # streams to the UI. "" disables partials. The final (authoritative)
    # transcription still uses `provider` + `fallback_providers`.
    partial_provider: str = "parakeet"
    # Reuse a completed partial as the FINAL transcript when it covered
    # exactly the finalized audio and the partial engine IS the final
    # engine (same model + same samples = same text) — skips the redundant
    # re-transcription on the answer's critical path.
    final_from_partial: bool = True
    # Architecture.md §"Dual-STT redundancy" — when True, transcribe()
    # fans out to a second engine and arbitrates. Off by default so
    # single-engine deploys are unaffected.
    dual_engine_enabled: bool = False
    secondary_provider: str | None = None      # e.g. "parakeet" — TODO
    # Architecture.md §"Vocabulary boosting" — feed an initial_prompt
    # to Whisper composed from resume + COPILOT.md + session terms.
    vocab_boost_enabled: bool = True
    # Also pass the booster's term list as faster-whisper `hotwords`
    # (a stronger bias than initial_prompt) for domain-name accuracy.
    hotwords_enabled: bool = True
    # Cloud STT fallback chain (used when provider is "groq"/"cloud"/"openai").
    # Each entry: {platform, model, base_url?}. Default = Groq turbo -> Groq
    # large-v3. Cloud-only: no local fallback. e.g.
    #   - {platform: groq, model: whisper-large-v3-turbo}
    #   - {platform: openai, model: gpt-4o-transcribe}
    cloud_chain: list | None = None
    # Whisper biasing prompt (OpenAI/Groq `/audio/transcriptions` `prompt`
    # field). Keep it SHORT and neutral: a long keyword list biases the
    # recogniser toward those exact words and causes hallucinated terms. A
    # brief domain hint is enough; the live path appends the session's recent
    # questions so it adapts to the actual interview. ~224-token cap.
    prompt: str | None = (
        "A technical software-engineering interview. The interviewer asks "
        "clear questions about computer science, programming, and system "
        "design."
    )


class VisionSection(BaseModel):
    """Local Vision Intelligence Layer (VisionAnalysis.md).

    A LOCAL vision model is the universal image/screenshot/document reader:
    it PARSES an image into a structured text representation ONCE (OCR text +
    layout + tables + a short description), which is then handed to whichever
    TEXT provider LLM answers the turn. No API/provider VISION model is ever
    used — provider models only ever receive the extracted TEXT. Mirrors the
    STT chain: one model resident at a time, chosen via a dropdown, lazily
    downloaded from HuggingFace on first use, GPU-first with CPU fallback.
    """
    # Master switch. On → every image is parsed to text before it reaches the
    # answer provider. Off → the raw-image path is used (legacy).
    enabled: bool = True
    # "local" (on-device VLM chain — the dropdown picks the model) or "cloud"
    # (send the image to a vision-capable provider LLM for extraction). Settings
    # toggle. Cloud reverses the "images never leave the machine" default — the
    # image still persists in Postgres; it's sent ONLY to the vision model, never
    # to the answer model.
    mode: str = "local"
    # CLOUD-mode upload tuning. The hosted vision model tiles an image down to
    # ~1568px internally, so uploading a full 1080p/4K screenshot just wastes
    # upload + prefill time. We downscale to `cloud_max_side` and JPEG-encode at
    # `cloud_jpeg_quality` before sending (the FULL image still persists in
    # Postgres — only the copy sent to the model shrinks). `cloud_max_tokens`
    # caps the extraction so the model returns a tight transcription, not a long
    # description (default llm.max_tokens is 10k). Local mode ignores these.
    cloud_max_side: int = 1568
    cloud_jpeg_quality: int = 85
    # The cloud read now only transcribes + names the language (the SOLVER does
    # the solving), so a tighter output cap ends it sooner. Was 1500.
    cloud_max_tokens: int = 1000
    # How many DIFFERENT vision models to try before giving up on cloud and
    # falling back to the local chain — a flaky/empty free model is retried on a
    # different one so cloud mode never silently returns nothing. Was 4.
    cloud_retries: int = 3
    # Per-attempt DEADLINE (seconds) for a single cloud vision read. Without it,
    # a slow/hanging free vision model blocks for the GLOBAL llm.timeout_seconds
    # (120s) × retries — the "Reading image is taking longer" stall. A tight cap
    # abandons a stuck model fast and rotates to a different one.
    cloud_attempt_timeout: float = 28.0
    # OCR sizing (RapidOCR): upscale a small capture to at least `ocr_min_side`
    # on the long edge so a tiny "<Lang> Auto" language chip is legible; clamp a
    # giant capture down to `ocr_max_side` so CPU stays bounded.
    ocr_min_side: int = 1600
    ocr_max_side: int = 2600
    # Extra latency weight applied to VISION model selection (vision-scoped
    # latency-aware routing) so the fastest CAPABLE vision model wins.
    vision_latency_weight: float = 0.2
    # A STRICT read-only extraction prompt used when the image looks like code:
    # read the selected-language chip + transcribe code EXACTLY, never solve/
    # complete/translate it (stops a capable model hallucinating a solution in
    # the wrong language). Falls back to `prompt` when empty.
    code_prompt: str = (
        "This is a screenshot of a coding site or IDE (e.g. LeetCode, HackerRank, "
        "VS Code). Your reply MUST BEGIN with one line exactly: 'Language: "
        "<name>'. Determine <name> by reading the editor's LANGUAGE SELECTOR — a "
        "small dropdown/label at the TOP of the code editor panel (often top-left "
        "or top-right, shown next to a small chevron '⌄', lock, or 'Auto', e.g. "
        "'Swift', 'Java', 'Python3', 'C++', 'Dart', 'Go'). Read that chip "
        "EXACTLY — do NOT guess and do NOT assume Python. Look carefully even if "
        "the chip is small. If no selector is visible at all, infer the language "
        "from the starter-code syntax and the file extension and name your best "
        "reading. After that first line, transcribe ONLY the coding problem and "
        "its code editor — the problem statement, the examples/constraints, and "
        "the code stub verbatim (preserve the exact class/method names, types and "
        "signature). IGNORE everything unrelated: the desktop, taskbar, browser "
        "tabs/chrome, other windows, menus, and side panels — do not describe "
        "them. Be concise: no commentary. Do NOT write, complete, translate, fix, "
        "or invent any code, and do NOT solve the problem — only transcribe the "
        "problem and its starter code."
    )
    # Routing tier for the cloud extraction call. "trivial" = speed-only (the
    # fastest vision model wins) — cloud mode exists to be FASTER than the local
    # VLM. Bump to "standard" if a fast model mis-reads dense screenshots.
    cloud_difficulty: str = "trivial"
    # Active local vision engine. Registered in app/vision/factory.py.
    # DEFAULT is Qwen2.5-VL (high-accuracy) for the 24GB GPU-server deploy. On a
    # small/laptop GPU this won't fit — the pre-flight memcheck refuses it (fails
    # OPEN, no crash); there, switch to "smolvlm_500m" (Settings → Vision, or set
    # `provider: smolvlm_500m`). memcheck always guards an impossible load.
    provider: str = "qwen2_5_vl"
    # Local fallback chain tried in order when the primary fails (load error /
    # OOM / mid-turn crash) — keeps a fully-local vision deploy alive. Empty by
    # default: the bigger fallbacks only fit larger machines (memcheck-gated).
    fallback_providers: list[str] = []
    # SmolVLM-500M (provider "smolvlm_500m") — the tiny, fast, fits-everywhere
    # default reader. SmolVLM-2.2B (provider "smolvlm") — more accurate, needs a
    # bigger GPU. Both use the standard transformers image-text-to-text API.
    smolvlm_small_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    smolvlm_model: str = "HuggingFaceTB/SmolVLM-Instruct"
    # Qwen2.5-VL (provider "qwen2_5_vl") — highest-accuracy local VLM. The 7B is
    # the GPU-server default; with `qwen_vl_load_8bit` it needs ~8 GB VRAM (vs
    # ~16 GB bf16), fitting a 24 GB card alongside STT. Use the 3B
    # ("Qwen/Qwen2.5-VL-3B-Instruct", ~7 GB bf16) on smaller GPUs.
    qwen_vl_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # Load the VLM in 8-bit (bitsandbytes) — halves VRAM so the 7B fits 24 GB
    # comfortably. Falls back to bf16 automatically if bitsandbytes is absent
    # (e.g. the CPU-only desktop build). Set false to force full precision.
    qwen_vl_load_8bit: bool = True
    # MiniCPM-V (provider "minicpm_v") — 8B; only for high-VRAM machines.
    minicpm_model: str = "openbmb/MiniCPM-V-2_6"
    # GPU-first: load on CUDA (bf16/fp16) when available — hundreds of ms vs
    # seconds on CPU; any GPU failure falls back to CPU transparently.
    use_gpu: bool = True
    # Reply-length cap for the extraction pass. The vision model DESCRIBES /
    # transcribes; it never answers the user, so this stays modest for speed.
    max_new_tokens: int = 512
    # Combine a dedicated OCR pass (RapidOCR on onnxruntime, provider
    # "rapid_ocr") with the VLM so text extraction is crisp even when the VLM
    # skips a region (dense code, a language chip). Runs CONCURRENTLY with the
    # VLM (no added latency); local-only. Off keeps it VLM-only.
    ocr_enabled: bool = True
    ocr_provider: str = "rapid_ocr"
    # Per-image cache: the structured representation is keyed by image hash, so
    # follow-up questions about the SAME screenshot/document skip the vision
    # stage entirely (VisionAnalysis.md "cache the vision output").
    cache_enabled: bool = True
    cache_max_entries: int = 128
    # The instruction that turns the VLM into a PARSER, not an answerer. Kept
    # task-agnostic so one representation serves chat, code, docs and live.
    # A TIGHT transcription prompt, deliberately: a verbose "describe/analyze"
    # instruction makes small local VLMs ramble and hallucinate (fabricated
    # numbers, fake "analysis"). Constraining them to pure transcription keeps
    # the parse faithful AND fast (fewer output tokens → lower latency).
    prompt: str = (
        "Transcribe all text in this image exactly as written, in reading "
        "order. Preserve layout with markdown — tables for tables, fenced code "
        "blocks for code. If a programming language is shown or selected (a "
        "language selector, a code editor, or code), name that language "
        "explicitly. If a non-text visual is present (chart, diagram, photo, "
        "UI control), add one short line naming it and what it shows. Output "
        "only what is visibly present. Do NOT summarize, analyze, answer, "
        "calculate, explain, or invent anything not shown."
    )


class AudioSection(BaseModel):
    source: str = "mic"               # system_loopback | mic | both
    sample_rate: int = 16000
    chunk_ms: int = 500
    vad: str = "silero"
    vad_threshold: float = 0.5
    endpoint_silence_ms: int = 800
    # Adaptive endpointing: a SHORT utterance (< min_utterance_ms of speech so
    # far, e.g. "What") is likely mid-question, so it needs a LONGER, clearly
    # intentional pause (short_utterance_gap_ms) before we finalize — this is
    # what stops "What <breath> is microservices?" being cut to just "What".
    # A longer utterance finalizes on the normal endpoint_silence_ms.
    min_utterance_ms: int = 700
    short_utterance_gap_ms: int = 1200
    # Hard ceiling on a single utterance: force-emit (transcribe) once speech
    # has run this long WITHOUT a natural silence gap. Without it a long,
    # gap-free question (or a noisy room that never dips below the VAD
    # threshold) would accumulate forever and never transcribe.
    max_utterance_ms: int = 15000
    # Streaming-partial cadence: don't emit interim transcripts before the
    # utterance is partial_min_ms long, then re-transcribe the growing buffer
    # at most every partial_interval_ms (see stt.partial_provider). With GPU
    # Parakeet (~50-120ms per pass) a tight cadence is essentially free and
    # gives the live-caption feel + early-finalize signal.
    partial_min_ms: int = 400
    partial_interval_ms: int = 500
    # END-OF-SPEECH partial: once this much trailing silence accumulates, one
    # extra partial is snapshotted immediately (bypassing the interval), so
    # the trailing '?' that unlocks the early endpoint gap + speculative
    # answering is seen fresh instead of up to partial_interval_ms stale.
    end_partial_trailing_ms: int = 160
    # Silence gap (ms) that ends an utterance whose latest partial reads
    # grammatically INCOMPLETE ("Can you tell me" — even with an ASR-guessed
    # '?'): the speaker paused mid-thought, so wait longer than the normal
    # endpoint before finalizing the fragment.
    incomplete_gap_ms: int = 1200
    # PRE-ROLL: how much pre-speech audio is prepended when an utterance
    # starts, so a soft onset the VAD only caught mid-word isn't clipped.
    preroll_ms: int = 240
    # VAD RE-ENTRY: within this window after prior speech, the start
    # threshold relaxes by vad_reentry_delta (floored at the end threshold)
    # so a quieter continuation ("…tell me <pause> in spring boot") still
    # opens the gate instead of being silently discarded.
    vad_reentry_window_ms: int = 3000
    vad_reentry_delta: float = 0.15


class QuestionDetectionSection(BaseModel):
    use_llm_classifier: bool = True
    min_question_length: int = 5
    followup_window_seconds: int = 60
    followup_similarity_threshold: float = 0.7
    recent_q_window: int = 3


class CodeSolverSection(BaseModel):
    default_language: str = "python"
    # When a coding SCREENSHOT's language can't be read AND the user named none,
    # ASK which language instead of silently solving in `default_language`
    # (which produced "solved in Python when Swift was selected in the editor").
    # False → fall back to `default_language` silently, as before.
    ask_when_language_unknown: bool = True
    # ── Differential (reference) testing ──────────────────────────────────────
    # After a solution PASSES the visible examples, optionally stress-test it
    # against an LLM-written brute-force reference on random inputs, to catch the
    # hidden/edge-case bugs the 1-3 visible examples miss. A found counterexample
    # only HARDENS the solution (a repair that ALSO passes the visible examples);
    # a correct solution is never downgraded to a false failure.
    differential_testing: bool = True
    # Always run it, vs. only when visible-example coverage is THIN (few cases) so
    # well-covered problems stay fast — the "fast gate, then differential when it
    # helps" policy.
    differential_always: bool = False
    differential_thin_examples: int = 2   # auto-run when visible examples ≤ this
    differential_cases: int = 50          # random inputs the fuzz harness tries
    # ── Complexity check (opt-in, ADVISORY only) ──────────────────────────────
    # Piggybacks on the differential harness: time the candidate on a large vs a
    # smaller generated input and, if the measured growth looks far worse than
    # the claimed Big-O, append a SOFT advisory note (never a failure, never a
    # repair). OFF by default — container timing is noisy, so it's opt-in.
    complexity_check: bool = False
    complexity_flag_ratio: float = 3.0    # observed_ratio > factor×this → advise
    include_complexity: bool = True
    include_edge_cases: bool = True
    include_alternatives: bool = True
    # Two-step image solve: OCR with vision model, then reason with code model.
    two_step_solve: bool = True
    ocr_max_tokens: int = 1500


class WebSearchSection(BaseModel):
    provider: str = "duckduckgo"
    max_results: int = 5


class GitWorkflowSection(BaseModel):
    """P2-7 — git workflow for the chat agent path.

    `enabled` (on) → after a successful build/edit, commit the change on a fresh
    feature branch with an auto-written message (always local + safe). `auto_push`
    + `open_pr` (off) → push to `origin` and produce a PR link; they only do
    anything when the workspace has a remote configured and (for HTTPS) a token.
    The token is read from here or the `ZAPTHETRICK_GIT_TOKEN` env var.
    """
    enabled: bool = True
    branch_prefix: str = "zapthetrick"
    auto_push: bool = False
    open_pr: bool = False
    token: str = ""


class UISection(BaseModel):
    themes_path: str = "./themes/"
    show_confidence: bool = True
    show_token_count: bool = False


class ServerSection(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    ws_path: str = "/ws/live"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class PostgresSection(BaseModel):
    """Postgres connection config.

    Intentionally has NO connection defaults — host/db/user/password
    start empty so a fresh install can't accidentally hit a stray
    local Postgres. The user configures everything via Settings →
    Database in the UI; `POST /api/settings` runs schema-create +
    migrations on save.

    Non-connection settings (pool sizing, extensions) keep sensible
    defaults — the user only fills the fields that actually identify
    their database.
    """
    host: str = ""
    port: int = 5432                      # standard Postgres port
    db: str = ""
    # Postgres schema (search_path). Created on demand if missing
    # (see `ensure_schema_exists`). Empty means "public".
    schema_name: str = ""
    user: str = ""
    password: str = ""
    password_ref: str = ""                # secure-storage handle
    pool_min: int = 5
    pool_max: int = 20
    enable_age: bool = True               # Apache AGE graph extension
    enable_pg_search: bool = False        # pg_search BM25; defaults off until image bakes it


class CacheSection(BaseModel):
    backend: str = "dragonfly"            # dragonfly | redis | memory
    url: str = "redis://localhost:6379"
    default_ttl_seconds: int = 3600


class StorageSection(BaseModel):
    blobs_backend: str = "postgres"       # postgres | filesystem | minio
    blobs_path: str = "./data/blobs"      # used by filesystem + the pg read-fallback
    minio_endpoint: str | None = None
    minio_access: str | None = None
    minio_secret: str | None = None
    minio_access_ref: str | None = None
    minio_secret_ref: str | None = None


class MigrationsSection(BaseModel):
    auto_apply: bool = True
    migrations_dir: str = "./storage/migrations"


class BackupSection(BaseModel):
    enabled: bool = True
    schedule_cron: str = "0 3 * * *"      # nightly 3 AM
    retention_days: int = 30
    target_dir: str = "./data/backups"


class DatabaseSection(BaseModel):
    """Per DataBaseArchitecture.md, the database section covers the
    full stack (Postgres/pgvector + cache + blob storage + migrations
    + backup). Each sub-section drives one factory in [app.storage.*];
    flipping a backend is a single config change. All vectors live in
    Postgres via pgvector — there is no separate vector database.
    """
    postgres: PostgresSection = Field(default_factory=PostgresSection)
    cache: CacheSection = Field(default_factory=CacheSection)
    storage: StorageSection = Field(default_factory=StorageSection)
    migrations: MigrationsSection = Field(default_factory=MigrationsSection)
    backup: BackupSection = Field(default_factory=BackupSection)


# ---- Architecture.md §9 sections ------------------------------------
# Drive the multi-agent mesh, advanced RAG, self-learning, and the new
# UI surfaces. Defaults match the spec; per-deployment overrides live in
# config.yaml.


class AgentsEnabled(BaseModel):
    supervisor: bool = True
    planner: bool = True
    clarifier: bool = True
    retriever: bool = True
    memory: bool = True
    persona: bool = True
    coder: bool = True
    vision: bool = True
    web: bool = False
    grounder: bool = True
    critic: bool = True
    reflector: bool = True
    suggester: bool = True


class AgentsPriorities(BaseModel):
    p0: list[str] = Field(
        default_factory=lambda: [
            "supervisor", "planner", "retriever", "persona", "coder", "grounder",
        ]
    )
    p1: list[str] = Field(default_factory=lambda: ["memory", "critic", "suggester"])
    p2: list[str] = Field(default_factory=lambda: ["reflector"])


class AgentsDeadlines(BaseModel):
    intent: int = 250
    plan: int = 200
    retrieve: int = 500
    first_token: int = 1500
    total: int = 8000
    p1_grace: int = 1500


class AgentsSection(BaseModel):
    enabled: AgentsEnabled = Field(default_factory=AgentsEnabled)
    priorities: AgentsPriorities = Field(default_factory=AgentsPriorities)
    deadlines_ms: AgentsDeadlines = Field(default_factory=AgentsDeadlines)


class AdvancedRagSection(BaseModel):
    """Advanced-RAG settings from Architecture.md §3.

    Lives alongside the existing `rag` section to avoid breaking the
    Phase-1 retriever's reads of `cfg.rag.top_k_retrieve`. New code
    should prefer the values here.
    """
    embedding_model: str = "BAAI/bge-m3"
    use_hierarchical_chunking: bool = True
    use_query_rewriting: bool = True
    use_hyde: bool = True
    use_contextual_compression: bool = True
    use_mmr: bool = True
    use_knowledge_graph: bool = True
    # Knowledge-graph "related concept" follow-up suggestions (Architecture §6):
    # extract entities/relations from the answer and suggest exploring the
    # topic's graph neighbours. Costs one extra LLM call per turn (runs before
    # `done`), so it's opt-in.
    kg_suggestions: bool = False
    small_to_big_retrieval: bool = True
    top_k_retrieve: int = 30
    top_k_rerank: int = 10
    top_k_final: int = 5
    compression_target_ratio: float = 0.4
    # Code knowledge graph: when a project archive (zip/tar/…) is uploaded,
    # parse it with tree-sitter and build a symbol/relationship graph, injecting
    # a project overview into the answer context (see app/codegraph/).
    use_code_knowledge_graph: bool = True
    code_graph_summary_chars: int = 6000
    # Workspace materializer (Phase 2): on a code-archive upload, extract it
    # into a real sandboxed project folder keyed by conversation so the chat
    # agent-run endpoint can read / edit / build / verify the actual codebase.
    materialize_workspace: bool = True
    # Phase 4.5 — security & abuse hardening for the chat agent-run path:
    #   max_concurrent_agent_runs — cap simultaneous workspace agent runs (a
    #     simple semaphore queue; extra runs wait their turn). Bounds CPU/mem on
    #     the single VPS.
    #   redact_agent_secrets — scrub secret-looking strings (keys/tokens/private
    #     keys/connection strings) from streamed + persisted agent step traces,
    #     so an uploaded `.env` / credential file is never echoed back.
    max_concurrent_agent_runs: int = 3
    redact_agent_secrets: bool = True
    # Phase 5 — context engineering (#1/#36): before an edit run, rank the
    # workspace's files by relevance to the task (codegraph centrality + name/
    # identifier overlap + entrypoint heuristics) and inject a budget-bounded
    # "relevant project context" preamble into the agent loop, so a large repo
    # fits a free model's context window and the agent starts on the right files.
    use_context_builder: bool = True
    context_budget_tokens: int = 6000
    # P2-3 — context engineering v2: a hierarchical repo digest (cached under
    # .zapthetrick/digest.md), recency-aware file ranking, and read compression
    # (signatures + relevant hunks) packed into the context preamble — plus a
    # cross-step scratchpad the agent jots findings to (the `note` tool) so a
    # fresh round of the long-horizon loop doesn't re-explore.
    #   context_v2          — use build_context_v2 (recency + compression) over
    #                         the Phase-5 builder for edit-run context.
    #   repo_digest         — prefix the context with the cached repo map.
    #   cross_step_scratchpad — carry working notes across goal-loop rounds.
    context_v2: bool = True
    repo_digest: bool = True
    cross_step_scratchpad: bool = True
    # P2-4 — long-horizon + live TODO checklist (TodoWrite parity):
    #   todo_list     — master switch for the live checklist (the `todo_write`
    #                   tool + the streamed `todo` SSE event are always wired;
    #                   this gates the upfront planner seed + injection).
    #   todo_planner  — before a build/edit run, ask the model to seed an
    #                   ordered checklist so the user sees a plan from step one.
    todo_list: bool = True
    todo_planner: bool = True
    # P2-5 — test-first rigor:
    #   test_first_rigor — inject the test-first directive into build/edit runs
    #                      (write tests for new/changed symbols; characterization
    #                      tests before a refactor) + fold test coverage of the
    #                      change into the confidence band. Non-blocking.
    #   strict_test_gate — opt-in: the goal loop won't mark 'done' while
    #                      added/changed symbols still have no test (bounded by
    #                      max_rounds; the final round is never blocked).
    test_first_rigor: bool = True
    strict_test_gate: bool = False
    # P2-6 — in-loop web tools: expose web_search (DuckDuckGo, no key) +
    # web_fetch (SSRF-guarded, untrusted-content wrapped) as agent tools the
    # loop can call mid-task. Off → the tools are hidden from the agent.
    agent_web_tools: bool = True
    # P2-10 — reliability + cognitive cache:
    #   cognitive_cache — reuse a stored completion for an identical low-temp
    #     request (prompt+options hash); a TTL+LRU process cache.
    #   prefer_recent_model — bias routing toward the model that most recently
    #     succeeded at a given difficulty (extends learning-lite into routing).
    cognitive_cache: bool = True
    cognitive_cache_ttl_s: int = 3600
    cognitive_cache_max: int = 512
    cognitive_cache_temp_ceiling: float = 0.5
    prefer_recent_model: bool = True
    # P2-11 — self-improvement on hard turns (free-only quality squeeze):
    #   self_improve — on an EXPERT agent turn, generate N candidate first
    #     actions from different free models and pick the best (self-consistency
    #     → judge). N× cost, so OFF by default; turn on for the quality squeeze.
    #   reflection_notes — append a one-line lesson from each run to the brain.
    self_improve: bool = False
    self_improve_n: int = 3
    reflection_notes: bool = True
    # P2-12 — untrusted-code safety (containment, not auth):
    #   content_safety  — refuse offensive-tooling requests (malware/keylogger/
    #     ransomware/phishing/…) while allowing defensive/educational security.
    #   injection_guard — scan uploaded code/docs for prompt-injection aimed at
    #     the agent and wrap the project context as untrusted data.
    content_safety: bool = True
    injection_guard: bool = True
    # Phase 8 — quality & trust on the chat agent-run path:
    #   red_team_review     — run an adversarial reviewer over the result
    #     (correctness/security/edge cases) and surface risks + feed confidence.
    #   surface_confidence  — compute + surface a confidence band + provenance.
    red_team_review: bool = True
    surface_confidence: bool = True
    # Phase 11 — project brain: a per-workspace `.zapthetrick/brain.md` the
    # agent reads each run (continuity) and appends decisions to (decision
    # ledger), plus a tiny per-workspace learning store that remembers which
    # model succeeded so future runs can be biased toward it.
    project_brain: bool = True
    # Phase 12 — multi-provider differentiator (B1/B2): after a chat agent-run,
    # have a DIFFERENT free model (or a council of N) independently judge whether
    # the result satisfies the task — cross-model verification a single-model
    # assistant can't do. council_size=1 → one cross-model verify (B1);
    # council_size>1 → a majority vote of N different models (B2). Disagreement
    # lowers the confidence band and is surfaced to the user.
    cross_model_verify: bool = True
    council_size: int = 1
    # Run the Clarifier + Grounder on the direct upload-stream path too, so a
    # file-upload turn gets the same clarification + hallucination-check as the
    # agent-mesh path (capability parity across the two answer pipelines).
    upload_quality_checks: bool = True
    # LLM-driven tool execution: let the agent dispatch registered tools
    # (web_search, code_search/callers/…) instead of leaving the registry idle.
    use_tool_executor: bool = True
    # Capability-aware routing: classify each turn's difficulty and route
    # hard/expert (computational) work to the strongest available model, plus a
    # rigor directive for correctness + elegance on demanding tasks.
    difficulty_aware_routing: bool = True
    # Self-refine pass on demanding turns: draft → verify (find errors) →
    # revise, computed in full BEFORE the first token, then faked as a stream.
    # OFF by default so every turn streams immediately (ChatGPT-like) — the
    # post-stream Critic still checks the answer. Enable + set `verify_levels`
    # only if you prefer correctness-over-latency on hard/expert turns.
    self_refine: bool = False
    verify_levels: list[str] = []
    # Max draft→verify→revise rounds for expert turns (hard turns always do 1).
    # Each round re-checks the latest draft on a DIFFERENT model and keeps a
    # substantive revision, so a genuinely hard problem keeps improving across
    # several models instead of one shot. Higher = better answers, more cost.
    # Default 1 keeps hard text turns fast (one verify→revise); raise for more
    # quality at the cost of latency. Image/Solve turns skip self-refine entirely
    # and stream directly for a fast first token.
    refine_rounds: int = 1
    # Per-step ceilings for the self-refine pass (seconds) — a slow provider
    # can't hang the turn; on timeout we use the draft we already have.
    verify_draft_timeout: float = 90.0
    verify_step_timeout: float = 45.0
    # Clarifier grace window (ms). The Clarifier runs CONCURRENTLY with the
    # answer (Option B): it normally only interrupts if it decides BEFORE the
    # first answer token. When the first token is ready but the Clarifier is
    # still deciding, we wait up to this long for its verdict before committing.
    # Confident cases (unspecified builds, planner-flagged turns) BLOCK instead,
    # so this only covers spontaneous clarifications — kept short so it adds
    # little latency to the many turns the gate ultimately declines. 0 disables.
    clarify_grace_ms: int = 900


class MemorySection(BaseModel):
    episodic_enabled: bool = True
    semantic_enabled: bool = True
    reflection_on_idle_minutes: int = 5
    reflection_on_session_end: bool = True
    user_owns_memory: bool = True

    # ── memory-graph spec (additive + flag-gated) ──────────────────────────
    # Structured, project-scoped Memory_Objects layered over episodic/semantic.
    # Off → today's episodic/semantic recall (Property 1/8).
    graph_enabled: bool = False
    # Inject relevance-ranked memories into the chat context (R3.2). Off by
    # default; hands the ranked set to the perceived-speed Context_Budget.
    inject_into_context: bool = False
    # Retrieval knobs (R3.2).
    retrieval_k: int = 6
    relevance_threshold: float = 0.35
    # Lifecycle (R4/R5): aging half-life (days), per-scope object cap.
    half_life_days: float = 30.0
    max_objects_per_scope: int = 500
    # Surface retrieved-memory provenance as additive `memory` meta.
    surface_memory: bool = False
    # §18 data lifecycle: purge episodes/skills older than this many days when a
    # retention sweep runs (0 = keep indefinitely — nothing is deleted silently).
    retention_days: int = 0


class LearningSection(BaseModel):
    online_critique: bool = True
    online_revision_threshold: float = 0.6
    feedback_capture: bool = True
    speculative_execution: bool = True
    # Fraction (0..1) of live turns sampled into the eval harness for accuracy /
    # regression tracking (Architecture §14). 0 = off.
    online_eval_sample_rate: float = 0.0
    # #12 answer-first gate v2: use the user's learned answerability (calibration
    # buckets) to upgrade a borderline DEFER to answer-first, cutting over-asking.
    answer_first_v2: bool = False


class UIAdvancedSection(BaseModel):
    layout_density: str = "comfortable"   # comfortable | compact
    show_tool_chips: bool = True
    show_source_citations: bool = True
    show_confidence: bool = True
    show_suggestions_rail: bool = True
    reduced_motion: str = "auto"          # auto | always | never
    font_scale: float = 1.0


class ThemesSection(BaseModel):
    default: str = "dark"
    available: list[str] = Field(
        default_factory=lambda: ["dark", "light", "midnight", "solarized"]
    )
    accent_color: str = "#7C5CFF"


class MCPSection(BaseModel):
    """Architecture.md §"MCP tool surface".

    Each entry is a server descriptor; the registry instantiates one
    per entry at startup. Most users won't touch this — the Tools
    screen handles add/remove via the registry route.
    """
    servers: list[dict] = Field(default_factory=list)
    auto_grant_low_danger: bool = True


class ResponseArchSection(BaseModel):
    """Architecture.md §"Response architecture"."""
    enabled: bool = True
    default_depth: str = "standard"   # tldr | standard | deeper | exhaustive


class ContinuitySection(BaseModel):
    """Architecture.md §"Conversation link graph"."""
    auto_link: bool = True
    auto_link_threshold: float = 0.65   # below this → suggest-confirm chip
    detect_followups: bool = True


class TechnicalPipelineSection(BaseModel):
    """Architecture.md §"Beyond DSA"."""
    enabled: bool = True
    # Force a particular domain (debug). None → classifier picks.
    force_domain: str | None = None


class ToolLoopSection(BaseModel):
    """Architecture §13 — iterative (mid-answer) tool use.

    Before the persona streams its answer, the model may call tools (compute /
    web_search / code_search / resume_lookup), see the UNTRUSTED result (§11),
    and call again — up to `max_rounds` — then answer with the evidence in
    context. Gated to `min_difficulty`+ (and the intent profile's tool allow-list
    when the registry is on) so trivial turns stay fast. Off by default.
    """
    enabled: bool = False
    max_rounds: int = 3
    # Minimum difficulty that unlocks the loop: trivial|standard|hard|expert.
    min_difficulty: str = "hard"
    # Default chat tool set when the intent profile doesn't constrain tools.
    tools: list[str] = Field(default_factory=lambda: [
        "code_solver", "web_search", "code_search", "resume_lookup"])


class CalibrationSection(BaseModel):
    """G1: outcome-driven threshold calibration. When on, thresholds like the
    semantic-intent `primary_threshold` adapt to observed good/bad outcomes
    instead of staying a hand-set literal. Off → static defaults. `min_samples`
    guards against swinging on thin data."""
    enabled: bool = False
    min_samples: int = 20


class PrivacySection(BaseModel):
    """§11/§18 privacy. `redact_egress` scrubs PII/secrets from messages just
    before they leave the device for a third-party LLM:
      off     — no redaction;
      secrets — API keys / tokens / private keys / credit cards / SSNs (default,
                safe: these are never the legitimate subject of a turn);
      strict  — also emails / phones / IPs (may reduce utility; opt-in).
    Deterministic + always-on (a safety property), fail-open."""
    redact_egress: str = "secrets"


class SynthesisSection(BaseModel):
    """Multi-model answer synthesis (Phase 3). On a composite/complex turn,
    decompose the answer into sections, route EACH to the free model best suited
    to it, and synthesize one coherent deliverable. Off by default → the normal
    single-model stream. Gated further to large/expert turns so simple asks stay
    fast + single-model."""
    enabled: bool = False
    max_sections: int = 5
    # Minimum output complexity that unlocks decomposition (small|medium|large).
    min_output_complexity: str = "large"
    # One critic pass over the merged result → a single revision when gaps found.
    self_eval: bool = False
    # G8: bound the fan-out — at most this many sections run at once (free-tier
    # rate-limit safety), and each section is capped at this many seconds.
    max_concurrency: int = 3
    section_timeout_s: int = 90


class UnderstandingSection(BaseModel):
    """Unified semantic Understanding pass (the 'brain'). One embedding of the
    turn drives intent + difficulty + task category + topic-shift + capabilities
    + output complexity, so the model router reads one coherent object instead of
    recomputing from keyword rules. Off by default → callers keep their existing
    per-signal paths; on → the semantic read is authoritative (fail-open)."""
    enabled: bool = False
    # Below this cosine between consecutive turns, the turn is an implicit
    # topic-shift (a subject change with no "new topic" phrase). bge-m3 scores
    # related follow-ups ~0.5-0.9; unrelated jumps drop well below.
    topic_shift_similarity: float = 0.35
    # Feed the Understanding's task_category + capabilities into the model router
    # (the "traffic controller"), and enable learned per-category success.
    route_from_understanding: bool = False


class IntentProfilesSection(BaseModel):
    """Architecture §4 — the Intent Profile Registry.

    Per-intent behavior profiles (which agents/graphs/tools, response shape,
    document eligibility, suggestion style). The code defaults live in
    `app/clarify/intent_profiles.py`; entries under `profiles` overlay them
    field-by-field, so a YAML author tweaks one field of one intent without
    restating the rest — e.g.::

        intent_profiles:
          enabled: true
          profiles:
            knowledge: {response_shape: prose, doc_eligible: false}
            debugging: {tools: [code_solver, code_search], suggestions: verify}

    Off by default: when off, `resolve()` still returns the code-default profile
    but no decision point reads it, so the pipeline is byte-for-byte unchanged.
    """
    enabled: bool = False
    profiles: dict[str, dict] = Field(default_factory=dict)


class ResilienceSection(BaseModel):
    """Architecture.md §15 — mid-stream failover ("always finishes").

    When a stream is cut off — the model stops with finish_reason 'length'
    (hit the output-token ceiling), or the transport drops after some tokens
    were already shown — re-prompt the next-best model with a *continuation
    contract* (the partial answer + "continue seamlessly") and de-dupe the
    seam, so the user sees one continuous answer.

    Off by default: it changes streaming behavior on the hot path, and a low
    `output_tokens` ceiling would make many normal turns end 'length' and get
    auto-continued (longer answers, more cost). Turn it on when you want the
    "always finishes" guarantee.
    """
    mid_stream_continuation: bool = False
    # Max continuation re-prompts per turn (bounds cost + latency).
    max_continuations: int = 2
    # Characters buffered at a continuation seam before de-duping the join.
    seam_buffer_chars: int = 200
    # Transient "no route" recovery: when a concurrent burst momentarily rate-
    # limits every model, `route_request` raises NoRouteAvailable. Instead of
    # erroring straight to the UI, the engine backs off (letting the 60s RPM/TPM
    # windows recover + de-syncing the burst) and retries route selection up to
    # `route_no_route_retries` times, `route_backoff_ms` (× attempt, + jitter)
    # apart. 0 retries → old behavior (immediate error).
    route_no_route_retries: int = 3
    route_backoff_ms: int = 400
    # Let the recovery planner drive the retry gate. It was already being
    # consulted on every provider failure and its verdict discarded — the gate
    # keyed purely on `exc.retryable`. With this on, the plan decides the backoff
    # (429s previously retried INSTANTLY) and whether to fall to a DIFFERENT model
    # instead of re-hitting the one that just failed. It also closes the
    # failure-KB learn loop: recovery outcomes are recorded, so the KB learns
    # which action actually works rather than only counting failures.
    # Off → the pure `exc.retryable` gate, i.e. today's behaviour.
    recovery_planner: bool = True
    # Cap on the plan's backoff. The planner's cooldown curve grows 1s→2s→4s, but
    # the engine falls to a different model anyway, so an uncapped wait would be
    # pure added latency on the hot path.
    recovery_backoff_max_ms: int = 1000
    # Reconnect/replay (§15): tag each SSE frame with an event id + tee it into a
    # bounded, TTL'd buffer so a dropped socket can replay what it missed. Purely
    # additive (no change to the live stream); on by default.
    replay_enabled: bool = True


class DocumentsSection(BaseModel):
    """Phase 10 — document intelligence (#9). Enhancements applied to generated
    PDF/DOCX/XLSX when the content warrants (gated by content heuristics too)."""
    # Number headings (1, 1.1, 1.1.1) + a Table of Contents on multi-section
    # PDF/Word documents (only when there are >= 3 headings).
    section_numbering: bool = True
    table_of_contents: bool = True
    # Excel: coerce numeric cells, add a SUM totals row + a bar chart per
    # data sheet that has a numeric column.
    excel_charts: bool = True
    # Strict generation gate: produce a downloadable document ONLY when the user
    # EXPLICITLY asks for a file (the deterministic detector fires). The LLM
    # triage classifier over-triggers ("here's a summary" → thinks doc), so when
    # this is True its `document:true` is not trusted on its own. Set False to
    # restore the looser LLM-or-explicit behavior.
    explicit_only: bool = True
    # Phase 1b — the export job manager's worker pool + per-render timeout.
    export_concurrency: int = 2
    export_timeout_s: float = 120.0
    # Bounded retry for a failed render. Only TRANSIENT failures are retried —
    # a deterministic one (unsupported format, TypeError, bad input) will fail
    # identically every time, so retrying it just burns the user's latency.
    # 0 disables retry entirely. Backoff is exponential from this base and is
    # interruptible: cancelling mid-backoff finishes the job CANCELLED and never
    # re-renders.
    export_max_retries: int = 1
    export_retry_backoff_s: float = 0.2
    # Re-emit an artifact the user already has instead of regenerating it.
    # DOWNLOAD_EXISTING ("where's the pdf?") was being classified correctly and
    # then ignored — the turn ran the full LLM generation anyway and re-created a
    # document the user already had. Falls through to normal generation when no
    # prior artifact exists, so an empty session still gets a guidance answer.
    reuse_existing_artifact: bool = True
    # DocGen roadmap — code-block linting inside review (ruff/eslint via
    # polyglot.linters); the staged multi-pass assembler (outline→content→
    # structure→format→validate); and the multi-reviewer LLM panel
    # (technical/grammar/formatting/consistency). The panel is real extra LLM
    # passes so it defaults OFF; lint + assembler are deterministic, default ON.
    code_lint_review: bool = True
    multi_pass_assembly: bool = True
    multi_reviewer: bool = False
    # Phase 4 — auto-enrich a rendered document with a TOC + glossary + smart
    # appendix + auto-diagram + figure/table numbering (model-driven). Default ON
    # so generated documents are actually structured; set False for the legacy
    # byte-identical (un-enriched) output.
    auto_structure: bool = True
    # Phase 3 — quality review. `quality_strict` makes the EXPORT endpoint block
    # (HTTP 422) when the deterministic review finds a hard error (empty section,
    # etc.); default OFF = warn-not-block (the report just rides a header).
    # `llm_review` layers a flag-gated LLM reviewer panel on top of the
    # deterministic checks; default OFF so no network call happens by default.
    quality_strict: bool = False
    llm_review: bool = False
    # Phase 1 — render the binary formats (PDF/DOCX/PPTX) directly from the
    # DocumentModel IR (single source of truth). Default ON; set False to fall
    # back to the legacy Markdown-tuple renderers if a regression appears.
    model_driven_render: bool = True
    # Phase 1b — execute the render in an isolated SUBPROCESS (crash/resource
    # isolation, hard timeout kills the process). Default OFF → render in-process
    # behind the async Job Manager (faster, no spawn cost).
    sandbox_render: bool = False
    # Functional post-render check (doc_verify): the rendered doc must contain at
    # least this fraction of the source's significant words, else it's flagged as
    # "dropped content". Only enforced when the source has ≥25 such words.
    verify_coverage_min: float = 0.6


class RoutingSection(BaseModel):
    """Multi-provider routing knobs."""
    # When True, the orchestrator routes across EVERY configured model that has
    # a usable key — including discovered ones the user never manually enabled —
    # ranked after the enabled fallback chain. Lets the full catalogue (hundreds
    # of models) serve any task, not just the curated/enabled few.
    route_all_models: bool = False
    # When True, route ONLY among free models whenever any free model is
    # available (paid models become a last resort). "Free" is detected from the
    # provider tier + OpenRouter `:free` suffix — no hardcoded model list.
    prefer_free: bool = True
    # Phase P2-1 — hybrid "strong tier": when True, the difficulties listed in
    # `strong_tier_difficulties` are allowed to use a NON-free (paid) strong
    # model instead of being held to free-only, so hard/expert turns get
    # frontier-grade quality while everything else stays free. Default OFF →
    # behavior is unchanged (free-first everywhere) until you opt in + add a
    # paid key. `monthly_paid_request_cap` caps paid requests per month
    # (0 = unlimited); over the cap, hard turns fall back to free-only.
    strong_tier_for_hard: bool = False
    strong_tier_difficulties: list[str] = Field(
        default_factory=lambda: ["expert"]
    )
    monthly_paid_request_cap: int = 0

    # ── intelligent-model-routing spec (all additive + flag-gated) ──────────
    # Per-category capability matching: adds a Task_Match term to the router
    # score. `capability_routing` turns it on; `task_match_weight` is the term's
    # weight (0 → byte-for-byte today's ranking, Property 2).
    capability_routing: bool = False
    task_match_weight: float = 0.0
    # Confidence-based escalation (R5): serve a faster model first, escalate to
    # a stronger one only on low aggregate confidence / failed verification.
    escalation: bool = False
    # Multi-model strategies + meta-router (R6/R7). Off → today's route_request.
    multi_model: bool = False
    meta_router: bool = False
    # Learning router (R8): bias ranking toward historically successful models
    # per task category. `learn_weight` is the additive term weight (0 = off).
    learning_router: bool = False
    learn_weight: float = 0.0
    # Phase 2 semantic learning: key the learned-success signal on the query's
    # EMBEDDING CLUSTER ("which model works on turns like this") instead of the
    # coarse task category. Reuses `learn_weight`; needs the Understanding pass
    # (its query embedding) — falls back to category-keyed when off/no embedding.
    semantic_learning: bool = False
    # Adaptive benchmarking (R9): rolling latency/success windows down-rank a
    # temporarily-degraded model.
    adaptive_benchmark: bool = False
    # Routing explainability (R10): emit the additive `route` trace meta.
    route_trace: bool = False

    # ── efficiency enhancements (2026-07-14; all additive + flag-gated) ──────
    # Latency-aware routing: fold each model's OBSERVED recent latency (p50 from
    # the perceived-health window) into the score, so genuinely fast models win
    # latency-sensitive turns instead of relying on the static speed_rank guess.
    # `latency_weight` is the additive term weight (0 → today's ranking).
    latency_aware: bool = False
    latency_weight: float = 0.0
    # A turn slower than this (seconds, p50) is treated as fully "slow" for the
    # latency term; faster is proportionally rewarded.
    latency_baseline_s: float = 8.0
    # Circuit breaker: after `circuit_fail_threshold` CONSECUTIVE hard failures
    # (provider errors / timeouts — NOT rate-limit 429s, which have their own
    # cooldown) a model is SKIPPED for `circuit_cooldown_s`, then half-opened so
    # one probe request can close it on success. Prevents wasting a call + retry
    # cycle on a model that's currently down. Off → today's behavior.
    circuit_breaker: bool = False
    circuit_fail_threshold: int = 3
    circuit_cooldown_s: float = 30.0
    # Cost mode — route to FREE models ONLY. Everything routes among free models;
    # a paid model is used only under the emergency policy below.
    free_only: bool = False
    # What free_only does when NO free model is routable right now (all rate-
    # limited / down):
    #   True  (default, GRACEFUL) → fall back to the best available PAID model as
    #          a one-off last resort so the request still succeeds; logged as a
    #          warning. You spend nothing in the normal case, but a transient
    #          free-tier outage doesn't break the app.
    #   False (STRICT) → fail with a clear "no free model routable" error and
    #          never spend a cent — the original hard guarantee.
    free_only_emergency_paid: bool = True
    # Proactive free-tier quota awareness. The quota ledger is already WRITTEN on
    # every dispatch; these make routing READ it, so we stop sending a request to
    # a provider we already know is spent (a guaranteed wasted round-trip + a
    # user-visible latency spike). Distinct from the reactive per-model 429
    # headroom above: that one only reacts AFTER the provider refuses us.
    #   quota_aware            → fold remaining headroom into the candidate score
    #   quota_skip_exhausted   → drop spent providers from the pool entirely
    # If every provider is spent the pool is restored untouched — availability
    # always beats the quota preference, so we degrade rather than fail.
    quota_aware: bool = True
    quota_weight: float = 12.0
    quota_exhausted_penalty: float = 250.0
    quota_skip_exhausted: bool = True


class PerceivedSection(BaseModel):
    """Perceived-speed knobs (perceived-speed spec). Everything defaults OFF so
    behavior is byte-for-byte today's until explicitly enabled.

    - `speculation_enabled` is the master kill-switch for ALL speculative work
      (prefetch/predictive-cache/drafting). When False, none of it runs (R19.4).
    - `max_concurrent_drafts` caps simultaneous speculative model generations.
    - `speculation_period_budget` caps speculative units per period (0 = off).
    - `connection_idle_timeout_s` releases idle pooled connections (R2.2).
    """
    speculation_enabled: bool = True   # latency batch 2026-07-11 (#4)
    max_concurrent_drafts: int = 2
    # 60 speculative units/hour (2026-07-12): racing every chat turn unbounded
    # exhausted free-tier keys.
    speculation_period_budget: int = 60      # 0 = no budget accounting / disabled
    speculation_period_seconds: int = 3600  # the accounting window
    connection_idle_timeout_s: float = 60.0
    # Predictive cache (R3) bound + similarity threshold for the semantic tier.
    predictive_cache_max_entries: int = 256
    cache_similarity_threshold: float = 0.95
    # Phase 2 — speculative multi-model drafting (R4). Off by default; even when
    # on it only runs while speculation is enabled + within budget.
    speculative_drafting: bool = True  # latency batch 2026-07-11 (#4)
    # Phase 2 — progressive context / incremental retrieval (R5/R6): overlap
    # retrieval with pre-generation prep. Off by default.
    progressive_context: bool = False
    # Phase 4 — background research follow-up (R15). Off by default; runs only
    # AFTER the initial answer has streamed.
    background_research: bool = False
    # Phase 5 #3/#10 — predictive context prefetch (precompute the query
    # embedding while typing) + the idle scheduler (summarize/embed/verify on
    # idle, pressure-gated). Default on; fail-open.
    predictive_prefetch: bool = True
    idle_work: bool = True
    # Stream pacing (R7): produce a concise-first reply when the provider's
    # first-byte latency exceeds this (seconds); 0 disables the concise-first path.
    slow_provider_threshold_s: float = 0.0
    # TTFT budget (R7.3): emit an immediate acknowledgment if the first token
    # would take longer than this (seconds); 0 disables the acknowledgment.
    ttft_ack_threshold_s: float = 0.0
    # Answer reuse cache (R14/R21): serve a previously-generated, revalidated
    # answer for an identical (per-user-scoped) prompt before re-generating, and
    # store completed answers for reuse. Off by default — flag-off is byte-for-
    # byte today's behavior (nothing served, nothing stored).
    answer_cache: bool = True          # latency batch 2026-07-11 (#4)
    # Latency observatory (R16): record per-stage TTFT telemetry on the live
    # path. Read-only (never affects a turn); off by default.
    observatory_enabled: bool = False


class FollowupSection(BaseModel):
    """Follow-up / conversation-state engine (followup-context-engine spec).
    Everything defaults OFF/neutral so behavior is byte-for-byte today's
    prompt-driven follow-ups until explicitly enabled (Property 1).

    The engine extends the existing `GoalLedger` record under
    `User.preferences.clarify_ledger[conversation_id]` — no schema migration —
    and every accumulated collection is bounded (R1.5/R4.3).
    """
    # Master switch. OFF → ConversationState load/observe/commit are no-ops and
    # the route uses today's continuity prompt.
    enabled: bool = False
    # Bounds for the accumulated collections (oldest/LRU eviction past these).
    max_decisions: int = 32
    max_constraints: int = 32
    max_entities: int = 64
    max_open_questions: int = 16
    max_enumerations: int = 12
    # Confidence gates (R2.3/R3.3/R12.3): at/above proceed, below defer/new-topic.
    followup_confidence_threshold: float = 0.6
    resolution_confidence_threshold: float = 0.6
    rewrite_confidence_threshold: float = 0.6
    # Phase 4 — emit the additive `interpretation`/`resolved_prompt` fields and
    # let the FE surface a dismissible "Understood as:" affordance (R11). Off by
    # default; purely additive when on.
    surface_interpretation: bool = False


class QualitySection(BaseModel):
    """Evaluation + reliability meta-layer (evaluation-and-reliability spec).
    Every runtime piece is flag-gated + fail-open → flags off = today's
    behavior byte-for-byte. The offline harness (scenario matrix + baseline) is
    dev/CI-only and needs no flag.
    """
    # Aggregate per-subsystem confidence into one band (R4). Additive meta only;
    # low band defers to the EXISTING clarifier (no new ask path).
    aggregate_confidence: bool = False
    # Grounder auto-correct (product decision 2026-07-08): when the fact-checker
    # flags unverified claims after the answer streamed, ONE bounded model pass
    # appends a short visible "Correction:" note — Claude's own pattern of
    # correcting itself in-turn rather than retro-editing streamed text. Never
    # in live mode (latency).
    grounder_autocorrect: bool = True
    # Tail-latency tuning (user report 2026-07-08: "stream sticks at the end
    # then resumes"): the verification pass is capped tighter and only runs
    # when enough claims were flagged to be worth the wait; a `verifying`
    # chip is emitted so the pause is visible work, not dead air.
    grounder_autocorrect_timeout_s: float = 3.5
    grounder_autocorrect_min_claims: int = 2
    # Mid-stream quality control (Phase 5 #13). Samples the partial response on a
    # char cadence (NOT per token) and flags refusal leakage / error spikes, and
    # stops only unpunctuated token-level degeneration — the one failure the
    # sentence-based `llm.stream_guard` is structurally blind to. Deliberately
    # conservative: a false stop on a good answer costs more than a missed catch,
    # so everything except high-confidence degeneration is flag-only.
    stream_control: bool = True
    # Request governor: pick a fast vs deep pipeline from difficulty (R5).
    governor: bool = False
    # Graceful degradation guard around non-critical subsystems (R6).
    degradation: bool = False
    # Non-blocking response-quality critic (R7).
    critic: bool = False
    # Emit the additive `aggregate_confidence`/`pipeline`/`degraded`/`quality`
    # SSE meta so the FE can surface them (R11). Off by default.
    surface_meta: bool = False


class WorkspaceSection(BaseModel):
    """Workspace + artifact runtime (workspace-and-artifacts spec). Additive +
    fail-open: with these off, the Default_Workspace path is byte-for-byte
    today's behavior and no artifact is ever created.
    """
    # Master switch for workspace grouping (default workspace is transparent).
    workspace_enabled: bool = False
    # Artifact runtime: substantial structured outputs become addressable,
    # versioned, editable artifacts reusing the doc generators + preview panel.
    artifacts_enabled: bool = False
    # Retained versions per artifact (oldest-first eviction, R5.3).
    max_artifact_versions: int = 20
    # Min answer length (chars) for an auto-artifact (an explicit request always
    # qualifies regardless of length).
    artifact_min_chars: int = 400
    # Emit the additive `artifact` SSE meta so the FE can surface the panel.
    surface_artifact: bool = False


class OrchestrationSection(BaseModel):
    """Multi-agent orchestration (agent-orchestration spec). Opt-in + flag-gated:
    off → today's single agent/answer path. All execution stays in the existing
    workspace sandbox under the existing concurrency/step/resource caps.
    """
    enabled: bool = False
    # Phase-5: PlannerAgent uses the deterministic decomposer for multi-goal
    # requests (single-goal turns keep the legacy linear plan). Independent of
    # `enabled` (which gates the full role-workflow surface).
    planner_decompose: bool = True
    # Bounds (on top of the existing max_concurrent_agent_runs / max_steps).
    max_subtasks: int = 6
    max_roles: int = 4
    max_tools: int = 8
    max_iterations: int = 3
    # Sandbox-verify generated code + auto-generate tests (Phase 3). Off by
    # default — they add runtime; honest verification status when off.
    sandbox_verify: bool = False
    generate_tests: bool = False
    # Emit the additive orchestration step events (workflow/role/tool/test).
    surface_orchestration: bool = False


class PersonalizationSection(BaseModel):
    """User-modeling + topic-risk policy + analytics (personalization-and-
    governance spec). Additive + flag-gated: neutral user + general topic =
    today's behavior. No second blocking LLM call; safety guards keep precedence.
    """
    # Infer + persist a per-user model (expertise/verbosity/style/frustration).
    user_model_enabled: bool = False
    # Feed expertise/style/verbosity into the existing answer-depth mechanic.
    adapt_depth: bool = False
    # Raise/decay a frustration signal → prefer concise/direct + fewer questions.
    frustration_detection: bool = False
    # Topic-risk policy gate: ADD caveats / "consult a professional" on sensitive
    # domains (composes with, never weakens, the existing safety guards).
    topic_policy: bool = False
    # Read-only analytics/audit view (dev build; aggregates existing telemetry).
    analytics_enabled: bool = False
    # Emit the additive `user_model`/`policy` SSE meta for the FE.
    surface_meta: bool = False
    # §17: inject the user's own standing instructions (User.preferences
    # ['custom_instructions']) into every turn, below the safety boundary. The
    # real gate is whether the user set any text; off disables it entirely.
    custom_instructions: bool = True
    # §6 Claude-style embedded follow-up: the answer ends with ONE short
    # contextual follow-up sentence generated inside the response (replaces
    # the templated suggestion chips in the UI).
    embedded_followup: bool = True


class LiveSection(BaseModel):
    """Live conversational intelligence (live-conversational-intelligence spec).
    Additive + flag-gated + fail-open: with every flag off the Live module
    behaves byte-for-byte as today (transcribe -> agent.predict -> answer). No
    new DB schema (per-session state is in-process); no second blocking LLM
    call (the event typer reuses the single agent.predict call).

    Phase 1 — structured events + interview state machine + turn-taking.
    """
    # Type each finalized utterance as a structured event and split multi-
    # question / boundary deterministically over the single agent.predict call.
    structured_events: bool = False
    # Maintain an explicit per-session interview state machine (never gates the
    # existing concurrency) and emit an additive {"type":"state"} frame.
    state_machine: bool = False
    # Hold a finalized utterance for a settle window so continued speech merges
    # into the same question instead of being answered twice.
    turn_taking: bool = False
    # End-of-turn settle window (ms) for turn-taking. 0 disables the wait.
    turn_settle_ms: int = 600
    # RETROACTIVE continuation merge: a fragment arriving within
    # continuation_window_s of a committed question that reads as its TAIL
    # ("in spring boot") cancels the in-flight answer and re-answers the
    # MERGED question — heals any premature commit a settle window missed.
    continuation_merge: bool = True
    continuation_window_s: float = 8.0
    # Hypothetical / assumption scenario probes ("Suppose the DB goes
    # down.") are answered even with no wh-word or '?' — detected
    # deterministically and promoted by the decision engine.
    hypothetical_question: bool = True
    # Tone-based promotion: a strong terminal pitch RISE on a multi-word
    # utterance answers even when the text reads like a statement (the
    # interviewer's delivery asked the question).
    prosody_promotion: bool = True
    # ACCURACY LEDGER: log every utterance decision (answered/skipped/
    # promoted/forced) + user feedback to an append-only JSONL, so detection
    # accuracy is measured on real sessions instead of guessed.
    accuracy_ledger: bool = True
    ledger_path: str = "data/live_ledger.jsonl"

    # Phase 2 — conversation memory graph.
    # Per-session topic tree + drift detection (widens the context tracker).
    topic_graph: bool = False
    # Multi-level memory (L1 recent / L2 current-topic / L3 rolling summary).
    multi_level_memory: bool = False

    # Phase 3 — deliberation before answering.
    # Detect the interview phase (introduction/technical/system-design/...).
    phase_detection: bool = False
    # Shape the answer by question type + phase (STAR / design / coding / ...).
    answer_strategy: bool = False
    # Fold an ordered answer outline into the same generation call.
    answer_planning: bool = False
    # Opt-in heavier two-step plan->generate (adds a call; default off).
    answer_planning_two_step: bool = False
    # Cap to concise + hedged when answer confidence < knowledge_gap_threshold.
    knowledge_gap_guard: bool = False
    knowledge_gap_threshold: float = 0.5
    # Fast vs deep answer path from the predicted difficulty / latency health.
    adaptive_latency: bool = False

    # Phase 4 — robustness, signals, event log.
    # Conservative domain-term transcript repair before detection.
    transcript_repair: bool = False
    # Context-aware STT question repair: feed the interview's DOMAIN (resume
    # skills, target role + JD, recent topics) to the question-cleaner so it can
    # confidently fix mis-transcribed technical terms ("Q proxy" -> "kube-proxy",
    # "spring" -> "string"). When a domain is present this also routes '?'-ending
    # questions through the cleaner instead of the no-cleanup fast path.
    context_repair: bool = True
    # 3-signal ensemble question detection (rule + agent + prosody).
    ensemble_detection: bool = False
    # Cancel the in-flight answer when the interviewer abandons/self-corrects.
    interruption_handling: bool = False
    # Detect interviewer satisfaction ("good" closes / "not quite" keeps open).
    satisfaction_detection: bool = False
    # Low STT/topic/speaker confidence lowers the surfaced answer confidence.
    uncertainty_tracking: bool = False
    # Learn the interviewer's style and tune detection thresholds.
    style_learning: bool = False
    # Append-only per-session replayable event log.
    event_log: bool = False

    # Phase 6 — trust, safety & privacy.
    # Require an explicit consent/disclaimer before capturing audio.
    consent: bool = False
    # Capture only the candidate's own audio (no other party).
    candidate_audio_only: bool = False
    # Redact PII before any third-party-LLM egress (persisted text unchanged).
    pii_redaction: bool = False
    # Neutralize instruction-like spans in the (untrusted) transcript.
    transcript_sanitization: bool = False
    # Retention policy for live transcripts/answers/embeddings (0 = keep).
    retention_days: int = 0
    # Hard budget (seconds) on one answer's generation: past it, the partial
    # answer is kept, the qid finalizes, and the session moves on.
    answer_timeout_s: float = 60.0

    # FAST QUESTION PATH: an audio transcript that is unambiguously a question
    # (terminal '?' + confident heuristic) skips the detection-LLM round trip
    # (seconds on free tiers) and goes straight to answer generation via a
    # deterministically-built event. Ambiguous utterances still take the full
    # LLM typing path.
    fast_question_path: bool = True
    # SPECULATIVE ANSWERING: when a streaming PARTIAL already reads as a
    # complete question, start the answer DURING the end-of-speech silence
    # with its frames buffered; flush instantly when the final transcript
    # matches. Overlaps the endpoint wait AND the LLM first-token wait —
    # speech-end → visible answer collapses to ~the endpoint gap. A mismatch
    # discards the speculation (costs one wasted LLM call on free quota).
    speculative_answers: bool = True

    # VERIFIER (post-LLM stage): after an answer finishes streaming, a fast
    # verification call scores relevance + hallucination risk and badges the
    # answer (meta.verify). Non-blocking — adds zero latency to the visible
    # stream. `answer_regenerate` additionally regenerates on a weak/garbled
    # verdict with the critique folded into the retry's directive.
    answer_verify: bool = False
    answer_regenerate: bool = False
    verify_min_relevance: float = 0.55
    verify_max_hallucination: float = 0.75
    # Max verification-driven retries. The FINAL retry ESCALATES: it bypasses
    # the pinned live model and forces the strongest tier so the auto-router
    # picks a different, more capable model. 0 disables retries (badge only).
    answer_max_retries: int = 2
    # Live code sandbox: run a coding answer's code in the sandbox AFTER it
    # streams; on a compile/run failure, regenerate once with the sandbox error
    # folded in — and prefer a language from the candidate's resume. Non-blocking
    # (the first answer streams live; broken code is replaced by a verified
    # revision, like the answer verifier). Runs Python/JS/Ruby/PHP + compiled
    # Java/Go; anything else is reported "not executed", never falsely verified.
    code_sandbox: bool = True
    # Max sandbox-driven code regenerations on a failed run (0 = badge only).
    code_max_fix: int = 1

    # Phase 7 — audio capture & transport reliability.
    # Topology-aware capture routing (candidate-mic / loopback / both).
    capture_topology: bool = False
    # Keep per-session state on disconnect so a reconnect resumes (no dup answers).
    session_resume: bool = False
    # Mobile mic-contention / audio-routing handling + overlay + degrade hooks.
    mobile_runtime: bool = False

    # Phase 8 — candidate experience & accessibility.
    # Surface concise talking-point bullets alongside the full answer.
    glanceable_surface: bool = False
    # Track candidate speech; suppress a competing full answer while they answer.
    candidate_awareness: bool = False
    # Detect language + answer in it (code-switch tolerant).
    multilingual: bool = False
    # Forced answer language ("auto" = follow detection).
    answer_language: str = "auto"

    # Phase 9 — economics & evaluation realism.
    # Cap concurrent answers (and optionally total) per live session.
    session_budget: bool = False
    max_concurrent_answers: int = 3
    max_answers_per_session: int = 0      # 0 = unlimited

    # Near-duplicate question guard (2026-07-08): endpoint splits / spec-final
    # mismatches / re-transcriptions of ONE spoken question are skipped
    # instead of answered twice. ON by default — a genuine re-ask after the
    # window still answers; superset continuations always pass.
    question_dedup: bool = True
    question_dedup_window_s: float = 20.0
    question_dedup_similarity: float = 0.87
    # Semantic layer (2026-07-09): embedding-cosine dedup catches PARAPHRASED
    # re-asks the char ratio misses. Uses the shared bge-m3 embedder when warm.
    question_dedup_semantic: bool = True
    question_dedup_semantic_sim: float = 0.90
    # Pre-generated resume answers (latency batch 2026-07-11 #2): common
    # interview questions answered once at upload; live matches by embedding
    # similarity and streams them INSTANTLY.
    prepared_answers: bool = True
    prepared_count: int = 64
    prepared_match_threshold: float = 0.86
    # Seconds between prepared-answer generations (free-tier safety).
    prepared_pacing_s: float = 2.0

    # Candidate-echo skip (2026-07-08 fix): these fields existed in
    # config.example.yaml but NOT in this model, so pydantic silently dropped
    # them and the guard read its getattr default (False) — echo skip was
    # never actually on. Declared + defaulted ON.
    candidate_echo_skip: bool = True
    candidate_echo_threshold: float = 0.72

    # Compound questions ("what is X and how do you use it?") answered as ONE
    # response with per-part headings instead of two bubbles (enhancement #4).
    combine_multi_questions: bool = True

    # Short factual questions ("what is …?", "difference between …") lead
    # with a one-sentence direct answer and use the concise depth — the
    # perceived-latency fast path (enhancement #6).
    factual_fast_path: bool = True

    # Phase 10 — conversational depth.
    # Speaker diarization / roles / panel threads.
    diarization: bool = False
    # Unified per-session interview world-model (assumptions + constraints).
    world_model: bool = False
    # Evaluation-objective + expected-depth estimation.
    objective_depth: bool = False
    # Revise the prior answer when a follow-up reinterprets the question.
    answer_revision: bool = False
    # Detect + correct a false/over-stated premise.
    false_premise: bool = False
    # Contradiction (challenge) + temporal-reference reasoning.
    contradiction: bool = False

    # Phase 11 — live knowledge, prediction & operability.
    # Predict likely follow-ups (+ optional speculative pre-warm).
    question_prediction: bool = False
    # Topic-triggered interview-knowledge retrieval (+ optional pack bias).
    interview_knowledge: bool = False
    knowledge_pack: str = ""
    # Periodic in-session state validation + recovery from the session summary.
    state_validation: bool = False
    # Real-time session-health warnings (STT/dropped/speaker/latency).
    session_health: bool = False
    # Optional candidate delivery coaching.
    delivery_coaching: bool = False

    # Phase 12 — candidate & organization intelligence (pre-interview).
    # Structured candidate profile + resume knowledge graph + scoped retrieval.
    candidate_profile: bool = False
    # Pre-generated interview assets + resume-reality grounding.
    interview_assets: bool = False
    # Organization/JD intelligence + fit analysis.
    org_intelligence: bool = False
    # Seniority-band calibration: pitch live answers at the candidate's band
    # (fresher → principal) inferred from the resume + target role, per
    # BandSpecific.md. Injects an ANSWER GUIDANCE directive (no extra LLM call).
    answer_calibration: bool = False

    # Phase 13 — HR, negotiation & specialized modes.
    # Interview_Mode detection/switching that biases answer strategy.
    interview_modes: bool = False
    # Fact-based, no-manipulation negotiation strategy for HR questions.
    negotiation: bool = False
    # Advisory Emotion_Signal from prosody (never decisive).
    emotion_signal: bool = False

    # Phase 14 — outcome analytics, replay, simulation & multimodal.
    # Advisory Outcome_Estimate (explicitly not a hiring decision).
    outcome_estimate: bool = False
    # Read-only Session_Replay route from the event log.
    session_replay: bool = False
    # Mock_Mode practice question generation.
    mock_mode: bool = False
    # Extensible Multimodal_Input adapter (audio-only when absent).
    multimodal: bool = False

    # Phase 15 — precision & explicit-coverage hardening (R48–R58).
    implicit_question: bool = False     # implicit/semantic-completion detection
    evidence: bool = False              # Supporting_Segments binding + stale hedge
    multi_pass: bool = False            # multi-pass understanding (objective refine)
    coreference: bool = False           # coreference resolution (defer low-conf)
    rhetorical: bool = False            # rhetorical-question suppression
    acoustic_adaptation: bool = False   # accent/noise/room → confidence penalty
    override_gate: bool = False         # gated override suggestion + confidence band
    stage_budget: bool = False          # per-stage soft time budgets
    feedback_loop: bool = False         # Feedback_Signal capture
    skill_gap: bool = False             # recurring-topic skill-gap retrieval boost
    cognitive_load: bool = False        # cognitive-load/pace → depth adaptation

    # Conversation signals (Phase 2 #13 readiness · #23 contract · #31
    # rhythm/fatigue · #32 steering). One flag gates the whole additive,
    # fail-open cluster in routes_ws; each signal only ever *adds* to `extra`
    # or appends to the answer directive, so a failure degrades to today's
    # behaviour rather than breaking the turn.
    conversation_signals: bool = False
    # BandSpecific — the 4th calibration dimension (industry vertical) folded
    # into the answer directive, and capability-over-title framing from the
    # career-readiness layer. Both advisory + fail-open; default on.
    industry_context: bool = True
    capability_framing: bool = True

    # Phase 16 — out-of-band extensions (R60, R61, R62).
    gpu_stt: bool = False               # GPU STT execution (latency-only; CPU fallback)
    career_intelligence: bool = False   # advisory career-prep coaching (off by default)
    enterprise_readiness: bool = False  # per-user scoping + shared libs + team analytics

    # Dual-source continuity — hear both voices, act on one.
    dual_source: bool = False           # accept source_role/candidate_text control frames
    candidate_channel: bool = False     # absorb candidate speech (never answered)
    role_memory: bool = False           # role-tagged shared conversation graph
    commitment_tracking: bool = False   # stated salary/offer/notice + interviewer signals

    # Capture/answering mode DEFAULT when the client sends no `mode` on connect.
    # False → "standard" (real interview: answer the interviewer only, absorb the
    # candidate, echo-skip on). True → "solo" (testing: one source, answer every
    # question regardless of who/what source; candidate-absorb + echo-skip off).
    # The Live UI toggle sends `mode` per-connection, overriding this.
    solo_mode: bool = False


class ConfidenceSection(BaseModel):
    """Centralized confidence bands for the clarifier gate + adaptation.

    Previously these were duplicated as module-level literals (clarifier
    `_BAND_*`, goal_ledger `_DEFAULT_BAND`, adaptation `_BAND_FLOOR`). Defaults
    here EQUAL those former literals, so behavior is byte-for-byte unchanged
    until someone tunes them — now in one place instead of several files.
    """
    band_high: float = 0.90        # >= → answer directly (no clarification panel)
    band_assume: float = 0.70      # [assume,high) → assumption-confirm
    band_targeted: float = 0.40    # [targeted,assume) → targeted (<=2 Qs); below → guided
    band_floor: float = 0.50       # adaptive-fatigue lower bound (adaptation.py)
    state_band_default: float = 0.90   # goal_ledger default slot band


class DecisionCoreSection(BaseModel):
    """Phase-1 decision-core upgrades (SeveralFeatures.md / ArchitectureVerdict
    Phase 1): structured requirement matrix, numeric risk scoring, assumption
    persistence. Additive + fail-open; any flag off restores prior behavior."""
    requirement_matrix: bool = True    # attach RequirementMatrix to Assessment
    risk_scoring: bool = True          # numeric risk → answer-band nudge
    risk_band_weight: float = 0.05     # band delta at HIGH risk (LOW gets -w/2)
    persist_assumptions: bool = True   # record assume-mode assumptions in ledger


class PolicySection(BaseModel):
    """Phase-3 declarative policy engine (app/policy). Builtin rules replicate
    the legacy pre-gate final cascade exactly, so `enabled: true` changes zero
    decisions until rules are added/overridden here. Each rule:
    {id, action: ANSWER|CLARIFY|DEFER, priority, weight, cost, benefit,
     when: [{field, op, value}, ...], reason, enabled}."""
    enabled: bool = True
    rules: list = Field(default_factory=list)   # declarative overlay by id


class SandboxSection(BaseModel):
    """Dedicated execution sandbox (app/sandbox) — Claude-style layered
    isolation: bubblewrap namespaces on Linux (no network, RO OS view,
    ephemeral workspace), rlimits on bare POSIX, constrained subprocess on
    Windows dev machines. `languages: []` = all supported runners allowed."""
    enabled: bool = True
    timeout_s: float = 10.0
    cpu_s: int = 8
    memory_mb: int = 512
    output_kb: int = 256
    max_files_mb: int = 32
    languages: list = Field(default_factory=list)   # allowlist; empty = all
    # Execution backend: "local" (host toolchains, resolved via tool_dirs/PATH)
    # or "docker" (everything compiled + run inside the `container` — one Linux
    # image with all 25 toolchains, see sandbox/Dockerfile + docker_exec.py).
    backend: str = "local"
    container: str = "zapthetrick_sandbox"
    # Bin dirs prepended to the sandbox PATH so a manually-installed current
    # toolchain wins over a stale machine-wide one that can't be removed without
    # admin (e.g. an old winget Scala pinned in the Machine PATH). Local backend
    # only.
    tool_dirs: list = Field(default_factory=list)
    # 2026-07-09: agent workspace build/test commands are bwrap-confined on
    # Linux (filesystem RO except the workspace; network KEPT for installs).
    harden_runner: bool = True


class ArtifactValidationSection(BaseModel):
    """Phase-4 closed artifact loop (app/documents/validators): every
    generated document/archive is structurally validated before delivery;
    an invalid render is re-rendered once (repair) and, failing that,
    degraded along the capability fallback chain (pdf→docx→md). The guard is
    fail-open — it never blocks delivery, it only catches provably-broken
    output."""
    enabled: bool = True
    repair: bool = True
    degrade: bool = True
    # Closed PROJECT loop for chat builds (app/verify): downloaded project
    # archives are syntax-checked + their tests RUN in the sandbox; a failed
    # build gets ONE model repair round; a VERIFICATION.txt report ships in
    # the archive either way.
    verify_projects: bool = True
    repair_with_model: bool = True
    run_tests: bool = True
    test_timeout_s: float = 60.0
    # 2026-07-09: the entrypoint is EXECUTED in the sandbox (import-time
    # crashes caught, long-running servers count as started).
    smoke_run: bool = True
    # Opt-in: pip-install requirements.txt into the sandbox workspace before
    # smoke/tests (network-allowed sandbox run) — catches broken dependency
    # lists. Off by default: slow + needs egress.
    install_deps: bool = False
    # Model repair rounds on a failed verify (was hardcoded to one).
    repair_rounds: int = 2


class TemperatureSection(BaseModel):
    """Per-task sampling-temperature policy (was inline literals per call site).

    Groups the ~26 scattered `temperature=` literals into four intents. Defaults
    match today's most-common values, so wiring a call to `temperature_for(kind)`
    is behavior-preserving. Lets math/classification stay deterministic and
    creative turns stay warmer, tunable centrally (gap G1.7).
    """
    classifier: float = 0.0    # triage/detect/difficulty/grounder/graders/executor — deterministic
    planning: float = 0.2      # planners / query-expansion / titles / history summary
    creative: float = 0.4      # follow-up suggestions and other generative-but-bounded turns
    verify: float = 0.5        # self-refine verify pass
    default: float = 0.2


class ModelMarkersSection(BaseModel):
    """Provider-agnostic id/name markers that BACKFILL model capabilities when a
    provider exposes no structured metadata (the common case for free OpenAI-
    compatible providers). Structured provider metadata ALWAYS wins; these are
    only the fallback. Externalized here so a new model family can be supported
    (or a mis-detection fixed) by editing config — no code change. Empty list →
    the code's built-in fallback list is used, so partial overrides are safe.
    Defaults equal the former in-code lists, so behavior is unchanged.
    """
    vision_markers: list[str] = Field(default_factory=lambda: [
        "vl", "vision", "gpt-4o", "gpt-4.1", "gpt-5", "o4-", "chatgpt-4o",
        "gemini", "claude-3", "claude-opus-4", "claude-sonnet-4", "claude-haiku-4",
        "llava", "pixtral", "qwen2-vl", "qwen2.5-vl", "qwen3-vl",
        "llama-3.2-11b-vision", "llama-3.2-90b-vision", "llama-4", "maverick",
        "scout", "gemma-3", "internvl", "minicpm-v", "phi-3.5-vision",
        "phi-4-multimodal", "grok-2-vision", "grok-4", "mistral-small-3.1",
        "mistral-small-3.2", "mistral-medium-3", "molmo", "aria", "deepseek-vl",
        "step-1v", "glm-4v", "glm-4.1v", "kimi-vl", "ernie-4.5-vl", "dots.vlm",
    ])
    moe_markers: list[str] = Field(default_factory=lambda: [
        "mixtral", "deepseek-v2", "deepseek-v3", "deepseek-chat", "deepseek-r1",
        "dbrx", "jamba", "arctic", "llama-4", "gpt-oss", "minimax", "olmoe",
        "-moe", "moe-",
    ])
    # Markers on a model id that imply NO tools/JSON support (base/embed/tiny).
    no_capability_markers: list[str] = Field(default_factory=lambda: [
        "base", "embed", "rerank", "-1b", "-2b", "tiny",
    ])
    # Category → strength markers (coding/math/reasoning/vision).
    category_markers: dict[str, list[str]] = Field(default_factory=lambda: {
        "coding": ["coder", "code", "codestral", "deepseek-coder", "qwen3-coder",
                   "starcoder", "codegemma"],
        "math": ["math", "deepseek-math", "qwen2.5-math", "wizardmath"],
        "reasoning": ["reason", "-r1", "o1", "o3", "o4", "thinking", "qwq",
                      "magistral", "deepseek-r1"],
        "vision": ["vl", "vision", "-vl-", "multimodal"],
    })


class OutputTokensSection(BaseModel):
    """Per-purpose output-token caps (`num_predict`) for the short auxiliary LLM
    calls (classifiers / graders / titles / pickers). These were inline literals
    at each call site; grouped here by PURPOSE so they're tuned centrally.
    Defaults equal the former literals, so behavior is unchanged until tuned.
    (Long generative calls keep sizing from `cfg.llm.max_tokens` / task-specific
    config and are not governed here.)
    """
    label: int = 32          # conversation / solve title generation
    micro_label: int = 24    # difficulty + doc-format tiny classifiers
    intent: int = 40         # triage + document-intent classify
    pattern: int = 100       # DSA pattern classifier
    short_json: int = 200    # tool-plan executor + follow-up suggestions
    verdict: int = 400       # grounder / complexity / history summary
    council_pick: int = 120  # council candidate picker
    council_gen: int = 500   # council candidate generation
    redteam: int = 700       # red-team safety review


class SemanticGatesSection(BaseModel):
    """Exemplar-embedding orchestration gates (app/semantics/gates.py,
    2026-07-09): every yes/no routing decision (document ask, profile
    question, implicit probe, …) is answered by semantic similarity to
    example phrasings — the old cue lists remain only as zero-latency
    fast-paths. `thresholds` overrides a gate's cosine cutoff by name."""
    enabled: bool = True
    thresholds: dict = Field(default_factory=dict)
    # Required positive-minus-negative similarity margin per gate.
    margins: dict = Field(default_factory=dict)


class SemanticIntentSection(BaseModel):
    """Embeddings-based (bge-m3) semantic intent classification as the PRIMARY
    intent classifier — intent is understood by MEANING (similarity to example
    phrasings), not keyword/regex rules.

    Decision path (`detect_intent_smart`):
      • embed the turn → nearest-exemplar intent + cosine score;
      • score ≥ primary_threshold → the semantic intent is authoritative;
      • below that (genuinely ambiguous) → consult the deterministic regex, and
        use the semantic best-guess only if the regex has no opinion (UNKNOWN);
      • embedder unavailable (not installed / load error) → the regex FALLBACK.

    So the keyword regex is no longer the brain — it is the deterministic
    safety-net for the low-confidence and model-unavailable cases. Fail-open by
    construction: nothing breaks if the model is absent.
    """
    enabled: bool = True
    # At/above this cosine the semantic intent decides directly (bge-m3 scores
    # correct intents ~0.5–0.9 on short turns). Below it, defer to the regex net.
    primary_threshold: float = 0.50
    # #12 self-improving intent: grow the exemplar set from user feedback. A 👍
    # adds a positive exemplar (reinforce); a 👎-with-correction adds a negative
    # one (demote). Off by default; the store persists to ~/.zapthetrick/.
    learn_exemplars: bool = False
    # How hard a learned NEGATIVE match demotes the winning intent's score.
    negative_penalty: float = 0.15
    # G4: on a low-confidence turn (semantic score < primary_threshold), ask the
    # model to disambiguate intent instead of falling to the keyword regex net.
    # Off by default (adds one classifier call on gray-zone turns only).
    llm_disambiguation: bool = False


class ToolsSection(BaseModel):
    """Tool dispatch."""
    # Let MEASURED per-tool reliability steer dispatch, not just be recorded.
    # A tool we have watched fail repeatedly is ordered behind a healthy
    # alternative and marked unreliable in the catalog shown to the model. It is
    # never hard-blocked: if a degraded tool is the ONLY path to a capability it
    # still runs, because a missing capability is worse than a flaky one. A tool
    # with no history is neutral — only observed failures demote it.
    reliability_routing: bool = True


class Config(BaseModel):
    """The full validated config tree."""
    app: AppSection = Field(default_factory=AppSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    embeddings: EmbeddingsSection = Field(default_factory=EmbeddingsSection)
    vector_store: VectorStoreSection = Field(default_factory=VectorStoreSection)
    reranker: RerankerSection = Field(default_factory=RerankerSection)
    rag: RAGSection = Field(default_factory=RAGSection)
    stt: STTSection = Field(default_factory=STTSection)
    vision: VisionSection = Field(default_factory=VisionSection)
    audio: AudioSection = Field(default_factory=AudioSection)
    question_detection: QuestionDetectionSection = Field(
        default_factory=QuestionDetectionSection
    )
    code_solver: CodeSolverSection = Field(default_factory=CodeSolverSection)
    web_search: WebSearchSection = Field(default_factory=WebSearchSection)
    git_workflow: GitWorkflowSection = Field(default_factory=GitWorkflowSection)
    ui: UISection = Field(default_factory=UISection)
    server: ServerSection = Field(default_factory=ServerSection)
    database: DatabaseSection = Field(default_factory=DatabaseSection)

    # Architecture.md §9 additions.
    agents: AgentsSection = Field(default_factory=AgentsSection)
    advanced_rag: AdvancedRagSection = Field(default_factory=AdvancedRagSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    learning: LearningSection = Field(default_factory=LearningSection)
    ui_advanced: UIAdvancedSection = Field(default_factory=UIAdvancedSection)
    themes: ThemesSection = Field(default_factory=ThemesSection)

    # Architecture.md additions (wiring pass).
    mcp: MCPSection = Field(default_factory=MCPSection)
    response_arch: ResponseArchSection = Field(default_factory=ResponseArchSection)
    documents: DocumentsSection = Field(default_factory=DocumentsSection)
    continuity: ContinuitySection = Field(default_factory=ContinuitySection)
    technical_pipeline: TechnicalPipelineSection = Field(default_factory=TechnicalPipelineSection)
    intent_profiles: IntentProfilesSection = Field(
        default_factory=IntentProfilesSection)
    understanding: UnderstandingSection = Field(
        default_factory=UnderstandingSection)
    synthesis: SynthesisSection = Field(default_factory=SynthesisSection)
    privacy: PrivacySection = Field(default_factory=PrivacySection)
    calibration: CalibrationSection = Field(default_factory=CalibrationSection)
    tool_loop: ToolLoopSection = Field(default_factory=ToolLoopSection)
    resilience: ResilienceSection = Field(default_factory=ResilienceSection)
    routing: RoutingSection = Field(default_factory=RoutingSection)
    perceived: PerceivedSection = Field(default_factory=PerceivedSection)
    followup: FollowupSection = Field(default_factory=FollowupSection)
    quality: QualitySection = Field(default_factory=QualitySection)
    tools: ToolsSection = Field(default_factory=ToolsSection)
    workspace: WorkspaceSection = Field(default_factory=WorkspaceSection)
    orchestration: OrchestrationSection = Field(default_factory=OrchestrationSection)
    personalization: PersonalizationSection = Field(
        default_factory=PersonalizationSection)
    live: LiveSection = Field(default_factory=LiveSection)
    # Centralized tuning knobs (was scattered code literals).
    confidence: ConfidenceSection = Field(default_factory=ConfidenceSection)
    decision_core: DecisionCoreSection = Field(
        default_factory=DecisionCoreSection)
    policy: PolicySection = Field(default_factory=PolicySection)
    artifact_validation: ArtifactValidationSection = Field(
        default_factory=ArtifactValidationSection)
    sandbox: SandboxSection = Field(default_factory=SandboxSection)
    temperature: TemperatureSection = Field(default_factory=TemperatureSection)
    output_tokens: OutputTokensSection = Field(default_factory=OutputTokensSection)
    model_markers: ModelMarkersSection = Field(default_factory=ModelMarkersSection)
    semantic_intent: SemanticIntentSection = Field(
        default_factory=SemanticIntentSection)
    semantic_gates: SemanticGatesSection = Field(
        default_factory=SemanticGatesSection)


# ---- Singleton -----------------------------------------------------------
_config: Config | None = None


def _drop_blanks(obj: Any) -> Any:
    """Recursively strip keys whose value is an empty string.

    Seeded / templated configs (e.g. the bundled `config.example.yaml`) leave
    some fields blank (`max_tokens: ""`). An empty string fails typed
    validation (int/bool fields) and would crash startup. Dropping the key lets
    Pydantic fall back to the field's default — i.e. "blank in YAML == default".
    Secrets that are legitimately empty (api keys) already default to "", so
    dropping them is a no-op.
    """
    if isinstance(obj, dict):
        return {k: _drop_blanks(v) for k, v in obj.items() if v != ""}
    if isinstance(obj, list):
        return [_drop_blanks(v) for v in obj]
    return obj


def _load_from_disk() -> Config:
    """Read CONFIG_PATH and parse into a Config. Missing file -> defaults.

    Reads defensively: decode UTF-8 tolerating a stray bad byte, then strip C1
    control chars before handing the text to PyYAML — one rogue character must
    never brick startup. Blank values fall back to model defaults (_drop_blanks).
    """
    if not CONFIG_PATH.exists():
        return Config()
    text = CONFIG_PATH.read_bytes().decode("utf-8", errors="replace")
    text = _C1_CONTROLS.sub("", text)
    raw = yaml.safe_load(text) or {}
    return Config(**_drop_blanks(raw))


def get_config() -> Config:
    """Return the active Config, loading from disk on first call."""
    global _config
    if _config is None:
        _config = _load_from_disk()
    return _config


def reload_config() -> Config:
    """Force a fresh read from disk. Used after external file edits."""
    global _config
    _config = _load_from_disk()
    return _config


def update_config(updates: dict[str, Any]) -> Config:
    """Deep-merge `updates` into the current config and persist to disk.

    Example: update_config({"llm": {"model": "qwen2.5:7b-instruct"}})
    keeps every other llm.* field intact and only replaces `model`.

    The merged dict is re-validated through the Pydantic Config before it
    is written, so an invalid update raises before any file IO happens.
    """
    current = get_config().model_dump()
    merged = _deep_merge(current, updates)
    new_cfg = Config(**merged)  # raises on invalid types

    with open(CONFIG_PATH, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(new_cfg.model_dump(), f, sort_keys=False, allow_unicode=True)

    global _config
    _config = new_cfg
    return new_cfg


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into `base`. Scalars overwrite, dicts recurse."""
    result = dict(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---- Convenience proxy --------------------------------------------------
class _CfgProxy:
    """Attribute-style access that always reflects the latest config.

    Lets callers write `cfg.llm.model` and pick up changes after a
    /api/settings update without re-importing or threading a Config arg.
    """
    def __getattr__(self, name: str) -> Any:
        return getattr(get_config(), name)


cfg = _CfgProxy()


def temperature_for(kind: str) -> float:
    """Resolve the sampling temperature for a task class from central config.

    `kind` ∈ {classifier, planning, creative, verify}; anything else → default.
    Replaces inline `temperature=<literal>` at call sites so the sampling policy
    is tuned in one place (config) rather than scattered across ~26 files.
    """
    t = cfg.temperature
    return {
        "classifier": t.classifier,
        "planning": t.planning,
        "creative": t.creative,
        "verify": t.verify,
    }.get(kind, t.default)
