#!/usr/bin/env bash
# ==============================================================================
# Runtime init for the SELF-CONTAINED RunPod image (deploy/runpod.Dockerfile).
#
# The image bakes the whole environment AND the app code (/opt/zapthetrick_be),
# so this does NOT clone anything. It just does the /workspace-VOLUME-persistent
# setup, then hands off to supervisor. Everything is driven by env vars — set
# them in the RunPod template (or deploy.sh) and the pod comes up fully
# configured with ZERO web-terminal steps. Recreate on any available GPU with
# the same volume → the same system, with data restored.
# ==============================================================================
set -euo pipefail

# ---- paths & config (all overridable via env; sane defaults for zero-config) -
VENV="${VENV:-/opt/venv}"
APP_DIR="${APP_DIR:-/opt/zapthetrick_be}"        # baked code (NOT on the volume)
# Postgres data on the pod's LOCAL disk (MooseFS /workspace can't honor 0700 for
# initdb). It's ephemeral — durability comes from the /workspace dump/restore.
PGDATA="${PGDATA:-/var/lib/pgdata}"
PGPASS="${POSTGRES_PASSWORD:-zaptrick}"
APP_PORT="${APP_PORT:-8888}"
CFG="${ZAPTHETRICK_CONFIG_PATH:-/workspace/config.yaml}"   # persists on volume
BACKUP_DIR="${BACKUP_DIR:-/workspace/pg_backups}"
BACKUP_INTERVAL_S="${BACKUP_INTERVAL_S:-1800}"    # 30 min
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export ZAPTHETRICK_CONFIG_PATH="$CFG"
export BACKUP_DIR
# ---- local generation floor (§2.1 T4) — ON by default (GPU generation) -------
# Runs an on-pod llama.cpp OpenAI server (CUDA-built → runs on the GPU) so the
# router ALWAYS has an instant, rate-limit-free route. Default ON: it's the
# answer to "generate on the 24GB GPU" and the fast floor a stalled free cloud
# model fails over to. Set LOCAL_LLM_ENABLED=0 to go cloud-only.
LOCAL_LLM_ENABLED="${LOCAL_LLM_ENABLED:-1}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-8081}"
LOCAL_LLM_MODEL_ID="${LOCAL_LLM_MODEL_ID:-qwen2.5-14b-instruct}"
LOCAL_LLM_GGUF="${LOCAL_LLM_GGUF:-/workspace/models/local-llm.gguf}"
# Qwen2.5-14B-Instruct Q4_K_M (~9GB) — the quality/latency sweet spot for a 24GB
# card shared with STT (~3GB) + local vision (~1-7GB). Single-file GGUF (reliable,
# no shard-merge). Override LOCAL_LLM_GGUF_URL for a smaller/faster model (e.g. the
# 7B) on a smaller GPU, or a 32B on a bigger one. A download failure degrades
# gracefully — the floor stays inert and answers fall back to cloud.
LOCAL_LLM_GGUF_URL="${LOCAL_LLM_GGUF_URL:-https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf}"
# A6/A2 — SPECULATIVE DECODING draft model. A tiny same-family model (Qwen2.5-0.5B,
# ~0.4GB) proposes tokens the 14B verifies in batches → ~1.5-2.5x faster generation,
# IDENTICAL output. Must share the target's tokenizer family (0.5B ↔ 14B both Qwen2.5).
# Best-effort: if the draft download fails OR the server build predates draft support,
# the launch falls back to the plain (non-speculative) command — never a crash.
LOCAL_LLM_DRAFT_GGUF="${LOCAL_LLM_DRAFT_GGUF:-/workspace/models/local-llm-draft.gguf}"
LOCAL_LLM_DRAFT_URL="${LOCAL_LLM_DRAFT_URL:-https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}"
# A1 — enable prompt-prefix KV caching (reuse the stable persona+resume prefix
# across questions) + flash-attention. Set 0 to launch with the plain command.
LOCAL_LLM_FAST="${LOCAL_LLM_FAST:-1}"
# A4 two-tier — an OPTIONAL smaller local model the router prefers for trivial /
# definition questions (the big model handles hard ones). OFF by default so it
# never risks OOM on a full 24GB card; set LOCAL_LLM_SMALL_MODEL_ID (+ URL) to a
# small instruct GGUF (e.g. qwen2.5-3b-instruct) to enable. Served by the SAME
# llama.cpp server (one port, switched by model_id).
LOCAL_LLM_SMALL_MODEL_ID="${LOCAL_LLM_SMALL_MODEL_ID:-}"
LOCAL_LLM_SMALL_GGUF="${LOCAL_LLM_SMALL_GGUF:-/workspace/models/local-llm-small.gguf}"
LOCAL_LLM_SMALL_URL="${LOCAL_LLM_SMALL_URL:-https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf}"
LLAMA_CFG="/workspace/llamacpp.json"
PGBIN="$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)"
mkdir -p /workspace "$HF_HOME" "$BACKUP_DIR"

echo "==> ssh"
(service ssh start 2>/dev/null || /usr/sbin/sshd 2>/dev/null || true)

# ---- 0b) native accounts — a PERSISTENT signing secret on the volume --------
# Without a secret `auth_mode()` infers "off", the app sees a server with no
# accounts and falls back to its device-local stub ("No local account for this
# email"), with no Google button. Generating the secret here (and exporting it
# as an env var, which beats config) fixes EXISTING volumes too — the
# config.yaml render below only runs on first boot.
#
# The secret lives on the volume, so it survives pod recreation: a 30-day login
# token stays valid even when the pod URL changes.
AUTH_SECRET_FILE="/workspace/auth_secret"
if [ -z "${ZAPTHETRICK_AUTH_SECRET:-}" ]; then
  if [ ! -s "$AUTH_SECRET_FILE" ]; then
    echo "==> generating a native auth secret (first time on this volume)"
    "$VENV/bin/python" -c "import secrets;print(secrets.token_urlsafe(48))" \
      > "$AUTH_SECRET_FILE"
    chmod 600 "$AUTH_SECRET_FILE"
  fi
  ZAPTHETRICK_AUTH_SECRET="$(cat "$AUTH_SECRET_FILE")"
fi
export ZAPTHETRICK_AUTH_SECRET
export ZAPTHETRICK_AUTH_MODE="${ZAPTHETRICK_AUTH_MODE:-native}"
# Sign-in required to use the pod unless the operator opts out.
export ZAPTHETRICK_AUTH_ENFORCE="${ZAPTHETRICK_AUTH_ENFORCE:-1}"

# ---- 1) config.yaml on the volume — MERGED with defaults on every boot -------
# This used to be skip-if-exists: rendered once on first boot, then never touched
# again. The consequence was silent and severe — every config key added after a
# volume was created stayed invisible to that pod forever. A pod could run
# `max_retries: 15` in the repo while actually using the code default of 6, and
# the `llm.local` block that makes the never-empty routing ladder work was simply
# absent, so the on-pod GPU model was never routable no matter what the image
# contained.
#
# Now defaults are re-applied on EVERY boot and the volume's own file is
# deep-merged ON TOP, so:
#   * new keys from config.example.yaml appear (that is the whole fix);
#   * anything the operator set — API keys, Settings, model picks — still wins;
#   * pod-shaped infrastructure (DB, ports, sandbox backend) is forced AFTER the
#     merge, because those describe THIS pod rather than a preference.
# A timestamped backup is kept, so a bad merge is always recoverable.
echo "==> merging $CFG with current defaults"
APP_PORT="$APP_PORT" PGPASS="$PGPASS" \
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" NVIDIA_API_KEY="${NVIDIA_API_KEY:-}" \
LOCAL_LLM_ENABLED="$LOCAL_LLM_ENABLED" LOCAL_LLM_MODEL_ID="$LOCAL_LLM_MODEL_ID" \
LOCAL_LLM_PORT="$LOCAL_LLM_PORT" LOCAL_LLM_SMALL_MODEL_ID="$LOCAL_LLM_SMALL_MODEL_ID" \
"$VENV/bin/python" "$APP_DIR/deploy/merge_config.py" \
  "$APP_DIR/config.example.yaml" "$CFG"

# ---- 2) Postgres: init on fresh disk, then RESTORE the latest dump if present -
sudo mkdir -p "$PGDATA" && sudo chown postgres:postgres "$PGDATA"
FRESH=0
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  FRESH=1
  sudo chmod 700 "$PGDATA"
  sudo -u postgres "$PGBIN/initdb" -D "$PGDATA" -E UTF8
fi
sudo -u postgres "$PGBIN/pg_ctl" -D "$PGDATA" -l /workspace/pg.log -o "-p 5432" -w start
sudo -u postgres psql -p 5432 -c "ALTER USER postgres PASSWORD '${PGPASS}';"
sudo -u postgres psql -p 5432 -c "CREATE EXTENSION IF NOT EXISTS vector;"
if [ "$FRESH" = 1 ] && [ -f "$BACKUP_DIR/latest.dump" ]; then
  echo "==> fresh DB + backup found → restoring $BACKUP_DIR/latest.dump"
  _restore_src="$BACKUP_DIR/latest.dump"
  # §11.1: if backups are AES-encrypted (PG_BACKUP_ENC_KEY set), decrypt to a
  # temp on the pod's LOCAL disk (never /workspace) before pg_restore.
  if [ -n "${PG_BACKUP_ENC_KEY:-}" ]; then
    _restore_src="/tmp/latest-decrypted.dump"
    if ! openssl enc -d -aes-256-cbc -pbkdf2 -pass env:PG_BACKUP_ENC_KEY \
          -in "$BACKUP_DIR/latest.dump" -out "$_restore_src" 2>/dev/null; then
      echo "   (could not decrypt backup — wrong PG_BACKUP_ENC_KEY? skipping restore)"
      _restore_src=""
    fi
  fi
  if [ -n "$_restore_src" ]; then
    PGPASSWORD="$PGPASS" "$PGBIN/pg_restore" --clean --if-exists --no-owner \
        -h 127.0.0.1 -p 5432 -U postgres -d postgres \
        "$_restore_src" 2>>/workspace/pg_restore.log || \
      echo "   (restore reported non-fatal errors — see /workspace/pg_restore.log)"
    [ "$_restore_src" = "/tmp/latest-decrypted.dump" ] && rm -f "$_restore_src"
  fi
fi
sudo -u postgres "$PGBIN/pg_ctl" -D "$PGDATA" -w stop   # supervisor owns it now

# ---- 3) supervisor: postgres + dragonfly + app + periodic backup + watchdog ---
# Local LLM floor (opt-in, §2.1 T4): fetch the GGUF once (to /workspace) and build
# a supervisor program serving an OpenAI-compatible API on 127.0.0.1. Empty when
# disabled → the config below is byte-identical to today.
LOCALLLM_PROGRAM=""
if [ "$LOCAL_LLM_ENABLED" = "1" ]; then
  mkdir -p "$(dirname "$LOCAL_LLM_GGUF")"
  # Model downloads run in the BACKGROUND (2026-07-28). They used to run inline
  # HERE — up to ~9.4GB before supervisord even started — so nothing listened on
  # the app port for minutes and RunPod's HTTP probe showed "Initializing… taking
  # longer than expected" (a dead-port window, not a broken app). Now the app
  # binds within seconds of boot and the llama server WAITS for its file (the
  # wait loop in its command below); draft/small models are picked up on the
  # next boot if they finish after the JSON render (best-effort extras).
  (
    if [ ! -f "$LOCAL_LLM_GGUF" ]; then
      echo "==> downloading local LLM GGUF -> $LOCAL_LLM_GGUF"
      curl -fsSL "$LOCAL_LLM_GGUF_URL" -o "$LOCAL_LLM_GGUF.part" \
        && mv "$LOCAL_LLM_GGUF.part" "$LOCAL_LLM_GGUF" \
        || echo "   (GGUF download failed — local floor stays inert until present)"
    fi
    if [ "$LOCAL_LLM_FAST" = "1" ] && [ ! -f "$LOCAL_LLM_DRAFT_GGUF" ]; then
      echo "==> downloading speculative draft model -> $LOCAL_LLM_DRAFT_GGUF"
      curl -fsSL "$LOCAL_LLM_DRAFT_URL" -o "$LOCAL_LLM_DRAFT_GGUF.part" \
        && mv "$LOCAL_LLM_DRAFT_GGUF.part" "$LOCAL_LLM_DRAFT_GGUF" \
        || echo "   (draft download failed — speculative decoding stays off)"
    fi
    if [ -n "$LOCAL_LLM_SMALL_MODEL_ID" ] && [ ! -f "$LOCAL_LLM_SMALL_GGUF" ]; then
      echo "==> downloading small two-tier model -> $LOCAL_LLM_SMALL_GGUF"
      curl -fsSL "$LOCAL_LLM_SMALL_URL" -o "$LOCAL_LLM_SMALL_GGUF.part" \
        && mv "$LOCAL_LLM_SMALL_GGUF.part" "$LOCAL_LLM_SMALL_GGUF" \
        || echo "   (small-model download failed — two-tier stays single-tier)"
    fi
    echo "==> model fetch done"
  ) >> /workspace/modelfetch.log 2>&1 &
  # Always configure the server when enabled (the old `if file exists` gate
  # meant a fresh volume got NO llama program at all until the next restart).
  if true; then
    # A1+A2: render a llama.cpp server config with flash-attention, prompt-prefix
    # KV cache, and (if present) the speculative draft model. Written as JSON so
    # bool/unknown-key handling is unambiguous. The supervisor command tries this
    # ENHANCED launch first and FALLS BACK to the plain known-good command if it
    # exits non-zero (e.g. an older server build rejects a key) — so a fast-path
    # flag can never crash-loop the model. Set LOCAL_LLM_FAST=0 to force plain.
    _DRAFT_JSON=""
    if [ "$LOCAL_LLM_FAST" = "1" ] && [ -f "$LOCAL_LLM_DRAFT_GGUF" ]; then
      _DRAFT_JSON=", \"draft_model\": \"${LOCAL_LLM_DRAFT_GGUF}\", \"draft_model_num_pred_tokens\": 10"
    fi
    # A4: a second served model entry for the small tier (same server/port).
    _SMALL_JSON=""
    if [ -n "$LOCAL_LLM_SMALL_MODEL_ID" ] && [ -f "$LOCAL_LLM_SMALL_GGUF" ]; then
      _SMALL_JSON=$(cat <<SMEOF
,
    {
      "model": "${LOCAL_LLM_SMALL_GGUF}",
      "model_alias": "${LOCAL_LLM_SMALL_MODEL_ID}",
      "n_gpu_layers": -1,
      "n_ctx": 8192,
      "n_batch": 512,
      "chat_format": "chatml",
      "flash_attn": true,
      "cache": true,
      "cache_type": "ram"
    }
SMEOF
)
    fi
    cat > "$LLAMA_CFG" <<JSONEOF
{
  "host": "127.0.0.1",
  "port": ${LOCAL_LLM_PORT},
  "models": [
    {
      "model": "${LOCAL_LLM_GGUF}",
      "model_alias": "${LOCAL_LLM_MODEL_ID}",
      "n_gpu_layers": -1,
      "n_ctx": 8192,
      "n_batch": 512,
      "chat_format": "chatml",
      "flash_attn": true,
      "cache": true,
      "cache_type": "ram"${_DRAFT_JSON}
    }${_SMALL_JSON}
  ]
}
JSONEOF
    _LL_ENHANCED="${VENV}/bin/python -m llama_cpp.server --config_file ${LLAMA_CFG}"
    _LL_PLAIN="${VENV}/bin/python -m llama_cpp.server --model ${LOCAL_LLM_GGUF} --host 127.0.0.1 --port ${LOCAL_LLM_PORT} --n_gpu_layers -1 --chat_format chatml --n_ctx 8192"
    # Wait for the (background) download to land the model — the `.part` → mv
    # rename is atomic, so seeing the final name means a COMPLETE file.
    _LL_WAIT="until [ -f ${LOCAL_LLM_GGUF} ]; do echo waiting for model download; sleep 5; done"
    _LL_CMD="bash -c '${_LL_WAIT}; ${_LL_PLAIN}'"
    [ "$LOCAL_LLM_FAST" = "1" ] && _LL_CMD="bash -c '${_LL_WAIT}; ${_LL_ENHANCED} || ${_LL_PLAIN}'"
    LOCALLLM_PROGRAM=$(cat <<LLEOF

[program:localllm]
command=${_LL_CMD}
autostart=true
autorestart=true
priority=15
startsecs=5
stdout_logfile=/workspace/localllm.log
stderr_logfile=/workspace/localllm.log
LLEOF
)
  fi
fi

echo "==> writing supervisor config"
cat > /etc/supervisor/conf.d/zaptrick.conf <<EOF
[program:postgres]
command=${PGBIN}/postgres -D ${PGDATA} -p 5432
user=postgres
autostart=true
autorestart=true
priority=10
stdout_logfile=/workspace/pg.log
stderr_logfile=/workspace/pg.log

[program:dragonfly]
command=/usr/local/bin/dragonfly --logtostderr --port 6379 --dir /workspace
autostart=true
autorestart=true
priority=10
stdout_logfile=/workspace/dragonfly.log
stderr_logfile=/workspace/dragonfly.log

[program:app]
command=${VENV}/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT} --proxy-headers --forwarded-allow-ips=*
directory=${APP_DIR}
environment=ZAPTHETRICK_CONFIG_PATH="${CFG}",PYTHONUNBUFFERED="1",HF_HOME="${HF_HOME}",ZAPTHETRICK_ENCRYPTION_KEY="${ZAPTHETRICK_ENCRYPTION_KEY:-}",ZAPTHETRICK_AUTH_SECRET="${ZAPTHETRICK_AUTH_SECRET}",ZAPTHETRICK_AUTH_MODE="${ZAPTHETRICK_AUTH_MODE}",ZAPTHETRICK_AUTH_ENFORCE="${ZAPTHETRICK_AUTH_ENFORCE}",GOOGLE_OAUTH_CLIENT_ID="${GOOGLE_OAUTH_CLIENT_ID:-}",GOOGLE_OAUTH_CLIENT_SECRET="${GOOGLE_OAUTH_CLIENT_SECRET:-}"
autostart=true
autorestart=true
priority=20
startsecs=8
stdout_logfile=/workspace/app.log
stderr_logfile=/workspace/app.log

[program:pgbackup]
command=bash -c 'while true; do sleep ${BACKUP_INTERVAL_S}; POSTGRES_PASSWORD="${PGPASS}" BACKUP_DIR="${BACKUP_DIR}" PG_BACKUP_ENC_KEY="${PG_BACKUP_ENC_KEY:-}" bash ${APP_DIR}/deploy/pg_backup.sh; done'
autostart=true
autorestart=true
priority=30
stdout_logfile=/workspace/pgbackup.log
stderr_logfile=/workspace/pgbackup.log

[program:watchdog]
# IMPORTANT: do NOT restart the app during first-boot model warmup. Loading the
# 7B vision model starves the event loop for a couple minutes, so /api/health
# stops answering — but the app is busy, not hung. So: wait until the app is
# healthy ONCE (however long the warmup/model-download takes), and only THEN
# start policing it. This stops the kill-mid-warmup -> reload -> slower loop.
command=bash -c 'until curl -fsS -m 10 http://127.0.0.1:${APP_PORT}/api/health >/dev/null 2>&1; do sleep 15; done; echo "watchdog: app healthy — monitoring"; f=0; while true; do sleep 60; if curl -fsS -m 10 http://127.0.0.1:${APP_PORT}/api/health >/dev/null 2>&1; then f=0; else f=\$((f+1)); if [ \$f -ge 5 ]; then echo "watchdog: \$f consecutive health failures -> restart app"; supervisorctl restart app; f=0; fi; fi; done'
autostart=true
autorestart=true
priority=40
stdout_logfile=/workspace/watchdog.log
stderr_logfile=/workspace/watchdog.log
${LOCALLLM_PROGRAM}
EOF

# ---- 4) durable shutdown: dump once on SIGTERM before stopping ---------------
term_handler() {
  echo "==> SIGTERM: final Postgres dump then shutdown"
  POSTGRES_PASSWORD="$PGPASS" BACKUP_DIR="$BACKUP_DIR" \
    PG_BACKUP_ENC_KEY="${PG_BACKUP_ENC_KEY:-}" \
    bash "$APP_DIR/deploy/pg_backup.sh" || true
  supervisorctl stop all || true
  kill -TERM "${SUPERVISOR_PID:-0}" 2>/dev/null || true
}
trap term_handler SIGTERM SIGINT

echo "==> starting supervisord"
supervisord -n -c /etc/supervisor/supervisord.conf &
SUPERVISOR_PID=$!
wait "$SUPERVISOR_PID"
