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
PGBIN="$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)"
mkdir -p /workspace "$HF_HOME" "$BACKUP_DIR"

echo "==> ssh"
(service ssh start 2>/dev/null || /usr/sbin/sshd 2>/dev/null || true)

# ---- 1) config.yaml on the volume — rendered from env once, then preserved ---
if [ ! -f "$CFG" ]; then
  echo "==> rendering $CFG from env (first boot on this volume)"
  APP_PORT="$APP_PORT" PGPASS="$PGPASS" \
  OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" NVIDIA_API_KEY="${NVIDIA_API_KEY:-}" \
  "$VENV/bin/python" - "$APP_DIR/config.example.yaml" "$CFG" <<'PY'
import os, sys, yaml
src, dst = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(src)) or {}
c.setdefault('database', {}).setdefault('postgres', {}).update(
    host='127.0.0.1', port=5432, db='postgres', schema_name='zapthetrick',
    user='postgres', password=os.environ['PGPASS'], enable_age=False)
c['database'].setdefault('cache', {})['url'] = 'redis://127.0.0.1:6379'
c.setdefault('vision', {})['mode'] = 'local'
c.setdefault('sandbox', {}).update(backend='local', enabled=True)
c.setdefault('server', {}).update(host='0.0.0.0', port=int(os.environ['APP_PORT']))
# Optional provider keys from env → so a fresh volume needs no UI to answer.
if os.environ.get('OPENROUTER_API_KEY'):
    c.setdefault('llm', {})['openrouter_api_key'] = os.environ['OPENROUTER_API_KEY']
if os.environ.get('NVIDIA_API_KEY'):
    c.setdefault('llm', {})['nvidia_api_key'] = os.environ['NVIDIA_API_KEY']
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
print('wrote', dst)
PY
else
  echo "==> $CFG exists — preserving Settings/keys already configured"
fi

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
  echo "==> fresh DB + backup found → pg_restore $BACKUP_DIR/latest.dump"
  PGPASSWORD="$PGPASS" "$PGBIN/pg_restore" --clean --if-exists --no-owner \
      -h 127.0.0.1 -p 5432 -U postgres -d postgres \
      "$BACKUP_DIR/latest.dump" 2>>/workspace/pg_restore.log || \
    echo "   (restore reported non-fatal errors — see /workspace/pg_restore.log)"
fi
sudo -u postgres "$PGBIN/pg_ctl" -D "$PGDATA" -w stop   # supervisor owns it now

# ---- 3) supervisor: postgres + dragonfly + app + periodic backup + watchdog ---
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
command=${VENV}/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT}
directory=${APP_DIR}
environment=ZAPTHETRICK_CONFIG_PATH="${CFG}",PYTHONUNBUFFERED="1",HF_HOME="${HF_HOME}",ZAPTHETRICK_ENCRYPTION_KEY="${ZAPTHETRICK_ENCRYPTION_KEY:-}"
autostart=true
autorestart=true
priority=20
startsecs=8
stdout_logfile=/workspace/app.log
stderr_logfile=/workspace/app.log

[program:pgbackup]
command=bash -c 'while true; do sleep ${BACKUP_INTERVAL_S}; POSTGRES_PASSWORD="${PGPASS}" BACKUP_DIR="${BACKUP_DIR}" bash ${APP_DIR}/deploy/pg_backup.sh; done'
autostart=true
autorestart=true
priority=30
stdout_logfile=/workspace/pgbackup.log
stderr_logfile=/workspace/pgbackup.log

[program:watchdog]
command=bash -c 'f=0; while true; do sleep 60; if curl -fsS -m 5 http://127.0.0.1:${APP_PORT}/api/health >/dev/null 2>&1; then f=0; else f=\$((f+1)); if [ \$f -ge 3 ]; then echo "watchdog: 3 health failures -> restart app"; supervisorctl restart app; f=0; fi; fi; done'
autostart=true
autorestart=true
priority=40
stdout_logfile=/workspace/watchdog.log
stderr_logfile=/workspace/watchdog.log
EOF

# ---- 4) durable shutdown: dump once on SIGTERM before stopping ---------------
term_handler() {
  echo "==> SIGTERM: final Postgres dump then shutdown"
  POSTGRES_PASSWORD="$PGPASS" BACKUP_DIR="$BACKUP_DIR" bash "$APP_DIR/deploy/pg_backup.sh" || true
  supervisorctl stop all || true
  kill -TERM "${SUPERVISOR_PID:-0}" 2>/dev/null || true
}
trap term_handler SIGTERM SIGINT

echo "==> starting supervisord"
supervisord -n -c /etc/supervisor/supervisord.conf &
SUPERVISOR_PID=$!
wait "$SUPERVISOR_PID"
