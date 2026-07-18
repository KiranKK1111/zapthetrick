#!/usr/bin/env bash
# Runtime init for the RunPod image (deploy/runpod.Dockerfile). The image already
# has Postgres 18 + pgvector, Dragonfly, the venv + GPU torch, toolchains and app
# deps baked in — this only does the /workspace-VOLUME-dependent setup, then
# hands off to supervisor (which becomes PID 1 and keeps the pod alive).
set -euo pipefail

VENV="${VENV:-/opt/venv}"
PGDATA="${PGDATA:-/workspace/pgdata}"
PGPASS="${POSTGRES_PASSWORD:-zaptrick}"
APP_PORT="${APP_PORT:-8888}"
BRANCH="${BRANCH:-main}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
PGBIN="$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)"
mkdir -p /workspace "$HF_HOME"

echo "==> ssh"
(service ssh start 2>/dev/null || /usr/sbin/sshd 2>/dev/null || true)

echo "==> code (pulled to /workspace; the image ships no app code)"
if [ -z "${REPO_URL:-}" ]; then
  echo "FATAL: REPO_URL is not set. This image is the ENVIRONMENT only — set" >&2
  echo "REPO_URL to your backend git URL so the code can be pulled." >&2
  exit 1
fi
[ -d /workspace/zapthetrick_be/.git ] || \
  git clone -b "$BRANCH" "$REPO_URL" /workspace/zapthetrick_be
APP_DIR=/workspace/zapthetrick_be
# Reconcile deps with the pulled code — fast if already satisfied by the image.
grep -viE '^(pywinpty|pytest)\b' "$APP_DIR/requirements.txt" > /tmp/req.txt || true
"$VENV/bin/pip" install -q -r /tmp/req.txt || true

echo "==> Postgres data dir on the persistent volume"
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  mkdir -p "$PGDATA" && chown postgres:postgres "$PGDATA" && chmod 700 "$PGDATA"
  sudo -u postgres "$PGBIN/initdb" -D "$PGDATA" -E UTF8
fi
sudo -u postgres "$PGBIN/pg_ctl" -D "$PGDATA" -l /workspace/pg.log -o "-p 5432" -w start
sudo -u postgres psql -p 5432 -c "ALTER USER postgres PASSWORD '${PGPASS}';"
sudo -u postgres psql -p 5432 -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres "$PGBIN/pg_ctl" -D "$PGDATA" -w stop   # supervisor owns it now

echo "==> config.yaml (written once; Settings changes are preserved)"
CFG="$APP_DIR/config.yaml"
if [ ! -f "$CFG" ]; then
  cp "$APP_DIR/config.example.yaml" "$CFG"
  APP_PORT="$APP_PORT" "$VENV/bin/python" - "$CFG" "$PGPASS" <<'PY'
import os, sys, yaml
path, pw = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(path)) or {}
c.setdefault('database', {}).setdefault('postgres', {}).update(
    host='127.0.0.1', port=5432, db='postgres', schema_name='zapthetrick',
    user='postgres', password=pw, enable_age=False)
c['database'].setdefault('cache', {})['url'] = 'redis://127.0.0.1:6379'
c.setdefault('vision', {})['mode'] = 'local'
c.setdefault('sandbox', {}).update(backend='local', enabled=True)
c.setdefault('server', {}).update(host='0.0.0.0', port=int(os.environ['APP_PORT']))
yaml.safe_dump(c, open(path, 'w'), sort_keys=False)
print('wrote', path)
PY
fi

echo "==> supervisor (postgres + dragonfly + app)"
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
environment=ZAPTHETRICK_CONFIG_PATH="${CFG}",PYTHONUNBUFFERED="1",HF_HOME="${HF_HOME}"
autostart=true
autorestart=true
priority=20
startsecs=8
stdout_logfile=/workspace/app.log
stderr_logfile=/workspace/app.log
EOF

echo "==> starting supervisord (foreground)"
exec supervisord -n -c /etc/supervisor/supervisord.conf
