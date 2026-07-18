#!/usr/bin/env bash
# =============================================================================
# One-shot, IDEMPOTENT bootstrap for zapthetrick_be on a bare RunPod GPU pod
# (no Docker). Brings the WHOLE stack up on first run and is safe to re-run:
#
#   Postgres 16 + pgvector   — data dir on the persistent /workspace volume
#   Dragonfly                — Redis-protocol cache on :6379
#   Python venv + GPU torch  — cu124 wheels for the RTX 3090
#   sandbox toolchains       — core interview languages for sandbox.backend=local
#   config.yaml              — wired to the local PG/cache, local vision, cloud LLM
#   supervisor               — keeps postgres + dragonfly + uvicorn running
#
# EVERYTHING that must survive a pod Stop/Start lives under /workspace (the
# RunPod network volume); the container disk is ephemeral.
#
# Usage (in the pod terminal, after the repo is at /workspace/zapthetrick_be):
#   POSTGRES_PASSWORD=yourpass bash deploy/runpod_bootstrap.sh
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/zapthetrick_be}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
# Local disk, NOT /workspace: RunPod's network volume (MooseFS) can't honor
# Postgres's required chown/0700 perms ("data directory has invalid
# permissions"). Local disk works; DB is wiped on full pod stop (models persist).
PGDATA="${PGDATA:-/var/lib/pgdata}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"
PGPASS="${POSTGRES_PASSWORD:-zaptrick}"
APP_PORT="${APP_PORT:-8888}"
export DEBIAN_FRONTEND=noninteractive

PG_MAJOR="${PG_MAJOR:-18}"

echo "==> [1/8] system packages + Postgres ${PG_MAJOR} (PGDG) + core toolchains"
apt-get update
apt-get install -y --no-install-recommends \
  build-essential git curl wget ca-certificates gnupg unzip sudo lsb-release
# PostgreSQL 18 comes from the official PGDG apt repo (Ubuntu ships only PG16).
install -d /usr/share/postgresql-common/pgdg
curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
  -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc
echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
  > /etc/apt/sources.list.d/pgdg.list
apt-get update
apt-get install -y --no-install-recommends \
  postgresql-${PG_MAJOR} postgresql-server-dev-${PG_MAJOR} \
  libgomp1 libmagic1 poppler-utils bubblewrap supervisor \
  python3-venv python3-dev \
  default-jdk nodejs npm ruby php-cli perl r-base sqlite3 golang-go
# Rust (sandbox) — rustup, minimal profile.
if ! command -v rustc >/dev/null 2>&1; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  ln -sf "$HOME/.cargo/bin/rustc" /usr/local/bin/rustc
  ln -sf "$HOME/.cargo/bin/cargo" /usr/local/bin/cargo
fi
# NOTE: Swift, Dart, .NET, Kotlin, Elixir, Racket, … are NOT installed here to
# keep first-boot fast. Add them from sandbox/Dockerfile's install lines as you
# need those languages verified; sandbox.backend=local just needs them on PATH.

PGBIN="$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)"

echo "==> [2/8] pgvector (build + install into the PG lib dir)"
if ! find /usr/lib/postgresql -name vector.so 2>/dev/null | grep -q .; then
  cd /tmp && rm -rf pgvector
  git clone --depth 1 https://github.com/pgvector/pgvector.git
  cd pgvector && make && make install
fi

echo "==> [3/8] Postgres cluster on the persistent volume ($PGDATA)"
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  mkdir -p "$PGDATA" && chown postgres:postgres "$PGDATA" && chmod 700 "$PGDATA"
  sudo -u postgres "$PGBIN/initdb" -D "$PGDATA" -E UTF8
fi
# Start it just long enough to set the password + enable pgvector; supervisor
# owns it from here on.
if ! sudo -u postgres "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
  sudo -u postgres "$PGBIN/pg_ctl" -D "$PGDATA" -l /workspace/pg.log \
    -o "-p 5432" -w start
fi
sudo -u postgres psql -p 5432 -c "ALTER USER postgres PASSWORD '${PGPASS}';"
sudo -u postgres psql -p 5432 -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -p 5432 -tAc "SELECT 'pgvector OK' FROM pg_extension WHERE extname='vector';"

echo "==> [4/8] Dragonfly cache"
if [ ! -x /usr/local/bin/dragonfly ]; then
  cd /tmp && wget -qO df.tgz \
    https://dragonflydb.gateway.scarf.sh/latest/dragonfly-x86_64.tar.gz
  tar xzf df.tgz
  mv dragonfly-x86_64 /usr/local/bin/dragonfly && chmod +x /usr/local/bin/dragonfly
fi

echo "==> [5/8] Python venv + GPU torch (cu124) + app deps"
[ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel
"$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124
grep -viE '^(pywinpty|pytest)\b' "$REPO_DIR/requirements.txt" > /tmp/req.linux.txt
"$VENV_DIR/bin/pip" install -r /tmp/req.linux.txt
# 8-bit quantization for the 7B VLM (GPU-only; not in requirements.txt).
"$VENV_DIR/bin/pip" install bitsandbytes

echo "==> [6/8] config.yaml (written once; Settings changes are preserved)"
CFG="$REPO_DIR/config.yaml"
if [ ! -f "$CFG" ]; then
  cp "$REPO_DIR/config.example.yaml" "$CFG"
  APP_PORT="$APP_PORT" "$VENV_DIR/bin/python" - "$CFG" "$PGPASS" <<'PY'
import os, sys, yaml
path, pgpass = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(path)) or {}
c.setdefault('database', {}).setdefault('postgres', {}).update(
    host='127.0.0.1', port=5432, db='postgres', schema_name='zapthetrick',
    user='postgres', password=pgpass, enable_age=False)
c['database'].setdefault('cache', {})['url'] = 'redis://127.0.0.1:6379'
c.setdefault('vision', {})['mode'] = 'local'        # powerful local VLM on the GPU
c.setdefault('sandbox', {})['backend'] = 'local'    # no docker daemon on the pod
c['sandbox']['enabled'] = True
c.setdefault('server', {}).update(host='0.0.0.0', port=int(os.environ['APP_PORT']))
yaml.safe_dump(c, open(path, 'w'), sort_keys=False)
print('wrote', path)
PY
fi
mkdir -p "$HF_HOME"

echo "==> [7/8] supervisor (postgres, dragonfly, app — all auto-restart)"
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
command=${VENV_DIR}/bin/uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT}
directory=${REPO_DIR}
environment=ZAPTHETRICK_CONFIG_PATH="${CFG}",PYTHONUNBUFFERED="1",HF_HOME="${HF_HOME}"
autostart=true
autorestart=true
priority=20
startsecs=8
stdout_logfile=/workspace/app.log
stderr_logfile=/workspace/app.log
EOF

echo "==> [8/8] start everything"
# Stop the ad-hoc postgres we started in step 3 so supervisor owns the one copy.
sudo -u postgres "$PGBIN/pg_ctl" -D "$PGDATA" -w stop >/dev/null 2>&1 || true
if ! supervisorctl status >/dev/null 2>&1; then
  supervisord -c /etc/supervisor/supervisord.conf
fi
supervisorctl reread && supervisorctl update
supervisorctl restart app || true

echo ""
echo "=========================================================="
echo " Bootstrap complete. The app runs its own DB migrations on"
echo " startup. Health check:  curl -s localhost:${APP_PORT}/api/health"
echo " Logs:  tail -f /workspace/app.log"
echo " Next: open Settings -> Vision / STT and pick the powerful"
echo "       local models; add your cloud LLM key in Providers."
echo "=========================================================="
