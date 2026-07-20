#!/usr/bin/env bash
# One Postgres dump + rotate. Called periodically (supervisor `pgbackup`) and on
# SIGTERM (entrypoint trap). Dumps the whole DB in custom format to the
# /workspace network volume so it survives pod terminate/recreate (§6.3).
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/workspace/pg_backups}"
KEEP="${BACKUP_KEEP:-48}"
PGBIN="$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)"

mkdir -p "$BACKUP_DIR"
ts="$(date +%Y%m%d-%H%M%S)"
tmp="$BACKUP_DIR/.dump-$ts.tmp"

# -Fc = custom format (compressed, restorable with pg_restore). Password via env.
if PGPASSWORD="${POSTGRES_PASSWORD:-zaptrick}" "$PGBIN/pg_dump" \
      -Fc -h 127.0.0.1 -p 5432 -U postgres postgres > "$tmp" 2>/dev/null; then
  mv "$tmp" "$BACKUP_DIR/dump-$ts.dump"
  ln -sfn "dump-$ts.dump" "$BACKUP_DIR/latest.dump"
  # Rotate: keep the newest $KEEP dumps.
  ls -1t "$BACKUP_DIR"/dump-*.dump 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
  echo "pg_backup: wrote $BACKUP_DIR/dump-$ts.dump"
else
  rm -f "$tmp"
  echo "pg_backup: dump skipped (Postgres not ready?)" >&2
fi
