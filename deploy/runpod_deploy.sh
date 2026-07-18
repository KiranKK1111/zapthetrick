#!/usr/bin/env bash
# Fast redeploy for an ALREADY-bootstrapped RunPod pod. Pulls the latest code,
# installs any new deps, and restarts just the app (postgres + dragonfly keep
# running under supervisor). This is what the `deploy` alias / auto-deploy cron
# calls on every push — NOT the full bootstrap.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/zapthetrick_be}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
BRANCH="${BRANCH:-main}"

cd "$REPO_DIR"
BEFORE="$(git rev-parse HEAD 2>/dev/null || echo none)"
git fetch origin "$BRANCH" -q
git reset --hard "origin/$BRANCH"
AFTER="$(git rev-parse HEAD)"

# Only reinstall deps when requirements actually changed (keeps redeploys fast).
if [ "$BEFORE" = none ] || ! git diff --quiet "$BEFORE" "$AFTER" -- requirements.txt; then
  echo "requirements changed → installing"
  grep -viE '^(pywinpty|pytest)\b' requirements.txt > /tmp/req.linux.txt
  "$VENV_DIR/bin/pip" install -r /tmp/req.linux.txt -q
fi

supervisorctl restart app
echo "deployed $AFTER"
