#!/usr/bin/env bash
# Build the self-contained ZapTheTrick pod image and push it to the registry.
# The image now CONTAINS the app source (baked) — push it to a PRIVATE repo.
#
#   ./deploy/build.sh                 # builds+pushes <IMAGE>:latest
#   ./deploy/build.sh v3              # builds+pushes <IMAGE>:v3 (and :latest)
#
# Reads IMAGE from deploy/.env (or the IMAGE env var). Run from the repo root
# (zapthetrick_be/) so the Docker build context is the whole backend.
set -euo pipefail
cd "$(dirname "$0")/.."                 # → zapthetrick_be/

# Load config (IMAGE=<dockerhub-user>/zapthetrick-runpod)
[ -f deploy/.env ] && set -a && . deploy/.env && set +a
: "${IMAGE:?set IMAGE in deploy/.env (e.g. IMAGE=youruser/zapthetrick-runpod)}"

TAG="${1:-latest}"
echo "==> building $IMAGE:$TAG (context: $(pwd))"
docker build -f deploy/runpod.Dockerfile -t "$IMAGE:$TAG" -t "$IMAGE:latest" .

echo "==> pushing $IMAGE:$TAG"
docker push "$IMAGE:$TAG"
[ "$TAG" != latest ] && docker push "$IMAGE:latest"

echo "✓ pushed $IMAGE:$TAG  — now: ./deploy/deploy.sh   (or Deploy from the saved RunPod template)"
