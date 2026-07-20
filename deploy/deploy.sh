#!/usr/bin/env bash
# One-command deploy to RunPod: creates a pod on the FIRST AVAILABLE GPU from a
# priority list, attaching your network volume + env — no RunPod UI needed.
#
#   ./deploy/deploy.sh
#
# Reads deploy/.env. Requires: RUNPOD_API_KEY, IMAGE, RUNPOD_VOLUME_ID.
# The pod's entrypoint self-configures (config from env, DB restore, services) —
# nothing to do after this returns except wait for /api/health.
#
# NOTE: RunPod's REST API field names occasionally change; this targets the v1
# endpoint (https://rest.runpod.io/v1/pods). If a field is rejected, check the
# current schema at https://docs.runpod.io/api-reference — the shape here is the
# stable core (image, gpu list, volume, ports, env).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . .env && set +a

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in deploy/.env}"
: "${IMAGE:?set IMAGE in deploy/.env}"
: "${RUNPOD_VOLUME_ID:?set RUNPOD_VOLUME_ID in deploy/.env (create the volume once in the RunPod UI/API)}"

POD_NAME="${POD_NAME:-zapthetrick}"
IMAGE_TAG="${IMAGE}:${IMAGE_TAG:-latest}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-40}"
# GPU priority: first one with stock wins (all run the cu124 image; 24–48 GB).
GPU_PRIORITY="${GPU_PRIORITY:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA GeForce RTX 4090,NVIDIA RTX A5000}"

# Env vars passed INTO the pod (the pod self-configures from these).
# Add/remove keys here — they land in the pod's environment.
ENV_JSON=$(jq -n \
  --arg app_port "${APP_PORT:-8888}" \
  --arg enc "${ZAPTHETRICK_ENCRYPTION_KEY:-}" \
  --arg pgpw "${POSTGRES_PASSWORD:-}" \
  --arg ork "${OPENROUTER_API_KEY:-}" \
  --arg nvk "${NVIDIA_API_KEY:-}" \
  '[
     {key:"APP_PORT", value:$app_port},
     {key:"ZAPTHETRICK_ENCRYPTION_KEY", value:$enc},
     {key:"POSTGRES_PASSWORD", value:$pgpw},
     {key:"OPENROUTER_API_KEY", value:$ork},
     {key:"NVIDIA_API_KEY", value:$nvk}
   ] | map(select(.value != ""))')

# Build the GPU id array from the priority list.
GPU_IDS=$(printf '%s' "$GPU_PRIORITY" | jq -R 'split(",") | map(gsub("^ +| +$";""))')

BODY=$(jq -n \
  --arg name "$POD_NAME" \
  --arg image "$IMAGE_TAG" \
  --arg vol "$RUNPOD_VOLUME_ID" \
  --argjson disk "$CONTAINER_DISK_GB" \
  --argjson gpus "$GPU_IDS" \
  --argjson env "$ENV_JSON" \
  '{
     name: $name,
     imageName: $image,
     gpuTypeIds: $gpus,
     gpuCount: 1,
     networkVolumeId: $vol,
     volumeMountPath: "/workspace",
     containerDiskInGb: $disk,
     ports: ["8888/http","22/tcp"],
     env: $env
   }')

echo "==> deploying $POD_NAME  (GPU priority: $GPU_PRIORITY)"
RESP=$(curl -fsS -X POST "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$BODY") || { echo "deploy failed — no GPU available or API error"; exit 1; }

POD_ID=$(printf '%s' "$RESP" | jq -r '.id // .pod.id // empty')
if [ -z "$POD_ID" ]; then echo "unexpected response:"; echo "$RESP" | jq .; exit 1; fi

echo "✓ pod $POD_ID deploying"
echo "  URL (once ready): https://${POD_ID}-8888.proxy.runpod.net"
echo "  health:           https://${POD_ID}-8888.proxy.runpod.net/api/health"
echo "  first boot on a NEW volume ~10–15 min (model download); after that ~2–3 min."
