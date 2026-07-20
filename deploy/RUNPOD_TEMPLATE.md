# Zero-config RunPod deploy — one-time setup, then recreate anywhere

The image is **self-contained** (code + env baked). A pod self-configures on boot:
renders `config.yaml` from env, restores Postgres from the `/workspace` backup,
starts Postgres + Dragonfly + the app under supervisor, and runs a health
watchdog — **no web-terminal steps, ever**. Recreate on any available GPU with
the same volume → the same system, data intact.

## One-time setup (do these once)

1. **Docker Hub (private) repo** — the image contains your source. Create a
   *private* repo, e.g. `youruser/zapthetrick-runpod`.
2. **Master key** — generate once, keep stable forever (it encrypts the provider
   keys you add in the UI; changing it makes them undecryptable). **64 hex chars:**
   `python -c "import secrets; print(secrets.token_hex(32))"`
   *(You do NOT set provider keys in env — you add them in the Providers screen;
   they're encrypted with this key and stored in Postgres, which the backups persist.)*
3. **RunPod network volume** — create a volume mounted at `/workspace`
   (50–100 GB) in a **datacenter that stocks several of your GPUs** (A40 / A6000 /
   L40S / RTX 4090 / A5000). Volumes are region-locked, so "any available GPU"
   means "any in that datacenter". Note its **volume id**.
4. `cp deploy/.env.example deploy/.env` and fill it in.

## Build + push (whenever code changes)

```bash
./deploy/build.sh            # or: ./deploy/build.sh v3
```

## Deploy — pick ONE

### A) One command (no UI)
```bash
./deploy/deploy.sh
```
Deploys on the first available GPU from `GPU_PRIORITY`, attaches the volume +
env, prints the URL. Needs `RUNPOD_API_KEY` + `RUNPOD_VOLUME_ID` in `.env`.

### B) Saved RunPod Template (UI-minimal)
Create a Template once (Templates → New), then every deploy is **pick GPU →
Deploy**:

| Field | Value |
|---|---|
| **Container Image** | `youruser/zapthetrick-runpod:latest` (+ registry creds for the private repo) |
| **Container Disk** | 40 GB |
| **Volume Mount Path** | `/workspace` |
| **Expose HTTP Ports** | `8888` |
| **Expose TCP Ports** | `22` |
| **Env variables** | `ZAPTHETRICK_ENCRYPTION_KEY` (required) · `APP_PORT=8888` · optional `POSTGRES_PASSWORD`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY` |
| **GPU Compatibility** | Min vRAM **24 GB**, Min RAM **32 GB**, CUDA **≥12.4**; compatible GPUs A40 / A6000 / L40S / RTX 4090 / RTX 3090 / A5000 (**not** RTX 50-series — Blackwell needs a cu128 image) |

To deploy: **Deploy a Pod → choose any available compatible GPU → select this
template → attach the volume → Deploy.** That's the only UI you touch.

## After deploy
- Ready at `https://<pod-id>-8888.proxy.runpod.net` — check `/api/health`.
- First boot on a **new** volume: ~10–15 min (model downloads to `/workspace`).
- Every boot after: ~2–3 min (models + code cached; DB restored from backup).
- **To move GPUs / recover from a stopped-pod-out-of-stock**: just **terminate**
  and **create a fresh pod** on any available compatible GPU with the same
  volume. Data is restored from `/workspace/pg_backups/latest.dump` (≤30 min old).

## What persists vs. what's ephemeral
| On `/workspace` volume (survives) | On pod local disk (ephemeral) |
|---|---|
| `config.yaml`, HF model cache, `pg_backups/` | Postgres data dir (`/var/lib/pgdata`) — but restored from the backup on boot |

Nothing else is required. Provider keys added via the app's Providers screen are
encrypted with the master key and stored in Postgres → they ride the backups too.
