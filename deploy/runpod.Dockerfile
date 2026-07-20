# =============================================================================
# RunPod GPU pod image for zapthetrick_be.
#
# The WHOLE environment AND the app code are baked at build time — Postgres 18 +
# pgvector, Dragonfly, the Python venv + GPU torch (cu128), the sandbox toolchains,
# the app's Python deps, and the source at /opt/zapthetrick_be — so the pod is
# fully self-contained: pull the image, set env vars, and it starts. No git clone,
# no REPO_URL. Only the /workspace-VOLUME-PERSISTENT bits (config.yaml, the model
# cache, and Postgres backups) are handled at RUNTIME by runpod_entrypoint.sh, so
# recreating on any available GPU with the same volume gives back the same system.
#
# Build + push (use deploy/build.sh), then deploy via a saved RunPod template or
# deploy/deploy.sh. The image contains source → publish to a PRIVATE registry.
#   docker build -f deploy/runpod.Dockerfile -t <you>/zapthetrick-runpod:latest .
#   docker push <you>/zapthetrick-runpod:latest
# =============================================================================
# CUDA 12.8 → torch cu128 wheels include Blackwell (sm_120) kernels AND stay
# backward-compatible down to Ampere/Ada, so ONE image runs on any modern RunPod
# GPU (RTX PRO Blackwell, RTX 50-series, A40/A6000/L40S/4090/A5000, A100, …).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PG_MAJOR=18 \
    PYTHONUNBUFFERED=1 \
    VENV=/opt/venv

# 1) Base tools, SSH, core sandbox toolchains, supervisor, and Postgres 18 (from
#    the official PGDG apt repo — Ubuntu ships only PG16).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl wget ca-certificates gnupg unzip sudo lsb-release \
      openssh-server \
      python3 python3-venv python3-dev python3-pip \
      libgomp1 libmagic1 poppler-utils bubblewrap supervisor \
      default-jdk nodejs npm ruby php-cli perl r-base sqlite3 golang-go \
  && install -d /usr/share/postgresql-common/pgdg \
  && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
       -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
  && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
       > /etc/apt/sources.list.d/pgdg.list \
  && apt-get update && apt-get install -y --no-install-recommends \
       postgresql-${PG_MAJOR} postgresql-server-dev-${PG_MAJOR} \
  && rm -rf /var/lib/apt/lists/*

# 2) pgvector, built against PG 18.
RUN cd /tmp && git clone --depth 1 https://github.com/pgvector/pgvector.git \
  && cd pgvector && make && make install && rm -rf /tmp/pgvector

# 3) Rust (sandbox toolchain).
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
  && ln -sf /root/.cargo/bin/rustc /usr/local/bin/rustc \
  && ln -sf /root/.cargo/bin/cargo /usr/local/bin/cargo

# 4) Dragonfly cache binary.
RUN cd /tmp && wget -qO df.tgz \
      https://dragonflydb.gateway.scarf.sh/latest/dragonfly-x86_64.tar.gz \
  && tar xzf df.tgz && mv dragonfly-x86_64 /usr/local/bin/dragonfly \
  && chmod +x /usr/local/bin/dragonfly && rm -f /tmp/df.tgz

# 5) Python venv + GPU torch (cu128) + the app's deps — baked so startup is fast.
#    Only requirements.txt is copied first, so this layer caches across code
#    changes. (`pywinpty` is Windows-only; `pytest` is dev-only.)
#    cu128 wheels carry Blackwell (sm_120) kernels + older archs — one build fits all.
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m venv "$VENV" \
  && "$VENV/bin/pip" install --no-cache-dir --upgrade pip wheel \
  && "$VENV/bin/pip" install --no-cache-dir torch torchvision \
       --index-url https://download.pytorch.org/whl/cu128 \
  && grep -viE '^(pywinpty|pytest)\b' /app/requirements.txt > /tmp/req.txt \
  && "$VENV/bin/pip" install --no-cache-dir -r /tmp/req.txt \
  # 8-bit quantization for the 7B VLM (GPU-only; kept out of requirements.txt so
  # the CPU/Windows desktop build isn't forced to install a CUDA wheel).
  && "$VENV/bin/pip" install --no-cache-dir bitsandbytes

# 6) Bake the app CODE into the image (self-contained — no git clone at boot,
#    no REPO_URL). Copied LAST so a code change doesn't bust the deps layer.
#    Lives at /opt/zapthetrick_be — NOT under /workspace, because the network
#    volume is mounted there at runtime and would hide baked files. `.dockerignore`
#    already keeps secrets/venv/caches/config.yaml out of the context.
#    NOTE: this image now contains your source — publish it to a PRIVATE registry.
COPY . /opt/zapthetrick_be

EXPOSE 8888 22

# 7) Runtime init: env→config, Postgres restore, services under supervisor.
ENTRYPOINT ["bash", "/opt/zapthetrick_be/deploy/runpod_entrypoint.sh"]
