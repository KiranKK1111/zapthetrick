# =============================================================================
# RunPod GPU pod image for zapthetrick_be.
#
# The WHOLE environment is baked at build time — Postgres 18 + pgvector,
# Dragonfly, the Python venv + GPU torch (cu124), the sandbox toolchains, and
# the app's Python deps — so a pod starts in seconds instead of running a
# multi-minute bootstrap. Only the /workspace-VOLUME-dependent bits (the DB data
# dir, config.yaml, model cache, and the code checkout for auto-deploy) are done
# at RUNTIME by deploy/runpod_entrypoint.sh.
#
# Build + push to a registry, then point the Terraform `image_name` at the tag
# and set `use_custom_image = true`:
#   docker build -f deploy/runpod.Dockerfile -t <you>/zapthetrick-runpod:latest .
#   docker push <you>/zapthetrick-runpod:latest
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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

# 5) Python venv + GPU torch (cu124) + the app's deps — baked so startup is fast.
#    Only requirements.txt is copied first, so this layer caches across code
#    changes. (`pywinpty` is Windows-only; `pytest` is dev-only.)
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m venv "$VENV" \
  && "$VENV/bin/pip" install --no-cache-dir --upgrade pip wheel \
  && "$VENV/bin/pip" install --no-cache-dir torch torchvision \
       --index-url https://download.pytorch.org/whl/cu124 \
  && grep -viE '^(pywinpty|pytest)\b' /app/requirements.txt > /tmp/req.txt \
  && "$VENV/bin/pip" install --no-cache-dir -r /tmp/req.txt \
  # 8-bit quantization for the 7B VLM (GPU-only; kept out of requirements.txt so
  # the CPU/Windows desktop build isn't forced to install a CUDA wheel).
  && "$VENV/bin/pip" install --no-cache-dir bitsandbytes

# 6) Deploy scripts only — these hold the ENTRYPOINT. The backend CODE is NOT
#    baked into the image; it's pulled from your repo to /workspace at runtime
#    (set REPO_URL). So the image is a reusable ENVIRONMENT, your code stays
#    private in Bitbucket, and a code change never needs an image rebuild.
COPY deploy/ /app/deploy/

EXPOSE 8888 22

# 7) Runtime init (DB dir + config + code + services on the persistent volume).
ENTRYPOINT ["bash", "/app/deploy/runpod_entrypoint.sh"]
