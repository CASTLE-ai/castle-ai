# =============================================================================
# CASTLE AI — Multi-stage Docker build
# GPU build (default):  docker build -t castle-ai/castle .
# CPU-only build:       docker build --build-arg DEVICE=cpu -t castle-ai/castle:cpu .
# Pre-download ckpts:   docker build --build-arg DOWNLOAD_CKPT=1 .
# =============================================================================

ARG DEVICE=gpu
ARG CUDA_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — base: CUDA runtime + Python 3.10 + system libraries
# ─────────────────────────────────────────────────────────────────────────────
FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3.10-dev \
        # Media processing
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        # Misc utilities
        curl \
        wget \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create and activate a venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — deps: install Python packages (GPU path)
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS deps-gpu

# PyTorch + torchvision with CUDA 12.6 wheels (compatible with CUDA 12.4 runtime)
RUN pip install torch torchvision \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
    && pip cache purge

# Copy requirements and install the rest (excluding GPU extras already installed)
COPY requirements.txt /tmp/requirements.txt

# Install everything EXCEPT gpu-only lines with special indexes
RUN grep -v '^\s*--extra-index-url' /tmp/requirements.txt \
        | grep -v 'xformers' \
        | grep -v 'cuml-cu12' \
    > /tmp/requirements-base.txt \
 && pip install -r /tmp/requirements-base.txt \
    --extra-index-url https://pypi.nvidia.com \
    && pip cache purge

# Install GPU-only extras
RUN pip install xformers \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
    && pip install cuml-cu12 \
        --extra-index-url https://pypi.nvidia.com \
    && pip cache purge

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 (alt) — deps: CPU-only path
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS deps-cpu

RUN pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip cache purge

COPY requirements.txt /tmp/requirements.txt

RUN grep -v '^\s*--extra-index-url' /tmp/requirements.txt \
        | grep -v 'xformers' \
        | grep -v 'cuml-cu12' \
    > /tmp/requirements-base.txt \
 && pip install -r /tmp/requirements-base.txt \
    && pip cache purge

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — checkpoint pre-download (optional, add --build-arg DOWNLOAD_CKPT=1)
# Runs after deps-gpu; for CPU builds the entrypoint handles downloads at runtime.
# ─────────────────────────────────────────────────────────────────────────────
FROM deps-gpu AS deps-with-ckpt

ARG DOWNLOAD_CKPT=0

RUN mkdir -p /models

RUN if [ "$DOWNLOAD_CKPT" = "1" ]; then \
        echo "Pre-downloading model checkpoints …" && \
        wget -q --show-progress \
            -O /models/sam_vit_b_01ec64.pth \
            https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth && \
        wget -q --show-progress \
            -O /models/dinov2_vitb14_reg4_pretrain.pth \
            https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth && \
        python -m gdown 1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ \
            -O /models/R50_DeAOTL_PRE_YTB_DAV.pth && \
        echo "Checkpoints downloaded." ; \
    else \
        echo "Skipping checkpoint pre-download (pass --build-arg DOWNLOAD_CKPT=1 to embed)." ; \
    fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — app: final image
# Select deps-with-ckpt (GPU) or deps-cpu depending on build arg
# ─────────────────────────────────────────────────────────────────────────────
FROM deps-with-ckpt AS app-gpu
FROM deps-cpu      AS app-cpu

# Determine which base to use (docker BuildKit ARG trick)
ARG DEVICE=gpu
FROM app-${DEVICE} AS app

# ── Non-root user ────────────────────────────────────────────────────────────
RUN groupadd --gid 1001 castle \
 && useradd  --uid 1001 --gid castle --shell /bin/bash --create-home castle

# ── Application code ─────────────────────────────────────────────────────────
WORKDIR /app

COPY --chown=castle:castle . .

# Link or create checkpoint/data directories
RUN mkdir -p /models /data \
 && chown -R castle:castle /models /data \
 && ln -sfn /models /app/ckpt

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PATH="/opt/venv/bin:$PATH" \
    CASTLE_DEVICE=auto \
    CASTLE_DATA=/data \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HDF5_USE_FILE_LOCKING=FALSE

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY --chown=castle:castle scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER castle

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:7860/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD []
