# syntax=docker/dockerfile:1
# Multi-arch (linux/amd64, linux/arm64). Torch ships CUDA 13 libs in its wheels, so the base
# image only needs to be compatible with driver >= 580 (orion: 595).
FROM nvidia/cuda:13.0.3-base-ubuntu24.04 AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

# gcc: Triton JIT-compiles a small launcher at runtime (flash-linear-attention kernels).
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./

# ---- runtime image (no dev deps) ----
FROM base AS runtime
RUN uv sync --frozen --no-dev --no-install-project
COPY scripts ./scripts
COPY config ./config

# ---- dev image: adds ruff + pytest; source is bind-mounted by compose ----
FROM base AS dev
RUN uv sync --frozen --no-install-project
COPY scripts ./scripts
COPY tests ./tests
COPY config ./config

# Default target = runtime (last stage must be the one built without --target).
FROM runtime
