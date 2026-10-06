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

# ---- builder: compile causal-conv1d against the locked torch (needs nvcc; not shipped) ----
# No prebuilt wheel exists for torch 2.14 + cu13 (upstream only publishes torch<=2.10 / NGC
# builds), and a wheel built for another torch minor is ABI-incompatible.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04 AS cc-builder
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH
ARG CAUSAL_CONV1D_VERSION=1.7.0
# Compile for orion's RTX 3090 (sm_86); override for other GPUs.
ARG TORCH_CUDA_ARCH_LIST=8.6
ARG MAX_JOBS=4
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
 && uv pip install pip setuptools wheel ninja packaging
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} \
    python -m pip wheel --no-build-isolation --no-deps "causal-conv1d==${CAUSAL_CONV1D_VERSION}" -w /wheels

# ---- runtime image (no dev deps) ----
FROM base AS runtime
RUN uv sync --frozen --no-dev --no-install-project
COPY --from=cc-builder /wheels /wheels
RUN uv pip install --no-deps /wheels/*.whl && rm -rf /wheels
COPY scripts ./scripts
COPY config ./config

# ---- dev image: adds ruff + pytest; source is bind-mounted by compose ----
FROM base AS dev
RUN uv sync --frozen --no-install-project
COPY --from=cc-builder /wheels /wheels
RUN uv pip install --no-deps /wheels/*.whl && rm -rf /wheels
COPY scripts ./scripts
COPY tests ./tests
COPY config ./config

# Default target = runtime (last stage must be the one built without --target).
FROM runtime
