# ── Build args ────────────────────────────────────────────────────────────
# PYTORCH_CHANNEL controls which PyTorch wheel index is used:
#   cu121   → CUDA 12.1  (NVIDIA — default for most HPC clusters)
#   rocm6.2 → ROCm 6.2   (AMD — required for LUMI supercomputer)
#   cpu     → CPU-only   (testing / preprocessing nodes)
#
# Example:
#   docker build --build-arg PYTORCH_CHANNEL=rocm6.2 -t csdp:lumi .
ARG PYTORCH_CHANNEL=cu121

FROM python:3.12-slim

WORKDIR /workspace

# System libs needed by MNE, h5py, wfdb
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first so the constraint in pyproject.toml is already met
# and pip won't pull the large CUDA default when resolving deps.
ARG PYTORCH_CHANNEL
RUN pip install --no-cache-dir \
    "torch~=2.11.0" \
    --index-url "https://download.pytorch.org/whl/${PYTORCH_CHANNEL}"

# Copy package source (data dirs and weights are excluded via .dockerignore)
COPY pyproject.toml README.md ./
COPY csdp_datastore/   csdp_datastore/
COPY csdp_pipeline/    csdp_pipeline/
COPY csdp_training/    csdp_training/

# NOTE: ml_architectures (USleep, LSeqSleepNet) is a private dependency
# from gitlab.au.dk. Install it separately before or after this image if needed:
#   pip install git+https://gitlab.au.dk/tech_ear-eeg/.../ml_architectures.git

RUN pip install --no-cache-dir -e .

# Verify the package imports cleanly at build time
RUN python -c "import csdp_datastore, csdp_pipeline, csdp_training; print('OK')"
