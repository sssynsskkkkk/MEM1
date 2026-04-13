FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    git \
    git-lfs \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    ninja-build && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MEM1
COPY . /workspace/MEM1

RUN python -m pip install --upgrade pip setuptools wheel

# Game24 training only needs HF-based rollout, not vLLM or retriever packages.
RUN python -m pip install --no-cache-dir --root-user-action=ignore \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir --root-user-action=ignore \
    "numpy<2" \
    "transformers==4.55.4" \
    accelerate \
    datasets \
    hydra-core \
    ray \
    "tensordict>=0.6,<0.10" \
    codetiming \
    dill \
    pybind11 \
    pandas \
    pyarrow \
    tqdm \
    scikit-learn \
    nltk \
    scipy \
    sympy \
    sentencepiece \
    protobuf \
    wandb \
    matplotlib \
    huggingface_hub \
    peft \
    IPython

# Install the local training package without its old dependency pins.
RUN python -m pip install --no-cache-dir --root-user-action=ignore -e /workspace/MEM1/Mem1/train --no-deps && \
    python -m nltk.downloader punkt punkt_tab

ENV PYTHONPATH=/workspace/MEM1:/workspace/MEM1/Mem1:/workspace/MEM1/Mem1/train:/workspace/MEM1/Mem1/inference

CMD ["bash"]
