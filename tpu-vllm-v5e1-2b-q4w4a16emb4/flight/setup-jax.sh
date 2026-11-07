#!/bin/bash
# The JAX side of the v5e-1 ladder, on the VM the rig's startup script brought up (vLLM image already patched).
# Run as root:  sudo LADDER=/opt/ladder bash setup-jax.sh
# Python 3.13 from deadsnakes with pip (as jax-dev's workload="jax" startup script), jax[tpu] at the version the
# 2026-11-03 v5e-1 runs used, the servers' packages, Rust nightly, the jaxrust_rs module, and every checkpoint
# downloaded once into /opt/hf (the vLLM container mounts it as its HF cache).
set -euo pipefail
L=${LADDER:-/opt/ladder}
export HOME=${HOME:-/root}  # systemd-run units have no HOME; rustup and cargo need it
export HF_HOME=/opt/hf HF_TOKEN=$(cat $L/hf_token)
mkdir -p /opt/hf
step() { echo "$(date +%T) START $1"; }

step python3.13
if ! command -v python3.13 >/dev/null; then
  apt-get update -qq && apt-get install -y -qq software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -qq
  apt-get install -y -qq python3.13 python3.13-dev
fi
python3.13 -m pip --version >/dev/null 2>&1 || { curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3.13 /tmp/get-pip.py; }

step jax
python3.13 -m pip install -q "jax[tpu]==0.12.0.1" numpy scipy ml_dtypes safetensors huggingface_hub transformers \
  sentencepiece 'jinja2>=3.1' fastapi uvicorn pydantic

step rust
apt-get install -y -qq build-essential pkg-config
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain nightly
. "$HOME/.cargo/env"
(cd $L/jax-dev/tpu-jaxrust-v5e1-2b && PYO3_PYTHON=python3.13 make rust)

step weights
python3.13 - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4", "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text",
             "google/gemma-4-E2B-it", "google/gemma-4-E2B-it-assistant"):
    print(repo, snapshot_download(repo), flush=True)
EOF

step versions
{
  python3.13 -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
  python3.13 -m pip show libtpu 2>/dev/null | grep -E '^(Name|Version)' | tr '\n' ' '; echo
  rustc --version
  # the vLLM image exists only on the rig's own VM; the TorchTPU VMs record their own versions elsewhere
  docker image inspect vllm-tpu-q4w4a16emb4:patched --format 'vllm image {{.Id}}' || echo "no vllm image here"
  docker image inspect vllm-tpu-q4w4a16emb4:patched >/dev/null 2>&1 && \
    docker run --rm --entrypoint python3 vllm-tpu-q4w4a16emb4:patched -c "import vllm; print('vllm', vllm.__version__)" || true
  python3.14 -c "import torch, torch_tpu; print('torch', torch.__version__, 'torch_tpu', torch_tpu.__version__)" 2>/dev/null || true
} > $L/versions-jaxvm.txt 2>&1
cat $L/versions-jaxvm.txt
echo "$(date +%T) SETUP-DONE"
