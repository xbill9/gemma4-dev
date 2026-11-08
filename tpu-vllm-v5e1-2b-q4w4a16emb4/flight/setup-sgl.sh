#!/bin/bash
# The SGLang-JAX arm of the v5e-1 ladder, on a bare v5e-1 TPU VM (v2-alpha-tpuv5-lite). Run as root:
#   sudo LADDER=/opt/ladder bash setup-sgl.sh
# Python 3.13 from deadsnakes, sglang-jax[tpu] at SGL_COMMIT (its own jax[tpu] pin), the ladder's client packages,
# google/gemma-4-E2B-it into /opt/hf.
set -euo pipefail
L=${LADDER:-/opt/ladder}
SGL_COMMIT=b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6  # sgl-project/sglang-jax main, 2026-10-20
export HOME=${HOME:-/root} HF_HOME=/opt/hf
[ -s $L/hf_token ] && export HF_TOKEN=$(cat $L/hf_token)
mkdir -p /opt/hf
step() { echo "$(date +%T) START $1"; }

step python3.13
if ! command -v python3.13 >/dev/null; then
  apt-get update -qq && apt-get install -y -qq software-properties-common git
  add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -qq
  apt-get install -y -qq python3.13 python3.13-dev
fi
python3.13 -m pip --version >/dev/null 2>&1 || { curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3.13 /tmp/get-pip.py; }

step sglang-jax
python3.13 -m pip install -q --ignore-installed blinker \
  "sglang-jax[tpu] @ git+https://github.com/sgl-project/sglang-jax@${SGL_COMMIT}#subdirectory=python"

step weights
python3.13 -c "from huggingface_hub import snapshot_download as s; print(s('google/gemma-4-E2B-it'), flush=True)"

step versions
{
  echo "sglang-jax $SGL_COMMIT"
  python3.13 -m pip show sglang-jax | grep -E '^Version'
  python3.13 -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
  python3.13 -m pip show libtpu 2>/dev/null | grep -E '^(Name|Version)' | tr '\n' ' '; echo
  python3.13 -c "import transformers; print('transformers', transformers.__version__)"
} > $L/versions-sgl.txt 2>&1
cat $L/versions-sgl.txt
echo "$(date +%T) SETUP-DONE"
