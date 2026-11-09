#!/bin/bash
# tpu-sglangjax-v5e1-2b: SGLang-JAX with the Gemma 4 E2B patch, on a bare v5e-1 TPU VM (v2-alpha-tpuv5-lite). Run as root:
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

step patch
# Gemma 4 E2B on sglang-jax: patches/gemma4-e2b.diff (null global KV heads, per-layer embeddings, double-wide MLP,
# KV-shared layers reading their source layer's K/V). Applied to the installed package.
SP=$(python3.13 -c "import sgl_jax, os; print(os.path.dirname(os.path.dirname(sgl_jax.__file__)))")
if patch -d "$SP" -p2 -R --dry-run -s < $L/patches/gemma4-e2b.diff >/dev/null 2>&1; then echo "patch already applied"
else patch -d "$SP" -p2 --forward < $L/patches/gemma4-e2b.diff; fi
python3.13 -c "import sgl_jax.srt.models.gemma4 as m; assert hasattr(m, '_kv_share_source'); print('patch ok')"

step torch-cpu
# transformers builds Gemma4Processor only with torch + torchvision; without them the server dies at
# TokenizerManager (2026-11-08 19:11, "Could not import module 'Gemma4Processor'"). CPU wheels: the TPU does the work.
python3.13 -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3.13 -c "from transformers import Gemma4Processor; print('Gemma4Processor ok')"

step weights
python3.13 -c "from huggingface_hub import snapshot_download as s; print(s('google/gemma-4-E2B-it'), flush=True)"

step versions
{
  echo "sglang-jax $SGL_COMMIT + patches/gemma4-e2b.diff sha256 $(sha256sum $L/patches/gemma4-e2b.diff | cut -c1-16)"
  python3.13 -m pip show sglang-jax | grep -E '^Version'
  python3.13 -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
  python3.13 -m pip show libtpu 2>/dev/null | grep -E '^(Name|Version)' | tr '\n' ' '; echo
  python3.13 -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
} > $L/versions-sgl.txt 2>&1
cat $L/versions-sgl.txt
echo "$(date +%T) SETUP-DONE"
