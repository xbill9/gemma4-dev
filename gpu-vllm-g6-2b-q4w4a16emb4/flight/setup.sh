#!/bin/bash
# Bring every engine of the G6 flight up on one instance, in dependency order. Run as root on the instance
# (gpu-pytorch-g6-2b-q4w4a16's create_g6_instance; that rig's bootstrap has already installed the PyTorch
# engine and written /opt/torch-g6-q4/PYTHON_BIN). The monorepo is at $GEMMA4_DEV.
#
#   GEMMA4_DEV=/opt/gemma4-dev MODEL_S3_URI=s3://... bash setup.sh [step...]
#
# Steps run in the order given (default: all). Each logs to /opt/flight-setup/<step>.log and stops the
# script on failure, so a rerun with the failed step and those after it picks up from there.
set -euo pipefail
R=${GEMMA4_DEV:-/opt/gemma4-dev}
LOG=/opt/flight-setup
mkdir -p "$LOG" /opt/models /opt/cache
# The 3050 flight's llama.cpp commit; gpu-llamacpp-g6-2b-q4_0/tpu.env records 77c4540ce, its 2026-10-26 runs.
LLAMA_CPP_COMMIT=${LLAMA_CPP_COMMIT:-7182ef71e}

step() {
  local name=$1; shift
  echo "$(date +%T) START $name"
  if ! "$@" > "$LOG/$name.log" 2>&1; then echo "$(date +%T) FAIL $name (see $LOG/$name.log)"; exit 1; fi
  echo "$(date +%T) END $name"
}

llamacpp()  { make -C "$R/gpu-llamacpp-g6-2b-q4_0" toolkit build download draft LLAMA_CPP_COMMIT="$LLAMA_CPP_COMMIT"; }
vllm()      { make -C "$R/gpu-vllm-g6-2b-q4w4a16emb4" pip install check drafter; }
sglang()    { make -C "$R/gpu-sglang-g6-2b-q4w4a16emb4" install check; }
mistralrs() { make -C "$R/gpu-mistralrs-g6-2b-q4_0" deps rust mmproj install; }
litert()    { make -C "$R/gpu-litertlm-g6-2b-int248" vulkan install download import
              make -C "$R/gpu-litertlm-g6-2b-q4w4a16emb4" fetch import MODEL_S3_URI="${MODEL_S3_URI:?set MODEL_S3_URI}"; }

STEPS=${*:-llamacpp vllm sglang mistralrs litert}
for s in $STEPS; do step "$s" "$s"; done
echo "$(date +%T) ready: python3 $R/gpu-vllm-g6-2b-q4w4a16emb4/flight/engines.py OUT_DIR ARMS"
