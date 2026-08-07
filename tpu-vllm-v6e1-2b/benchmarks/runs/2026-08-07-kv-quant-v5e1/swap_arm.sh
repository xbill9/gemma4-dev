#!/usr/bin/env bash
# Swap the live vLLM container between the bf16 baseline and a quantized-KV arm.
#
# Runs ON THE TPU VM HOST. Implements the procedure in ROLLBACK.md: the original container is
# stopped and renamed, never deleted, so reverting is `docker start` on the untouched original
# and no flag has to be retranscribed.
#
# The image is pinned by ID, not by the :nightly tag — the tag moves, and a newer image pulled
# between arms would confound the comparison with an engine change.
#
# The HF token is read from Secret Manager into a local variable and passed straight to
# `docker run -e`. It is never echoed, never written to disk, and tracing is off for the whole
# section. Pre-flight checks that the fetch works BEFORE anything is stopped, so a missing IAM
# grant cannot strand the rig with no serving container.
#
# Usage:  ./swap_arm.sh forward <kv-dtype>   # e.g. ./swap_arm.sh forward fp8_e4m3
#         ./swap_arm.sh back
set -euo pipefail

IMAGE=sha256:2a4a1f82793f748e02af54d77a62e590d34d2c9c68e833a8bb00d26a878a684c
MODEL=google/gemma-4-E2B-it
BASE=vllm-gemma4
BF16=vllm-gemma4-bf16
QUANT=vllm-gemma4-quant

mode="${1:-}"

show_cache_config() {
  curl -s --max-time 10 localhost:8000/metrics 2>/dev/null | grep '^vllm:cache_config_info' || echo "(endpoint not answering)"
}

case "$mode" in
forward)
  KV_DTYPE="${2:?usage: swap_arm.sh forward <kv-dtype>}"

  echo "== pre-flight: secret access =="
  set +x
  if ! HF_TOKEN_VALUE="$(gcloud secrets versions access latest --secret=hf-token 2>/dev/null)"; then
    echo "FATAL: cannot read hf-token from Secret Manager. Nothing stopped; rig untouched." >&2
    exit 1
  fi
  if [ -z "$HF_TOKEN_VALUE" ]; then
    echo "FATAL: hf-token is empty. Nothing stopped; rig untouched." >&2
    exit 1
  fi
  echo "ok (${#HF_TOKEN_VALUE} chars)"

  echo "== pre-flight: image present =="
  sudo docker image inspect "$IMAGE" --format 'image ok: {{.Id}}' >/dev/null
  echo "ok"

  echo "== baseline cache_config_info (bf16) =="
  show_cache_config

  echo "== stopping and renaming baseline (NOT deleting) =="
  sudo docker stop "$BASE"
  sudo docker rename "$BASE" "$BF16"

  echo "== starting quantized arm: kv-cache-dtype=$KV_DTYPE =="
  sudo docker rm -f "$QUANT" 2>/dev/null || true
  sudo docker run -d \
    --name "$QUANT" \
    --privileged \
    --network host \
    --shm-size 10737418240 \
    -v /dev/shm:/dev/shm \
    -e HF_HOME=/dev/shm \
    -e HF_TOKEN="$HF_TOKEN_VALUE" \
    "$IMAGE" \
    vllm serve "$MODEL" \
    --max-model-len 16384 \
    --tensor-parallel-size 1 \
    --disable_chunked_mm_input \
    --max_num_batched_tokens 4096 \
    --limit-mm-per-prompt '{"image":4,"audio":1}' \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --kv-cache-dtype "$KV_DTYPE"
  unset HF_TOKEN_VALUE
  ;;

back)
  echo "== stopping quantized arm =="
  sudo docker stop "$QUANT" 2>/dev/null || true
  echo "== restoring untouched baseline =="
  sudo docker start "$BF16"
  sudo docker rename "$BF16" "$BASE"
  ;;

*)
  echo "usage: $0 forward <kv-dtype> | $0 back" >&2
  exit 2
  ;;
esac

echo "== containers =="
sudo docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}'
