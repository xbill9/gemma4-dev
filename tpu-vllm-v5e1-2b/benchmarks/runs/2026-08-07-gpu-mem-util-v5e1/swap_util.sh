#!/usr/bin/env bash
# Swap the live vLLM container between the 0.92 baseline and a --gpu-memory-utilization arm.
#
# Runs ON THE TPU VM HOST. Same discipline as the KV-quant run's swap_arm.sh: the baseline
# container is stopped and renamed, never deleted, so reverting is `docker start` on the
# untouched original and no flag has to be retranscribed. Its JAX compile cache survives, so
# the revert is warm.
#
# The image is pinned by ID, not by the :nightly tag — the tag moves, and a newer image pulled
# between arms would confound a memory comparison with an engine change.
#
# The HF token is read from Secret Manager into a local variable and passed straight to
# `docker run -e`. It is never echoed, never written to disk, and tracing is off for the whole
# section. Pre-flight checks that the fetch works BEFORE anything is stopped, so a missing IAM
# grant cannot strand the rig with no serving container.
#
# Container names are per-configuration, not a shared "quant" slot. The qwix run lost arm 1's
# full log because a second arm reused the same container name and `docker rm -f`'d it.
#
# Usage:  ./swap_util.sh forward <util>   # e.g. ./swap_util.sh forward 0.95
#         ./swap_util.sh back
set -euo pipefail

IMAGE=sha256:2a4a1f82793f748e02af54d77a62e590d34d2c9c68e833a8bb00d26a878a684c
MODEL=google/gemma-4-E2B-it
BASE=vllm-gemma4
PARK=vllm-gemma4-park-092

mode="${1:-}"

show_cache_config() {
  curl -s --max-time 10 localhost:8000/metrics 2>/dev/null \
    | grep -o 'gpu_memory_utilization="[^"]*"\|kv_cache_size_tokens="[^"]*"\|num_gpu_blocks="[^"]*"\|cache_dtype="[^"]*"' \
    || echo "(endpoint not answering)"
}

case "$mode" in
forward)
  UTIL="${2:?usage: swap_util.sh forward <util>}"
  ARM="vllm-gemma4-util${UTIL//./}"

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

  echo "== pre-flight: park name free =="
  if sudo docker inspect "$PARK" >/dev/null 2>&1; then
    echo "FATAL: $PARK already exists — a previous swap did not complete its 'back'." >&2
    echo "Resolve by hand before proceeding. Nothing stopped; rig untouched." >&2
    exit 1
  fi
  echo "ok"

  echo "== baseline config (0.92) =="
  show_cache_config

  echo "== stopping and renaming baseline (NOT deleting) =="
  sudo docker stop "$BASE"
  sudo docker rename "$BASE" "$PARK"

  echo "== starting arm: gpu-memory-utilization=$UTIL as $ARM =="
  sudo docker rm -f "$ARM" 2>/dev/null || true
  sudo docker run -d \
    --name "$ARM" \
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
    --gpu-memory-utilization "$UTIL"
  unset HF_TOKEN_VALUE
  ;;

back)
  echo "== stopping every util arm =="
  for c in $(sudo docker ps -q --filter "name=vllm-gemma4-util"); do
    sudo docker stop "$c"
  done
  echo "== restoring untouched baseline =="
  sudo docker start "$PARK"
  sudo docker rename "$PARK" "$BASE"
  ;;

*)
  echo "usage: $0 forward <util> | $0 back" >&2
  exit 2
  ;;
esac

echo "== containers =="
sudo docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}'
