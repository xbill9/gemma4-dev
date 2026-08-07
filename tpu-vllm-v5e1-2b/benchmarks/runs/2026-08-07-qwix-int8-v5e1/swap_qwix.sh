#!/usr/bin/env bash
# Swap the live vLLM container between the bf16 baseline and a qwix-quantized-weights arm.
#
# Runs ON THE TPU VM HOST. Same procedure as the KV-quant run's swap_arm.sh: the original
# container is stopped and renamed, never deleted, so reverting is `docker start` on the
# untouched original and no flag has to be retranscribed. Reverting is also warm — the original
# container keeps its own JAX compile cache on its filesystem layer.
#
# The image is pinned by ID, not by the :nightly tag — the tag moves, and a newer image pulled
# between arms would confound the comparison with an engine change.
#
# The HF token is read from Secret Manager into a local variable and passed straight to
# `docker run -e`. It is never echoed, never written to disk, and tracing is off for the whole
# section. Pre-flight checks that the fetch works BEFORE anything is stopped.
#
# Usage:  ./swap_qwix.sh forward <weight_qtype> [tile_size|-] [abstract]
#           e.g. ./swap_qwix.sh forward int8
#                ./swap_qwix.sh forward int8 - abstract     # use_abstract_model=true
#                ./swap_qwix.sh forward int4 128 abstract
#         ./swap_qwix.sh back
#
# The concrete path (no `abstract`) loads bf16 weights and quantizes them in place, so it needs
# room for both. Measured 2026-08-07: that OOMs on a v5e-1 even for E2B, whose bf16 weights fit.
set -euo pipefail

IMAGE=sha256:2a4a1f82793f748e02af54d77a62e590d34d2c9c68e833a8bb00d26a878a684c
MODEL=google/gemma-4-E2B-it
BASE=vllm-gemma4
BF16=vllm-gemma4-bf16
QUANT=vllm-gemma4-qwix

mode="${1:-}"

case "$mode" in
forward)
  QTYPE="${2:?usage: swap_qwix.sh forward <weight_qtype> [tile_size|-] [abstract]}"
  TILE="${3:-}"
  ABSTRACT="${4:-}"
  [ "$TILE" = "-" ] && TILE=""

  # qwix rules are the serialization of additional_config["quantization"]["qwix"]["rules"].
  # tile_size is omitted entirely when not asked for — a null would not mean the same thing.
  if [ -n "$TILE" ]; then
    RULE="{\"module_path\":\".*\",\"weight_qtype\":\"$QTYPE\",\"tile_size\":$TILE}"
  else
    RULE="{\"module_path\":\".*\",\"weight_qtype\":\"$QTYPE\"}"
  fi
  # use_abstract_model sits beside `rules`, not inside one — qwix_utils.py reads it from
  # additional_config["quantization"]["qwix"]["use_abstract_model"] (default False).
  if [ "$ABSTRACT" = "abstract" ]; then
    QWIX="{\"rules\":[$RULE],\"use_abstract_model\":true}"
  else
    QWIX="{\"rules\":[$RULE]}"
  fi
  ADDITIONAL_CONFIG="{\"quantization\":{\"qwix\":$QWIX}}"
  echo "== additional-config =="
  echo "$ADDITIONAL_CONFIG"

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
  curl -s --max-time 10 localhost:8000/metrics 2>/dev/null | grep '^vllm:cache_config_info' || echo "(endpoint not answering)"

  # Idempotent: a second forward (e.g. retrying with different rules) finds the baseline already
  # stopped and renamed from the first one. Preserving it is the point; renaming twice is not.
  if sudo docker ps -a --format '{{.Names}}' | grep -qx "$BASE"; then
    echo "== stopping and renaming baseline (NOT deleting) =="
    sudo docker stop "$BASE"
    sudo docker rename "$BASE" "$BF16"
  else
    echo "== baseline already preserved as $BF16; leaving it alone =="
    sudo docker ps -a --format '{{.Names}}\t{{.Status}}' | grep "$BF16" || {
      echo "FATAL: neither $BASE nor $BF16 exists. Refusing to proceed." >&2; exit 1; }
  fi

  echo "== starting qwix arm: weight_qtype=$QTYPE tile_size=${TILE:-none} =="
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
    --additional-config "$ADDITIONAL_CONFIG"
  unset HF_TOKEN_VALUE
  ;;

back)
  echo "== stopping qwix arm =="
  sudo docker stop "$QUANT" 2>/dev/null || true
  echo "== restoring untouched baseline =="
  sudo docker start "$BF16"
  sudo docker rename "$BF16" "$BASE"
  ;;

*)
  echo "usage: $0 forward <weight_qtype> [tile_size] | $0 back" >&2
  exit 2
  ;;
esac

echo "== containers =="
sudo docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}'
