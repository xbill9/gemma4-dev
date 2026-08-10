#!/bin/bash
# Fix-up pass for the 64K context row: the main sweep skipped ctx=65536 because
# input+output cannot fit inside max-model-len. Re-measure that row with the
# input clamped to (max_model_len - output - 32) per model, recorded as
# c65536-u<N>.json with the effective input length inside the result JSON
# (vllm bench serve records random_input_len). Reuses each model's boot ladder
# result from the main sweep (boot.json); skips models that never booted or
# whose max_model_len is too small to be interesting (< 16384).
set -u
USERS="1 2 4 8 16 32 64"
OUT=/var/tmp/sweep
IMG=vllm/vllm-tpu:nightly
OUTPUT_LEN=128

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

PROJECT=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
ACCESS_TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/hf-token/versions/latest:access" \
  | python3 -c 'import json,sys,base64; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())')
[ -z "$HF_TOKEN" ] && { log "FATAL: no HF token"; echo FIXUP-DONE; exit 1; }

serve_model() {
  sudo docker rm -f vllm-gemma4 vllm-sweep >/dev/null 2>&1
  sudo docker run --name vllm-sweep --privileged --net=host -d \
    -v /dev/shm:/dev/shm --shm-size 10gb \
    -e HF_HOME=/dev/shm -e HF_TOKEN="$HF_TOKEN" "$IMG" vllm serve "$1" \
    --max-model-len "$2" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 --max_num_batched_tokens 4096 \
    --disable_chunked_mm_input --limit-mm-per-prompt '{"image":0,"audio":0}' \
    >/dev/null || return 1
  for i in $(seq 1 150); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && return 0
    [ "$(sudo docker inspect -f '{{.State.Running}}' vllm-sweep 2>/dev/null)" = "true" ] || return 1
    sleep 10
  done
  return 1
}

for M in google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-12B-it; do
  MSHORT=$(basename "$M" | tr '[:upper:]' '[:lower:]')
  BOOT="$OUT/$MSHORT/boot.json"
  [ -s "$BOOT" ] || { log "no boot.json for $MSHORT; skipping"; continue; }
  ML=$(python3 -c "import json; print(json.load(open('$BOOT'))['max_model_len'])")
  [ "$ML" -lt 16384 ] && { log "$MSHORT max_model_len=$ML too small for a 64K-row fixup; skipping"; continue; }
  EFF=$(( ML - OUTPUT_LEN - 32 ))
  # Anything already measured at the clamped length is done.
  NEED=0
  for U in $USERS; do [ -s "$OUT/$MSHORT/c65536-u${U}.json" ] || NEED=1; done
  [ "$NEED" -eq 0 ] && { log "$MSHORT 64K row already complete"; continue; }
  log "booting $M (max-model-len=$ML) for 64K fixup, effective input=$EFF"
  serve_model "$M" "$ML" || { log "$MSHORT failed to boot for fixup"; continue; }
  for U in $USERS; do
    F="$OUT/$MSHORT/c65536-u${U}"
    [ -s "$F.json" ] && continue
    rm -f "$F.skip"
    NP=$U; [ "$NP" -lt 4 ] && NP=4; [ "$NP" -gt 64 ] && NP=64
    log "CELL $MSHORT ctx=65536(eff=$EFF) u=$U np=$NP"
    sudo timeout 1800 docker run --rm --net=host \
      -v /dev/shm:/dev/shm -v "$OUT:$OUT" \
      -e HF_HOME=/dev/shm -e HF_TOKEN="$HF_TOKEN" "$IMG" \
      vllm bench serve --model "$M" --dataset-name random \
      --random-input-len "$EFF" --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$NP" --max-concurrency "$U" --ignore-eos \
      --save-result --result-filename "$F.json" > "$F.log" 2>&1
    RC=$?
    if [ $RC -ne 0 ] || [ ! -s "$F.json" ]; then
      mv "$F.log" "$F.fail" 2>/dev/null
      log "FAIL $MSHORT 64K-fixup u=$U rc=$RC"
    fi
  done
done
log "fixup complete"
echo FIXUP-DONE
