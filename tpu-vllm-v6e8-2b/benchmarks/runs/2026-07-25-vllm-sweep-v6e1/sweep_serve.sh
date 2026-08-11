#!/bin/bash
# vLLM serving sweep on a v6e-1: models x concurrency x context length.
# Runs on the VM under nohup; writes one JSON (or .fail/.skip marker) per cell
# to /var/tmp/sweep/<model-short>/c<ctx>-u<users>.*, resumable (existing .json
# cells are not rerun). Prints SWEEP-DONE at the end.
set -u
MODELS="google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-12B-it"
USERS="1 2 4 8 16 32 64"
CTXS="8 16 32 64 128 256 512 2048 4096 8192 16384 32768 65536"
OUT=/var/tmp/sweep
IMG=vllm/vllm-tpu:nightly
OUTPUT_LEN=128
mkdir -p "$OUT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# --- HF token from Secret Manager via the metadata server (never logged) -----
PROJECT=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
ACCESS_TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/hf-token/versions/latest:access" \
  | python3 -c 'import json,sys,base64; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())')
if [ -z "$HF_TOKEN" ]; then log "FATAL: no HF token"; echo SWEEP-DONE; exit 1; fi

serve_model() { # $1=model $2=max_len -> 0 when /health responds
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

for M in $MODELS; do
  MSHORT=$(basename "$M" | tr '[:upper:]' '[:lower:]')
  mkdir -p "$OUT/$MSHORT"
  # Boot ladder: shrink max-model-len until the model fits KV+weights in HBM.
  MAXLEN_USED=0
  for ML in 65536 32768 16384 8192; do
    log "booting $M with max-model-len=$ML"
    T0=$(date +%s)
    if serve_model "$M" "$ML"; then
      MAXLEN_USED=$ML
      log "$M healthy at max-model-len=$ML in $(( $(date +%s) - T0 ))s"
      echo "{\"model\": \"$M\", \"max_model_len\": $ML, \"time_to_healthy_s\": $(( $(date +%s) - T0 ))}" > "$OUT/$MSHORT/boot.json"
      break
    fi
    log "$M failed to become healthy at max-model-len=$ML"
    sudo docker logs vllm-sweep 2>&1 | tail -30 > "$OUT/$MSHORT/boot-fail-$ML.log" || true
  done
  if [ "$MAXLEN_USED" -eq 0 ]; then
    log "$M unbootable on this chip; recording and moving on"
    echo "unbootable" > "$OUT/$MSHORT/UNBOOTABLE"
    continue
  fi

  for CTX in $CTXS; do
    # Cell budget scales down as prefill cost scales up.
    if   [ "$CTX" -le 512 ];  then NP_MULT=8; NP_MIN=16; NP_MAX=128; TMO=600
    elif [ "$CTX" -le 8192 ]; then NP_MULT=3; NP_MIN=8;  NP_MAX=96;  TMO=900
    else                           NP_MULT=1; NP_MIN=4;  NP_MAX=64;  TMO=1800; fi
    for U in $USERS; do
      F="$OUT/$MSHORT/c${CTX}-u${U}"
      [ -s "$F.json" ] && { log "skip existing $F"; continue; }
      if [ $(( CTX + OUTPUT_LEN + 32 )) -gt "$MAXLEN_USED" ]; then
        echo "ctx+output exceeds max_model_len=$MAXLEN_USED" > "$F.skip"
        log "SKIP $MSHORT ctx=$CTX u=$U (exceeds max-model-len)"
        continue
      fi
      NP=$(( U * NP_MULT )); [ "$NP" -lt "$NP_MIN" ] && NP=$NP_MIN; [ "$NP" -gt "$NP_MAX" ] && NP=$NP_MAX
      log "CELL $MSHORT ctx=$CTX u=$U np=$NP"
      sudo timeout "$TMO" docker run --rm --net=host \
        -v /dev/shm:/dev/shm -v "$OUT:$OUT" \
        -e HF_HOME=/dev/shm -e HF_TOKEN="$HF_TOKEN" "$IMG" \
        vllm bench serve --model "$M" --dataset-name random \
        --random-input-len "$CTX" --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NP" --max-concurrency "$U" --ignore-eos \
        --save-result --result-filename "$F.json" > "$F.log" 2>&1
      RC=$?
      if [ $RC -ne 0 ] || [ ! -s "$F.json" ]; then
        mv "$F.log" "$F.fail" 2>/dev/null
        log "FAIL $MSHORT ctx=$CTX u=$U rc=$RC"
      fi
    done
  done
done
log "sweep complete"
echo SWEEP-DONE
