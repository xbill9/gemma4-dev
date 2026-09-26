#!/bin/bash
# serve.sh <model> [extra vllm flags...] -- (re)start the vLLM TPU server on :8000 with the
# pre-registered flags, then wait up to $BOOT_TIMEOUT seconds for /v1/models. Exit 0 when serving, 1 when
# the boot failed or timed out; the full container log is kept in $LOGS either way.
# IMAGE and LOGS override the image and log directory (w4a16_probe.sh serves a patched image).
# BOOT_TIMEOUT (seconds, default 1500) bounds the wait; XLA_CACHE_DIR, when set, keeps vLLM's
# JAX compile cache on the host so the next boot of the same model skips recompiling.
# MM_LIMIT sets --limit-mm-per-prompt (default: every modality at 0); empty omits the flag.
set -u
IMAGE=${IMAGE:-vllm/vllm-tpu:nightly}
LOGS=${LOGS:-/opt/jev-tpu-logs}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-1500}
MM_LIMIT=${MM_LIMIT-'{"image":0,"audio":0,"video":0}'}
MM_ARGS=(); [ -n "$MM_LIMIT" ] && MM_ARGS=(--limit-mm-per-prompt "$MM_LIMIT")
CACHE_ARGS=()
if [ -n "${XLA_CACHE_DIR:-}" ]; then
  mkdir -p "$XLA_CACHE_DIR"
  CACHE_ARGS=(-e VLLM_XLA_CACHE_PATH=/xla-cache -v "$XLA_CACHE_DIR":/xla-cache)
fi
MODEL="$1"; shift
TAG=$(echo "$MODEL" | tr '/' '_')
mkdir -p "$LOGS"
docker rm -f vllm >/dev/null 2>&1
docker run -d --name vllm --privileged --net=host --shm-size 10gb -v /dev/shm:/dev/shm \
  -e HF_TOKEN="$(cat /root/hf_token)" -e HF_HOME=/dev/shm/hf -v /opt/jev-tpu:/work "${CACHE_ARGS[@]}" \
  "$IMAGE" \
  vllm serve "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 \
    --generation-config vllm "${MM_ARGS[@]}" --enable-prefix-caching "$@" >/dev/null
START=$(date +%s)
while :; do
  if curl -sf localhost:8000/v1/models | grep -q '"id"'; then
    echo "READY $MODEL after $(( $(date +%s) - START ))s"; docker logs vllm > "$LOGS/$TAG.boot.log" 2>&1; exit 0
  fi
  if [ -z "$(docker ps -q -f name=vllm)" ] || [ $(( $(date +%s) - START )) -gt "$BOOT_TIMEOUT" ]; then
    docker logs vllm > "$LOGS/$TAG.boot.log" 2>&1
    # Quote the last raised exception (skipping vLLM's generic wrapper), not every line that
    # mentions "error".
    if [ -z "$(docker ps -q -f name=vllm)" ]; then WHY="exited"; else WHY="timed out after ${BOOT_TIMEOUT}s"; fi
    echo "FAILED $MODEL ($WHY) after $(( $(date +%s) - START ))s: $(grep -av tpu_info "$LOGS/$TAG.boot.log" | grep -aoE '[A-Za-z_.]*(Error|Exception): .*|RESOURCE_EXHAUSTED.*' | grep -av 'See root cause above' | tail -1 | cut -c1-400)"
    docker rm -f vllm >/dev/null 2>&1; exit 1
  fi
  sleep 15
done
