#!/bin/bash
# serve.sh <model> [extra vllm flags...] -- (re)start the vLLM TPU server on :8000 with the
# pre-registered flags, then wait up to 25 minutes for /v1/models. Exit 0 when serving, 1 when
# the boot failed or timed out; the full container log is kept in /opt/jev-tpu-logs either way.
set -u
MODEL="$1"; shift
TAG=$(echo "$MODEL" | tr '/' '_')
mkdir -p /opt/jev-tpu-logs
docker rm -f vllm >/dev/null 2>&1
docker run -d --name vllm --privileged --net=host --shm-size 10gb -v /dev/shm:/dev/shm \
  -e HF_TOKEN="$(cat /root/hf_token)" -e HF_HOME=/dev/shm/hf -v /opt/jev-tpu:/work \
  vllm/vllm-tpu:nightly \
  vllm serve "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 \
    --generation-config vllm --limit-mm-per-prompt '{"image":0,"audio":0}' --enable-prefix-caching "$@" >/dev/null
START=$(date +%s)
while :; do
  if curl -sf localhost:8000/v1/models | grep -q '"id"'; then
    echo "READY $MODEL after $(( $(date +%s) - START ))s"; docker logs vllm > "/opt/jev-tpu-logs/$TAG.boot.log" 2>&1; exit 0
  fi
  if [ -z "$(docker ps -q -f name=vllm)" ] || [ $(( $(date +%s) - START )) -gt 1500 ]; then
    docker logs vllm > "/opt/jev-tpu-logs/$TAG.boot.log" 2>&1
    echo "FAILED $MODEL after $(( $(date +%s) - START ))s: $(grep -E 'Error|error|RESOURCE_EXHAUSTED|NotImplemented' "/opt/jev-tpu-logs/$TAG.boot.log" | tail -3 | tr '\n' ' ' | cut -c1-600)"
    docker rm -f vllm >/dev/null 2>&1; exit 1
  fi
  sleep 15
done
