#!/usr/bin/env bash
# Arm E: the EXACT recommended config from the article, incl. the compile-cache mount.
sudo docker rm -f vllm-rec >/dev/null 2>&1 || true
sudo docker run -d --name vllm-rec --privileged --net=host \
  -v /dev/shm:/dev/shm --shm-size 10gb \
  -v /home/xbill/.cache/vllm:/root/.cache/vllm \
  -e HF_HOME=/dev/shm -e HF_TOKEN="$(cat ~/.hf_token)" \
  vllm/vllm-tpu:nightly \
  vllm serve google/gemma-4-E2B-it \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --disable-chunked-mm-input \
    --limit-mm-per-prompt '{"image":4,"audio":1}' \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 >/dev/null
echo "started vllm-rec"
