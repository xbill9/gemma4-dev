#!/usr/bin/env bash
set -euo pipefail
docker rm -f vllm-neuron 2>/dev/null || true
docker run -d --name vllm-neuron --restart unless-stopped --ipc=host \
  --device=/dev/neuron0 \
  -v /opt/qat-e2b/model:/workspace/real-gemma4-E2B-it:ro \
  -v /opt/qat-e2b/qat_e2b_prefill.pt:/workspace/qat_pre.pt:ro \
  -v /opt/qat-e2b/qat_e2b_decode.pt:/workspace/qat_dec.pt:ro \
  -e KV_MAX=128 -e KV_BUCKET=32 \
  -e KV_PRE_OUT=/workspace/qat_pre.pt -e KV_DEC_OUT=/workspace/qat_dec.pt \
  -e SELFTEST=1 -v /opt/qat-e2b/optb_server_qat.py:/app/optb_server_slim.py:ro \
  -p 8080:8080 \
  docker.io/xbill9/gemma4-optb:slim
