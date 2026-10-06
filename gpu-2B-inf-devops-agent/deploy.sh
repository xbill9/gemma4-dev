#!/bin/bash
set -e

echo "Stopping and removing existing vllm-server container..."
docker stop vllm-server || true
docker rm vllm-server || true

echo "Actively clearing corrupt compiler cache on host to force correct JIT graph compilation..."
sudo rm -rf /var/tmp/neuron-compile-cache || true
sudo rm -rf /home/ubuntu/.cache/neuron/* || true


# Dynamic device mapping on the host
DEVICES=""
for dev in /dev/neuron*; do
    if [ -e "$dev" ]; then
        DEVICES="$DEVICES --device $dev"
    fi
done
if [ -z "$DEVICES" ]; then
    DEVICES="--device /dev/neuron0"
fi

echo "Launching vLLM container with devices: $DEVICES"

docker run -d --name vllm-server \
  --no-healthcheck \
  $DEVICES \
  --ipc=host \
  --restart no \
  -p 8080:8080 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e NEURON_CC_FLAGS="--model-type=gemma4 --enable-mixed-shapes=False --target=inf2 --hbm-scratchpad-page-size=1024" \
  -e NEURON_SCRATCHPAD_PAGE_SIZE=1024 \
  -e NEURON_CORES_PER_WORKER=2 \
  -e NEURON_COMPILER_WORKERS=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  -v /home/ubuntu/.cache/neuron:/root/.cache/neuron \
  -v /home/ubuntu/apply_all_patches.py:/apply_all_patches.py \
  -v /home/ubuntu/patch_and_run.sh:/patch_and_run.sh \
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 \
  bash /patch_and_run.sh
