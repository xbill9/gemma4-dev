#!/bin/bash
# the PyTorch TPU backend batched decode sweep (lockstep static batching) on gemma-4-E2B-it.
# Run on the sweep VM AFTER the vLLM sweep: requires the PyTorch TPU backend stack
# installed for python3.12 and /tmp/tpu_decode_bench.py in place.
# The vLLM containers must be stopped first — torch and vLLM cannot share the TPU.
set -u
sudo docker rm -f vllm-sweep vllm-gemma4 >/dev/null 2>&1 || true
sleep 5  # let libtpu release the chip
for B in 1 2 4 8 16 32 64; do
  echo "===== BATCH=$B ====="
  python3.12 /tmp/tpu_decode_bench.py --model google/gemma-4-E2B-it --batch "$B" 2>&1 \
    | grep -E "Loaded|Warmup done|compiled static-shape|Traceback|Error" || true
done
echo BATCH-SWEEP-DONE
