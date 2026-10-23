#!/bin/bash
# GSM8K (1,319, 2,048-token limit) and BFCL simple (400) on the PyTorch int4 engine through jev-tpu-v5e1/gen_eval.py,
# one request at a time, on the engine with the sliding-layer prompt-attention fix (w4attn.py). Same harness, sets and token limit as the llama.cpp 1650 Ti reference
# (local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-accuracy-1650ti, arm e2b-none).
# Expects openai_server.py on :8100 (started with --max-ctx 8192); nothing else may hold the card.
set -u
cd "$(dirname "$0")"
J=$HOME/gemma4-dev/jev-tpu-v5e1
for t in bfcl_simple gsm8k; do
  echo "$(date +%T) START $t gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" | tee -a driver.log
  (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8100 google/gemma-4-E2B-it $t "$OLDPWD/$t-pytorch-fixed.jsonl" --concurrency 1) > $t-pytorch-fixed.log 2>&1
  echo "$(date +%T) END $t rc=$? $(tail -1 $t-pytorch-fixed.log)" | tee -a driver.log
done
