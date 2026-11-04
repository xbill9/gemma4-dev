#!/bin/bash
# After the mistral.rs chain: SGLang (no drafter, W4A16 drafter at 1-4 tokens) and vLLM as the same-day
# reference, through the all-engine runner.
F=/home/xbill/gemma4-dev/local-mistralrs-3050oem-2b-q4_0/benchmarks/runs/chain-2026-11-04.out
until grep -q "accuracy exit" $F; do sleep 30; done
while pgrep -x mistralrs >/dev/null; do sleep 2; done
cd /home/xbill/gemma4-dev/local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem
python3 engines.py /home/xbill/gemma4-dev/local-sglang-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-04-engines-3050oem \
  sglang,sglang-d1,sglang-d2,sglang-d3,sglang-d4,vllm,vllm-d4
echo "engines exit $?"
