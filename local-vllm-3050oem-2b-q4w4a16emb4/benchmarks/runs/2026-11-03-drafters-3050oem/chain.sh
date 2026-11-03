#!/bin/bash
# Waits for the vLLM accuracy run to release the card, then runs drafters.py on both engines,
# one after the other: vLLM (no drafter, bf16 and W4A16 at 1-4 tokens, no drafter again),
# then llama.cpp (no drafter, exact Q4_0, Q8_0 and bf16 at 1-4 tokens, no drafter again).
set -u
cd "$(dirname "$0")"
ACC=../2026-11-03-accuracy-3050oem/driver.log
until grep -qE "^DONE|died" $ACC; do sleep 30; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 5; done
say() { echo "$(date +%F' '%T) $*" >> chain.log; }
say START vllm
python3 drafters.py vllm . none,bf16-n1,bf16-n2,bf16-n3,bf16-n4,w4a16-n1,w4a16-n2,w4a16-n3,w4a16-n4,none-again > vllm.out 2>&1
say END vllm rc=$?
say START llama
python3 drafters.py llama . none,q4x-n1,q4x-n2,q4x-n3,q4x-n4,q8-n1,q8-n2,q8-n3,q8-n4,bf16-n1,bf16-n2,bf16-n3,bf16-n4,none-again > llama.out 2>&1
say END llama rc=$?
say DONE
